"""RGB-derived MoGe-2 history for the two frozen FMB forecast windows.

One metric-depth map is inferred for each historical RGB frame. No future frame,
annotation or MolmoMotion forecast is read while preparing these inputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, digest, verify_frozen, write_json


MODEL_FILE = ROOT.parent / "models/moge-2-vitl/model.pt"
MOGE_SOURCE = ROOT.parent / "third_party/MoGe"
EXPECTED_MODEL_SHA256 = "3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"


def extract_depth(model, frame_bgr: np.ndarray, fx_candidate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).cuda()
    fov_x = float(np.degrees(2 * np.arctan(256 / (2 * fx_candidate))))
    with torch.inference_mode():
        output = model.infer(tensor.float() / 255, num_tokens=1200,
                             use_fp16=True, apply_mask=False, fov_x=fov_x)
    depth = output["depth"].float().cpu().numpy()
    points = output["points"].float().cpu().numpy()
    intrinsics = output["intrinsics"].float().cpu().numpy()
    intrinsics[0] *= 256
    intrinsics[1] *= 192
    depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
    points = cv2.resize(points, (256, 256), interpolation=cv2.INTER_LINEAR)
    intrinsics[1] *= 256 / 192
    if depth.shape != (256, 256) or points.shape != (256, 256, 3):
        raise ValueError("Unexpected MoGe output geometry")
    return depth, points, intrinsics


def local_depth(depth: np.ndarray, uv: np.ndarray) -> tuple[float, int]:
    x, y = np.rint(uv).astype(int)
    if not (2 <= x < 254 and 2 <= y < 254):
        raise ValueError("Point too close to image edge")
    patch = depth[y-2:y+3, x-2:x+3]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if len(valid) < 15:
        raise ValueError("Too few valid MoGe depths at historical point")
    return float(np.median(valid)), len(valid)


def unproject(uv: np.ndarray, z: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.stack(((uv[..., 0] - k[0, 2]) * z / k[0, 0],
                     (uv[..., 1] - k[1, 2]) * z / k[1, 1], z), axis=-1)


def internal_consistency(depth: np.ndarray, points: np.ndarray, k: np.ndarray) -> float:
    yy, xx = np.indices(depth.shape)
    uv = np.stack((xx, yy), axis=-1)
    derived = unproject(uv, depth, k)
    error = np.linalg.norm(derived - points, axis=2)
    valid = np.isfinite(error) & np.isfinite(depth) & (depth > 0)
    valid[:8] = False
    valid[-8:] = False
    valid[:, :8] = False
    valid[:, -8:] = False
    return float(np.median(error[valid]) * 1000)


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if digest(MODEL_FILE) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("MoGe-2 checkpoint hash mismatch")
    sys.path.insert(0, str(MOGE_SOURCE))
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(MODEL_FILE).cuda().eval()
    for name, run in RUNS.items():
        verify_frozen(run)
        dest = run / "moge2_history_v1"
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / "manifest.json").exists():
            previous = json.loads((dest / "manifest.json").read_text(encoding="utf8"))
            if (digest(run / "history_moge2_raw_3d.npy") != previous["raw_xyz_sha256"] or
                digest(run / "history_moge2_history_scaled_3d.npy") != previous["scaled_xyz_sha256"] or
                digest(dest / "moge2_history_maps.npz") != previous["maps_sha256"]):
                raise RuntimeError(f"Previous MoGe history is incomplete or changed: {dest}")
            print(name, "already prepared and hashes verified", flush=True)
            continue
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source_path = ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"]
        if digest(source_path) != config["source_sha256"]:
            raise RuntimeError("Source FMB file hash mismatch")
        source = np.load(source_path, allow_pickle=True).item()
        uv = np.load(run / "history_points_2d.npy").astype(float)
        k_nominal = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        depths, points, estimated_k = [], [], []
        sampled = np.empty((3, 8), dtype=float)
        counts = np.empty((3, 8), dtype=int)
        for i, step in enumerate(config["history_steps"]):
            frame = source["obs/side_1"][step]  # Stored BGR.
            d, p, k_model = extract_depth(model, frame, k_nominal[0, 0])
            depths.append(d.astype("float32"))
            points.append(p.astype("float32"))
            estimated_k.append(k_model.astype("float32"))
            for j in range(8):
                sampled[i, j], counts[i, j] = local_depth(d, uv[i, j])
            print(name, step, "median peg depth m", float(np.median(sampled[i])), flush=True)
        raw_xyz = unproject(uv, sampled, k_nominal)
        sensor = np.load(run / "history_sensor_3d.npy").astype(float)
        scale = float(np.median(sensor[..., 2] / sampled))
        scaled_xyz = unproject(uv, sampled * scale, k_nominal)
        static_roi = (slice(20, 80), slice(20, 80))
        sensor_static_medians = []
        moge_static_medians = []
        for step, d in zip(config["history_steps"], depths):
            sensor_patch = source["obs/side_1_depth"][step][static_roi].astype(float) * 1e-4
            model_patch = d[static_roi]
            sensor_static_medians.append(float(np.median(sensor_patch[sensor_patch > 0])))
            moge_static_medians.append(float(np.median(model_patch[np.isfinite(model_patch) & (model_patch > 0)])))
        np.savez_compressed(dest / "moge2_history_maps.npz",
                            steps=np.asarray(config["history_steps"]), depth=np.asarray(depths),
                            points=np.asarray(points), K_model=np.asarray(estimated_k),
                            sampled_depth=sampled.astype("float32"), valid_patch_count=counts)
        np.save(run / "history_moge2_raw_3d.npy", raw_xyz.astype("float32"))
        np.save(run / "history_moge2_history_scaled_3d.npy", scaled_xyz.astype("float32"))
        report = {
            "episode": name, "status": "PREPARED_WITH_CALIBRATION_UNCERTAINTY",
            "model": "MoGe-2 ViT-L", "model_checkpoint_sha256": EXPECTED_MODEL_SHA256,
            "MoGe_source_commit": "74fbce054ebed49800de42d0ad0e83495065719a",
            "preprocessing": "source BGR to RGB; inverse 256x256->256x192 aspect stretch; candidate horizontal FOV from K_nominal fx; outputs depth/points resized to 256x256",
            "history_steps_only": config["history_steps"], "future_frames_read": False,
            "FMB_source_sha256": config["source_sha256"],
            "depth_semantics": "MoGe predicted metric camera Z as used in prior metric-depth experiment",
            "nominal_K_status": "SUPPORTED_NOT_EXACT",
            "sampling": "5x5 median of positive finite predicted depth at frozen 2D points",
            "valid_patch_counts": counts.tolist(),
            "raw_depth_median_m": float(np.median(sampled)),
            "sensor_depth_median_m": float(np.median(sensor[..., 2])),
            "raw_median_abs_depth_gap_to_sensor_mm": float(np.median(np.abs(sampled - sensor[..., 2])) * 1000),
            "single_history_scale_to_sensor": scale,
            "scaled_median_abs_depth_gap_to_sensor_mm": float(np.median(np.abs(sampled * scale - sensor[..., 2])) * 1000),
            "scale_fit_uses_only_historical_sensor_depth": True,
            "fixed_background_patch_xyxy": [20, 20, 80, 80],
            "fixed_patch_sensor_depth_median_by_frame_m": sensor_static_medians,
            "fixed_patch_moge_depth_median_by_frame_m": moge_static_medians,
            "fixed_patch_sensor_median_range_mm": float(np.ptp(sensor_static_medians) * 1000),
            "fixed_patch_moge_median_range_mm": float(np.ptp(moge_static_medians) * 1000),
            "fixed_patch_caveat": "A visually static tabletop ROI; no independently registered geometry ground truth",
            "raw_xyz_sha256": digest(run / "history_moge2_raw_3d.npy"),
            "scaled_xyz_sha256": digest(run / "history_moge2_history_scaled_3d.npy"),
            "maps_sha256": digest(dest / "moge2_history_maps.npz"),
            "MoGe_output_K_px_by_frame": [k.tolist() for k in estimated_k],
            "MoGe_K_interpretation": "Horizontal FOV was supplied from candidate nominal K; these output K values are conditioned, not independently estimated camera calibration",
            "MoGe_depth_points_Z_max_abs_m_by_frame": [float(np.max(np.abs(p[..., 2] - d)))
                                                        for p, d in zip(points, depths)],
            "MoGe_points_vs_depth_plus_model_K_median_mm_by_frame": [
                internal_consistency(d, p, k) for d, p, k in zip(depths, points, estimated_k)],
            "internal_consistency_caveat": "Checks pinhole interpretation after image resize; low residual does not establish physical metric accuracy",
        }
        write_json(dest / "manifest.json", report)
        print(json.dumps({"episode": name, "scale": scale,
                          "raw_gap_mm": report["raw_median_abs_depth_gap_to_sensor_mm"],
                          "scaled_gap_mm": report["scaled_median_abs_depth_gap_to_sensor_mm"]}, indent=2))


if __name__ == "__main__":
    main()
