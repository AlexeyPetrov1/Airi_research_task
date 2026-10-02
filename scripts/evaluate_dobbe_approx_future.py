"""Track sealed cup points in future HoNY RGB-D after prediction is frozen.

All 3D predictions and GT use the optical frame of history frame 98. Missing or
unreliable measured-depth samples remain NaN and are excluded from metrics.
This is an exploratory transferred-K evaluation, not a validated calibration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from inspect_hony_scenes import ROOT, liblzfse
from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, unproject


RUN = ROOT / "runs/dobbe_rgbd_study/approx_history"
RAW = ROOT / "data/dobbe_oxe/target_raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def red_surface_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 65, 25), (14, 255, 255))
    mask |= cv2.inRange(hsv, (167, 65, 25), (179, 255, 255))
    return cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))


def metric(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    errors = np.linalg.norm(pred - gt, axis=2)
    errors[~valid] = np.nan
    finite = np.isfinite(errors)
    final = finite[:, -1]
    observed_ade = float(np.nanmean(errors)) if finite.any() else None
    observed_fde = float(np.nanmean(errors[:, -1])) if final.any() else None
    return {"valid_point_timestamps": int(finite.sum()),
            "valid_final_points": int(final.sum()),
            "ADE_3D_m_full_8x30": observed_ade if finite.all() else None,
            "FDE_3D_m_full_8_points": observed_fde if final.all() else None,
            "observed_only_ADE_3D_m": observed_ade,
            "observed_only_FDE_3D_m": observed_fde,
            "error_by_future_frame_m": [
                float(np.nanmean(errors[:, t])) if finite[:, t].any() else None
                for t in range(errors.shape[1])]}


def resample_model(prediction: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    # The model's 15 Hz cadence is an assumption, documented in the freeze.
    source_times = np.arange(31, dtype=float) / 15.0
    target_times = np.arange(1, 31, dtype=float) / 30.0
    source = np.concatenate((anchor[:, None, :], prediction), axis=1)
    target = np.empty_like(prediction)
    for point in range(8):
        for axis in range(3):
            target[point, :, axis] = np.interp(
                target_times, source_times, source[point, :, axis])
    return target


def main() -> None:
    freeze = json.loads((RUN / "input_freeze.json").read_text(encoding="utf-8"))
    model = json.loads((RUN / "model_run.json").read_text(encoding="utf-8"))
    pred_path = RUN / "prediction_15hz.npy"
    if not (model["status"] == "COMPLETE" and model["success"]
            and sha256(pred_path) == model["prediction_sha256"]):
        raise RuntimeError("Prediction must be complete and frozen before future GT")
    for name, expected in freeze["sha256"].items():
        if sha256(RUN / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    prediction = np.load(pred_path).astype(np.float64)
    assert prediction.shape == (8, 30, 3)
    anchor_history = np.load(RUN / "points_3d_history_candidate.npy")
    anchor_uv = np.load(RUN / "points_2d_at_t0_candidate.npy")
    assert anchor_history.shape == (3, 8, 3) and anchor_uv.shape == (8, 2)
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    k = np.asarray(manifest["K_256x256"], dtype=np.float64)
    history = freeze["history_frames"]
    future = list(range(freeze["future_evaluation_only"][0],
                        freeze["future_evaluation_only"][1] + 1))
    assert history == [96, 97, 98] and future == list(range(99, 129))

    video = cv2.VideoCapture(str(RAW / "compressed_video_h264.mp4"))
    future_rgb = []
    for frame_id in future:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = video.read()
        if not ok or image.shape[:2] != (256, 256):
            raise RuntimeError(f"Missing future RGB frame {frame_id}")
        future_rgb.append(image)
    video.release()
    depth = np.frombuffer(liblzfse.decompress(
        (RAW / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)[:future[-1]+1]
    labels = json.loads((RAW / "labels.json").read_text(encoding="utf-8"))
    pose = np.repeat(np.eye(4)[None], future[-1]+1, axis=0)
    pose[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(future[-1]+1)]).as_matrix()
    pose[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(future[-1]+1)]

    gt = np.full((8, 30, 3), np.nan, dtype=np.float64)
    uv = np.full((8, 30, 2), np.nan, dtype=np.float64)
    valid = np.zeros((8, 30), dtype=bool)
    diagnostics = []
    previous_gray = cv2.cvtColor(cv2.imread(str(RUN / "rgb_0098.png")),
                                 cv2.COLOR_BGR2GRAY)
    previous_uv = anchor_uv.reshape(8, 1, 2).astype(np.float32)
    active = np.ones(8, dtype=bool)
    ref_pose = pose[history[-1]]
    for t, (frame_id, image) in enumerate(zip(future, future_rgb)):
        current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        next_uv, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, previous_uv, None,
            winSize=(17, 17), maxLevel=3)
        return_uv, back_ok, _ = cv2.calcOpticalFlowPyrLK(
            current_gray, previous_gray, next_uv, None,
            winSize=(17, 17), maxLevel=3)
        mask = red_surface_mask(image)
        point_status = []
        for point in range(8):
            if not active[point]:
                point_status.append("track_lost_previous")
                continue
            if not (forward_ok[point, 0] and back_ok[point, 0]):
                active[point] = False
                point_status.append("lk_failed")
                continue
            if np.linalg.norm(previous_uv[point, 0] - return_uv[point, 0]) > 1.5:
                active[point] = False
                point_status.append("lk_round_trip_failed")
                continue
            u, v = next_uv[point, 0]
            if not (2 <= u <= 253 and 2 <= v <= 253):
                active[point] = False
                point_status.append("outside_rgb")
                continue
            uv[point, t] = [u, v]
            if mask[round(v), round(u)] == 0:
                point_status.append("outside_red_cup")
                continue
            z, spread = depth_at(depth[frame_id], float(u), float(v))
            if not np.isfinite(z) or spread > .03:
                point_status.append("invalid_measured_depth")
                continue
            optical = unproject(np.array([[u, v]], dtype=np.float64),
                                np.array([z]), k, np.zeros(4), "z")[0]
            label_xyz = optical @ CAMERA_TO_LABEL.T
            world = label_xyz @ pose[frame_id, :3, :3].T + pose[frame_id, :3, 3]
            in_ref_label = (world - ref_pose[:3, 3]) @ ref_pose[:3, :3]
            gt[point, t] = in_ref_label @ CAMERA_TO_LABEL
            valid[point, t] = True
            point_status.append("valid")
        diagnostics.append({"frame": frame_id, "valid_points": int(valid[:, t].sum()),
                            "point_status": point_status})
        previous_gray = current_gray
        previous_uv = next_uv

    np.save(RUN / "future_rgb_uv.npy", uv.astype(np.float32))
    np.save(RUN / "future_3d_gt_candidate.npy", gt.astype(np.float32))
    np.save(RUN / "future_gt_valid.npy", valid)
    panels = []
    for frame_id in (99, 105, 110, 115, 120, 128):
        t = frame_id - 99
        canvas = future_rgb[t].copy()
        for point in range(8):
            if not np.isfinite(uv[point, t]).all():
                continue
            u, v = np.rint(uv[point, t]).astype(int)
            color = (0, 255, 0) if valid[point, t] else (0, 220, 255)
            cv2.circle(canvas, (u, v), 4, color, -1)
            cv2.putText(canvas, str(point), (u+4, v-4),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        cv2.putText(canvas, f"{frame_id}: {int(valid[:, t].sum())}/8 depth GT",
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .43,
                    (255, 255, 255), 1)
        panels.append(canvas)
    cv2.imwrite(str(RUN / "future_tracking_review.jpg"),
                np.vstack((np.hstack(panels[:3]), np.hstack(panels[3:]))))
    anchor = anchor_history[-1].astype(np.float64)
    velocity = anchor_history[-1] - anchor_history[-2]
    steps = np.arange(1, 31, dtype=float)
    stationary = np.broadcast_to(anchor[:, None, :], (8, 30, 3)).copy()
    constant_velocity = anchor[:, None, :] + steps[None, :, None] * \
        velocity[:, None, :]
    aligned = resample_model(prediction, anchor)
    methods = {"MolmoMotion_index_aligned": prediction,
               "MolmoMotion_15_to_30Hz_assumed": aligned,
               "Stationary": stationary,
               "Constant_velocity_30Hz": constant_velocity}
    metrics = {name: metric(points, gt, valid) for name, points in methods.items()}
    report = {"status": "EXPLORATORY_APPROXIMATE_RGBD_GT",
              "calibration_status": "main-scene K not independently validated",
              "prediction_sha256": model["prediction_sha256"],
              "future_first_last": [future[0], future[-1]],
              "K_source": manifest["K_source"],
              "depth_semantics": manifest["depth_semantics"],
              "coordinate_frame": manifest["history_3d_frame"],
              "valid_point_timestamps": int(valid.sum()),
              "possible_point_timestamps": 240,
              "valid_final_points": int(valid[:, -1].sum()),
              "full_eight_point_30_frame_gt": bool(valid.all()),
              "per_frame": diagnostics,
              "metrics": metrics,
              "timing_note": "Index-aligned and assumed 15-to-30 Hz alignments are both diagnostics; no exact model timestamp metadata."}
    (RUN / "future_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    for name, result in metrics.items():
        data = [np.nan if x is None else x * 100 for x in
                result["error_by_future_frame_m"]]
        ax.plot(np.arange(1, 31), data, label=name)
    ax.set(xlabel="Future RGB frame after H2", ylabel="Observed-point error (cm)",
           title="Exploratory Dobb-E red-cup forecast, transferred K")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.savefig(RUN / "future_error_t.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"valid": int(valid.sum()),
                      "final_valid": int(valid[:, -1].sum()),
                      "metrics": {name: {k: result[k] for k in
                              ("observed_only_ADE_3D_m",
                               "observed_only_FDE_3D_m")}
                                  for name, result in metrics.items()}}, indent=2))


if __name__ == "__main__":
    main()
