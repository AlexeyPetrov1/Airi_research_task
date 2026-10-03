"""Future-only evaluation after a successful sealed forecast.

Observed 2D tracks are independent evaluation data. Projection of the forecast
uses the published DobbE pose convention hypothesis and causal ViPE K, so its
pixel errors are conditional diagnostics, never certified metric ground truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from dobbe_vipe_v1 import ROOT, ALLTRACKER, scene_dir, read, write, sha, freeze, load_video, encode
from dobbe_vipe_geometry import project, sample_depth, stats


def relative_label_poses(labels, ids, t0):
    from scipy.spatial.transform import Rotation
    from validate_dobbe_geometry import CAMERA_TO_LABEL
    basis = np.eye(4)
    basis[:3, :3] = CAMERA_TO_LABEL
    def pose(i):
        result = np.eye(4)
        result[:3, :3] = Rotation.from_quat(labels[str(i)]["quats"]).as_matrix()
        result[:3, 3] = labels[str(i)]["xyz"]
        return result @ basis
    anchor_inv = np.linalg.inv(pose(t0))
    return np.stack([anchor_inv @ pose(i) for i in ids])


def evaluate(scene, branch, variant):
    dest = scene_dir(scene, branch)
    folder = dest / variant
    model_run = read(folder / "model_run.json")
    if not model_run.get("success") or model_run["status"] != "COMPLETE":
        raise RuntimeError("Future remains sealed until a successful prediction exists")
    prediction_path = folder / "prediction_15hz.npy"
    if sha(prediction_path) != model_run["prediction_sha256"]:
        raise RuntimeError("Prediction changed")
    input_freeze = read(folder / "input_freeze.json")
    if sha(folder / "input_freeze.json") != model_run["input_freeze_sha256"]:
        raise RuntimeError("Model input freeze changed")
    if sha(dest / "protocol.json") != input_freeze["protocol_sha256"]:
        raise RuntimeError("Frozen protocol changed")
    for name, digest in input_freeze["input_sha256"].items():
        if sha(dest / name) != digest:
            raise RuntimeError(f"Sealed model input changed: {name}")
    p = read(dest / "protocol.json")
    all_control_predictions = {}
    if branch == "sift_audited":
        for name in ["smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "paired_vipe", "paired_hybrid"]:
            receipt = read(dest / name / "model_run.json")
            if not receipt.get("success") or receipt["status"] != "COMPLETE":
                raise RuntimeError("All predeclared A forecasts must finish before A future access")
            digest = sha(dest / name / "prediction_15hz.npy")
            if digest != receipt["prediction_sha256"]:
                raise RuntimeError("Another predeclared A prediction changed")
            all_control_predictions[name] = digest
    output = folder / "evaluation"
    output.mkdir(exist_ok=True)
    ids = [p["t0"], *p["future_raw_ids_evaluation_only"]]
    # The first future read occurs here, after the prediction and inputs are checked.
    frames = load_video(ROOT / p["raw"] / "compressed_video_h264.mp4", ids[-1]+1)[ids]
    video = output / "future_15hz.mp4"
    if not video.exists():
        encode(video, frames)
    future_receipt = {
        "raw_ids": ids, "prediction_sha256_before_future_access": sha(prediction_path),
        "model_input_freeze_sha256": sha(folder / "input_freeze.json"),
        "purpose": "evaluation only; no prediction inputs, point IDs or thresholds modified"}
    if all_control_predictions:
        future_receipt["predeclared_A_predictions_sha256"] = all_control_predictions
    freeze(output / "future_access_receipt.json", future_receipt)
    points = np.load(folder / "points_2d_t0.npy")
    sys.path.insert(0, str(ALLTRACKER))
    import torch
    from nets.alltracker import Net
    torch.manual_seed(0)
    torch.hub.set_dir(str(ROOT.parent / ".cache/torch/hub"))
    weights = ROOT.parent / ".cache/torch/hub/checkpoints/alltracker.pth"
    model = Net(16)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=False)["model"], strict=True)
    model = model.cuda().eval()
    rgbs = torch.from_numpy(frames[..., ::-1].copy()).permute(0, 3, 1, 2)[None].float().cuda()
    with torch.inference_mode():
        flows, confidence, _, _ = model(rgbs, iters=4, is_training=False)
    grid = torch.from_numpy(points/255*2-1).to(flows.device)[None, None].expand(len(frames), -1, -1, -1)
    tracks = torch.nn.functional.grid_sample(flows[0], grid, align_corners=True)[:, :, 0].permute(0, 2, 1).cpu().numpy()+points[None]
    confidence = torch.nn.functional.grid_sample(confidence[0, :, :1], grid,
                         align_corners=True)[:, 0, 0].cpu().numpy()
    tracks[0] = points
    confidence[0] = 1
    visible = (confidence > .5) & np.isfinite(tracks).all(-1) & ((tracks >= 0) & (tracks <= 255)).all(-1)
    variant_gate = read(folder / "geometry_gate.json")
    point_ids = variant_gate.get("selected_ids", list(range(8)))
    np.savez(output / "observed_future_tracks.npz", raw_ids=ids, tracks=tracks,
             visibility=visible, confidence=confidence, point_ids=point_ids)
    write(output / "alltracker_execution.json", {
        "source": "vendored author nets.alltracker.Net(16)", "iters": 4,
        "checkpoint_sha256": sha(weights), "query_sha256": sha(folder / "points_2d_t0.npy"),
        "tracks_sha256": sha(output / "observed_future_tracks.npz"),
        "direction": "forward from t0, evaluation only", "raw_ids": ids})
    del model, rgbs, flows
    torch.cuda.empty_cache()
    # No calibration is fitted to future observations.
    labels = read(ROOT / p["raw"] / "labels.json")
    poses = relative_label_poses(labels, ids, p["t0"])
    causal = np.load(dest / "geometry_all.npz")
    k = causal["intrinsics"][-1]
    prediction = np.load(prediction_path)
    initial = np.load(folder / "points_3d_camera_t0_diagnostic.npy")[-1]
    predicted_uv = np.stack([project(prediction[:, t], poses[t+1], k) for t in range(30)])
    stationary_uv = np.stack([project(initial, poses[t+1], k) for t in range(30)])
    valid = visible[1:] & np.isfinite(predicted_uv).all(-1) & np.isfinite(stationary_uv).all(-1)
    errors = np.linalg.norm(predicted_uv-tracks[1:], axis=-1)
    stationary_errors = np.linalg.norm(stationary_uv-tracks[1:], axis=-1)
    per_frame = [stats(errors[t, valid[t]]) for t in range(30)]
    metric = stats(errors[valid])
    metric["mean_ADE_px"] = float(errors[valid].mean()) if valid.any() else None
    metric["FDE_visible_mean_px"] = float(errors[-1, valid[-1]].mean()) if valid[-1].any() else None
    baseline = stats(stationary_errors[valid])
    baseline["mean_ADE_px"] = float(stationary_errors[valid].mean()) if valid.any() else None
    # Audit the same pose hypothesis against observed-prefix static tracks.
    history_tracks = np.load(dest / "tracks.npz")
    bg = np.arange(int(history_tracks["object_count"]), int(history_tracks.get("background_end", len(history_tracks["tracks"][0]))))
    measured = np.load(dest / "measured_depth_prefix.npy")[-1]
    measured = cv2.resize(measured, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
    query = history_tracks["tracks"][-1, bg]
    z, _ = sample_depth(measured, query)
    world = np.column_stack([(query[:, 0]-k[2])*z/k[0], (query[:, 1]-k[3])*z/k[1], z])
    observed_pose = relative_label_poses(labels, p["observed_raw_ids"], p["t0"])
    static_errors = []
    for t in np.linspace(0, len(observed_pose)-4, 6).astype(int):
        estimate = project(world, observed_pose[t], k)
        keep = history_tracks["visibility"][t, bg] & np.isfinite(estimate).all(-1)
        static_errors.extend(np.linalg.norm(estimate[keep]-history_tracks["tracks"][t, bg][keep], axis=-1))
    pose_audit = stats(static_errors)
    summary = {"prediction_sha256": sha(prediction_path), "point_frames_visible": int(visible[1:].sum()),
        "comparison_point_frames": int(valid.sum()),
        "forecast_projection_invalid_point_frames": int((~np.isfinite(predicted_uv).all(-1)).sum()),
        "stationary_projection_invalid_point_frames": int((~np.isfinite(stationary_uv).all(-1)).sum()),
        "total_point_frames": 240, "conditional_forecast_pixel_errors": metric,
        "conditional_stationary_baseline_pixel_errors": baseline,
        "per_future_frame": per_frame, "observed_prefix_published_pose_audit_px": pose_audit,
        "projection_pose_source": "published DobbE labels c2w * CAMERA_TO_LABEL, relative to t0",
        "K_source": "causal ViPE only; fixed for evaluation", "future_fit": False,
        "metric_3d_ground_truth": False,
        "interpretation": "Pixel comparison is conditional on the published pose/basis and estimated K; future 2D tracking is model-derived, not manual GT."}
    write(output / "metrics.json", summary)
    np.savez(output / "conditional_projection.npz", predicted_uv=predicted_uv,
        stationary_uv=stationary_uv, observed_uv=tracks[1:], valid=valid, relative_poses=poses)
    render(output, frames, ids, predicted_uv, tracks, visible, errors, valid)
    print(scene, branch, variant, "evaluation", metric, flush=True)


def render(output, frames, ids, predicted, observed, visible, errors, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    times = [1, 5, 10, 15, 20, 25, 29, 30]
    for ax, t in zip(axes.flat, times):
        ax.imshow(cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB))
        for i in range(8):
            color = plt.cm.tab10(i)
            if visible[t, i]:
                ax.scatter(*observed[t, i], color=color, s=18)
            if np.isfinite(predicted[t-1, i]).all():
                ax.scatter(*predicted[t-1, i], color=color, marker="x", s=35)
        ax.set_xlim(0, 255); ax.set_ylim(255, 0)
        ax.set_title(f"raw {ids[t]}; +{t/15:.2f}s"); ax.axis("off")
    fig.suptitle("dots: observed AllTracker; crosses: forecast, published-pose hypothesis")
    fig.tight_layout(); fig.savefig(output / "future_comparison.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(8):
        y = errors[:, i].copy(); y[~valid[:, i]] = np.nan
        ax.plot(np.arange(1, 31)/15, y, color=plt.cm.tab10(i), label=f"point {i+1}")
    ax.set_xlabel("seconds after t0"); ax.set_ylabel("conditional projection error, pixels")
    ax.legend(ncol=2, fontsize=8); fig.tight_layout()
    fig.savefig(output / "conditional_error_over_time.png", dpi=150); plt.close(fig)
    # Browser-compatible display video; never used as a model input.
    import imageio_ffmpeg
    import subprocess
    panels = []
    for t in range(1, 31):
        panel = cv2.resize(frames[t], (512, 512), interpolation=cv2.INTER_NEAREST)
        on_screen = 0
        for i in range(8):
            color = tuple(int(x) for x in (np.array(plt.cm.tab10(i)[:3])[::-1]*255))
            if visible[t, i]:
                cv2.circle(panel, tuple((observed[t, i]*2).round().astype(int)), 4, color, -1)
            uv = predicted[t-1, i]
            if np.isfinite(uv).all() and ((uv >= 0) & (uv <= 255)).all():
                on_screen += 1
                cv2.drawMarker(panel, tuple((uv*2).round().astype(int)), color,
                               cv2.MARKER_TILTED_CROSS, 12, 2)
        panel = cv2.copyMakeBorder(panel, 0, 56, 0, 0, cv2.BORDER_CONSTANT)
        cv2.putText(panel, f"raw {ids[t]} +{t/15:.2f}s; forecast in image {on_screen}/8",
                    (6, 533), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
        cv2.putText(panel, "dot=AllTracker; x=forecast, conditional camera poses",
                    (6, 554), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        panels.append(panel)
    argv = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "512x568", "-r", "15", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output / "future_overlay.mp4")]
    subprocess.run(argv, input=np.stack(panels).tobytes(), check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["A", "B", "C"], required=True)
    parser.add_argument("--branch", choices=["no_vda", "default", "rectified", "sift_audited"], default="no_vda")
    parser.add_argument("--variant", default="pure_vipe")
    args = parser.parse_args()
    evaluate(args.scene, args.branch, args.variant)
