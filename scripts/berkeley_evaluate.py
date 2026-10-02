"""Evaluate frozen Berkeley predictions with independent author AllTracker tracks.

Usage: python scripts/berkeley_evaluate.py SCENE_DIR --stage all
Run ``track`` in the tracker environment, then ``evaluate`` on CPU if necessary.
Only evaluation/ and viz/ plus metrics.json are written. Future RGB is opened
only after validating and hashing the saved predictions and causal inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = np.arange(2, 30, 3, dtype=np.int64)
TIMES = np.arange(1, 11, dtype=np.float64) / 5


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_file(scene: Path, *names: str) -> Path:
    for name in names:
        path = scene / name
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {names} exists under {scene}")


def load_npy(scene: Path, *names: str) -> np.ndarray:
    return np.load(find_file(scene, *names), allow_pickle=False)


def freeze_inputs(scene: Path) -> tuple[np.ndarray, dict]:
    """Refuse evaluation without actual saved, complete, nonzero model output."""
    prediction_path = find_file(scene, "predictions/future_3d.npy", "predictions/prediction_15hz.npy")
    prediction = np.load(prediction_path, allow_pickle=False).astype(np.float64)
    if (prediction.ndim != 3 or prediction.shape[1:] != (30, 3)
            or prediction.shape[0] not in (8, 16, 24)):
        raise ValueError(f"Expected [8/16/24,30,3], received {prediction.shape}")
    if not np.isfinite(prediction).all() or not np.any(prediction):
        raise ValueError("Nonfinite or zero-filled predictions are not successful inference")
    receipt = read_json(find_file(scene, "predictions/model_run.json", "predictions/inference.json"))
    if receipt.get("success") is not True:
        raise ValueError("Prediction receipt must explicitly declare success=true")
    if receipt.get("prediction_completeness", 1.0) != 1.0:
        raise ValueError("Incomplete model point/time identifiers")
    successful_chunks = receipt.get("successful_chunks", receipt.get("number_of_successful_model_chunks"))
    if successful_chunks != len(prediction)//8:
        raise ValueError("Successful model chunk receipt disagrees with prediction count")
    groups = receipt.get("groups", [])
    if len(groups) != successful_chunks or not all(group.get("success") is True for group in groups):
        raise ValueError("All successful model chunks require individual receipts")
    paths = [prediction_path]
    for folder in ("observed", "groups", "geometry", "predictions"):
        paths.extend(sorted(p for p in (scene / folder).rglob("*")
                            if p.is_file() and p.suffix.lower() in (".npy", ".npz", ".pt", ".json", ".txt", ".png")))
    hashes = {str(p.relative_to(scene)).replace("\\", "/"): sha256(p) for p in sorted(set(paths))}
    dest = scene / "evaluation/prediction_freeze.json"
    existing = read_json(dest)
    if existing and existing.get("sha256") != hashes:
        raise ValueError("Frozen predictions/causal inputs changed since evaluation began")
    if not existing:
        write_json(dest, {"created_utc": datetime.now(timezone.utc).isoformat(),
                          "future_access_after_this_receipt": True,
                          "prediction_shape": list(prediction.shape), "sha256": hashes})
    return prediction, receipt


def scene_inputs(scene: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history = load_npy(scene, "observed/history_rgb.npy")
    uv = load_npy(scene, "observed/points_2d_history.npy")
    xyz = load_npy(scene, "observed/points_3d_history.npy")
    ids = load_npy(scene, "observed/selected_point_ids.npy").astype(np.int64)
    k = load_npy(scene, "geometry/K_median.npy").astype(np.float64)
    if history.ndim != 4 or history.shape[0] != 3 or history.shape[-1] != 3:
        raise ValueError(f"Bad RGB history {history.shape}")
    if uv.shape != (3, len(ids), 2) or xyz.shape != (3, len(ids), 3):
        raise ValueError(f"Bad history coordinates {uv.shape}, {xyz.shape}, {ids.shape}")
    if ids.shape != (len(ids),) or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= 100)):
        raise ValueError("Selected point IDs must uniquely identify the 100 grounded candidates")
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    if not np.isfinite(uv).all() or not np.isfinite(xyz).all() or (xyz[..., 2] <= 0).any():
        raise ValueError("Invalid causal history")
    return history, uv, xyz, ids, k


def track_future(scene: Path, max_side: int = 512, iters: int = 4, device: str = "cuda") -> None:
    prediction, _ = freeze_inputs(scene)
    history, uv, _, ids, _ = scene_inputs(scene)
    metadata = read_json(scene / "metadata.json")
    timestamp_audit = validate_timestamps(metadata, scene)
    destination = scene / "evaluation/evaluation_tracks_2d.npz"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing independent tracks: {destination}")
    future = load_npy(scene, "evaluation/future_rgb.npy", "evaluation/rgb.npy")
    if future.shape != (10, *history.shape[1:]):
        raise ValueError(f"Expected ten real future RGB frames, got {future.shape}")
    if len(ids) != len(prediction):
        raise ValueError("Prediction and query counts disagree")
    import torch
    import torch.nn.functional as functional

    tracker_root = ROOT / "data_generation/third_party/alltracker"
    sys.path.insert(0, str(tracker_root))
    from nets.alltracker import Net

    checkpoint = ROOT.parent / ".cache/torch/hub/checkpoints/alltracker.pth"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Expected installed author AllTracker checkpoint: {checkpoint}")
    frames = np.concatenate([history[-1:], future], axis=0)
    h0, w0 = frames.shape[1:3]
    scale = min(1., max_side / max(h0, w0))
    h, w = max(8, int(h0 * scale) // 8 * 8), max(8, int(w0 * scale) // 8 * 8)
    if (h, w) != (h0, w0):
        frames = np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR) for frame in frames])
    torch.manual_seed(0)
    model = Net(16)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"], strict=True)
    model = model.to(device).eval()
    rgbs = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2)[None].float().to(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.monotonic()
    with torch.inference_mode():
        flow, visconf, _, _ = model(rgbs, iters=iters, sw=None, is_training=False)
    if flow.shape != (1, 11, 2, h, w) or visconf.ndim != 5 or visconf.shape[:2] != (1, 11):
        raise ValueError(f"Unexpected AllTracker output {flow.shape}, {visconf.shape}")
    # Bilinear dense-flow sampling preserves the exact physical query at t0.
    # Author CLI rounds dense map coordinates; this adapter avoids anchor quantization.
    points = uv[-1].astype(np.float32) * np.array([w / w0, h / h0], np.float32)
    normalized = points / np.array([w - 1, h - 1], np.float32) * 2 - 1
    grid = torch.from_numpy(normalized).to(device)[None, None].expand(11, -1, -1, -1)
    flow_samples = functional.grid_sample(flow[0], grid, align_corners=True)[:, :, 0].permute(0, 2, 1)
    scores = functional.grid_sample(visconf[0], grid, align_corners=True)[:, :, 0].permute(0, 2, 1)
    xy = (flow_samples.cpu().numpy() + points[None]) / np.array([w / w0, h / h0])
    scores = scores.cpu().numpy()
    visibility_probability = scores[..., 0]
    confidence = scores[..., 1] if scores.shape[-1] >= 2 else scores[..., 0]
    xy[0], visibility_probability[0], confidence[0] = uv[-1], 1., 1.
    visible = visibility_probability > .5
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    np.savez_compressed(destination, tracks=xy.astype(np.float32), visibility=visible,
                        visibility_probability=visibility_probability, confidence=confidence,
                        point_ids=ids, dim=np.array([h0, w0]),
                        time_from_t0_s=np.r_[0., TIMES],
                        source_indices=np.r_[timestamp_audit["history_source_indices"][-1], timestamp_audit["future_source_indices"]],
                        source_timestamps_s=np.r_[timestamp_audit["history_timestamps_s"][-1], timestamp_audit["future_timestamps_s"]])
    write_json(scene / "evaluation/alltracker_execution.json", {
        "success": True, "tracker": "official vendored nets.alltracker.Net(16)",
        "checkpoint_sha256": sha256(checkpoint), "direction": "t0 -> ten real future frames",
        "independent_of_observed_tracking": True, "model_prediction_frozen_first": True,
        "visibility_rule": "official channel 0 > 0.5; confidence channel 1 saved separately",
        "sampling": "bilinear dense flow at exact fixed t0 query; no future smoothing",
        "input_shape": list(rgbs.shape), "original_dimensions_hw": [h0, w0],
        "iterations": iters, "runtime_seconds": elapsed,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device.startswith("cuda") else 0.,
        "prediction_sha256": sha256(find_file(scene, "predictions/future_3d.npy", "predictions/prediction_15hz.npy")),
        "track_sha256": sha256(destination)})
    print(json.dumps({"scene": str(scene), "tracking_seconds": elapsed,
                      "future_visible_pairs": int(visible[1:].sum())}), flush=True)


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([k[0, 0] * xyz[..., 0] / xyz[..., 2] + k[0, 2],
                         k[1, 1] * xyz[..., 1] / xyz[..., 2] + k[1, 2]], axis=-1)


def velocity(history: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    centered = timestamps - timestamps.mean()
    return np.sum(centered[:, None, None] * (history - history.mean(axis=0)), axis=0) / np.sum(centered**2)


def validate_timestamps(metadata: dict, scene: Path | None = None) -> dict:
    """Accept only the exact 5 Hz real-frame protocol, allowing float32 roundoff."""
    t0 = metadata.get("t0_source_frame_index", metadata.get("t0_source_frame", metadata.get("t0")))
    history_indices = np.asarray(metadata["history_source_indices"])
    future_indices = np.asarray(metadata["future_source_indices"])
    if not np.array_equal(history_indices, np.arange(t0-2, t0+1)):
        raise ValueError("History is not the three consecutive real frames ending at t0")
    if not np.array_equal(future_indices, np.arange(t0+1, t0+11)):
        raise ValueError("Evaluation is not the next ten consecutive real frames")
    history_times = np.asarray(metadata["history_timestamps"], dtype=float)
    future_times = np.asarray(metadata["future_timestamps"], dtype=float)
    if history_times.shape != (3,) or future_times.shape != (10,):
        raise ValueError("Exact source timestamps are required for H3 and F10")
    if not np.allclose(history_times-history_times[-1], [-.4, -.2, 0.], atol=2e-6, rtol=0):
        raise ValueError("Observed physical timestamps differ from the 5 Hz protocol")
    if not np.allclose(future_times-history_times[-1], TIMES, atol=2e-6, rtol=0):
        raise ValueError("Future physical timestamps differ from the 5 Hz protocol")
    if scene is not None:
        for alternatives in (("evaluation/timestamps.npy", "evaluation/future_timestamps.npy"),
                             ("evaluation/source_indices.npy", "evaluation/future_source_indices.npy")):
            if not any((scene / name).exists() for name in alternatives):
                raise ValueError(f"Exact future export timestamps/indices are required: {alternatives}")
        for relative_path, expected in (("observed/history_timestamps.npy", history_times),
                                        ("observed/history_source_indices.npy", history_indices),
                                        ("evaluation/timestamps.npy", future_times),
                                        ("evaluation/source_indices.npy", future_indices),
                                        ("evaluation/future_timestamps.npy", future_times),
                                        ("evaluation/future_source_indices.npy", future_indices)):
            path = scene / relative_path
            if path.exists() and not np.array_equal(np.load(path, allow_pickle=False), expected):
                raise ValueError(f"Source metadata and saved timestamps/indices disagree: {relative_path}")
    return {"history_source_indices": history_indices.tolist(), "future_source_indices": future_indices.tolist(),
            "history_timestamps_s": history_times.tolist(), "future_timestamps_s": future_times.tolist(),
            "history_exact_relative_times_s": (history_times-history_times[-1]).tolist(),
            "future_exact_relative_times_s": (future_times-history_times[-1]).tolist(),
            "timestamp_alignment_tolerance_s": 2e-6,
            "max_nominal_alignment_error_s": float(np.max(np.abs(future_times-history_times[-1]-TIMES)))}


def metric_values(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray, suffix: str) -> tuple[dict, np.ndarray]:
    error = np.linalg.norm(prediction - truth, axis=-1)
    if not np.isfinite(error[mask]).all():
        raise ValueError("A method is nonfinite on the common GT visibility mask; do not drop its bad predictions")
    by_time = [float(error[:, t][mask[:, t]].mean()) if mask[:, t].any() else None for t in range(10)]
    return {f"ADE_{suffix}": float(error[mask].mean()) if mask.any() else None,
            f"FDE_{suffix}": float(error[:, -1][mask[:, -1]].mean()) if mask[:, -1].any() else None,
            "error_by_horizon": by_time, "valid_pairs": int(mask.sum()),
            "valid_point_coverage": float(mask.mean()), "valid_points_by_horizon": mask.sum(axis=0).tolist(),
            "FDE_valid_points": int(mask[:, -1].sum())}, error


def lift_native_depth(depth: np.ndarray, uv: np.ndarray, k: np.ndarray, radius: int = 2) -> np.ndarray:
    """Point-major future XYZ using a robust 5x5 native metric depth median."""
    result = np.full((*uv.shape[:-1], 3), np.nan, np.float64)
    for point in range(len(uv)):
        for t in range(10):
            u, v = uv[point, t]
            if not np.isfinite([u, v]).all():
                continue
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < depth.shape[2] and 0 <= y < depth.shape[1]):
                continue
            patch = depth[t, max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            # Match observed lifting: exclude sensor saturation (65.535 m)
            # and require at least 13 valid samples for a full 5x5 patch.
            valid = patch[np.isfinite(patch) & (patch > 0) & (patch < 10)]
            if len(valid) < max(3, (patch.size+1)//2):
                continue
            z = float(np.median(valid))
            result[point, t] = [(u-k[0, 2])/k[0, 0]*z, (v-k[1, 2])/k[1, 1]*z, z]
    return result


def label(frame: np.ndarray, text: str, row: int = 0) -> None:
    y = 22 + row * 22
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)


def points_on(frame: np.ndarray, uv: np.ndarray, ids: np.ndarray, color: tuple[int, int, int],
              valid: np.ndarray | None = None, marker: str = "circle", labels: bool = True) -> None:
    height, width = frame.shape[:2]
    for index, pair in enumerate(uv):
        if (valid is not None and not valid[index]) or not np.isfinite(pair).all():
            continue
        x, y = np.rint(pair).astype(np.int64)
        if not (0 <= x < width and 0 <= y < height):
            continue
        if marker == "cross":
            cv2.drawMarker(frame, (int(x), int(y)), color, cv2.MARKER_TILTED_CROSS, 8, 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)
        if labels:
            text = str(int(ids[index]))
            cv2.putText(frame, text, (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, .32, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)


def trails_on(frame: np.ndarray, trajectories: np.ndarray, color: tuple[int, int, int],
              mask: np.ndarray | None = None, thickness: int = 1) -> None:
    height, width = frame.shape[:2]
    for p, trajectory in enumerate(trajectories):
        for t in range(1, len(trajectory)):
            pair = trajectory[t-1:t+1]
            if not np.isfinite(pair).all() or (mask is not None and not mask[p, t-1:t+1].all()):
                continue
            # Clip drawings only: forecasts outside the image remain in metrics.
            a, b = np.rint(np.clip(pair, -2**25, 2**25)).astype(np.int64)
            ok, start, end = cv2.clipLine((0, 0, width, height), tuple(map(int, a)), tuple(map(int, b)))
            if ok:
                cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)


def save_rgb(path: Path, frame: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Failed to write {path}")


def write_video(path: Path, frames: list[np.ndarray], fps: float = 5) -> dict:
    import imageio.v2 as imageio
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8,
                            macro_block_size=2, pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if count != len(frames):
        raise ValueError(f"Video validation failed for {path}: {count} frames")
    return {"frame_count": count, "fps": actual_fps, "bytes": path.stat().st_size}


def visualize(scene: Path, history: np.ndarray, history_uv: np.ndarray, history_xyz: np.ndarray,
              ids: np.ndarray, k: np.ndarray, prediction: np.ndarray, future_rgb: np.ndarray,
              gt: np.ndarray, mask: np.ndarray, methods: dict, metrics: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    viz = scene / "viz"
    viz.mkdir(exist_ok=True)
    gt_color, pred_color, history_color = (40, 245, 100), (255, 65, 170), (20, 190, 255)
    hero = min(8, len(ids))
    hero_ids = ids[:hero]
    h, w = history.shape[1:3]
    panels = []
    for t, rgb in enumerate(history):
        panel = rgb.copy()
        points_on(panel, history_uv[t], ids, gt_color)
        label(panel, f"Observed t={(t-2)/5:+.1f}s | {len(ids)} selected points")
        panels.append(panel)
    save_rgb(viz / "history_contact_sheet.png", np.concatenate(panels, axis=1))
    mask_path = find_file(scene, "observed/mask.png")
    segmentation = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
    if segmentation.shape != (h, w):
        raise ValueError("Segmentation resolution mismatch")
    candidates = load_npy(scene, "observed/query_points_100.npy")
    segmentation_panel = history[-1].copy()
    segmentation_panel[segmentation] = (.5*segmentation_panel[segmentation] + .5*np.array([255, 190, 0])).astype(np.uint8)
    points_on(segmentation_panel, candidates, np.arange(len(candidates)), (80, 255, 255), labels=False)
    label(segmentation_panel, f"t0 target mask | {len(candidates)} K-means queries")
    save_rgb(viz / "mask_and_100_queries.png", segmentation_panel)
    selected = history[-1].copy()
    palette = [(40, 245, 100), (255, 190, 20), (130, 190, 255)]
    for group in range(len(ids)//8):
        selection = slice(8*group, 8*(group+1))
        points_on(selected, history_uv[-1, selection], ids[selection], palette[group])
    label(selected, f"{len(ids)} selected queries | persistent candidate IDs")
    save_rgb(viz / "selected_24_points.png", selected)
    depth = load_npy(scene, "geometry/depth_observed.npy")[-1].squeeze()
    depth_valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
    valid_depth = depth[depth_valid]
    low, high = np.percentile(valid_depth, [2, 98])
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(np.ma.masked_invalid(np.where(depth_valid, depth, np.nan)), cmap="viridis", vmin=low, vmax=high)
    ax.scatter(history_uv[-1, :, 0], history_uv[-1, :, 1], s=12, c="red")
    for point, (u, v) in zip(ids, history_uv[-1]):
        ax.text(u+3, v-3, str(point), color="white", fontsize=6)
    ax.set_title(f"t0 depth (m): {metrics['depth_source']}\nInvalid depth masked (requires 0 < Z < 10 m)")
    ax.set_axis_off()
    fig.colorbar(im, ax=ax, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(viz / "depth_t0.png", dpi=150)
    plt.close(fig)
    ks = load_npy(scene, "geometry/K_per_frame.npy")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, name, row, col in zip(axes.ravel(), ("fx", "fy", "cx", "cy"), (0, 1, 0, 1), (0, 1, 2, 2)):
        values = ks[:, row, col]
        ax.plot(np.arange(len(values)), values, "o-")
        ax.axhline(k[row, col], color="tab:red", linestyle="--", label="robust median")
        ax.set(xlabel="Observed frame index", ylabel=f"{name} (px)")
        ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Intrinsics estimated using observed RGB only")
    fig.tight_layout()
    fig.savefig(viz / "intrinsics_stability.png", dpi=150)
    plt.close(fig)
    stability = read_json(scene / "geometry/intrinsics_stability.json")
    write_json(viz / "intrinsics_stability.json", stability or {"K_median": k.tolist(), "K_per_frame": ks.tolist()})
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    for p in range(len(ids)):
        color = plt.cm.turbo(p/max(1, len(ids)-1))
        past = history_xyz[:, p]
        predicted = np.vstack([past[-1:], prediction[p]])
        ax.plot(*past.T, color=color, linewidth=2.8, marker="o", markersize=3)
        ax.plot(*predicted.T, color=color, linewidth=.9, linestyle="--")
        ax.text(*past[-1], str(ids[p]), fontsize=6)
    ax.set(xlabel="Camera X (m)", ylabel="Camera Y (m)", zlabel="Camera Z (m)",
           title="Observed H3 (solid) and all 30 forecast steps (dashed)")
    ax.legend(handles=[Line2D([0], [0], color="black", lw=3, label="Observed history"),
                       Line2D([0], [0], color="black", ls="--", label="MolmoMotion forecast")])
    fig.tight_layout()
    fig.savefig(viz / "trajectory_3d.png", dpi=160)
    plt.close(fig)
    predicted_full = project(prediction, k)
    gt_with_anchor = np.concatenate([history_uv[-1, :, None], gt], axis=1)
    visible_with_anchor = np.concatenate([np.ones((len(ids), 1), bool), mask], axis=1)
    pred_with_anchor = np.concatenate([history_uv[-1, :, None], predicted_full], axis=1)
    overlay = history[-1].copy()
    trails_on(overlay, gt_with_anchor[:hero], gt_color, visible_with_anchor[:hero], 2)
    trails_on(overlay, pred_with_anchor[:hero], pred_color, thickness=1)
    trails_on(overlay, history_uv[:, :hero].transpose(1, 0, 2), history_color, thickness=2)
    points_on(overlay, history_uv[-1, :hero], hero_ids, history_color)
    label(overlay, "GT: green | MolmoMotion: pink | observed: blue")
    label(overlay, f"Hero group: {hero} points; metrics use all {len(ids)}", 1)
    save_rgb(viz / "final_overlay_t0.png", overlay)
    # Retain the full forecast when it exits the native image. The ordinary
    # overlay above remains useful for interpreting the original camera view.
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(history[-1], extent=(0, w, h, 0), zorder=0)
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="black", linewidth=1.2,
            label="Source image boundary", zorder=2)
    for p in range(hero):
        past = history_uv[:, p]
        projected = pred_with_anchor[p]
        reference = gt_with_anchor[p].copy()
        reference[~visible_with_anchor[p]] = np.nan
        ax.plot(past[:, 0], past[:, 1], color="#14beff", linewidth=2.2,
                label="Observed H3" if p == 0 else None)
        ax.plot(reference[:, 0], reference[:, 1], color="#19a84b", linewidth=1.7,
                label="Tracked GT" if p == 0 else None)
        ax.plot(projected[:, 0], projected[:, 1], color="#ee348f", linewidth=1.2,
                label="MolmoMotion (all 30 steps)" if p == 0 else None)
        ax.scatter(*projected[-1], color="#ee348f", marker="x", s=22)
        ax.annotate(str(ids[p]), projected[-1], xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color="#a61b62")
        ax.annotate(str(ids[p]), past[-1], xytext=(4, -9), textcoords="offset points",
                    fontsize=7, color="#006b9e")
    bounds = np.concatenate([history_uv[:, :hero].reshape(-1, 2),
                             pred_with_anchor[:hero].reshape(-1, 2),
                             gt_with_anchor[:hero].reshape(-1, 2), [[0, 0], [w, h]]])
    bounds = bounds[np.isfinite(bounds).all(axis=-1)]
    lower, upper = bounds.min(axis=0), bounds.max(axis=0)
    margin = np.maximum((upper-lower)*.05, 10)
    ax.set(xlim=(lower[0]-margin[0], upper[0]+margin[0]),
           ylim=(upper[1]+margin[1], lower[1]-margin[1]),
           xlabel="u (native image pixels)", ylabel="v (native image pixels)",
           title=f"Complete 2D trajectories, including forecasts outside image | hero {hero}/{len(ids)} points")
    ax.set_aspect("equal")
    ax.grid(alpha=.18)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(viz / "trajectory_2d_full_extent.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, color in (("MolmoMotion", "#d62c81"), ("static", "#087db8"), ("constant_velocity", "#bc7f04")):
        ax.plot(TIMES, metrics["methods"][name]["error_by_horizon"], "o-", label=name, color=color)
    ax.set(xlabel="Physical time after t0 (s)", ylabel="Mean 2D projection error (px)", xticks=TIMES)
    ax.grid(alpha=.25)
    second = ax.twinx()
    second.plot(TIMES, mask.sum(axis=0), "k--", alpha=.5, label="Visible points")
    second.set(ylabel="Valid tracked points", ylim=(0, len(ids)+1))
    ax.legend(loc="upper left")
    second.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(viz / "error_by_time.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(mask.astype(float), interpolation="nearest", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set(xticks=np.arange(10), xticklabels=[f"{x:.1f}" for x in TIMES], yticks=np.arange(len(ids)),
           yticklabels=ids, xlabel="Physical time after t0 (s)", ylabel="Persistent point ID",
           title=f"Common evaluation visibility: {int(mask.sum())}/{mask.size} pairs")
    fig.tight_layout()
    fig.savefig(viz / "visibility_coverage.png", dpi=150)
    plt.close(fig)
    comparison, side_by_side = [], []
    for t, rgb in enumerate(future_rgb):
        panel, left, right = rgb.copy(), rgb.copy(), rgb.copy()
        gt_path = gt_with_anchor[:hero, :t+2]
        gt_valid = visible_with_anchor[:hero, :t+2]
        pred_path = pred_with_anchor[:hero, :ALIGNMENT[t]+2]
        for destination in (panel, left):
            trails_on(destination, gt_path, gt_color, gt_valid, 2)
            points_on(destination, gt[:hero, t], hero_ids, gt_color, mask[:hero, t])
        for destination in (panel, right):
            trails_on(destination, pred_path, pred_color, thickness=1)
            trails_on(destination, history_uv[:, :hero].transpose(1, 0, 2), history_color, thickness=2)
            points_on(destination, methods["MolmoMotion"][:hero, t], hero_ids, pred_color, marker="cross")
        label(panel, f"t=+{TIMES[t]:.1f}s | GT green / prediction pink X")
        label(panel, f"Hero IDs {','.join(map(str, hero_ids))} | visible {int(mask[:, t].sum())}/{len(ids)}", 1)
        label(left, f"REAL RGB + tracked GT | t=+{TIMES[t]:.1f}s")
        label(right, f"REAL RGB + MolmoMotion | step {ALIGNMENT[t]+1}")
        comparison.append(panel)
        side_by_side.append(np.concatenate([left, right], axis=1))
    videos = {"pred_vs_gt_2d.mp4": write_video(viz / "pred_vs_gt_2d.mp4", comparison),
              "side_by_side.mp4": write_video(viz / "side_by_side.mp4", side_by_side)}
    write_json(viz / "visualization_manifest.json", {"hero_point_ids": hero_ids.tolist(),
               "metrics_point_ids": ids.tolist(), "RGB_frames_are_real": True,
               "future_frame_interpolation": False, "videos": videos})


def evaluate(scene: Path) -> dict:
    started = time.monotonic()
    prediction, model_run = freeze_inputs(scene)
    history, history_uv, history_xyz, ids, k = scene_inputs(scene)
    metadata = read_json(scene / "metadata.json")
    timestamp_audit = validate_timestamps(metadata, scene)
    future = load_npy(scene, "evaluation/future_rgb.npy", "evaluation/rgb.npy")
    if future.shape != (10, *history.shape[1:]) or len(prediction) != len(ids):
        raise ValueError("RGB/prediction dimensions disagree")
    data = np.load(scene / "evaluation/evaluation_tracks_2d.npz", allow_pickle=False)
    if not np.array_equal(data["point_ids"], ids):
        raise ValueError("Independent tracking point order differs from inference")
    tracks, visibility = data["tracks"], data["visibility"]
    if tracks.shape != (11, len(ids), 2) or visibility.shape != (11, len(ids)):
        raise ValueError("Wrong independent future track shapes")
    if not np.allclose(tracks[0], history_uv[-1], atol=1e-3):
        raise ValueError("Independent future tracker is not anchored at exact observed queries")
    gt = tracks[1:].transpose(1, 0, 2).astype(np.float64)
    h, w = history.shape[1:3]
    finite = np.isfinite(gt).all(axis=-1)
    in_frame = (gt[..., 0] >= 0) & (gt[..., 0] < w) & (gt[..., 1] >= 0) & (gt[..., 1] < h)
    mask = visibility[1:].T & finite & in_frame
    history_times = np.asarray(timestamp_audit["history_exact_relative_times_s"], dtype=np.float64)
    future_xyz = prediction[:, ALIGNMENT]
    method_xyz = {"MolmoMotion": future_xyz,
                  "static": np.repeat(history_xyz[-1, :, None], 10, axis=1),
                  "constant_velocity": history_xyz[-1, :, None] + velocity(history_xyz, history_times)[:, None] * TIMES[None, :, None]}
    methods = {name: project(xyz, k) for name, xyz in method_xyz.items()}
    measurements, errors = {}, {}
    for name, values in methods.items():
        measurements[name], errors[name] = metric_values(values, gt, mask, "2D_px")
    geometry_metadata = read_json(scene / "geometry/depth_source.json")
    depth_source = metadata.get("depth_source", geometry_metadata.get("depth_source", "estimated: UniDepthV2"))
    stability = read_json(scene / "geometry/intrinsics_stability.json")
    grounding = read_json(scene / "observed/grounding_metadata.json")
    camera_audit = read_json(scene / "geometry/camera_motion_audit.json")
    filtering = read_json(scene / "geometry/filter_metadata.json")
    limitations = list(metadata.get("warnings", [])) + list(metadata.get("limitations", []))
    limitations.extend(["Berkeley history uses real 5 FPS frames; MolmoMotion training/inference forecast is 15 FPS.",
                        "GT is independently inferred AllTracker correspondence, not human-labeled or instrumented motion ground truth.",
                        "Common visibility mask uses tracker channel 0 > 0.5, finite in-image GT; errors retain out-of-image predictions.",
                        "Camera intrinsics are estimated from observed RGB, not published calibration.",
                        "GT tracks may drift during gripper/object/container occlusion; coverage and video require joint interpretation."])
    if "estimated" in str(depth_source).lower() or "unidepth" in str(depth_source).lower():
        limitations.append("Metric input depth is estimated; no 3D_est accuracy is claimed without native future depth.")
    if mask[:, -1].sum() == 0:
        limitations.append("No valid visible final points: FDE is unavailable, not zero.")
    if not mask.any():
        limitations.append("No valid independent future tracks: all accuracy metrics unavailable.")
    if stability.get("focal_stable_within_15_percent") is False:
        limitations.append("Observed UniDepth focal estimates vary by more than 15%; robust median K does not remove calibration uncertainty.")
    outside_pred = ((methods["MolmoMotion"][..., 0] < 0) | (methods["MolmoMotion"][..., 0] >= w)
                    | (methods["MolmoMotion"][..., 1] < 0) | (methods["MolmoMotion"][..., 1] >= h))
    metrics = {"source_episode_id": metadata.get("source_episode_id", metadata.get("episode_id")),
               "task": metadata.get("instruction", metadata.get("task")),
               "t0": metadata.get("t0_source_frame_index", metadata.get("t0")),
               "t0_timestamp_s": metadata.get("t0_timestamp_s", metadata.get("t0_seconds")),
               "depth_source": depth_source, "K_source": metadata.get("K_source", geometry_metadata.get("K_source", "UniDepthV2 observed-frame robust median")),
               "K_median": k.tolist(), "number_of_valid_points": len(ids),
               "intrinsics_stability": stability, "camera_motion_audit": camera_audit,
               "preprocessing": {"grounding": grounding, "filtering": filtering,
                                 "geometry": geometry_metadata,
                                 "tracking_sampling_deviation": "bilinear dense flow at exact query instead of author CLI nearest-pixel query rounding"},
               "number_of_successful_model_chunks": model_run["successful_chunks"], "point_ids": ids.tolist(),
               "prediction_shape": list(prediction.shape), "evaluation_prediction_indices_zero_based": ALIGNMENT.tolist(),
               "evaluation_times_from_t0_s": TIMES.tolist(), "gt_interpolated": False,
               "timestamp_audit": timestamp_audit,
               "methods": measurements, "ADE_2D_px": measurements["MolmoMotion"]["ADE_2D_px"],
               "FDE_2D_px": measurements["MolmoMotion"]["FDE_2D_px"],
               "static_baseline": measurements["static"], "constant_velocity_baseline": measurements["constant_velocity"],
               "baseline_definition": "static and ordinary-least-squares constant velocity in camera XYZ using exact observed timestamps, projected through same K",
               "valid_point_coverage": float(mask.mean()), "valid_pairs": int(mask.sum()),
               "points_with_any_valid_future": int(mask.any(axis=1).sum()),
               "valid_points_by_horizon": mask.sum(axis=0).tolist(),
               "prediction_outside_image_by_horizon": outside_pred.sum(axis=0).tolist(),
               "prediction_nonpositive_Z_count": int((prediction[..., 2] <= 0).sum()),
               "timing": {"model": model_run.get("prediction_seconds", model_run.get("runtime_seconds", model_run.get("total_runtime_seconds"))),
                          "model_scene_elapsed_seconds": model_run.get("elapsed_seconds"),
                          "model_load_seconds": model_run.get("model_load_seconds"),
                          "tracker": read_json(scene / "evaluation/alltracker_execution.json").get("runtime_seconds")},
               "GPU_memory": {"model_peak_gib": model_run.get("peak_gpu_memory_gib", model_run.get("peak_cuda_allocated_gib")),
                              "tracker_peak_gib": read_json(scene / "evaluation/alltracker_execution.json").get("peak_gpu_memory_gib")},
               "model_receipt": model_run, "warnings_limitations": list(dict.fromkeys(limitations))}
    native = scene / "evaluation/native_depth_future.npy"
    if native.exists():
        depth_metadata = read_json(scene / "evaluation/native_depth_metadata.json")
        if depth_metadata.get("measured_metric_depth") is not True:
            raise ValueError("Native future depth requires explicit measured_metric_depth provenance")
        native_depth = np.load(native, allow_pickle=False)
        if native_depth.shape == (10, h, w, 1):
            native_depth = native_depth[..., 0]
        if native_depth.shape != (10, h, w):
            raise ValueError("Native depth must be aligned at RGB resolution")
        gt_xyz = lift_native_depth(native_depth, gt, k)
        depth_mask = mask & np.isfinite(gt_xyz).all(axis=-1)
        metrics["3D_est"] = {name: metric_values(values, gt_xyz, depth_mask, "3D_est_m")[0]
                             for name, values in method_xyz.items()}
        metrics["ADE_3D_est_m"] = metrics["3D_est"]["MolmoMotion"]["ADE_3D_est_m"]
        metrics["FDE_3D_est_m"] = metrics["3D_est"]["MolmoMotion"]["FDE_3D_est_m"]
        metrics["3D_est"]["depth_patch_window"] = "5x5 median of finite native metric depth with 0<Z<10 m; >=max(3,ceil(patch_area/2)) valid samples (13/25 for full patch)"
        np.savez_compressed(scene / "evaluation/ground_truth_3d_est.npz", xyz=gt_xyz, valid=depth_mask, point_ids=ids)
    else:
        metrics["3D_est"] = None
    np.savez_compressed(scene / "evaluation/evaluation_results.npz", point_ids=ids, ground_truth_2d=gt,
                        common_visibility_mask=mask, prediction_3d_at_5hz=future_xyz,
                        prediction_2d=methods["MolmoMotion"], static_2d=methods["static"],
                        constant_velocity_2d=methods["constant_velocity"],
                        prediction_indices=ALIGNMENT, timestamps_from_t0_s=TIMES,
                        **{f"{name}_error_px": error for name, error in errors.items()})
    with (scene / "evaluation/error_by_horizon.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "prediction_index_zero_based", "valid_points", *methods])
        for t in range(10):
            writer.writerow([TIMES[t], ALIGNMENT[t], int(mask[:, t].sum()),
                             *[measurements[name]["error_by_horizon"][t] for name in methods]])
    write_json(scene / "metrics.json", metrics)
    visualize(scene, history, history_uv, history_xyz, ids, k, prediction, future, gt, mask, methods, metrics)
    metrics["timing"]["evaluation_and_visualization_seconds"] = time.monotonic() - started
    write_json(scene / "metrics.json", metrics)
    print(json.dumps({"scene": str(scene), "ADE_2D_px": metrics["ADE_2D_px"],
                      "FDE_2D_px": metrics["FDE_2D_px"], "coverage": metrics["valid_point_coverage"]}), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--stage", choices=("track", "evaluate", "all"), default="all")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.stage in ("track", "all"):
        track_future(args.scene, args.max_side, args.iters, args.device)
    if args.stage in ("evaluate", "all"):
        evaluate(args.scene)


if __name__ == "__main__":
    main()
