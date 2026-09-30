"""Evaluate and visualize the saved single-episode F30 prediction without GPU."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.lines import Line2D
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/author_davis_bmx_trees_f30"
SAMPLE = ROOT / "examples/data/davis_bmx_trees"
SOURCE = ROOT / "data/pointmotionbench/davis"
METHODS = ["MolmoMotion", "Static", "Constant velocity"]
COLORS = {"MolmoMotion": "#e63946", "Static": "#3267d6", "Constant velocity": "#dd8b17"}
POINT_COLORS = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93", "#ff924c", "#40c9a2", "#f72585"]
LABEL_OFFSETS = [(-62, -48), (20, -57), (35, -8), (-70, 8), (40, 30), (-65, 48), (-72, -17), (40, 55)]


def check_original_hashes() -> None:
    """Guard the saved inference and input artifacts against accidental writes."""
    baseline = json.loads((RUN / "correction_baseline_hashes.json").read_text(encoding="utf-8-sig"))
    for name, expected in baseline.items():
        actual = hashlib.sha256((RUN / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Original artifact changed: {name}")


def parse_raw_forecast(text: str, points: int = 8, horizon: int = 30) -> np.ndarray:
    """Strictly decode the saved quantized text without the model's permissive parser."""
    match = re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*', text)
    if match is None:
        raise ValueError("Expected exactly one complete <tracks> block")
    frames = match.group(1).split(";")
    if len(frames) != horizon:
        raise ValueError(f"Expected {horizon} timestamps, got {len(frames)}")
    delta = np.empty((points, horizon, 3), dtype=np.float32)
    for step, frame in enumerate(frames):
        tokens = frame.split()
        if len(tokens) != 1 + 4 * points:
            raise ValueError(f"Timestamp {step}: expected {points} complete point records")
        if Decimal(tokens[0]) != Decimal(step + 3):
            raise ValueError(f"Unexpected or repeated timestamp at step {step}: {tokens[0]}")
        seen = set()
        for offset in range(1, len(tokens), 4):
            record = tokens[offset:offset + 4]
            if any(re.fullmatch(r"[+-]?\d+", token) is None for token in record):
                raise ValueError(f"Non-integer point record: {record}")
            point_id, x, y, z = map(int, record)
            if point_id not in range(1, points + 1) or point_id in seen:
                raise ValueError(f"Missing, duplicate or unexpected point ID at step {step + 3}")
            seen.add(point_id)
            delta[point_id - 1, step] = (x / 1000.0, y / 1000.0, z / 1000.0)
        if seen != set(range(1, points + 1)):
            raise ValueError(f"Missing point IDs at step {step + 3}")
    if not np.isfinite(delta).all():
        raise ValueError("Dequantized answer contains NaN/Inf")
    return delta


def verify_saved_data(pred: np.ndarray, saved: dict) -> dict:
    """Check official DAVIS arrays, frame indices, the raw answer, and anchor."""
    meta = json.loads((SAMPLE / "meta.json").read_text())
    ids = np.asarray(meta["point_indices"], dtype=np.int64)
    assert meta["video"] == "bmx-trees" and meta["obj"] == "bike_rider"
    assert meta["history_frame_indices"] == [0, 1, 2] and meta["t0_absolute"] == 2
    assert np.array_equal(ids, [9, 12, 17, 18, 27, 29, 32, 34])
    with np.load(SOURCE / "bmx-trees_2d.npz", allow_pickle=True) as file:
        source_2d = file["tracks"].item()["bike_rider"]
        source_vis = file["visibility"].item()["bike_rider"]
    with np.load(SOURCE / "bmx-trees_3d.npz", allow_pickle=True) as file:
        source_3d = file["points_3d"].item()["bike_rider"]
    future_frames = np.arange(3, 33)
    expected_history = source_3d[ids, :3].transpose(1, 0, 2)
    expected_future = source_3d[ids][:, future_frames]
    expected_2d = source_2d[future_frames][:, ids].transpose(1, 0, 2)
    expected_mask = source_vis[future_frames][:, ids].T & np.isfinite(expected_future).all(axis=-1)
    checks = {
        "point_indices": np.array_equal(saved["point_indices"], ids),
        "history_3d": np.array_equal(saved["history_3d"], expected_history, equal_nan=True),
        "history_2d": np.array_equal(saved["history_2d"], source_2d[:3, ids]),
        "future_3d": np.array_equal(saved["future_3d"], expected_future, equal_nan=True),
        "future_2d": np.array_equal(saved["future_2d"], expected_2d),
        "future_frame_indices": np.array_equal(saved["future_frame_indices"], future_frames),
        "future_times": np.array_equal(saved["future_seconds_from_t0"], np.arange(1, 31) / 24),
        "visibility_mask": np.array_equal(saved["valid"], expected_mask),
        "query_points_2d": np.array_equal(
            torch.load(SAMPLE / "points_2d_at_t0.pt", weights_only=True).numpy(), source_2d[2, ids]),
        "query_history_3d": np.array_equal(
            torch.load(SAMPLE / "points_3d_history.pt", weights_only=True).numpy(), expected_history),
        "intrinsics": np.array_equal(
            torch.load(SAMPLE / "intrinsics_K.pt", weights_only=True).numpy(), saved["intrinsics_K"]),
    }
    if not all(checks.values()):
        raise ValueError(f"Saved reference differs from source: {checks}")
    processor_inputs = torch.load(RUN / "processor_inputs.pt", map_location="cpu", weights_only=False)
    anchor = processor_inputs["anchor_3d"].detach().numpy().reshape(3)
    assert processor_inputs["future_horizon"] == 30 and processor_inputs["history_size"] == 3
    if not np.array_equal(anchor, expected_history[-1, 0]):
        raise ValueError("Saved processor anchor differs from the author history")
    delta = parse_raw_forecast((RUN / "model_output_raw.txt").read_text())
    reconstructed = delta + anchor
    if pred.shape != (8, 30, 3) or not np.isfinite(pred).all():
        raise ValueError("Saved model forecast is incomplete or nonfinite")
    max_diff = float(np.max(np.abs(reconstructed - pred)))
    if max_diff > 1e-7:
        raise ValueError(f"Decoded raw answer differs from saved prediction by {max_diff} m")
    valid = saved["valid"]
    if valid.shape != (8, 30) or valid.sum() != 215 or valid[:, -1].sum() != 8 or valid[:, 14].any():
        raise ValueError("Unexpected 3D reference coverage")
    return {
        "source_comparisons_exact": checks,
        "raw_timestamps": list(range(3, 33)),
        "raw_point_ids_per_timestamp": list(range(1, 9)),
        "raw_position_count": 240,
        "max_raw_vs_saved_difference_m": max_diff,
        "anchor_3d_m": anchor.tolist(),
        "valid_reference_pairs": int(valid.sum()),
        "valid_reference_points_at_frame_32": int(valid[:, -1].sum()),
        "valid_reference_points_at_frame_17": int(valid[:, 14].sum()),
        "horizon_seconds": float(saved["future_seconds_from_t0"][-1]),
    }


def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("prediction and GT must have matching (P,F,3) shape")
    if valid.shape != pred.shape[:2] or valid.dtype != np.bool_:
        raise ValueError("valid mask must be bool (P,F)")
    if not np.isfinite(pred).all():
        raise ValueError("prediction has NaN/Inf; cannot silently mask model omissions")
    if not np.isfinite(gt[valid]).all():
        raise ValueError("GT is nonfinite where mask is true")
    errors = np.linalg.norm(pred.astype(np.float64) - gt.astype(np.float64), axis=-1)
    counts = valid.sum(axis=0)
    per_time = [float(errors[valid[:, t], t].mean()) if counts[t] else None for t in range(valid.shape[1])]
    return {
        "ADE_m": float(errors[valid].mean()) if valid.any() else None,
        "FDE_m": float(errors[valid[:, -1], -1].mean()) if valid[:, -1].any() else None,
        "FDE_frame": pred.shape[1],
        "valid_pairs": int(valid.sum()),
        "total_pairs": int(valid.size),
        "FDE_valid_points": int(valid[:, -1].sum()),
        "error_by_horizon_m": per_time,
        "valid_points_by_horizon": counts.tolist(),
    }


def self_test() -> None:
    gt = np.zeros((2, 3, 3), dtype=np.float32)
    full = np.ones((2, 3), dtype=bool)
    assert metrics(gt, gt, full)["ADE_m"] == 0.0
    shifted = gt.copy()
    shifted[..., 0] = 2
    assert metrics(shifted, gt, full)["ADE_m"] == 2.0
    assert metrics(shifted, gt, full)["FDE_m"] == 2.0
    history = np.array([[[0, 0, 1]], [[1, 0, 1]], [[2, 0, 1]]], dtype=np.float32)
    motion = history[-1, 0] + np.arange(1, 4)[:, None] * (history[-1, 0] - history[-2, 0])
    assert np.array_equal(motion[:, 0], [3, 4, 5])
    uniform_gt = motion[None, :, :]
    assert metrics(uniform_gt, uniform_gt, np.ones((1, 3), bool))["ADE_m"] == 0.0
    partial = full.copy()
    partial[:, -1] = False
    assert metrics(shifted, gt, partial)["FDE_m"] is None
    empty = np.zeros_like(full)
    assert metrics(gt, gt, empty)["ADE_m"] is None
    assert metrics(gt, gt, empty)["FDE_m"] is None
    try:
        bad = gt.copy()
        bad[0, 0, 0] = np.nan
        metrics(bad, gt, partial)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN prediction was accepted")
    for bad_value in (np.inf, -np.inf):
        bad = gt.copy()
        bad[0, 0, 0] = bad_value
        try:
            metrics(bad, gt, partial)
        except ValueError:
            pass
        else:
            raise AssertionError("infinite prediction was accepted")
    bad_gt = gt.copy()
    bad_gt[0, -1, 0] = np.nan
    try:
        metrics(gt, bad_gt, full)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite valid GT was accepted")
    assert metrics(gt, bad_gt, partial)["FDE_m"] is None
    try:
        metrics(gt[:, :2], gt, full)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete forecast was accepted")
    print("synthetic metric checks: PASS")


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = xyz[..., 2]
    px = np.full((*xyz.shape[:-1], 2), np.nan, dtype=np.float64)
    good = np.isfinite(xyz).all(axis=-1) & (z > 0)
    px[..., 0][good] = k[0, 0] * xyz[..., 0][good] / z[good] + k[0, 2]
    px[..., 1][good] = k[1, 1] * xyz[..., 1][good] / z[good] + k[1, 2]
    return px


def plot_error(rows: dict, seconds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name in METHODS:
        y = [np.nan if x is None else x for x in rows[name]["error_by_horizon_m"]]
        ax.plot(seconds, y, marker=".", label=name, color=COLORS[name])
    ax.set(xlabel="Seconds after t0 (24 fps)", ylabel="Mean 3D error (m)",
           title="bmx-trees: error by forecast horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "error_vs_horizon.png", dpi=170)
    plt.close(fig)


def plot_tracks(pred: np.ndarray, gt: np.ndarray, history: np.ndarray,
                valid: np.ndarray, ids: np.ndarray) -> None:
    """Compare trajectories in their stored common 3D coordinate system."""
    fig = plt.figure(figsize=(17, 9.5))
    visible_gt = np.where(valid[..., None], gt, np.nan)
    finite = np.concatenate((pred.reshape(-1, 3), visible_gt.reshape(-1, 3), history.reshape(-1, 3)))
    finite = finite[np.isfinite(finite).all(axis=1)]
    low, high = finite.min(axis=0), finite.max(axis=0)
    center = (low + high) / 2
    half_span = max(float((high - low).max()) / 2, 0.01) * 1.08
    for i, point_id in enumerate(ids):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        h = history[:, i]
        ax.plot(h[:, 0], h[:, 1], h[:, 2], color=POINT_COLORS[i], lw=2, marker="o", markersize=3)
        ax.scatter(*h[-1], color="black", marker="*", s=72, depthshade=False)
        ax.plot(pred[i, :, 0], pred[i, :, 1], pred[i, :, 2], color=COLORS["MolmoMotion"], lw=1.8)
        ax.plot(visible_gt[i, :, 0], visible_gt[i, :, 1], visible_gt[i, :, 2],
                color="#198754", lw=1.8, linestyle="--")
        ax.set(xlim=(center[0] - half_span, center[0] + half_span),
               ylim=(center[1] - half_span, center[1] + half_span),
               zlim=(center[2] - half_span, center[2] + half_span),
               xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title=f"ID {point_id}")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-65)
    legend = [
        Line2D([0], [0], color=POINT_COLORS[0], marker="o", label="Observed frames 0–2 (point color varies)"),
        Line2D([0], [0], color="black", marker="*", linestyle="", label="t0 = frame 2"),
        Line2D([0], [0], color=COLORS["MolmoMotion"], label="MolmoMotion, frames 3–32"),
        Line2D([0], [0], color="#198754", linestyle="--", label="3D reference where valid"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("DAVIS bmx-trees: saved 3D trajectories in common coordinates; gaps mean no valid reference", fontsize=14)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.10, wspace=0.02, hspace=0.18)
    fig.savefig(RUN / "trajectory_comparison.png", dpi=150)
    plt.close(fig)


def draw_frame(t: int, history_xy: np.ndarray, gt_xy: np.ndarray,
               valid: np.ndarray, ids: np.ndarray) -> Image.Image:
    with Image.open(RUN / f"frames/{t:05d}.jpg") as file:
        frame = file.convert("RGB")
    canvas = Image.new("RGB", (854, 520), "#111111")
    canvas.paste(frame, (0, 40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    stage = "INPUT" if t < 3 else "2D REFERENCE"
    t0_label = "  (t0)" if t == 2 else ""
    draw.text((12, 9), f"DAVIS bmx-trees  {stage} frame {t:02d}/32{t0_label}  {(t-2)/24:+.3f}s", fill="white", font=font)
    draw.text((12, 495), "Colored markers: original 2D point annotations, labeled by point ID", fill="white", font=font)
    step = t - 3
    shown = 0
    for i, point_id in enumerate(ids):
        if t >= 3 and not valid[i, step]:
            continue
        xy = history_xy[t, i] if t < 3 else gt_xy[i, step]
        if np.isfinite(xy).all() and -20 <= xy[0] < 874 and -20 <= xy[1] < 500:
            x, y = float(xy[0]), float(xy[1] + 40)
            color = POINT_COLORS[i]
            draw.ellipse((x-5, y-5, x+5, y+5), fill=color, outline="black", width=1)
            dx, dy = LABEL_OFFSETS[i]
            label_x = min(max(x + dx, 2), 815)
            label_y = min(max(y + dy, 42), 465)
            draw.line((x, y, label_x + 8, label_y + 8), fill=color, width=2)
            draw.rounded_rectangle((label_x, label_y, label_x+34, label_y+22), radius=3, fill="#111111", outline=color, width=2)
            draw.text((label_x+4, label_y+1), str(point_id), fill=color, font=font)
            shown += 1
    if t >= 3 and shown == 0:
        draw.text((12, 48), "No valid 2D reference points on this frame", fill="white", font=font)
    return canvas


def main() -> None:
    self_test()
    check_original_hashes()
    with np.load(RUN / "prediction.npz", allow_pickle=False) as file:
        pred = file["future_3d"]
    with np.load(RUN / "ground_truth.npz", allow_pickle=False) as file:
        saved = {name: file[name] for name in file.files}
    audit = verify_saved_data(pred, saved)
    gt = saved["future_3d"]
    gt_xy = saved["future_2d"]
    valid = saved["valid"]
    ids = saved["point_indices"]
    seconds = saved["future_seconds_from_t0"]
    history = saved["history_3d"]
    history_xy = saved["history_2d"]
    k = saved["intrinsics_K"]
    steps = np.arange(1, 31, dtype=np.float32)
    static = np.broadcast_to(history[-1, :, None, :], pred.shape).copy()
    velocity = history[-1, :, None, :] + steps[None, :, None] * (history[-1] - history[-2])[:, None, :]
    predictions = {"MolmoMotion": pred, "Static": static, "Constant velocity": velocity}
    rows = {name: metrics(p, gt, valid) for name, p in predictions.items()}
    previous = json.loads((RUN / "metrics.json").read_text())
    for name in METHODS:
        for field in ("ADE_m", "FDE_m"):
            if abs(rows[name][field] - previous[name][field]) > 1e-7:
                raise ValueError(f"Recomputed {name} {field} differs from original result")
        if rows[name]["valid_pairs"] != 215 or rows[name]["FDE_valid_points"] != 8:
            raise ValueError(f"Unexpected metric mask for {name}")
    (RUN / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (RUN / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "ADE_m", "FDE_m_at_t30", "valid_pairs", "total_pairs", "FDE_valid_points"])
        for name in METHODS:
            row = rows[name]
            writer.writerow([name, row["ADE_m"], row["FDE_m"], row["valid_pairs"], row["total_pairs"], row["FDE_valid_points"]])
    plot_error(rows, seconds)
    history_px = project(history, k)
    history_err = np.linalg.norm(history_px - history_xy, axis=-1)
    if not np.isfinite(history_err).all():
        raise ValueError("Historical 3D reprojection is nonfinite")
    history_mean = history_err.mean(axis=1)
    if not (history_mean[0] < 1 and history_mean[1] > 20 and history_mean[2] > 50):
        raise ValueError("Unexpected coordinate frame evidence")
    gt_px = project(gt, k)
    projection_err = np.linalg.norm(gt_px - gt_xy, axis=-1)
    (RUN / "projection_check.json").write_text(json.dumps({
        "historical_frame_indices": [0, 1, 2],
        "history_3d_reprojected_with_saved_K_mean_error_px_by_frame": history_mean.tolist(),
        "history_3d_reprojected_with_saved_K_error_px_by_point": history_err.tolist(),
        "future_3d_reprojected_with_saved_K_mean_error_px_where_valid": float(projection_err[valid].mean()),
        "future_3d_reprojected_with_saved_K_max_error_px_where_valid": float(projection_err[valid].max()),
        "interpretation": "Saved 3D coordinates are consistent with the first sequence frame's camera/world anchor, not camera t0=frame 2. A single K without per-frame extrinsics cannot project common-frame 3D coordinates onto later moving RGB frames.",
    }, indent=2) + "\n")
    plot_tracks(pred, gt, history, valid, ids)
    input_sheet = Image.new("RGB", (854 * 3, 520), "black")
    for j, t in enumerate((0, 1, 2)):
        input_sheet.paste(draw_frame(t, history_xy, gt_xy, valid, ids), (854*j, 0))
    input_sheet.save(RUN / "input_history_points.png")
    selected = [0, 1, 2, 3, 17, 32]
    sheet = Image.new("RGB", (854 * 3, 520 * 2), "black")
    for j, t in enumerate(selected):
        sheet.paste(draw_frame(t, history_xy, gt_xy, valid, ids), (854*(j % 3), 520*(j // 3)))
    sheet.save(RUN / "observed_and_future.png")
    with imageio.get_writer(RUN / "real_continuation_gt.mp4", fps=24, codec="libx264", quality=8,
                            macro_block_size=2) as writer:
        for t in range(33):
            writer.append_data(np.asarray(draw_frame(t, history_xy, gt_xy, valid, ids)))
    with imageio.get_reader(RUN / "real_continuation_gt.mp4") as reader:
        assert reader.count_frames() == 33
        assert abs(reader.get_meta_data()["fps"] - 24) < 1e-6
        decoded_sheet = Image.new("RGB", (854 * 2, 520 * 2), "black")
        for j, t in enumerate((0, 2, 17, 32)):
            decoded = reader.get_data(t)
            assert decoded.shape == (520, 854, 3)
            expected = np.asarray(draw_frame(t, history_xy, gt_xy, valid, ids))
            assert np.abs(decoded.astype(np.int16) - expected.astype(np.int16)).mean() < 15
            decoded_sheet.paste(Image.fromarray(decoded), (854 * (j % 2), 520 * (j // 2)))
    decoded_sheet.save(RUN / "video_decoded_check.png")
    check_original_hashes()
    (RUN / "correction_validation.json").write_text(json.dumps({
        "saved_data_audit": audit,
        "metrics_match_original_within_m": 1e-7,
        "original_sha256_unchanged": True,
        "video_frames": 33,
        "video_fps": 24,
        "decoded_video_frames_checked": [0, 2, 17, 32],
    }, indent=2) + "\n")
    print(json.dumps({name: {k: rows[name][k] for k in ("ADE_m", "FDE_m", "valid_pairs", "FDE_valid_points")}
                      for name in METHODS}, indent=2))


if __name__ == "__main__":
    main()
