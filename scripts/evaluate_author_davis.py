"""Evaluate and visualize the saved single-episode F30 prediction without GPU."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/author_davis_bmx_trees_f30"
METHODS = ["MolmoMotion", "Static", "Constant velocity"]
COLORS = {"MolmoMotion": "#e63946", "Static": "#3267d6", "Constant velocity": "#dd8b17"}


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
    partial = full.copy()
    partial[:, -1] = False
    assert metrics(shifted, gt, partial)["FDE_m"] is None
    empty = np.zeros_like(full)
    assert metrics(gt, gt, empty)["ADE_m"] is None
    try:
        bad = gt.copy()
        bad[0, 0, 0] = np.nan
        metrics(bad, gt, partial)
    except ValueError:
        pass
    else:
        raise AssertionError("NaN prediction was accepted")
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
    ax.set(xlabel="Seconds after t₀ (24 fps)", ylabel="Mean 3D error (m)",
           title="bmx-trees: error by forecast horizon")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "error_vs_horizon.png", dpi=170)
    plt.close(fig)


def plot_tracks(pred_px: np.ndarray, gt_px: np.ndarray, valid: np.ndarray, ids: np.ndarray) -> None:
    with Image.open(RUN / "frames/00002.jpg") as file:
        frame = np.asarray(file.convert("RGB"))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(frame)
    for i, point_id in enumerate(ids):
        idx = np.flatnonzero(valid[i])
        ax.plot(pred_px[i, :, 0], pred_px[i, :, 1], color=COLORS["MolmoMotion"], alpha=0.5, lw=1.7)
        ax.plot(gt_px[i, idx, 0], gt_px[i, idx, 1], color="#21a366", alpha=0.65, lw=1.7)
        ax.text(gt_px[i, 0, 0], gt_px[i, 0, 1], str(point_id), color="white", fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.5, "pad": 1})
    ax.plot([], [], color=COLORS["MolmoMotion"], label="MolmoMotion")
    ax.plot([], [], color="#21a366", label="3D reference where visible")
    all_x = np.concatenate([pred_px[..., 0].ravel(), gt_px[..., 0].ravel()])
    right = max(854, float(np.nanmax(all_x)) + 30)
    ax.set(xlim=(0, right), ylim=(480, 0), xlabel="x in fixed t₀ camera (pixels)", ylabel="y (pixels)",
           title="3D forecast and 3D reference in the fixed t₀ camera")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RUN / "trajectory_comparison.png", dpi=170)
    plt.close(fig)


def draw_frame(t: int, gt_xy: np.ndarray, valid: np.ndarray, ids: np.ndarray) -> Image.Image:
    with Image.open(RUN / f"frames/{t:05d}.jpg") as file:
        frame = file.convert("RGB")
    canvas = Image.new("RGB", (854, 520), "#111111")
    canvas.paste(frame, (0, 40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    draw.text((12, 9), f"DAVIS bmx-trees  frame {t:02d}/32  +{(t-2)/24:.3f}s from t₀", fill="white", font=font)
    draw.text((12, 495), "Green cross: visible 2D reference on actual frame; label: point index", fill="white", font=font)
    if t < 3:
        return canvas
    step = t - 3
    for i, point_id in enumerate(ids):
        if valid[i, step]:
            g = gt_xy[i, step]
            if np.isfinite(g).all() and -20 <= g[0] < 874 and -20 <= g[1] < 500:
                x, y = g[0], g[1] + 40
                draw.line((x-5, y-5, x+5, y+5), fill="#35ee75", width=3)
                draw.line((x-5, y+5, x+5, y-5), fill="#35ee75", width=3)
                draw.text((x+6, y-8), str(point_id), fill="#35ee75", font=font)
    return canvas


def main() -> None:
    self_test()
    with np.load(RUN / "prediction.npz", allow_pickle=False) as file:
        pred = file["future_3d"]
    with np.load(RUN / "ground_truth.npz", allow_pickle=False) as file:
        gt = file["future_3d"]
        gt_xy = file["future_2d"]
        valid = file["valid"]
        ids = file["point_indices"]
        seconds = file["future_seconds_from_t0"]
        history = file["history_3d"]
        k = file["intrinsics_K"]
    if pred.shape != (8, 30, 3) or not np.isfinite(pred).all():
        raise ValueError("saved model forecast is incomplete")
    steps = np.arange(1, 31, dtype=np.float32)
    static = np.broadcast_to(history[-1, :, None, :], pred.shape).copy()
    velocity = history[-1, :, None, :] + steps[None, :, None] * (history[-1] - history[-2])[:, None, :]
    predictions = {"MolmoMotion": pred, "Static": static, "Constant velocity": velocity}
    rows = {name: metrics(p, gt, valid) for name, p in predictions.items()}
    (RUN / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (RUN / "metrics.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "ADE_m", "FDE_m_at_t30", "valid_pairs", "total_pairs", "FDE_valid_points"])
        for name in METHODS:
            row = rows[name]
            writer.writerow([name, row["ADE_m"], row["FDE_m"], row["valid_pairs"], row["total_pairs"], row["FDE_valid_points"]])
    plot_error(rows, seconds)
    pred_px = project(pred, k)
    gt_px = project(gt, k)
    projection_err = np.linalg.norm(gt_px - gt_xy, axis=-1)
    (RUN / "projection_check.json").write_text(json.dumps({
        "gt_3d_to_2d_mean_error_px": float(projection_err[valid].mean()),
        "gt_3d_to_2d_max_error_px": float(projection_err[valid].max()),
        "pred_positive_depth_count": int((pred[..., 2] > 0).sum()),
        "conclusion": "Future RGB frames use changing camera coordinates. No per-frame extrinsics are available; do not overlay fixed-t0 3D projections onto them.",
    }, indent=2) + "\n")
    plot_tracks(pred_px, gt_px, valid, ids)
    selected = [2, 3, 8, 17, 32]
    sheet = Image.new("RGB", (854 * 3, 520 * 2), "black")
    for j, t in enumerate(selected):
        sheet.paste(draw_frame(t, gt_xy, valid, ids), (854*(j % 3), 520*(j // 3)))
    sheet.save(RUN / "observed_and_future.png")
    with imageio.get_writer(RUN / "real_continuation_gt.mp4", fps=24, codec="libx264", quality=8,
                            macro_block_size=2) as writer:
        for t in range(33):
            writer.append_data(np.asarray(draw_frame(t, gt_xy, valid, ids)))
    with imageio.get_reader(RUN / "real_continuation_gt.mp4") as reader:
        assert reader.count_frames() == 33
        assert reader.get_data(0).shape == (520, 854, 3)
        assert reader.get_data(32).shape == (520, 854, 3)
    print(json.dumps({name: {k: rows[name][k] for k in ("ADE_m", "FDE_m", "valid_pairs", "FDE_valid_points")}
                      for name in METHODS}, indent=2))


if __name__ == "__main__":
    main()
