"""Postprocess frozen FMB forecasts with a per-frame proper rigid fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_fmb_quantitative_2d import GT_TIMES, interpolate, measure, project
from run_fmb_ablation_suite import RUNS, write_json


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-vector Kabsch: fitted = source @ rotation + translation."""
    a = np.asarray(source, dtype=float)
    b = np.asarray(target, dtype=float)
    if a.shape != (8, 3) or b.shape != (8, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected finite 8x3 source and target")
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = bc - ac @ rotation
    return a @ rotation + translation, rotation, translation


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    return np.asarray([np.linalg.norm(points[..., i, :] - points[..., j, :], axis=-1)
                       for i, j in pairs])


def evaluate_one(name: str, run: Path) -> dict:
    history = np.load(run / "history_sensor_3d.npy").astype(float)
    source = history[-1]
    original = np.load(run / "prediction_15hz.npy").astype(float)
    if original.shape != (8, 30, 3):
        raise ValueError("Unexpected forecast shape")
    fixed = np.empty_like(original)
    proper = []
    residual = []
    for t in range(30):
        fixed[:, t], r, _ = rigid_fit(source, original[:, t])
        proper.append(float(np.linalg.det(r)))
        residual.append(float(np.sqrt(np.mean(np.sum((fixed[:, t] - original[:, t])**2, axis=1)))))
    dest = run / "rigid_correction_v1"
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / "corrected_prediction_15hz.npy", fixed.astype("float32"))
    original10, fixed10 = interpolate(original), interpolate(fixed)
    np.save(dest / "corrected_prediction_10hz.npy", fixed10.astype("float32"))
    k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
    uv_original, uv_fixed = project(original10, k), project(fixed10, k)
    np.save(dest / "corrected_prediction_2d.npy", uv_fixed.astype("float32"))
    gt = np.load(run / "gt_2d.npy")
    mask = np.load(run / "validity_mask.npy")
    alt = np.load(run / "gt_dense_flow_alternative.npy")
    original_metrics, original_err = measure(uv_original, gt, mask)
    fixed_metrics, fixed_err = measure(uv_fixed, gt, mask)
    original_alt, _ = measure(uv_original, alt, mask)
    fixed_alt, _ = measure(uv_fixed, alt, mask)
    baseline_distances = pairwise_distances(source)[:, None]
    rigid_error = pairwise_distances(original.transpose(1, 0, 2)) - baseline_distances
    rigid_after = pairwise_distances(fixed.transpose(1, 0, 2)) - baseline_distances
    report = {
        "episode": name,
        "status": "EVALUATED",
        "method": "proper SO(3) Kabsch per forecast frame, source is final historical 3D points",
        "fit_uses_future_ground_truth": False,
        "original": original_metrics,
        "rigid_corrected": fixed_metrics,
        "original_alternative_dense_flow_GT": original_alt,
        "rigid_corrected_alternative_dense_flow_GT": fixed_alt,
        "mean_pairwise_distance_change_before_mm": float(np.abs(rigid_error).mean() * 1000),
        "mean_pairwise_distance_change_after_mm": float(np.abs(rigid_after).mean() * 1000),
        "mean_fit_residual_mm": float(np.mean(residual) * 1000),
        "max_fit_residual_mm": float(np.max(residual) * 1000),
        "rotation_determinant_range": [float(min(proper)), float(max(proper))],
        "mean_projected_point_shift_px": float(np.linalg.norm(uv_fixed - uv_original, axis=2)[mask].mean()),
    }
    write_json(dest / "evaluation.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(GT_TIMES, original_err.mean(axis=0), label="Original", color="tab:orange")
    axes[0].plot(GT_TIMES, fixed_err.mean(axis=0), label="Rigid fit", color="tab:blue")
    axes[0].set(xlabel="Future time (s, nominal)", ylabel="2D error (px)")
    axes[0].grid(alpha=.3)
    axes[0].legend()
    axes[1].plot(np.arange(1, 31) / 15, np.abs(rigid_error).mean(axis=0) * 1000,
                 label="Before", color="tab:orange")
    axes[1].plot(np.arange(1, 31) / 15, np.abs(rigid_after).mean(axis=0) * 1000,
                 label="After", color="tab:blue")
    axes[1].set(xlabel="Future time (s, nominal)", ylabel="Mean pair distance change (mm)")
    axes[1].grid(alpha=.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(dest / "rigid_correction.png", dpi=160)
    plt.close(fig)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    args = parser.parse_args()
    for name in args.episodes:
        print(json.dumps(evaluate_one(name, RUNS[name]), indent=2))


if __name__ == "__main__":
    main()
