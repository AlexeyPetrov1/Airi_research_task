"""Relate frozen historical geometry diagnostics to observed 2D forecast error."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from evaluate_fmb_quantitative_2d import interpolate, measure, project
from run_fmb_ablation_suite import ROOT, RUNS, write_json


METHODS = {
    "sensor": ("history_sensor_3d.npy", "prediction_15hz.npy"),
    "planar PnP": ("history_pnp_3d.npy", "variants/pnp/prediction_15hz.npy"),
    "CAD silhouette + TCP": ("history_cad_silhouette_tcp_3d.npy",
                             "variants/cad_silhouette/prediction_15hz.npy"),
    "MoGe-2 raw": ("history_moge2_raw_3d.npy",
                   "ablation_suite_v1/moge2_raw/prediction_15hz.npy"),
    "MoGe-2 history scaled": ("history_moge2_history_scaled_3d.npy",
                              "ablation_suite_v1/moge2_scaled/prediction_15hz.npy"),
}


def pair_distance(history: np.ndarray) -> np.ndarray:
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    return np.stack([np.linalg.norm(history[:, i] - history[:, j], axis=1) for i, j in pairs])


def main() -> None:
    rows = []
    for episode, run in RUNS.items():
        sensor = np.load(run / "history_sensor_3d.npy").astype(float)
        k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        gt = np.load(run / "gt_2d.npy")
        mask = np.load(run / "validity_mask.npy")
        for method, (history_file, prediction_file) in METHODS.items():
            if not (run / history_file).exists() or not (run / prediction_file).exists():
                continue
            history = np.load(run / history_file).astype(float)
            future = np.load(run / prediction_file).astype(float)
            if history.shape != (3, 8, 3) or future.shape != (8, 30, 3):
                raise ValueError(f"Unexpected shape: {episode}/{method}")
            if not np.isfinite(history).all() or not np.isfinite(future).all():
                raise ValueError(f"Nonfinite values: {episode}/{method}")
            geometry_gap = np.linalg.norm(history - sensor, axis=2) * 1000
            rigidity = pair_distance(history)
            pair_delta = np.abs(np.diff(rigidity, axis=1)) * 1000
            metrics, _ = measure(project(interpolate(future), k), gt, mask)
            rows.append({
                "episode": episode, "geometry": method,
                "history_file": history_file, "prediction_file": prediction_file,
                "median_XYZ_disagreement_with_sensor_mm": float(np.median(geometry_gap)),
                "median_abs_Z_disagreement_with_sensor_mm": float(np.median(np.abs(history[..., 2] - sensor[..., 2])) * 1000),
                "median_adjacent_pair_distance_change_mm": float(np.median(pair_delta)),
                "p90_adjacent_pair_distance_change_mm": float(np.percentile(pair_delta, 90)),
                "ADE_2D_px": metrics["ADE_2D_px"],
                "FDE_2D_px": metrics["FDE_2D_px"],
                "valid_pairs": metrics["valid_pairs"],
            })
    dest = ROOT / "runs/fmb_geometry_forecast_comparison_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(dest / "comparison.json", {
        "rows": rows,
        "interpretation": "Sensor depth is an imperfect reference, not independent 3D ground truth. CAD/PnP points are approximate correspondences on a smooth face. Two episodes and few methods do not support a population correlation estimate.",
        "geometry_prepared_without_future_GT": True,
        "same_nominal_K_for_projection": True,
    })
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    colors = {"sensor": "tab:blue", "planar PnP": "tab:orange", "CAD silhouette + TCP": "tab:green",
              "MoGe-2 raw": "tab:purple", "MoGe-2 history scaled": "tab:red"}
    marks = {"first": "o", "second": "s"}
    for row in rows:
        for ax, key in zip(axes, ("median_abs_Z_disagreement_with_sensor_mm",
                                  "median_adjacent_pair_distance_change_mm")):
            ax.scatter(row[key], row["ADE_2D_px"], marker=marks[row["episode"]],
                       color=colors[row["geometry"]], s=80)
    inset = axes[0].inset_axes([.48, .51, .49, .43])
    for row in rows:
        inset.scatter(row["median_abs_Z_disagreement_with_sensor_mm"], row["ADE_2D_px"],
                      marker=marks[row["episode"]], color=colors[row["geometry"]], s=34)
    inset.set(xlim=(-1, 35), ylim=(20, 135), title="Z gap 0..35 mm")
    inset.tick_params(labelsize=7)
    inset.grid(alpha=.2)
    axes[0].set(xlabel="Median historical Z gap to sensor (mm)", ylabel="ADE 2D (px)")
    axes[1].set(xlabel="Median pair-distance change between historical frames (mm)",
                ylabel="ADE 2D (px)")
    for ax, key in zip(axes, ("median_abs_Z_disagreement_with_sensor_mm",
                              "median_adjacent_pair_distance_change_mm")):
        maximum = max(row[key] for row in rows)
        ax.set_xlim(-max(1, maximum * .03), max(1, maximum * 1.22))
    for ax in axes:
        top = max(row["ADE_2D_px"] for row in rows)
        ax.set_ylim(0, top * 1.16)
        ax.grid(alpha=.3)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=color,
                      label=name, markersize=7) for name, color in colors.items()]
    handles += [Line2D([0], [0], marker=mark, linestyle="none", color="black",
                       label=f"{episode} episode", markersize=7)
                for episode, mark in marks.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=4, fontsize=8)
    fig.suptitle("Historical geometry diagnostics vs saved 2D forecasts", y=.99)
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.23, wspace=.24)
    fig.savefig(dest / "geometry_vs_forecast.png", dpi=170)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
