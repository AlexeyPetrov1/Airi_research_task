"""Additional visible-motion metrics from the frozen FMB 2D annotations."""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_fmb_quantitative_2d import interpolate, project
from inspect_fmb_2d_window import red_component
from run_fmb_ablation_suite import ROOT, RUNS, write_json


def angle_degrees(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.linalg.norm(a) < 1e-8 or np.linalg.norm(b) < 1e-8:
        return None
    cosine = float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    return float(np.degrees(np.arccos(cosine)))


def main() -> None:
    heatmaps = []
    rows = []
    for name, run in RUNS.items():
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source = np.load(ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"],
                         allow_pickle=True).item()
        _, bbox = red_component(source["obs/side_1"][config["t0"]])
        x, y, width, height = [int(v) for v in bbox[:4]]
        size_px = float(np.hypot(width, height))
        gt = np.load(run / "gt_2d.npy").astype(float)
        mask = np.load(run / "validity_mask.npy")
        t0 = np.load(run / "points_2d_at_t0.npy").astype(float)
        k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        model = project(interpolate(np.load(run / "prediction_15hz.npy")), k)
        predictions = {"MolmoMotion": model,
                       "stationary": np.load(run / "baseline_stationary_2d.npy"),
                       "constant_velocity": np.load(run / "baseline_constant_velocity_2d.npy")}
        repeat = json.loads((run / "annotation_repeatability.json").read_text(encoding="utf8"))
        threshold = max(2., 2 * repeat["p90_px"])
        for method, pred in predictions.items():
            error = np.linalg.norm(pred - gt, axis=2)
            good = mask[:, -1]
            gt_delta = gt[:, -1] - t0
            pred_delta = pred[:, -1] - t0
            eligible = good & (np.linalg.norm(gt_delta, axis=1) > threshold)
            angles = [angle_degrees(gt_delta[i], pred_delta[i]) for i in range(8)
                      if eligible[i] and np.linalg.norm(pred_delta[i]) > threshold]
            angles = [a for a in angles if a is not None]
            ratios = [float(np.linalg.norm(pred_delta[i]) / np.linalg.norm(gt_delta[i]))
                      for i in range(8) if eligible[i]]
            rows.append({
                "episode": name, "method": method, "object_bbox_xywh": [x, y, width, height],
                "object_bbox_diagonal_px": size_px,
                "ADE_over_t0_object_bbox_diagonal": float(error[mask].mean() / size_px),
                "FDE_over_t0_object_bbox_diagonal": float(error[:, -1][good].mean() / size_px),
                "within_5px_fraction": float(np.mean(error[mask] <= 5)),
                "within_10px_fraction": float(np.mean(error[mask] <= 10)),
                "within_20px_fraction": float(np.mean(error[mask] <= 20)),
                "final_direction_error_median_deg": float(np.median(angles)) if angles else None,
                "final_displacement_ratio_median": float(np.median(ratios)) if ratios else None,
                "final_direction_evaluable_points": len(angles),
                "final_displacement_evaluable_points": len(ratios),
                "movement_threshold_px": threshold,
                "valid_pairs": int(mask.sum()),
            })
        heatmaps.append(np.linalg.norm(model - gt, axis=2))
    dest = ROOT / "runs/fmb_extended_2d_metrics_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "metrics.csv").open("w", newline="", encoding="utf8") as f:
        keys = list(rows[0])
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(dest / "metrics.json", {
        "rows": rows,
        "object_scale": "Diagonal of red peg mask bounding box at t0, same segmentation for each method within episode",
        "movement_threshold": "max(2 px, 2 times p90 direct-repeat algorithmic discrepancy)",
        "caveat": "Object mask, 2D tracks and their repeatability share RGB segmentation assumptions. Direction is undefined when either observed or predicted displacement is at or below the movement threshold; it is not imputed as 0 degrees.",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, layout="constrained")
    vmax = float(max(np.max(e) for e in heatmaps))
    for ax, name, error in zip(axes, RUNS, heatmaps):
        im = ax.imshow(error, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax,
                       cmap="magma")
        ax.set(title=f"{name}: MolmoMotion error", xlabel="Future time (s, nominal)",
               xticks=[0, 4, 9, 14, 19], xticklabels=["0.1", "0.5", "1.0", "1.5", "2.0"])
    axes[0].set(ylabel="Point ID", yticks=list(range(8)))
    fig.colorbar(im, ax=axes, label="2D error (px)", shrink=.85)
    fig.savefig(dest / "point_time_error_heatmap.png", dpi=170)
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
