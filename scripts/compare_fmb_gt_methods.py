"""Re-score the same frozen forecasts against both RGB-derived 2D trackers."""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_fmb_quantitative_2d import interpolate, measure, project
from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json


def predictions(run, k):
    tracks = {
        "MolmoMotion sensor": np.load(run / "prediction_2d.npy"),
        "stationary": np.load(run / "baseline_stationary_2d.npy"),
        "constant velocity": np.load(run / "baseline_constant_velocity_2d.npy"),
        "MolmoMotion PnP": project(interpolate(np.load(run / "variants/pnp/prediction_15hz.npy")), k),
        "MolmoMotion CAD": project(interpolate(np.load(run / "variants/cad_silhouette/prediction_15hz.npy")), k),
    }
    for name in ("moge2_raw", "moge2_scaled"):
        path = run / SUITE / name / "prediction_2d.npy"
        if path.exists():
            tracks[f"MolmoMotion {name}"] = np.load(path)
    return tracks


def main() -> None:
    rows = []
    for episode, run in RUNS.items():
        k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        mask = np.load(run / "validity_mask.npy")
        references = {"sequential red-mask ECC": np.load(run / "gt_2d.npy"),
                      "RGB dense optical flow": np.load(run / "gt_dense_flow_alternative.npy")}
        for name, track in predictions(run, k).items():
            for reference, gt in references.items():
                metrics, _ = measure(track, gt, mask)
                rows.append({"episode": episode, "forecast": name, "reference": reference,
                             "ADE_2D_px": metrics["ADE_2D_px"],
                             "FDE_2D_px": metrics["FDE_2D_px"],
                             "valid_pairs": metrics["valid_pairs"]})
    dest = ROOT / "runs/fmb_gt_sensitivity_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(dest / "comparison.json", {
        "rows": rows, "same_mask_for_both_reference_tracks": True,
        "reference_limit": "Both tracks are algorithmic RGB proxies on a low-texture part, not independent material-point ground truth",
        "no_method_selected_by_evaluation_future": True,
    })
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    for ax, episode in zip(axes, RUNS):
        subset = [row for row in rows if row["episode"] == episode]
        names = list(dict.fromkeys(row["forecast"] for row in subset))
        x = np.arange(len(names))
        for offset, ref, color in ((-.19, "sequential red-mask ECC", "#4e79a7"),
                                    (.19, "RGB dense optical flow", "#f28e2b")):
            ax.bar(x + offset, [next(row["ADE_2D_px"] for row in subset
                                      if row["forecast"] == name and row["reference"] == ref)
                                for name in names], width=.37, label=ref, color=color)
        ax.set(title=f"{episode}: ADE under two frozen RGB-derived references",
               ylabel="ADE 2D (px)", xticks=x, xticklabels=names)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8)
    fig.savefig(dest / "comparison.png", dpi=160)
    plt.close(fig)
    print(len(rows), "GT sensitivity rows")


if __name__ == "__main__":
    main()
