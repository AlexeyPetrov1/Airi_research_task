"""Fixed-depth 2x2 comparison: sensor/MoGe-2 depth and nominal/0.9 focal K."""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json


def metric(run, variant: str | None) -> dict:
    if variant is None:
        return json.loads((run / "metrics.json").read_text(encoding="utf8"))["metrics"]["MolmoMotion"]
    evaluation = json.loads((run / SUITE / variant / "evaluation.json").read_text(encoding="utf8"))
    return evaluation["metrics"]


def main() -> None:
    records = []
    interactions = {}
    for episode, run in RUNS.items():
        conditions = {
            ("sensor", "nominal"): None,
            ("sensor", "focal_090"): "focal_090",
            ("MoGe-2 raw", "nominal"): "moge2_raw",
            ("MoGe-2 raw", "focal_090"): "moge2_focal_090",
        }
        values = {}
        for (depth, k), variant in conditions.items():
            measured = metric(run, variant)
            values[(depth, k)] = measured
            records.append({"episode": episode, "depth_source": depth, "K": k,
                            "ADE_2D_px": measured["ADE_2D_px"],
                            "FDE_2D_px": measured["FDE_2D_px"],
                            "valid_pairs": measured["valid_pairs"]})
        interactions[episode] = {
            measure: (values[("MoGe-2 raw", "focal_090")][measure] -
                      values[("MoGe-2 raw", "nominal")][measure]) -
                     (values[("sensor", "focal_090")][measure] -
                      values[("sensor", "nominal")][measure])
            for measure in ("ADE_2D_px", "FDE_2D_px")
        }
    dest = ROOT / "runs/fmb_depth_k_factorial_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(dest / "comparison.json", {
        "records": records, "difference_in_K_effects_px": interactions,
        "design": "fixed MoGe-2 predicted depth maps from historical RGB conditioned on nominal FOV; K variation changes only unprojection and output projection; sensor depth held fixed likewise",
        "K_alt": "fx and fy multiplied by 0.9; principal point fixed",
        "no_future_fitted": True,
        "calibration_status": "neither K is established as active color calibration",
        "interpretation_limit": "Two trials and one model do not identify a general depth-by-K interaction or separate network geometry error from camera-model error",
    })
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), layout="constrained")
    maxima = {key: max(r[key] for r in records) for key in ("ADE_2D_px", "FDE_2D_px")}
    for row, ep in enumerate(RUNS):
        selected = [r for r in records if r["episode"] == ep]
        for col, key in enumerate(("ADE_2D_px", "FDE_2D_px")):
            matrix = np.asarray([[next(r[key] for r in selected if r["depth_source"] == depth and r["K"] == k)
                                  for k in ("nominal", "focal_090")]
                                 for depth in ("sensor", "MoGe-2 raw")])
            ax = axes[row, col]
            ax.imshow(matrix, cmap="viridis", vmin=0, vmax=maxima[key])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white",
                            bbox={"facecolor": "black", "alpha": .45, "edgecolor": "none"})
            ax.set(title=f"{ep}: {key}", xticks=[0, 1], xticklabels=["K nominal", "0.9 focal"],
                   yticks=[0, 1], yticklabels=["sensor", "MoGe-2 raw"])
    fig.savefig(dest / "factorial.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"records": records, "interaction": interactions}, indent=2))


if __name__ == "__main__":
    main()
