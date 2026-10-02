"""Compare all frozen full-inference FMB K hypotheses on both episodes."""

from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json


NAMES = ("Nominal", "Principal point", "Prior focal 0.925*", "Focal 0.9", "Focal 1.1")


def main() -> None:
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, (episode, run) in enumerate(RUNS.items()):
        with (run / "calibration_sensitivity.csv").open(newline="", encoding="utf8") as f:
            prior = {r["K_variant"]: r for r in csv.DictReader(f)}
        base = json.loads((run / "metrics.json").read_text(encoding="utf8"))["metrics"]
        values = []
        for label, variant in zip(NAMES[:3], ("K_nominal", "K_simple", "K_focal_0925")):
            record = prior[variant]
            if record["evaluation_mode"] != "full_inference":
                raise ValueError(f"K variant lacks full inference: {episode}/{variant}")
            values.append((label, float(record["ADE_2D_px"]), float(record["FDE_2D_px"]),
                           "previous diagnostic" if variant == "K_focal_0925" else "frozen hypothesis"))
        for label, variant in zip(NAMES[3:], ("focal_090", "focal_110")):
            measured = json.loads((run / SUITE / variant / "evaluation.json").read_text(encoding="utf8"))["metrics"]
            values.append((label, measured["ADE_2D_px"], measured["FDE_2D_px"], "predeclared ±10%"))
        for label, ade, fde, kind in values:
            records.append({"episode": episode, "K_hypothesis": label,
                            "ADE_2D_px": ade, "FDE_2D_px": fde, "status": kind,
                            "evaluation_mode": "full_inference"})
        for col, key in enumerate(("ADE_2D_px", "FDE_2D_px")):
            ax = axes[row, col]
            metric_index = 1 if col == 0 else 2
            colors = ["#4e79a7", "#4e79a7", "#bab0ac", "#f28e2b", "#f28e2b"]
            ax.bar(np.arange(5), [v[metric_index] for v in values], color=colors)
            ax.axhline(base["stationary"][key], linestyle="--", color="black", label="stationary")
            ax.axhline(base["constant_velocity"][key], linestyle=":", color="black", label="constant velocity")
            ax.set(title=f"{episode}: {key}", ylabel="Pixel error",
                   xticks=np.arange(5), xticklabels=NAMES)
            ax.tick_params(axis="x", rotation=25)
            ax.grid(axis="y", alpha=.3)
    fig.legend(handles=[Patch(color="#4e79a7", label="Existing fixed candidates"),
                        Patch(color="#bab0ac", label="Previous depth-fitted diagnostic*"),
                        Patch(color="#f28e2b", label="Prescribed focal ±10%")],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=3, fontsize=8)
    fig.suptitle("Full MolmoMotion inference under candidate K; * not an independent color-camera calibration",
                 y=.99, fontsize=10)
    fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.93, hspace=.60, wspace=.24)
    dest = ROOT / "runs/fmb_K_sensitivity_combined_v1"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "comparison.png", dpi=160)
    plt.close(fig)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(dest / "comparison.json", {
        "records": records,
        "all_full_inference": True,
        "interpretation": "Scores under candidate K values; no K was selected by future GT or independently confirmed as active color intrinsics",
    })
    print(len(records), "K comparison records")


if __name__ == "__main__":
    main()
