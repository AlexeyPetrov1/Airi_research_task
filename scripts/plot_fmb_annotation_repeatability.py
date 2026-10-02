"""Visualize existing blind algorithmic repeat checks without changing the GT."""

from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS


def main() -> None:
    for name, run in RUNS.items():
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        annotation = json.loads((run / "annotation_repeatability.json").read_text(encoding="utf8"))
        observations = annotation["observations"]
        source = np.load(ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"],
                         allow_pickle=True).item()
        records = sorted(observations, key=lambda r: (r["point_id"], r["step"]))
        if len(records) != 10 or len(set(r["step"] for r in records)) != 5:
            raise ValueError("Expected 2 point IDs on 5 preselected future frames")
        fig = plt.figure(figsize=(14, 5.2))
        grid = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 1, 1.8])
        for index, row in enumerate(records):
            ax = fig.add_subplot(grid[index // 5, index % 5])
            frame = cv2.cvtColor(source["obs/side_1"][row["step"]], cv2.COLOR_BGR2RGB)
            a = np.asarray(row["primary_uv"])
            b = np.asarray(row["repeat_uv"])
            center = (a + b) / 2
            x0 = int(round(center[0])) - 15
            y0 = int(round(center[1])) - 15
            x0 = int(np.clip(x0, 0, 226))
            y0 = int(np.clip(y0, 0, 226))
            crop = frame[y0:y0+30, x0:x0+30]
            ax.imshow(crop, interpolation="nearest")
            ax.scatter([a[0] - x0], [a[1] - y0], s=65, facecolors="none",
                       edgecolors="#1f77b4", linewidths=1.6, label="sequential ECC")
            ax.scatter([b[0] - x0], [b[1] - y0], s=70, marker="x",
                       color="#ff7f0e", linewidths=1.7, label="direct ECC")
            ax.set(title=f"step {row['step']}, id {row['point_id']}\nΔ {row['difference_px']:.2f} px",
                   xticks=[], yticks=[])
        ax = fig.add_subplot(grid[:, 5])
        distances = [r["difference_px"] for r in records]
        ax.hist(distances, bins=np.linspace(0, max(2.5, max(distances) + .1), 9),
                color="#4e79a7", edgecolor="white")
        ax.axvline(annotation["median_px"], color="black", linestyle="--",
                   label=f"median {annotation['median_px']:.2f} px")
        ax.axvline(annotation["p90_px"], color="#e15759", linestyle="--",
                   label=f"p90 {annotation['p90_px']:.2f} px")
        ax.set(xlabel="Distance between repeat paths (px)", ylabel="Count",
               title=f"{name}: 5/20 future frames")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.25)
        fig.suptitle("RGB-derived proxy points: blue circle = sequential ECC, orange cross = direct ECC. Crops are 30x30 source pixels.",
                     fontsize=10)
        fig.subplots_adjust(left=.03, right=.98, bottom=.12, top=.83, wspace=.25, hspace=.5)
        fig.savefig(run / "annotation_repeatability_v1.png", dpi=170)
        plt.close(fig)
        print(name, annotation["median_px"], annotation["p90_px"], annotation["max_px"])


if __name__ == "__main__":
    main()
