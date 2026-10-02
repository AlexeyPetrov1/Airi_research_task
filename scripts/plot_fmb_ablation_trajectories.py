"""Plot actual saved point trajectories for prescribed FMB input perturbations."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, SUITE


SCENARIOS = ("nominal", "scale_090", "scale_110", "perm_keep_anchor", "perm_new_anchor",
             "focal_090", "focal_110")
COLORS = {"nominal": "#e15759", "scale_090": "#f28e2b", "scale_110": "#b07aa1",
          "perm_keep_anchor": "#59a14f", "perm_new_anchor": "#76b7b2",
          "focal_090": "#4e79a7", "focal_110": "#9c755f", "GT": "black"}


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for row, (episode, run) in enumerate(RUNS.items()):
        gt = np.load(run / "gt_2d.npy")
        valid = np.load(run / "validity_mask.npy")
        tracks = {"nominal": np.load(run / "prediction_2d.npy")}
        for name in SCENARIOS[1:]:
            tracks[name] = np.load(run / SUITE / name / "prediction_2d.npy")
        for col, point_id in enumerate((0, 7)):
            ax = axes[row, col]
            for label, data in (("GT", gt), *tracks.items()):
                points = data[point_id]
                if label == "GT":
                    points = points[valid[point_id]]
                ax.plot(points[:, 0], points[:, 1], color=COLORS[label],
                        linewidth=2 if label in ("GT", "nominal") else 1.2,
                        alpha=1 if label in ("GT", "nominal") else .85,
                        label=label)
                ax.scatter(points[-1, 0], points[-1, 1], color=COLORS[label], s=22)
            ax.add_patch(Rectangle((0, 0), 256, 256, fill=False,
                                   edgecolor="gray", linestyle="--", linewidth=.8))
            ax.set(title=f"{episode}: point {point_id}", xlabel="u (px)", ylabel="v (px)")
            ax.invert_yaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .005),
               ncol=4, fontsize=8)
    fig.suptitle("Saved forecasts under prescribed scale, point-order and K changes; dashed box = 256x256 image",
                 y=.99)
    fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.93, hspace=.28, wspace=.22)
    fig.savefig(ROOT / "runs/fmb_ablation_trajectory_scenarios.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"status": "PASS", "episodes": list(RUNS), "scenarios": SCENARIOS,
                      "point_ids": [0, 7], "source": "saved 2D predictions only"}, indent=2))


if __name__ == "__main__":
    main()
