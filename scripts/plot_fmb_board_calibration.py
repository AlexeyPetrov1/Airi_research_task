"""Visual summary of RGB-only board K fit and independent depth check."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT


def main() -> None:
    data = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    fit = np.asarray(data["best_fit"]["K"])
    nominal = np.array([K_NOMINAL[0, 0], K_NOMINAL[1, 1],
                        K_NOMINAL[0, 2], K_NOMINAL[1, 2]])
    best = np.array([fit[0, 0], fit[1, 1], fit[0, 2], fit[1, 2]])
    intervals = np.asarray(data["monte_carlo_K_5_50_95_percentiles"])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for i, (ax, label) in enumerate(zip(axes.ravel(), ("fx", "fy", "cx", "cy"))):
        ax.scatter(nominal[i], 0, color="#356ca8", s=60, label="nominal")
        ax.scatter(best[i], 1, color="#ca5042", s=60, label="RGB board fit")
        ax.plot((intervals[0, i], intervals[2, i]), (1, 1),
                color="#ca5042", lw=4, alpha=.5)
        ax.set_yticks((0, 1), ("nominal", "RGB board fit"))
        ax.set_ylim(-.5, 1.5)
        ax.set_xlabel(f"{label} (px)")
        ax.grid(alpha=.2)
    fig.suptitle("Five FMB board placements: K and 5–95% response to 2 px landmark noise")
    fig.savefig(OUT / "board_K_stability.png", dpi=160)
    plt.close(fig)

    episodes = [row["episode"] for row in data["episodes"]]
    rgb_new = [row["rmse_px"] for row in data["heldout_frames_train_features"]]
    rgb_nom = [row["rmse_px"] for row in data["fixed_nominal_heldout_frames"]]
    dep_new = [row["median_signed_Z_error_mm"] for row in data["depth_crosscheck_new"]]
    dep_nom = [row["median_signed_Z_error_mm"] for row in data["depth_crosscheck_fixed_nominal"]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.arange(len(episodes))
    axs[0].bar(x-.18, rgb_new, width=.36, label="RGB board fit", color="#ca5042")
    axs[0].bar(x+.18, rgb_nom, width=.36, label="nominal", color="#356ca8")
    axs[0].set(xlabel="raw FMB n", ylabel="held-out RGB reprojection RMSE (px)",
               xticks=x, xticklabels=episodes)
    axs[0].legend()
    axs[1].bar(x-.18, dep_new, width=.36, color="#ca5042")
    axs[1].bar(x+.18, dep_nom, width=.36, color="#356ca8")
    axs[1].axhline(0, color="black", lw=1)
    axs[1].set(xlabel="raw FMB n", ylabel="held-out CAD Z − sensor Z, median (mm)",
               xticks=x, xticklabels=episodes)
    fig.savefig(OUT / "board_RGB_vs_depth.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
