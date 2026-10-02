"""Show disagreement between two independent RGB tracking algorithms, not 3D truth."""

from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, write_json


SELECTED_OFFSETS = (0, 5, 10, 15, 19)  # five of 20 future frames, fixed in advance


def main() -> None:
    for episode, run in RUNS.items():
        primary = np.load(run / "gt_2d.npy")
        alternative = np.load(run / "gt_dense_flow_alternative.npy")
        valid = np.load(run / "validity_mask.npy")
        if primary.shape != (8, 20, 2) or alternative.shape != primary.shape:
            raise ValueError("Unexpected track shape")
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source = np.load(ROOT.parent / "data/fmb/single_object_manipulation_dataset" /
                         config["source_file"], allow_pickle=True).item()
        error = np.linalg.norm(primary - alternative, axis=2)
        selected = error[:, SELECTED_OFFSETS]
        fig = plt.figure(figsize=(14.5, 6.3), layout="constrained")
        grid = fig.add_gridspec(2, 5, height_ratios=[1.6, 1])
        ax = fig.add_subplot(grid[0, :])
        image = ax.imshow(error, cmap="magma", vmin=0, vmax=32, aspect="auto")
        ax.set(title=f"{episode}: sequential red-mask ECC vs RGB dense optical flow",
               xlabel="Future frame (1..20)", ylabel="Input point ID",
               xticks=np.arange(20), xticklabels=np.arange(1, 21))
        ax.set_xticks(np.asarray(SELECTED_OFFSETS), minor=True)
        ax.grid(which="minor", axis="x", color="white", linewidth=.8, alpha=.7)
        fig.colorbar(image, ax=ax, label="2D tracking disagreement (px)", shrink=.86)
        for column, offset in enumerate(SELECTED_OFFSETS):
            point_id = int(np.argmax(error[:, offset]))
            step = config["future_steps"][offset]
            frame = cv2.cvtColor(source["obs/side_1"][step], cv2.COLOR_BGR2RGB)
            p = primary[point_id, offset]
            q = alternative[point_id, offset]
            x0 = int(np.clip(round(p[0]) - 36, 0, 184))
            y0 = int(np.clip(round(p[1]) - 36, 0, 184))
            crop = frame[y0:y0+72, x0:x0+72]
            a = fig.add_subplot(grid[1, column])
            a.imshow(crop, interpolation="nearest")
            a.scatter([p[0] - x0], [p[1] - y0], s=65, facecolors="none",
                      edgecolors="#1f77b4", linewidths=1.7)
            a.scatter([q[0] - x0], [q[1] - y0], s=65, marker="x",
                      color="#ff7f0e", linewidths=1.7)
            a.set(title=f"step {step}, id {point_id}: {error[point_id, offset]:.1f} px",
                  xticks=[], yticks=[])
        dest = run / "tracking_uncertainty_v1"
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "tracker_disagreement.png", dpi=160)
        plt.close(fig)
        report = {
            "episode": episode, "method_A": "sequential affine ECC on red object mask",
            "method_B": "RGB dense optical flow from t0", "future_frames_shown":
            [config["future_steps"][offset] for offset in SELECTED_OFFSETS],
            "selected_pairs": int(selected.size),
            "selected_median_px": float(np.median(selected)),
            "selected_p90_px": float(np.percentile(selected, 90)),
            "selected_max_px": float(np.max(selected)),
            "all_future_median_px": float(np.median(error[valid])),
            "all_future_p90_px": float(np.percentile(error[valid], 90)),
            "interpretation": "Alternative algorithm disagreement on smooth object. Neither trajectory is independently verified physical point truth; both use RGB and share visibility assumptions.",
            "GT_and_predictions_unchanged": True,
        }
        write_json(dest / "tracker_disagreement.json", report)
        print(episode, report["selected_median_px"], report["selected_p90_px"])


if __name__ == "__main__":
    main()
