"""Show frozen historical FMB sensor and MoGe-2 depth with common scales."""

from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inspect_fmb_2d_window import red_component
from run_fmb_ablation_suite import ROOT, RUNS, write_json


def main() -> None:
    for episode, run in RUNS.items():
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source = np.load(ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"],
                         allow_pickle=True).item()
        with np.load(run / "moge2_history_v1/moge2_history_maps.npz") as maps:
            mo_depth = maps["depth"]
            model_k = maps["K_model"]
        uv = np.load(run / "history_points_2d.npy")
        fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout="constrained")
        rows = []
        for i, step in enumerate(config["history_steps"]):
            rgb_bgr = source["obs/side_1"][step]
            rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            sensor = source["obs/side_1_depth"][step].astype(float) * 1e-4
            network = mo_depth[i].astype(float)
            mask, _ = red_component(rgb_bgr)
            interior = cv2.erode(mask.astype("uint8"), np.ones((5, 5), np.uint8)).astype(bool)
            valid = interior & np.isfinite(sensor) & (sensor > 0) & np.isfinite(network) & (network > 0)
            difference = np.abs(sensor - network)
            rows.append({"step": step, "interior_valid_pixels": int(valid.sum()),
                         "median_abs_depth_gap_mm": float(np.median(difference[valid]) * 1000),
                         "MAE_depth_gap_mm": float(np.mean(difference[valid]) * 1000),
                         "MoGe_output_K_px_given_candidate_FOV": model_k[i].tolist()})
            ax = axes[i]
            ax[0].imshow(rgb)
            ax[0].scatter(uv[i, :, 0], uv[i, :, 1], s=12, color="cyan")
            ax[0].set_title(f"RGB step {step} / points")
            ax[1].imshow(np.ma.masked_where(sensor <= 0, sensor), cmap="turbo", vmin=.1, vmax=.9)
            ax[1].set_title("Sensor depth / m")
            ax[2].imshow(np.ma.masked_where(~np.isfinite(network) | (network <= 0), network),
                         cmap="turbo", vmin=.1, vmax=.9)
            ax[2].set_title("MoGe-2 depth / m")
            im = ax[3].imshow(np.ma.masked_where(~valid, difference), cmap="magma", vmin=0, vmax=.2)
            ax[3].set_title(f"|difference|, peg interior\nmedian {rows[-1]['median_abs_depth_gap_mm']:.0f} mm")
            for item in ax:
                item.set(xticks=[], yticks=[])
        fig.colorbar(axes[0, 1].images[0], ax=axes[:, 1:3], shrink=.55,
                     label="Depth (m), common scale")
        fig.colorbar(im, ax=axes[:, 3], shrink=.55, label="Absolute gap (m)")
        fig.suptitle(f"{episode}: historical depth; RGB-depth alignment of published arrays unverified")
        dest = run / "moge2_history_v1"
        fig.savefig(dest / "depth_comparison.png", dpi=150)
        plt.close(fig)
        write_json(dest / "depth_comparison.json", {
            "episode": episode, "rows": rows,
            "comparison_mask": "eroded red object component, valid positive sensor and MoGe depth",
            "sensor_is_imperfect_reference": True,
            "RGB_depth_pixel_registration_unverified": True,
            "depth_display_scale_m": [.1, .9], "difference_display_scale_m": [0, .2],
        })
        print(episode, [(r["step"], r["median_abs_depth_gap_mm"]) for r in rows])


if __name__ == "__main__":
    main()
