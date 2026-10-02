"""Compare the actual frozen H3 point clouds on common metric axes."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fmb_ablation_suite import RUNS, write_json


SOURCES = {
    "Sensor": "history_sensor_3d.npy",
    "Planar PnP": "history_pnp_3d.npy",
    "CAD silhouette + TCP": "history_cad_silhouette_tcp_3d.npy",
    "MoGe-2 raw": "history_moge2_raw_3d.npy",
    "MoGe-2 history scaled": "history_moge2_history_scaled_3d.npy",
}
FRAME_COLORS = ("#4e79a7", "#f28e2b", "#e15759")


def main() -> None:
    for episode, run in RUNS.items():
        histories = {name: np.load(run / path)
                     for name, path in SOURCES.items() if (run / path).exists()}
        if not histories:
            raise ValueError(f"No geometry: {run}")
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        all_points = np.concatenate(list(histories.values()), axis=1).reshape(-1, 3)
        x_span = (float(all_points[:, 0].min()), float(all_points[:, 0].max()))
        y_span = (float(all_points[:, 1].min()), float(all_points[:, 1].max()))
        z_span = (float(all_points[:, 2].min()), float(all_points[:, 2].max()))
        ranges = []
        for low, high in (x_span, y_span, z_span):
            pad = max(.01, (high - low) * .08)
            ranges.append((low - pad, high + pad))
        fig, axes = plt.subplots(2, len(histories), figsize=(3.4 * len(histories), 6.2),
                                 layout="constrained", squeeze=False)
        for column, (name, xyz) in enumerate(histories.items()):
            if xyz.shape != (3, 8, 3):
                raise ValueError(f"Unexpected {name} history shape")
            for frame_index, step in enumerate(config["history_steps"]):
                color = FRAME_COLORS[frame_index]
                axes[0, column].scatter(xyz[frame_index, :, 0], xyz[frame_index, :, 1],
                                        color=color, s=17, label=f"step {step}")
                axes[1, column].scatter(xyz[frame_index, :, 0], xyz[frame_index, :, 2],
                                        color=color, s=17)
            axes[0, column].set(title=name, xlabel="X (m)", ylabel="Y (m)",
                                xlim=ranges[0], ylim=ranges[1])
            axes[1, column].set(xlabel="X (m)", ylabel="Z (m)",
                                xlim=ranges[0], ylim=ranges[2])
            for row in range(2):
                axes[row, column].set_aspect("equal", adjustable="box")
                axes[row, column].grid(alpha=.2)
        axes[0, 0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"{episode}: historical H3 point clouds on identical metric axes; candidate K")
        dest = run / "geometry_bundle_v1"
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / "history_clouds_common_axes.png", dpi=160)
        plt.close(fig)
        write_json(dest / "history_clouds_common_axes.json", {
            "episode": episode, "methods": list(histories), "history_steps": config["history_steps"],
            "common_X_m": ranges[0], "common_Y_m": ranges[1], "common_Z_m": ranges[2],
            "projection_K_status": "SUPPORTED_NOT_EXACT",
            "MoGe_scale_caveat": "history-scaled variant uses one sensor-depth scale fitted on H3 only",
        })
        print(episode, list(histories), ranges)


if __name__ == "__main__":
    main()
