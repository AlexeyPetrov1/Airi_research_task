"""Plot H3 static-background depth stability and object shape consistency."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, write_json


def pair_changes(xyz: np.ndarray) -> float:
    pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
    distances = np.stack([np.linalg.norm(xyz[:, i] - xyz[:, j], axis=1)
                          for i, j in pairs])
    return float(np.median(np.abs(np.diff(distances, axis=1))) * 1000)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    rows = []
    colors = {"sensor": "#4e79a7", "MoGe raw": "#e15759",
              "MoGe H3 scale": "#59a14f"}
    for row_index, (episode, run) in enumerate(RUNS.items()):
        report = json.loads((run / "moge2_history_v1/manifest.json").read_text(encoding="utf8"))
        scale = report["single_history_scale_to_sensor"]
        times = report["history_steps_only"]
        background = {
            "sensor": report["fixed_patch_sensor_depth_median_by_frame_m"],
            "MoGe raw": report["fixed_patch_moge_depth_median_by_frame_m"],
            "MoGe H3 scale": (np.asarray(report["fixed_patch_moge_depth_median_by_frame_m"]) * scale).tolist(),
        }
        geometry = {
            "sensor": np.load(run / "history_sensor_3d.npy"),
            "MoGe raw": np.load(run / "history_moge2_raw_3d.npy"),
            "MoGe H3 scale": np.load(run / "history_moge2_history_scaled_3d.npy"),
        }
        ax = axes[row_index, 0]
        for name, values in background.items():
            ax.plot(times, np.asarray(values) * 1000, "o-", color=colors[name], label=name)
        ax.set(title=f"{episode}: visually static background ROI (20:80, 20:80)",
               ylabel="Median depth (mm)", xlabel="Historical frame", xticks=times)
        ax.grid(alpha=.25)
        if row_index == 0:
            ax.legend(fontsize=8)
        rigidity = {name: pair_changes(xyz) for name, xyz in geometry.items()}
        ax = axes[row_index, 1]
        ax.bar(np.arange(3), list(rigidity.values()), color=[colors[k] for k in rigidity])
        ax.set(title=f"{episode}: H3 object pair-distance change",
               ylabel="Median adjacent change (mm)", xticks=np.arange(3),
               xticklabels=list(rigidity))
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=.25)
        rows.append({"episode": episode, "history_steps": times,
                     "background_depth_m": background,
                     "background_range_mm": {k: float(np.ptp(v) * 1000) for k, v in background.items()},
                     "median_object_pair_distance_change_mm": rigidity,
                     "single_scale_fitted_on_H3_only": scale})
    maximum = max(max(row["median_object_pair_distance_change_mm"].values()) for row in rows)
    for ax in axes[:, 1]:
        ax.set_ylim(0, maximum * 1.15)
    fig.suptitle("Sensor vs RGB-derived MoGe-2 history; common rigidity scale, candidate FOV/K")
    dest = ROOT / "runs/fmb_moge2_temporal_geometry_v1"
    dest.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest / "temporal_consistency.png", dpi=160)
    plt.close(fig)
    write_json(dest / "temporal_consistency.json", {
        "rows": rows, "background_caveat": "Same-pixel sensor/RGB registration unverified; ROI visually static only",
        "rigidity_caveat": "Small pair-distance changes test internal consistency, not absolute 3D accuracy",
        "MoGe_K_caveat": "Output K is conditioned on supplied candidate horizontal FOV",
    })
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
