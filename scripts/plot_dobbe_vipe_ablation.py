"""Standalone paired-depth figure from sealed observed inputs and predictions."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dobbe_vipe_v1 import RUN, read, write


def plot():
    dest = RUN / "A/sift_audited"
    names = ["paired_vipe", "paired_hybrid"]
    history = [np.load(dest / name / "points_3d_camera_t0_diagnostic.npy") for name in names]
    prediction = [np.load(dest / name / "prediction_15hz.npy") for name in names]
    for file in ["points_2d_t0.npy", "c2w_t0.npy"]:
        assert np.array_equal(np.load(dest / names[0] / file), np.load(dest / names[1] / file))
    ids = read(dest / "paired_selection_audit.json")["selected_ids"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    result = {}
    for j, (name, hist, pred) in enumerate(zip(names, history, prediction)):
        label = "ViPE depth" if j == 0 else "DobbE measured-depth hypothesis"
        color = "tab:blue" if j == 0 else "tab:orange"
        axes[0].plot(np.arange(8), hist[-1, :, 2], "o-", color=color, label=label)
        displacement = np.linalg.norm(pred-hist[-1, :, None], axis=-1)
        axes[1].plot(np.arange(1, 31)/15, np.median(displacement, axis=0), color=color, label=label+" median")
        axes[1].plot(np.arange(1, 31)/15, displacement.max(0), "--", color=color, label=label+" max")
        result[name] = {"final_displacement_median": float(np.median(displacement[:, -1])),
            "final_displacement_max": float(displacement[:, -1].max()),
            "max_temporal_coordinate_variation": float(np.abs(pred-pred[:, :1]).max())}
    axes[0].set_xticks(np.arange(8), ids); axes[0].set_xlabel("common object query ID")
    axes[0].set_ylabel("t0 optical Z, estimated depth units")
    axes[0].set_title("Same RGB queries / K / c2w; both unsmoothed"); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("seconds after t0"); axes[1].set_ylabel("forecast displacement from t0, estimated units")
    axes[1].set_title("Model response to changing depth"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(dest / "paired_depth_comparison.png", dpi=160); plt.close(fig)
    write(dest / "paired_forecast_summary.json", {"metric_ground_truth": False, "results": result,
        "interpretation": "Input/forecast sensitivity only; absolute 3D differences are not prediction error against GT."})


if __name__ == "__main__":
    plot()
