"""Diagnostic comparison of sensor-depth and alternative CAD-geometry runs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_fmb_quantitative_2d import GT_TIMES, interpolate, measure, project


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))


def compare_variant(variant: str, prefix: str, caveat: str) -> dict:
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    nominal = json.loads((RUN / "model_run.json").read_text(encoding="utf8"))
    other_dir = RUN / "variants" / variant
    other_run = json.loads((other_dir / "model_run.json").read_text(encoding="utf8"))
    assert nominal["success"] and other_run["success"]
    assert nominal["t0"] == other_run["t0"]
    assert nominal["action_text"] == other_run["action_text"]
    assert nominal["model_revision"] == other_run["model_revision"]
    sensor_future = interpolate(np.load(RUN / "prediction_15hz.npy"))
    other_future = interpolate(np.load(other_dir / "prediction_15hz.npy"))
    np.save(other_dir / "prediction_10hz.npy", other_future.astype("float32"))
    k = np.asarray(json.loads((RUN / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
    sensor_uv = project(sensor_future, k)
    other_uv = project(other_future, k)
    xyz_gap = np.linalg.norm(sensor_future-other_future, axis=2)
    uv_gap = np.linalg.norm(sensor_uv-other_uv, axis=2)
    gt = np.load(RUN / "gt_2d.npy")
    mask = np.load(RUN / "validity_mask.npy")
    other_metrics, _ = measure(other_uv, gt, mask)
    sensor_metrics, _ = measure(sensor_uv, gt, mask)
    rows = [{"time_s": float(t), "source_step": config["t0"]+1+i,
             "mean_3D_disagreement_m": float(xyz_gap[:, i].mean()),
             "mean_2D_disagreement_px": float(uv_gap[:, i].mean())}
            for i, t in enumerate(GT_TIMES)]
    with (RUN / f"{prefix}_disagreement_by_time.csv").open(
            "w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "DIAGNOSTIC_INPUT_METHOD_SENSITIVITY",
              "variant": variant,
              "same_t0_points_action_model": True,
              "sensor_history_file": "history_sensor_3d.npy",
              "other_history_file": other_run["geometry_file"],
              "other_geometry_caveat": caveat,
              "input_sensor_vs_other": json.loads((RUN / ("sensor_vs_pnp.json" if variant == "pnp" else "sensor_vs_cad_silhouette_tcp.json")).read_text(encoding="utf8")),
              "mean_3D_prediction_disagreement_m": float(xyz_gap.mean()),
              "median_3D_prediction_disagreement_m": float(np.median(xyz_gap)),
              "final_mean_3D_prediction_disagreement_m": float(xyz_gap[:, -1].mean()),
              "mean_projected_2D_prediction_disagreement_px": float(uv_gap.mean()),
              "final_mean_projected_2D_prediction_disagreement_px": float(uv_gap[:, -1].mean()),
              "sensor_ADE_FDE_px": [sensor_metrics["ADE_2D_px"], sensor_metrics["FDE_2D_px"]],
              "other_ADE_FDE_px_diagnostic_not_for_geometry_selection": [
                  other_metrics["ADE_2D_px"], other_metrics["FDE_2D_px"]]}
    (RUN / f"{prefix}_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    fig, left = plt.subplots(figsize=(8, 4.5))
    left.plot(GT_TIMES, xyz_gap.mean(axis=0)*1000, label="3D disagreement", color="tab:blue")
    left.set(xlabel="Future time (s, nominal)", ylabel="Mean 3D disagreement (mm)")
    right = left.twinx()
    right.plot(GT_TIMES, uv_gap.mean(axis=0), label="projected 2D disagreement",
               color="tab:red")
    right.set(ylabel="Mean projected 2D disagreement (px)")
    left.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RUN / f"{prefix}_disagreement.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_sensor_vs_other",)},
                     indent=2))
    return result


def main() -> None:
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    pnp_warning = (
        "Planar RGB tracked correspondences are weak: frame 124 PnP implies large rotation/motion inconsistent with robot TCP; diagnostic stress case."
        if config["t0"] == 126 else
        "Planar RGB/CAD correspondences on a textureless face are only approximate. At this second window the PnP history agrees with sensor depth locally, but 2 px RGB perturbations move reconstructed points by about 11-12 mm median; diagnostic, not exact calibration."
    )
    compare_variant("pnp", "geometry_prediction", pnp_warning)
    if (RUN / "variants/cad_silhouette/prediction_15hz.npy").exists():
        compare_variant("cad_silhouette", "cad_silhouette_prediction", "RGB rounded-CAD silhouette with TCP motion magnitude prior, no sensor depth or future. Object surface points are inferred from a featureless face and this is not independently measured camera calibration.")


if __name__ == "__main__":
    main()
