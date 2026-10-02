"""Descriptive side-by-side audit of two frozen FMB 2D experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"
SECOND_BASE = ROOT / "runs/fmb_second_example_1_M_L_3_vertical_n_3"
SECOND = SECOND_BASE / "quantitative_2d_sensor_t130"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf8"))


def one(name: str, run: Path) -> dict:
    config = read(run / "experiment_config.json")
    model = read(run / "model_run.json")
    metrics = read(run / "metrics.json")
    k = read(run / "k_nominal.json")["K_nominal"]
    gt = np.load(run / "gt_2d.npy")
    alt = np.load(run / "gt_dense_flow_alternative.npy")
    mask = np.load(run / "validity_mask.npy")
    p0 = np.load(run / "points_2d_at_t0.npy")
    projection = np.load(run / "prediction_2d.npy")
    history = np.load(run / "history_sensor_3d.npy")
    sensor_cad = read(run / "sensor_vs_cad_silhouette_tcp.json")
    labels = {method: {"ADE_px": value["ADE_2D_px"], "FDE_px": value["FDE_2D_px"]}
              for method, value in metrics["metrics"].items()}
    return {"trial": name, "source": config["source_file"],
            "source_identity": metrics["source_episode"], "t0": config["t0"],
            "history_steps": config["history_steps"], "future_steps": config["future_steps"],
            "action_exact": model["action_text"], "model_revision": model["model_revision"],
            "checkpoint_config_sha256": model["checkpoint_config_sha256"],
            "model_pt_sha256_previously_verified": model["model_pt_sha256_previously_verified"],
            "camera": config["camera"], "K_nominal": k,
            "source_hz_nominal": config["source_hz_nominal"],
            "model_prediction_hz_assumed": config["model_prediction_hz_assumed"],
            "history_shape": list(history.shape),
            "forecast_shape": list(np.load(run / "prediction_15hz.npy").shape),
            "GT_shape": list(gt.shape),
            "GT_method": read(run / "gt_tracking_diagnostics.json")["method"],
            "depth_rule": read(run / "history_depth_probe.json")["rule"],
            "depth_scale_m_per_unit": read(run / "preflight.json")["depth_scale_m_per_unit"],
            "GT_coverage": int(mask.sum()),
            "prediction_completeness": model["prediction_completeness"],
            "metrics": labels,
            "model_minus_stationary_ADE_px": labels["MolmoMotion"]["ADE_px"]-labels["stationary"]["ADE_px"],
            "model_minus_constant_velocity_ADE_px": labels["MolmoMotion"]["ADE_px"]-labels["constant_velocity"]["ADE_px"],
            "mean_GT_final_displacement_px": float(np.linalg.norm(gt[:, -1]-p0, axis=1).mean()),
            "mean_model_final_displacement_px": float(np.linalg.norm(projection[:, -1]-p0, axis=1).mean()),
            "GT_ECC_vs_dense_flow_median_px": float(np.median(np.linalg.norm(alt-gt, axis=2)[mask])),
            "sensor_vs_RGB_CAD_median_XYZ_mm": sensor_cad["median_XYZ_disagreement_mm"],
            "model_outside_image_pairs": read(run / "projection_check.json")["outside_image_pairs"]}


def main() -> None:
    first, second = one("ShareRobot episode_5201 / FMB trial 2", FIRST), one(
        "additional FMB trial 3", SECOND)
    identical = {"model_revision": first["model_revision"] == second["model_revision"],
                 "checkpoint_config_sha256": first["checkpoint_config_sha256"] == second["checkpoint_config_sha256"],
                 "model_pt_sha256_record": first["model_pt_sha256_previously_verified"] == second["model_pt_sha256_previously_verified"],
                 "action_exact": first["action_exact"] == second["action_exact"],
                 "camera_name": first["camera"] == second["camera"],
                 "nominal_timeline_rates": first["source_hz_nominal"] == second["source_hz_nominal"] == 10 and
                     first["model_prediction_hz_assumed"] == second["model_prediction_hz_assumed"] == 15,
                 "tensor_shapes": first["history_shape"] == second["history_shape"] == [3, 8, 3] and
                     first["forecast_shape"] == second["forecast_shape"] == [8, 30, 3] and
                     first["GT_shape"] == second["GT_shape"] == [8, 20, 2],
                 "GT_method": first["GT_method"] == second["GT_method"] == "RGB_RED_MASK_SEQUENTIAL_AFFINE_ECC",
                 "depth_aggregation_rule": all(
                     token in rule.lower() for rule in (first["depth_rule"], second["depth_rule"])
                     for token in ("median", "positive", "5x5", "15")),
                 "K_nominal": bool(np.allclose(first["K_nominal"], second["K_nominal"])),
                 "depth_scale": first["depth_scale_m_per_unit"] == second["depth_scale_m_per_unit"],
                 "prediction_shape_coverage": first["prediction_completeness"] == second["prediction_completeness"] == 1.,
                 "GT_coverage": first["GT_coverage"] == second["GT_coverage"] == 160}
    if not all(identical.values()):
        raise ValueError(f"Trial protocol mismatch: {identical}")
    rows = []
    for result in (first, second):
        for method, metric in result["metrics"].items():
            rows.append({"trial": result["trial"], "source": result["source"],
                         "method": method, **metric})
    with (SECOND_BASE / "two_trial_comparison.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = {"status": "DESCRIPTIVE_TWO_TRIAL_COMPARISON",
              "same_protocol_checks": identical,
              "important_difference": "Trials differ in source episode, t0, object motion, point locations, and unverified effective camera/depth preprocessing. The additional raw FMB file has no verified ShareRobot episode mapping.",
              "no_population_inference": True,
              "trials": [first, second]}
    (SECOND_BASE / "two_trial_comparison.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    methods = ("MolmoMotion", "stationary", "constant_velocity")
    colors = ("tab:blue", "tab:orange", "tab:green")
    for ax, result, title in zip(axes, (first, second), ("Trial 2 / ShareRobot 5201", "Trial 3 / raw FMB")):
        for index, (method, color) in enumerate(zip(methods, colors)):
            ax.bar(np.arange(2)+(index-1)*.22,
                   [result["metrics"][method]["ADE_px"], result["metrics"][method]["FDE_px"]],
                   width=.22, label=method, color=color)
        ax.set_xticks([0, 1], ["ADE", "FDE"])
        ax.set(title=title, ylabel="2D error (px)")
        ax.grid(axis="y", alpha=.2)
    axes[1].legend(fontsize=8)
    fig.suptitle("Two local FMB trials, same prediction/evaluation protocol")
    fig.tight_layout()
    fig.savefig(SECOND_BASE / "two_trial_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"same_protocol_checks": identical,
                      "trial_1_ADE_FDE": first["metrics"],
                      "trial_2_ADE_FDE": second["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
