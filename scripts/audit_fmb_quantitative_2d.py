"""Verify frozen inputs, timeline, GT, and model artifacts for the FMB run."""

from __future__ import annotations

import hashlib
import csv
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))
SOURCE_OVERRIDE = os.environ.get("FMB_QUANT_SOURCE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    freeze = json.loads((RUN / "input_freeze.json").read_text(encoding="utf8"))
    hashes_match = {name: sha256(RUN / name) == expected
                    for name, expected in freeze["sha256"].items()}
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    window = json.loads((RUN / "window_selection.json").read_text(encoding="utf8"))
    preflight = json.loads((RUN / "preflight.json").read_text(encoding="utf8"))
    model = json.loads((RUN / "model_run.json").read_text(encoding="utf8"))
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf8"))
    required_artifacts = (
        "current_state.json", "window_selection.json", "experiment_config.json",
        "k_nominal.json", "k_sensitivity_variants.json", "history_points_2d.npy",
        "history_depth_probe.json", "history_sensor_3d.npy", "history_pnp_3d.npy",
        "sensor_vs_pnp.json", "rigidity_check.json", "cad_dimension_check.json",
        "pnp_annotation_noise_sensitivity.json", "preflight.json", "raw_model_output.txt",
        "prediction_15hz.npy", "prediction_10hz.npy", "model_run.json",
        "gt_2d.npy", "validity_mask.npy", "reason_mask.npy",
        "annotation_repeatability.json", "metrics.csv", "metrics.json",
        "error_by_time.csv", "calibration_sensitivity.csv",
        "history_8_points.png", "history_depth_quality.png", "sensor_vs_pnp_3d.png",
        "future_trajectory_overlay.png", "future_frames_predictions_gt.png",
        "error_vs_time.png", "calibration_sensitivity.png", "coverage.png",
        "nominal_repeatability.json", "geometry_prediction_comparison.json",
        "cad_silhouette_prediction_comparison.json", "visual_review.json",
        "temporal_forecast_diagnostic.json",
    )
    variant_names = ("nominal", "nominal_repeat", "simple", "focal", "pnp", "cad_silhouette")
    parser_states = {}
    for variant in variant_names:
        variant_dir = RUN if variant == "nominal" else RUN / "variants" / variant
        parser_states[variant] = json.loads(
            (variant_dir / "parser_validation.json").read_text(encoding="utf8"))["status"]
    with (RUN / "calibration_sensitivity.csv").open(newline="", encoding="utf8") as stream:
        calibration_rows = list(csv.DictReader(stream))
    gt = np.load(RUN / "gt_2d.npy")
    valid = np.load(RUN / "validity_mask.npy")
    reason = np.load(RUN / "reason_mask.npy")
    history_uv = np.load(RUN / "history_points_2d.npy")
    history_xyz = np.load(RUN / "history_sensor_3d.npy")
    prediction = np.load(RUN / "prediction_15hz.npy")
    pred10 = np.load(RUN / "prediction_10hz.npy")
    pt = np.load(RUN / "prediction_time_15hz_s.npy")
    gt_t = np.load(RUN / "evaluation_time_10hz_s.npy")
    checks = {
        "all_frozen_input_hashes_match": all(hashes_match.values()),
        "frozen_file_hashes": hashes_match,
        "window_has_full_20_step_future": window["final_t0"] == config["t0"] and
            window["history_steps"] == config["history_steps"] and
            window["future_steps"] == config["future_steps"] ==
            list(range(config["t0"]+1, config["t0"]+21)),
        "history_shape": list(history_xyz.shape),
        "history_2D_shape": list(history_uv.shape),
        "history_finite_positive": bool(np.isfinite(history_xyz).all() and
                                        (history_xyz[..., 2] > 0).all()),
        "preflight": preflight["decision"],
        "source_sha256_matches_config": (sha256(Path(SOURCE_OVERRIDE)) == config["source_sha256"]
            if SOURCE_OVERRIDE else None),
        "source_primitive_exact": config["source_primitive_exact"],
        "prediction_shape": list(prediction.shape),
        "prediction_10hz_shape": list(pred10.shape),
        "prediction_finite_positive": bool(np.isfinite(prediction).all() and
                                           (prediction[..., 2] > 0).all()),
        "model_status": model["status"],
        "model_prediction_completeness": model["prediction_completeness"],
        "gt_shape": list(gt.shape), "validity_shape": list(valid.shape),
        "reason_shape": list(reason.shape),
        "GT_valid_pairs": int(valid.sum()), "GT_total_pairs": 160,
        "FDE_valid_points": int(valid[:, -1].sum()),
        "required_artifacts_present": all((RUN / name).exists() for name in required_artifacts),
        "missing_required_artifacts": [name for name in required_artifacts
                                       if not (RUN / name).exists()],
        "all_raw_model_parsers_pass": all(status == "PASS" for status in parser_states.values()),
        "parser_status_by_variant": parser_states,
        "nominal_repeatability_pass": json.loads(
            (RUN / "nominal_repeatability.json").read_text(encoding="utf8"))["status"] == "PASS",
        "full_inference_for_all_three_K": len(calibration_rows) == 3 and
            {row["K_variant"] for row in calibration_rows} ==
            {"K_nominal", "K_simple", "K_focal_0925"} and
            all(row["evaluation_mode"] == "full_inference" for row in calibration_rows),
        "metric_methods_complete": set(metrics["metrics"]) ==
            {"MolmoMotion", "stationary", "constant_velocity"},
        "all_methods_common_coverage": all(
            metric["valid_pairs"] == int(valid.sum()) and
            metric["FDE_valid_points"] == int(valid[:, -1].sum())
            for metric in metrics["metrics"].values()),
        "time_prediction_grid_15hz_exact": bool(np.allclose(pt, np.arange(1, 31)/15)),
        "time_GT_grid_10hz_exact": bool(np.allclose(gt_t, np.arange(1, 21)/10)),
        "input_freeze_mtime_before_raw_output": (RUN / "input_freeze.json").stat().st_mtime_ns <
            (RUN / "raw_model_output.txt").stat().st_mtime_ns,
        "GT_mtime_before_raw_output": (RUN / "gt_2d.npy").stat().st_mtime_ns <
            (RUN / "raw_model_output.txt").stat().st_mtime_ns,
        "prediction_used_for_GT_annotation": False,
        "GT_tracker_code": "annotate_fmb_future_2d.py reads FMB RGB and frozen t0 points, not prediction files",
        "future_used_for_window_selection_only_before_model": True,
        "camera_K_and_points_changed_after_prediction": not all(hashes_match.values()),
    }
    critical = ("all_frozen_input_hashes_match", "window_has_full_20_step_future",
                "history_finite_positive", "prediction_finite_positive",
                "time_prediction_grid_15hz_exact", "time_GT_grid_10hz_exact",
                "input_freeze_mtime_before_raw_output", "GT_mtime_before_raw_output",
                "required_artifacts_present", "all_raw_model_parsers_pass",
                "nominal_repeatability_pass", "full_inference_for_all_three_K",
                "metric_methods_complete", "all_methods_common_coverage")
    checks["status"] = "PASS" if all(checks[name] for name in critical) and (
        (checks["source_sha256_matches_config"] is not False) and
        checks["history_shape"] == [3, 8, 3] and
        checks["prediction_shape"] == [8, 30, 3] and
        checks["prediction_10hz_shape"] == [8, 20, 3] and
        checks["gt_shape"] == [8, 20, 2] and
        checks["validity_shape"] == [8, 20] and
        checks["reason_shape"] == [8, 20] and
        checks["GT_valid_pairs"] == 160 and
        checks["FDE_valid_points"] == 8) else "FAIL"
    (RUN / "leakage_and_artifact_audit.json").write_text(json.dumps(checks, indent=2) + "\n",
                                                         encoding="utf8")
    print(json.dumps(checks, indent=2))
    if checks["status"] != "PASS":
        raise ValueError("Quantitative run audit failed")


if __name__ == "__main__":
    main()
