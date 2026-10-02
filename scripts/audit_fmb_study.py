"""Final integrity audit for the frozen two-episode FMB study."""

from __future__ import annotations

import csv
import json
import re

import cv2
import numpy as np

from evaluate_fmb_quantitative_2d import measure
from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, digest, verify_frozen, write_json


def check_variant(run, variant: str) -> dict:
    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array

    dest = run / SUITE / variant
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    status = json.loads((dest / "model_run.json").read_text(encoding="utf8"))
    parser = json.loads((dest / "parser_validation.json").read_text(encoding="utf8"))
    evaluation = json.loads((dest / "evaluation.json").read_text(encoding="utf8"))
    xyz = np.load(dest / "history_3d.npy")
    model_order = np.load(dest / "prediction_15hz_model_order.npy")
    original_order = np.load(dest / "prediction_15hz.npy")
    raw_file = dest / "raw_model_output.txt"
    raw = raw_file.read_text(encoding="utf8")
    parsed = parse_tracks_text(raw)
    if parsed is None:
        raise ValueError(f"No complete tracks block: {dest}")
    delta, visible = tracks_to_array(parsed, num_points=8, num_frames=30, start_timestamp=3.0)
    rebuilt = np.asarray(delta) + xyz[-1, 0]
    gt = np.load(run / "gt_2d.npy")
    mask = np.load(run / "validity_mask.npy")
    uv = np.load(dest / "prediction_2d.npy")
    remeasured, _ = measure(uv, gt, mask)
    order = np.asarray(manifest["point_order_model_to_original"])
    checks = {
        "prepared_hashes": digest(dest / "history_3d.npy") == manifest["history_3d_sha256"] and
                           digest(dest / "points_2d_at_t0.npy") == manifest["points_2d_at_t0_sha256"],
        "manifest_hash_in_run": digest(manifest_path) == status["manifest_sha256"],
        "model_success": bool(status["success"] and status["status"] == "COMPLETE"),
        "model_revision": status["model_revision"] == manifest["model_revision"],
        "raw_sha256": digest(raw_file) == status["raw_output_sha256"],
        "official_parser_all_visible": bool(np.asarray(visible).all()),
        "official_parser_matches_model_order_tensor": bool(np.max(np.abs(rebuilt-model_order)) < 1e-4),
        "saved_parser_pass": parser["status"] == "PASS",
        "shape_finite_positive": (model_order.shape == (8, 30, 3) and
                                  np.isfinite(model_order).all() and
                                  (model_order[..., 2] > 0).all() and
                                  not np.all(model_order == 0)),
        "permutation_reversed": bool(np.array_equal(original_order, model_order[np.argsort(order)])),
        "common_coverage": evaluation["metrics"]["valid_pairs"] == int(mask.sum()) == 160,
        "recomputed_ADE": abs(remeasured["ADE_2D_px"] - evaluation["metrics"]["ADE_2D_px"]) < 1e-3,
        "recomputed_FDE": abs(remeasured["FDE_2D_px"] - evaluation["metrics"]["FDE_2D_px"]) < 1e-3,
        "GT_hash": digest(run / "gt_2d.npy") == evaluation["gt_sha256"],
        "mask_hash": digest(run / "validity_mask.npy") == evaluation["evaluation_mask_sha256"],
    }
    return {"variant": variant, "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "raw_sha256": status["raw_output_sha256"]}


def audit_media(run) -> dict:
    media = run / "study_media_v1"
    manifest = json.loads((media / "media_manifest.json").read_text(encoding="utf8"))
    expected = {"input_and_geometry.mp4": 18,
                "forecast_vs_observed.mp4": 60,
                "geometry_methods.mp4": 60}
    checks = {}
    for filename, frames in expected.items():
        path = media / filename
        cap = cv2.VideoCapture(str(path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        checks[filename] = path.exists() and count == frames
    checks["all_overview_images"] = all((media / name).exists() for name in manifest["still_overviews"])
    checks["saved_outputs_only"] = manifest["visuals_generated_from_saved_predictions_only"]
    neural = run / "moge2_history_v1"
    neural_video = neural / "forecast_comparison.mp4"
    cap = cv2.VideoCapture(str(neural_video))
    checks["moge2_comparison_video"] = neural_video.exists() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 60
    cap.release()
    checks["moge2_comparison_overview"] = (neural / "forecast_comparison_overview.png").exists()
    return checks


def audit_report_links() -> dict:
    report = ROOT / "report/fmb_geometry_forecast_study.md"
    content = report.read_text(encoding="utf8")
    local = [link for link in re.findall(r"\]\(([^)]+)\)", content)
             if not link.startswith(("https://", "http://"))]
    missing = [link for link in local if not (report.parent / link).exists()]
    return {"report_exists": report.exists(), "local_link_count": len(local),
            "missing_local_links": missing, "all_local_links_resolve": not missing}


def main() -> None:
    episodes = {}
    moge_checkpoint_sha = digest(ROOT.parent / "models/moge-2-vitl/model.pt")
    for name, run in RUNS.items():
        verify_frozen(run)
        base = json.loads((run / "model_run.json").read_text(encoding="utf8"))
        gt = np.load(run / "gt_2d.npy")
        mask = np.load(run / "validity_mask.npy")
        variants = [check_variant(run, variant) for variant in VARIANTS]
        input_audit = json.loads((run / SUITE / "input_audit.json").read_text(encoding="utf8"))
        neural = json.loads((run / "moge2_history_v1/manifest.json").read_text(encoding="utf8"))
        bundle = json.loads((run / "geometry_bundle_v1/manifest.json").read_text(encoding="utf8"))
        media = audit_media(run)
        checks = {"base_model_success": base["success"],
                  "GT_shape": gt.shape == (8, 20, 2),
                  "GT_mask_shape_coverage": mask.shape == (8, 20) and int(mask.sum()) == 160,
                  "all_variant_inputs_pass": input_audit["status"] == "PASS",
                  "all_variant_results_pass": all(v["status"] == "PASS" for v in variants),
                  "variant_count": len(variants) == len(VARIANTS),
                  "media": all(media.values()),
                  "rigid_result": (run / "rigid_correction_v1/evaluation.json").exists(),
                  "geometry_bundle": {"moge2_raw", "moge2_history_scaled"}.issubset(bundle["methods"]),
                  "moge2_history_hashes": (
                      neural["model_checkpoint_sha256"] == moge_checkpoint_sha and
                      digest(run / "history_moge2_raw_3d.npy") == neural["raw_xyz_sha256"] and
                      digest(run / "history_moge2_history_scaled_3d.npy") == neural["scaled_xyz_sha256"] and
                      digest(run / "moge2_history_v1/moge2_history_maps.npz") == neural["maps_sha256"]),
                  "moge2_depth_figure": (run / "moge2_history_v1/depth_comparison.png").exists(),
                  "common_axis_history_figure": (run / "geometry_bundle_v1/history_clouds_common_axes.png").exists()}
        episodes[name] = {"status": "PASS" if all(checks.values()) else "FAIL",
                          "checks": checks, "variants": variants, "media": media}
    summary_path = ROOT / "runs/fmb_ablation_suite_v1_summary.csv"
    with summary_path.open(newline="", encoding="utf8") as f:
        summary = list(csv.DictReader(f))
    table_expectations = {
        "ablation_summary": (summary_path, len(RUNS) * len(VARIANTS)),
        "geometry_vs_forecast": (ROOT / "runs/fmb_geometry_forecast_comparison_v1/comparison.csv", 10),
        "depth_K_factorial": (ROOT / "runs/fmb_depth_k_factorial_v1/comparison.csv", 8),
        "GT_tracker_sensitivity": (ROOT / "runs/fmb_gt_sensitivity_v1/comparison.csv", 28),
        "paired_ablation_effects": (ROOT / "runs/fmb_ablation_conclusions_v1/paired_effects.csv", 26),
    }
    table_counts = {}
    for name, (path, expected) in table_expectations.items():
        with path.open(newline="", encoding="utf8") as f:
            observed = len(list(csv.DictReader(f)))
        table_counts[name] = {"observed": observed, "expected": expected,
                              "status": "PASS" if observed == expected else "FAIL"}
    link_check = audit_report_links()
    overall = (all(row["status"] == "PASS" for row in episodes.values()) and
               len(summary) == len(RUNS) * len(VARIANTS) and
               all(row["status"] == "PASS" for row in table_counts.values()) and
               link_check["all_local_links_resolve"])
    result = {"status": "PASS" if overall else "FAIL",
              "episodes": episodes, "summary_row_count": len(summary),
              "table_counts": table_counts,
              "moge2_checkpoint_sha256": moge_checkpoint_sha,
              "report_links": link_check,
              "interpretation": "Checks artifact consistency only; does not establish physical 3D ground truth or camera calibration"}
    write_json(ROOT / "runs/fmb_study_audit.json", result)
    print(result["status"], {key: value["status"] for key, value in episodes.items()},
          "summary rows", len(summary), "missing links", link_check["missing_local_links"])
    if not overall:
        raise RuntimeError("FMB study audit failed")


if __name__ == "__main__":
    main()
