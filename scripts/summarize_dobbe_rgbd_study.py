"""Collect reproducible tables and record the geometry gate outcome."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_rgbd_study"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    colmap_rows = []
    for scene in ("main", "second"):
        for path in sorted((OUT / scene).glob("colmap_*/summary.json")):
            s = read(path)
            models = s["reconstructions"]
            row = {"scene": scene, "run": path.parent.name,
                   "model": s["model"], "subset": s["subset"],
                   "input_frames": len(s["input_frames"]),
                   "registered": 0, "points3D": 0,
                   "fx": "", "fy": "", "cx": "", "cy": "",
                   "distortion": "", "mean_colmap_reprojection_px": "",
                   "image_size": json.dumps(s.get("image_size", [256, 256])),
                   "published_256x256_K": ""}
            if models:
                best = max(models, key=lambda m: m["registered"])
                cam = best["cameras"][0]
                row.update({"registered": best["registered"],
                            "points3D": best["points3D"],
                            "fx": cam["fx"], "fy": cam["fy"],
                            "cx": cam["cx"], "cy": cam["cy"],
                            "distortion": json.dumps(cam["params"][4:]),
                            "published_256x256_K": json.dumps(
                                cam.get("published_256x256_K", [
                                    cam["fx"], cam["fy"], cam["cx"], cam["cy"]])),
                            "mean_colmap_reprojection_px":
                                best["mean_reprojection_px"]})
            colmap_rows.append(row)
    with (OUT / "colmap_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=colmap_rows[0].keys())
        writer.writeheader()
        writer.writerows(colmap_rows)

    geometry_rows = []
    for scene in ("main", "second"):
        s = read(OUT / f"{scene}_static_geometry.json")
        for name, candidate in s["candidates"].items():
            for variant, metric in candidate["variants"].items():
                geometry_rows.append({"scene": scene, "K": name,
                                      "pose_depth": variant,
                                      "n": metric["n_3d"],
                                      "median_3d_m": metric["median_3d_m"],
                                      "p90_3d_m": metric["p90_3d_m"],
                                      "median_reprojection_px":
                                          metric["median_reprojection_px"],
                                      "p90_reprojection_px":
                                          metric["p90_reprojection_px"]})
    with (OUT / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=geometry_rows[0].keys())
        writer.writeheader()
        writer.writerows(geometry_rows)

    main = read(OUT / "main_static_geometry.json")
    second = read(OUT / "second_static_geometry.json")
    baseline = main["candidates"]["naive"]["variants"]["c2w_z"]
    pinhole = main["candidates"]["colmap_pinhole_all"]["variants"]["c2w_z"]
    second_baseline = second["candidates"]["naive"]["variants"]["c2w_z"]
    second_colmap = second["candidates"]["colmap_pinhole_all_fixedpp_init62_107"]
    second_colmap_metric = second_colmap["variants"]["c2w_z"]
    second_rectified = second["candidates"][
        "colmap_simple_pinhole_all_fixedpp_rectified"]
    second_rectified_metric = second_rectified["variants"]["c2w_z"]
    approximate_dir = OUT / "approx_history"
    approximate_model = (read(approximate_dir / "model_run.json")
                         if (approximate_dir / "model_run.json").exists() else None)
    approximate_future = (read(approximate_dir / "future_evaluation.json")
                          if (approximate_dir / "future_evaluation.json").exists()
                          else None)
    decision = {
        "status": "BLOCKED", "molmo_ready": False,
        "protocol": "protocol.json",
        "reasons": [
            "Legacy main-scene runs cover at most 9/96 and have unstable intrinsics, but used a guessed focal prior, early principal-point refinement, relaxed mapper thresholds and no foreground masks; this is not proof of general COLMAP self-calibration failure.",
            "Main-scene COLMAP intrinsics improve measured-depth median 3D error modestly but greatly worsen reprojection compared with naive K; the estimated intrinsics are physically implausible. Undoing the documented RGB aspect stretch and fitting SIMPLE_PINHOLE also yields no stable main-scene model.",
            "Depth Z-versus-ray remains unresolved to a trustworthy metric precision; RGB-depth spatial overlay alone cannot settle it. Accounting for the exporter's clockwise RGB rotation and the source-defined P branch supports camera-to-world labels.",
            "Second-scene fixed-center PINHOLE and aspect-rectified SIMPLE_PINHOLE register the full RGB prefix and improve median static 3D and reprojection errors, but their even subsets collapse and P90 static/reprojection errors remain high. They do not validate the main-scene K.",
            "A user-requested exploratory transfer of control-scene RGB-only COLMAP K through F2-NeRF produced an approximate main-scene [3,8,3] and model run, but the transfer assumes an unverified shared camera and does not change the formal geometry gate.",
        ],
        "main_quantitative_evidence": {
            "calibration_frames": 96,
            "pinhole_all_registered": next(r["registered"] for r in colmap_rows
                                           if r["run"] == "colmap_pinhole_all"
                                           and r["scene"] == "main"),
            "naive_static_median_m": baseline["median_3d_m"],
            "colmap_pinhole_static_median_m": pinhole["median_3d_m"],
            "naive_reprojection_median_px": baseline["median_reprojection_px"],
            "colmap_pinhole_reprojection_median_px":
                pinhole["median_reprojection_px"],
        },
        "second_quantitative_evidence": {
            "calibration_frames": 121,
            "fixed_pp_pinhole_K": second_colmap["camera"]["params"][:4],
            "naive_static_median_m": second_baseline["median_3d_m"],
            "colmap_static_median_m": second_colmap_metric["median_3d_m"],
            "naive_reprojection_median_px":
                second_baseline["median_reprojection_px"],
            "colmap_reprojection_median_px":
                second_colmap_metric["median_reprojection_px"],
            "rectified_simple_pinhole_K_published_rgb":
                second_rectified["camera"]["params"][:4],
            "rectified_simple_pinhole_static_median_m":
                second_rectified_metric["median_3d_m"],
            "rectified_simple_pinhole_reprojection_median_px":
                second_rectified_metric["median_reprojection_px"],
        },
        "points_3d_history_created": (
            approximate_dir / "points_3d_history_candidate.npy").exists(),
        "molmomotion_invoked": approximate_model is not None,
        "future_depth_opened_for_gt": approximate_future is not None,
        "future_prediction_metrics_computed": approximate_future is not None,
        "validated_points_3d_history_created": False,
        "validated_full_3d_gt_and_metrics": False,
        "colmap_profile_for_historical_evidence": "legacy",
        "historical_pixel_convention": "Legacy metrics passed COLMAP principal points directly to OpenCV; corrected clean validation subtracts 0.5 px",
        "clean_rerun": (
            "../dobbe_colmap_clean_rerun_v1/clean_decision.json"
            if (OUT.parent / "dobbe_colmap_clean_rerun_v1/clean_decision.json").exists() else None),
        "claim_scope": "No independently validated main K has been obtained from recorded candidates; impossibility of COLMAP self-calibration is not established",
        "exploratory_approximate_run": (
            "approx_history/manifest.json" if approximate_model else None),
        "exploratory_prediction_status": (
            approximate_model["status"] if approximate_model else None),
        "exploratory_future_gt_coverage": (
            [approximate_future["valid_point_timestamps"],
             approximate_future["possible_point_timestamps"]]
            if approximate_future else None),
    }
    (OUT / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
