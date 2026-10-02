"""Estimate current-window planar CAD/PnP sensitivity to 2 px RGB annotation noise."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np


RUN = Path(os.environ.get(
    "FMB_QUANT_RUN",
    Path(__file__).resolve().parents[1] /
    "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))
NOISE_SIGMA_PX = 2.0
DRAWS = 300


def camera_points(cad: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    return cad @ rotation.T + np.asarray(tvec).reshape(1, 3)


def distribution(values: np.ndarray) -> dict:
    return {"median": float(np.median(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
            "min": float(np.min(values)),
            "max": float(np.max(values))}


def main() -> None:
    records = json.loads((RUN / "pnp_correspondences.json").read_text(encoding="utf8"))
    k = np.asarray(json.loads((RUN / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"],
                   dtype="float64")
    rng = np.random.default_rng(5201)
    frames = []
    for frame in records["frames"]:
        uv = np.asarray(frame["observed_uv"], dtype="float64")
        cad = np.asarray(frame["object_points_CAD_m"], dtype="float64")
        refined_reference = np.asarray(
            frame["branches"][frame["selected_branch"]]["camera_points_m"],
            dtype="float64")
        selected_median_depth = float(np.median(refined_reference[:, 2]))
        original_count, original_rvecs, original_tvecs, _ = cv2.solvePnPGeneric(
            cad, uv, k, None, flags=cv2.SOLVEPNP_IPPE)
        if not original_count:
            raise ValueError(f"No noiseless IPPE solution for step {frame['step']}")
        original_candidates = [camera_points(cad, rv, tv)
                               for rv, tv in zip(original_rvecs, original_tvecs)]
        original_index = int(np.argmin([
            np.mean(np.linalg.norm(xyz-refined_reference, axis=1))
            for xyz in original_candidates]))
        baseline = original_candidates[original_index]
        branch_conditioned_z = []
        branch_conditioned_xyz_shift_mm = []
        minimum_reprojection_z = []
        failed = 0
        ambiguous = 0
        for _ in range(DRAWS):
            uv_noisy = (uv + rng.normal(0, NOISE_SIGMA_PX, uv.shape)).astype("float64")
            try:
                count, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    cad, uv_noisy, k, None, flags=cv2.SOLVEPNP_IPPE)
            except cv2.error:
                failed += 1
                continue
            if not count:
                failed += 1
                continue
            candidates = [camera_points(cad, rv, tv) for rv, tv in zip(rvecs, tvecs)]
            prior_gaps = [float(np.mean(np.linalg.norm(xyz-baseline, axis=1)))
                          for xyz in candidates]
            nearest = int(np.argmin(prior_gaps))
            best_reprojection = int(np.argmin(np.asarray(errors).reshape(-1)))
            ambiguous += int(nearest != best_reprojection)
            chosen = candidates[nearest]
            if not np.isfinite(chosen).all():
                failed += 1
                continue
            branch_conditioned_z.append(float(np.median(chosen[:, 2])))
            branch_conditioned_xyz_shift_mm.append(
                float(np.median(np.linalg.norm(chosen-baseline, axis=1))*1000))
            minimum_reprojection_z.append(float(np.median(candidates[best_reprojection][:, 2])))
        if not branch_conditioned_z:
            raise ValueError(f"No valid Monte Carlo PnP solutions for step {frame['step']}")
        frames.append({
            "step": frame["step"],
            "refined_reference_selected_branch_median_Z_m": selected_median_depth,
            "noiseless_IPPE_reference_median_Z_m": float(np.median(baseline[:, 2])),
            "noiseless_IPPE_vs_refined_reference_median_XYZ_gap_mm": float(
                np.median(np.linalg.norm(baseline-refined_reference, axis=1))*1000),
            "valid_draws": len(branch_conditioned_z),
            "failed_draws": failed,
            "different_branch_from_minimum_reprojection_draws": ambiguous,
            "branch_conditioned_median_Z_m": distribution(np.asarray(branch_conditioned_z)),
            "branch_conditioned_median_point_XYZ_shift_mm": distribution(
                np.asarray(branch_conditioned_xyz_shift_mm)),
            "minimum_reprojection_branch_median_Z_m": distribution(
                np.asarray(minimum_reprojection_z)),
        })
    result = {
        "method": "MONTE_CARLO_RGB_ANNOTATION_NOISE_PLANAR_IPPE",
        "noise_sigma_px": NOISE_SIGMA_PX,
        "draws_per_frame": DRAWS,
        "seed": 5201,
        "sensor_depth_used": False,
        "future_frames_used": False,
        "branch_conditioning_warning": "Nearest camera XYZ to frozen PnP branch uses the original RGB result as a prior. Minimum-reprojection branch distribution is also reported; neither establishes true pose.",
        "frames": frames,
    }
    (RUN / "pnp_annotation_noise_sensitivity.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
