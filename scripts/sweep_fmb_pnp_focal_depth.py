"""Conditional focal sweep for the FMB CAD/RGB fit against interior depth.

This cannot calibrate color intrinsics: it assumes D405 default depth units,
published RGB/depth interior registration, and a uniform focal adjustment.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import ANNOTATIONS, D, FRONT, H, K, OUT, SOURCE, W


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    probe = json.loads((OUT / "cad_pnp_probe.json").read_text(encoding="utf8"))
    prior = probe["joint_constant_orientation_fit"]
    x0 = np.r_[Rotation.from_matrix(prior["shared_R"]).as_rotvec(),
               np.asarray([f["t_m"] for f in prior["frames"]]).ravel()]
    steps = list(ANNOTATIONS)
    observations = np.asarray([ANNOTATIONS[s] for s in steps], dtype=np.float64)
    interior = np.asarray([[(a - .5) * W, -D / 2, (1 - b) * H]
                           for a in (.2, .5, .8) for b in (.25, .5, .75)])
    tcp = source["obs/tcp_pose"]
    tcp_distance = float(np.linalg.norm(tcp[140, :3] - tcp[130, :3]))
    records = []

    for factor in np.arange(.8, 1.2001, .025):
        k = K.copy()
        k[:2, :2] *= factor

        def reprojection_residual(x: np.ndarray) -> np.ndarray:
            return np.concatenate([
                (cv2.projectPoints(FRONT, x[:3], x[3+3*i:6+3*i], k, None)[0]
                 .reshape(4, 2) - observations[i]).ravel()
                for i in range(len(steps))])

        fit = least_squares(reprojection_residual, x0, method="lm")
        rot = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        translations = fit.x[3:].reshape(len(steps), 3)
        errors_by_frame = []
        counts = []
        for i, step in enumerate(steps):
            uv = cv2.projectPoints(interior, fit.x[:3], translations[i], k, None)[0]
            uv = np.rint(uv.reshape(-1, 2)).astype(int)
            predicted_z = (rot @ interior.T).T[:, 2] + translations[i, 2]
            depth = source["obs/side_1_depth"][step]
            errors = []
            for (u, v), z in zip(uv, predicted_z):
                patch = depth[v-2:v+3, u-2:u+3]
                valid = patch[patch > 0]
                if len(valid) >= 3:
                    errors.append(float(z - np.median(valid) * .0001))
            errors_by_frame.append(errors)
            counts.append(len(errors))
        all_errors = np.concatenate(errors_by_frame)
        center_motion = float(np.linalg.norm(translations[2] - translations[0]))
        cad_z_by_step = [float(np.median((rot @ interior.T).T[:, 2] + translations[i, 2]))
                         for i in range(len(steps))]
        records.append({
            "focal_factor": round(float(factor), 3),
            "rgb_corner_rmse_px": float(np.sqrt(np.mean(reprojection_residual(fit.x)**2))),
            "cad_interior_median_Z_m_by_step": cad_z_by_step,
            "cad_interior_median_Z_m": float(np.median(cad_z_by_step)),
            "valid_interior_depth_samples_by_step": counts,
            "median_signed_Z_residual_m_by_step": [float(np.median(e)) for e in errors_by_frame],
            "median_absolute_Z_residual_m": float(np.median(np.abs(all_errors))),
            "pnp_130_to_140_displacement_m": center_motion,
            "difference_from_tcp_displacement_m": center_motion - tcp_distance,
        })

    best = min(records, key=lambda r: r["median_absolute_Z_residual_m"])
    result = {
        "status": "CONDITIONAL_FOCAL_SWEEP_NOT_RGB_CALIBRATION",
        "rgb_K_baseline": K.tolist(),
        "depth_units_assumed_m_per_count": .0001,
        "depth_sample": "median of positive 5x5 raw depth around each projected CAD interior point",
        "sensor_depth_used_to_select_focal_factor": True,
        "tcp_130_to_140_displacement_m": tcp_distance,
        "best_factor_by_interior_depth_residual": best["focal_factor"],
        "limitations": [
            "Published 256x256 RGB/depth registration is not exact or documented.",
            "The focal factor is shared by fx and fy, with fixed principal point and zero distortion.",
            "Depth units are the D405 SDK default, not recorded episode metadata.",
            "The color intrinsics cannot be inferred uniquely from three planar corner sets.",
        ],
        "records": records,
    }
    (OUT / "cad_pnp_focal_depth_sweep.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"status": result["status"],
                      "best_factor": best["focal_factor"],
                      "best_depth_median_abs_error_mm": best["median_absolute_Z_residual_m"] * 1000,
                      "best_rgb_corner_rmse_px": best["rgb_corner_rmse_px"],
                      "best_tcp_motion_difference_mm": best["difference_from_tcp_displacement_m"] * 1000,
                      "baseline_depth_median_abs_error_mm": records[8]["median_absolute_Z_residual_m"] * 1000},
                     indent=2))


if __name__ == "__main__":
    main()
