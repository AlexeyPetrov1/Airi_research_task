"""Propagate RGB-silhouette K sensitivity into the MolmoMotion 3D history.

All fits use RGB masks and rounded CAD only. Sensor depth is not used here.
The result describes sensitivity to the specified 2px contour perturbation,
not a calibrated probability distribution of the true 3D points.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import OUT, SOURCE
from probe_fmb_cad_silhouette_k import initial_pose, project_model, red_hull, support


HISTORY = (130, 135, 140)


def one_history(k: np.ndarray, frames: np.ndarray, targets: dict[int, np.ndarray],
                cad_points: np.ndarray) -> tuple[np.ndarray, list[float]]:
    points = []
    errors = []
    for step in HISTORY:
        p0 = initial_pose(step, frames[step], k)
        residual = lambda p: support(project_model(p[:3], p[3:], k)) - targets[step]
        fit = least_squares(residual, p0, loss="soft_l1", f_scale=2., max_nfev=120)
        r = Rotation.from_rotvec(fit.x[:3]).as_matrix()
        points.append((r @ cad_points.T).T + fit.x[3:])
        errors.append(float(np.sqrt(np.mean(residual(fit.x)**2))))
    return np.asarray(points), errors


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frames = source["obs/side_1"]
    targets = {s: support(red_hull(frames[s])) for s in HISTORY}
    manifest = json.loads((OUT / "cad_history_candidate/manifest.json").read_text(encoding="utf8"))
    cad_points = np.asarray(manifest["cad_points_m"], dtype=float)
    calibration = json.loads((OUT / "effective_k_rounded_cad_silhouette.json").read_text(encoding="utf8"))
    k_best = np.asarray(calibration["best_fit"]["K"])
    best, best_errors = one_history(k_best, frames, targets, cad_points)
    histories = []
    reprojections = []
    for fx, fy, cx, cy in calibration["monte_carlo_K_fx_fy_cx_cy"]:
        k = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        try:
            history, errors = one_history(k, frames, targets, cad_points)
        except (RuntimeError, ValueError):
            continue
        if np.isfinite(history).all():
            histories.append(history)
            reprojections.append(errors)
    all_histories = np.asarray(histories)
    all_reproj = np.asarray(reprojections)
    centroid_z = all_histories[:, :, :, 2].mean(axis=2)
    point_shift = np.linalg.norm(all_histories - best[None], axis=3)
    median_point_shift = np.median(point_shift, axis=2)
    temporal_motion = np.linalg.norm(all_histories[:, 2].mean(1) -
                                     all_histories[:, 0].mean(1), axis=1)
    tcp = source["obs/tcp_pose"]
    tcp_motion = float(np.linalg.norm(tcp[140, :3] - tcp[130, :3]))
    centers = all_histories.mean(axis=2)
    robot_motion_gates = {}
    for tolerance_mm in (2, 3, 5):
        accepted = np.ones(len(all_histories), dtype=bool)
        for a, b in ((0, 1), (1, 2), (0, 2)):
            observed_motion = np.linalg.norm(centers[:, b] - centers[:, a], axis=1)
            expected_motion = np.linalg.norm(tcp[HISTORY[b], :3] - tcp[HISTORY[a], :3])
            accepted &= np.abs(observed_motion - expected_motion) <= tolerance_mm * 0.001
        robot_motion_gates[str(tolerance_mm)] = {
            "accepted": int(accepted.sum()),
            "of": len(all_histories),
            "mean_Z_m_5_50_95_percentiles_by_step":
                np.percentile(centroid_z[accepted], [5, 50, 95], axis=0).tolist()
                if accepted.any() else None,
        }
    result = {
        "status": "SILHOUETTE_K_UNCERTAINTY_PROPAGATION_NOT_VALIDATED_3D",
        "source_calibration": "effective_k_rounded_cad_silhouette.json",
        "history_steps": HISTORY, "history_shape": list(best.shape),
        "contour_noise_sigma_px": 2.,
        "successful_mc_histories": len(histories),
        "best_K_history_mean_Z_m_by_step": best[:, :, 2].mean(axis=1).tolist(),
        "best_K_history_reprojection_support_rmse_px_by_step": best_errors,
        "mc_history_mean_Z_m_5_50_95_percentiles_by_step":
            np.percentile(centroid_z, [5, 50, 95], axis=0).tolist(),
        "mc_median_point_XYZ_shift_from_best_m_5_50_95_percentiles_by_step":
            np.percentile(median_point_shift, [5, 50, 95], axis=0).tolist(),
        "mc_reprojection_support_rmse_px_5_50_95_percentiles_by_step":
            np.percentile(all_reproj, [5, 50, 95], axis=0).tolist(),
        "mc_center_displacement_130_to_140_m_5_50_95_percentiles":
            np.percentile(temporal_motion, [5, 50, 95]).tolist(),
        "tcp_displacement_130_to_140_m": tcp_motion,
        "tcp_displacement_magnitude_gate_all_three_pairs": robot_motion_gates,
        "note": "All eight points are rigid CAD interior locations; pairwise distances are invariant by construction. RGB silhouettes do not independently identify each interior surface point."
    }
    (OUT / "effective_k_rounded_cad_history_uncertainty.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    np.savez_compressed(OUT / "effective_k_rounded_cad_history_candidates.npz",
                        best=best.astype("float32"), mc=all_histories.astype("float32"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
