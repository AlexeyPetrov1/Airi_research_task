"""RGB-only rounded-CAD K probe with one late object orientation.

FMB TCP turns only 2.43 degrees from frames 128 to 146. This deliberately
strong shared-orientation hypothesis tests whether robot rigidity can make
effective K identifiable without using sensor depth during fitting.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import K as K0, OUT, SOURCE
from probe_fmb_cad_silhouette_k import (DIRS, HOLDOUT, TRAIN, initial_pose,
                                        k_from_x, project_model, red_hull, support)
from probe_fmb_effective_k import INTERIOR


def fit_shared(steps: tuple[int, ...], obs: dict[int, np.ndarray], frames: np.ndarray,
               kseed: np.ndarray, rotation_seed: int = 0, max_nfev: int = 150) -> dict:
    assert steps[0] == 0
    late = steps[1:]
    starts = [initial_pose(s, frames[s], kseed) for s in steps]
    x0 = np.r_[np.log(kseed[0, 0]), np.log(kseed[1, 1]), kseed[0, 2], kseed[1, 2],
               starts[0], starts[1 + rotation_seed][:3],
               np.asarray([p[3:] for p in starts[1:]]).ravel()]

    def residual(x: np.ndarray) -> np.ndarray:
        k = k_from_x(x)
        e = [support(project_model(x[4:7], x[7:10], k)) - obs[0]]
        for i, s in enumerate(late):
            e.append(support(project_model(x[10:13], x[13+3*i:16+3*i], k)) - obs[s])
        return np.concatenate(e)

    lower = np.r_[np.log(30), np.log(40), -128, -128,
                  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, .02,
                  -np.inf, -np.inf, -np.inf,
                  np.tile([-np.inf, -np.inf, .02], len(late))]
    upper = np.r_[np.log(800), np.log(1000), 384, 384,
                  np.inf, np.inf, np.inf, np.inf, np.inf, 2.,
                  np.inf, np.inf, np.inf,
                  np.tile([np.inf, np.inf, 2.], len(late))]
    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=2., max_nfev=max_nfev, x_scale="jac")
    error = residual(result.x).reshape(len(steps), len(DIRS))
    return {"K": k_from_x(result.x).tolist(), "cost": float(result.cost),
            "rmse_support_px": float(np.sqrt(np.mean(error**2))),
            "per_frame_rmse_support_px": np.sqrt(np.mean(error**2, axis=1)).tolist(),
            "success": bool(result.success),
            "early_pose": result.x[4:10].tolist(),
            "late_shared_rotation": result.x[10:13].tolist(),
            "late_translations": result.x[13:].reshape(-1, 3).tolist()}


def heldout_check(fit: dict, source: dict, obs: dict[int, np.ndarray]) -> list[dict]:
    k = np.asarray(fit["K"])
    rv = np.asarray(fit["late_shared_rotation"])
    frames = source["obs/side_1"]
    depth = source["obs/side_1_depth"]
    rows = []
    for s in HOLDOUT:
        p0 = initial_pose(s, frames[s], k)[3:]
        even = np.arange(0, len(DIRS), 2)
        odd = np.arange(1, len(DIRS), 2)
        def residual(tv: np.ndarray) -> np.ndarray:
            return support(project_model(rv, tv, k)) - obs[s]
        partial = least_squares(lambda tv: residual(tv)[even], p0,
                                loss="soft_l1", f_scale=2., max_nfev=100)
        full = least_squares(residual, p0, loss="soft_l1", f_scale=2., max_nfev=100)
        uv = cv2.projectPoints(INTERIOR, rv, full.x, k, None)[0].reshape(-1, 2)
        z = (Rotation.from_rotvec(rv).as_matrix() @ INTERIOR.T).T[:, 2] + full.x[2]
        hsv = cv2.cvtColor(frames[s], cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
               cv2.inRange(hsv, (170, 75, 90), (179, 255, 255))) > 0
        errors = []
        interior_red = []
        for (u_float, v_float), z_cad in zip(uv, z):
            u, v = np.rint([u_float, v_float]).astype(int)
            if not (2 <= u < 254 and 2 <= v < 254):
                continue
            interior_red.append(bool(red[v, u]))
            if not red[v, u]:
                continue
            patch = depth[s, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            if len(valid) >= 3:
                errors.append(float(z_cad - np.median(valid)*.0001))
        rows.append({"step": s, "support_rmse_px": float(np.sqrt(np.mean(residual(full.x)**2))),
                     "heldout_odd_direction_rmse_px":
                     float(np.sqrt(np.mean(residual(partial.x)[odd]**2))),
                     "translation_m": full.x.tolist(),
                     "interior_red_fraction": float(np.mean(interior_red)) if interior_red else None,
                     "median_abs_interior_sensor_Z_error_m":
                     float(np.median(np.abs(errors))) if errors else None,
                     "sensor_depth_samples": len(errors)})
    return rows


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frames = source["obs/side_1"]
    hulls = {s: red_hull(frames[s]) for s in TRAIN + HOLDOUT}
    obs = {s: support(hulls[s]) for s in TRAIN + HOLDOUT}
    seeds = []
    for factor in (.7, 1., 1.4):
        k = K0.copy()
        k[0, 0] *= factor
        k[1, 1] *= factor
        for rotation_seed in (0, len(TRAIN)-2):
            seeds.append(fit_shared(TRAIN, obs, frames, k, rotation_seed))
    best = min(seeds, key=lambda x: x["cost"])
    k = np.asarray(best["K"])
    leave_one_out = []
    for s in TRAIN[1:]:
        subset = tuple(x for x in TRAIN if x != s)
        result = fit_shared(subset, obs, frames, k)
        leave_one_out.append({"excluded_step": s, "K": result["K"],
                              "rmse_support_px": result["rmse_support_px"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        noisy = {s: support(hulls[s] + rng.normal(0, 2., hulls[s].shape)) for s in TRAIN}
        trial = fit_shared(TRAIN, noisy, frames, k, max_nfev=100)
        if trial["success"]:
            km = np.asarray(trial["K"])
            mc.append([float(km[0, 0]), float(km[1, 1]),
                       float(km[0, 2]), float(km[1, 2])])
    mc_array = np.asarray(mc)
    result = {"status": "EXPLORATORY_SHARED_LATE_ORIENTATION_CAD_SILHOUETTE_K",
              "train_frames": TRAIN, "heldout_frames": HOLDOUT,
              "shared_orientation_assumption": "Frames 128-146; TCP rotates <=2.43 deg",
              "seed_fits": seeds, "best_fit": best,
              "leave_one_late_train_frame_out": leave_one_out,
              "monte_carlo_silhouette_vertex_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K_fx_fy_cx_cy": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc_array, [5, 50, 95], axis=0).tolist() if len(mc) else None,
              "heldout": heldout_check(best, source, obs),
              "limitations": ["Rigid-grasp orientation assumption is stronger than the source data guarantee.",
                              "Red mask hull is incomplete under gripper occlusion.",
                              "Sensor depth is only used in held-out diagnostics."]}
    (OUT / "effective_k_rounded_cad_rigid_probe.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"best_K": best["K"], "best_rmse": best["rmse_support_px"],
                      "seed_cost": [x["cost"] for x in seeds],
                      "leave_one_out_K": [x["K"] for x in leave_one_out],
                      "monte_carlo_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "monte_carlo_successes": len(mc), "heldout": result["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
