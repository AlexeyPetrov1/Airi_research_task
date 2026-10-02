"""Experimental RGB-only effective-K fit using the exact rounded CAD silhouette.

The published STEP solid 48 has a 2.88 mm corner radius. Bounding-box corners
used by the earlier point fit are virtual; this probe compares projected
rounded-rectangle silhouettes with segmented red RGB masks instead.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import D, EIGHT, FRONT, H, K as K0, OUT, SOURCE, W
from probe_fmb_effective_k import INTERIOR, landmarks, pose_for_k


RADIUS = .00288
TRAIN = (0, 128, 134, 140, 146)
HOLDOUT = (130, 138, 144)
ANGLES = np.arange(0, 2 * np.pi, np.pi / 8)
DIRS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


def rounded_cuboid_points() -> np.ndarray:
    points = []
    for z in (0., H):
        for sx, sy, start in ((1, 1, 0), (-1, 1, np.pi/2),
                              (-1, -1, np.pi), (1, -1, 3*np.pi/2)):
            cx = sx * (W/2 - RADIUS)
            cy = sy * (D/2 - RADIUS)
            for theta in np.linspace(start, start + np.pi/2, 13):
                points.append([cx + RADIUS*np.cos(theta),
                               cy + RADIUS*np.sin(theta), z])
    return np.asarray(points, dtype=np.float64)


MODEL = rounded_cuboid_points()


def red_hull(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
            cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(float)


def support(points: np.ndarray) -> np.ndarray:
    return np.max(points @ DIRS.T, axis=0)


def project_model(rv: np.ndarray, tv: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.projectPoints(MODEL, rv, tv, k, None)[0].reshape(-1, 2)


def k_from_x(x: np.ndarray) -> np.ndarray:
    return np.asarray([[np.exp(x[0]), 0, x[2]], [0, np.exp(x[1]), x[3]], [0, 0, 1]])


def initial_pose(step: int, bgr: np.ndarray, k: np.ndarray) -> np.ndarray:
    if step >= 128:
        rv, tv = pose_for_k(k, landmarks(bgr)[0])
    else:
        uv = np.asarray([[92, 13], [104, 15], [106, 78], [98, 78]], dtype=float)
        _, rvs, tvs, _ = cv2.solvePnPGeneric(FRONT, uv, k, None, flags=cv2.SOLVEPNP_IPPE)
        rv, tv = rvs[0].ravel(), tvs[0].ravel()
    return np.r_[rv, tv]


def fit(steps: tuple[int, ...], obs: dict[int, np.ndarray],
        frames: np.ndarray, kseed: np.ndarray, max_nfev: int = 150) -> dict:
    poses = np.asarray([initial_pose(s, frames[s], kseed) for s in steps])
    x0 = np.r_[np.log(kseed[0, 0]), np.log(kseed[1, 1]),
               kseed[0, 2], kseed[1, 2], poses.ravel()]

    def residual(x: np.ndarray) -> np.ndarray:
        k = k_from_x(x)
        return np.concatenate([support(project_model(x[4+6*i:7+6*i],
                                                   x[7+6*i:10+6*i], k)) - obs[s]
                               for i, s in enumerate(steps)])

    lower = np.r_[np.log(30), np.log(40), -128, -128,
                  np.tile([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, .02], len(steps))]
    upper = np.r_[np.log(800), np.log(1000), 384, 384,
                  np.tile([np.inf, np.inf, np.inf, np.inf, np.inf, 2.0], len(steps))]
    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=2., max_nfev=max_nfev, x_scale="jac")
    errors = residual(result.x).reshape(len(steps), len(DIRS))
    return {"K": k_from_x(result.x).tolist(), "cost": float(result.cost),
            "rmse_support_px": float(np.sqrt(np.mean(errors ** 2))),
            "per_frame_rmse_support_px": np.sqrt(np.mean(errors ** 2, axis=1)).tolist(),
            "success": bool(result.success), "poses": result.x[4:].reshape(-1, 6).tolist()}


def validate(k: np.ndarray, steps: tuple[int, ...], obs: dict[int, np.ndarray],
             frames: np.ndarray, depth_frames: np.ndarray) -> list[dict]:
    rows = []
    for step in steps:
        pose0 = initial_pose(step, frames[step], k)
        even = np.arange(0, len(DIRS), 2)
        odd = np.arange(1, len(DIRS), 2)
        fit_partial = least_squares(
            lambda x: (support(project_model(x[:3], x[3:], k)) - obs[step])[even],
            pose0, loss="soft_l1", f_scale=2, max_nfev=120)
        partial_error = support(project_model(fit_partial.x[:3], fit_partial.x[3:], k)) - obs[step]
        fit_pose = least_squares(lambda x: support(project_model(x[:3], x[3:], k)) - obs[step],
                                 pose0, loss="soft_l1", f_scale=2, max_nfev=120)
        error = support(project_model(fit_pose.x[:3], fit_pose.x[3:], k)) - obs[step]
        uv = cv2.projectPoints(INTERIOR, fit_pose.x[:3], fit_pose.x[3:], k, None)[0].reshape(-1, 2)
        hsv = cv2.cvtColor(frames[step], cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
               cv2.inRange(hsv, (170, 75, 90), (179, 255, 255))) > 0
        rot = Rotation.from_rotvec(fit_pose.x[:3]).as_matrix()
        z = (rot @ INTERIOR.T).T[:, 2] + fit_pose.x[5]
        z_errors = []
        interior_red = []
        for (u_float, v_float), z_cad in zip(uv, z):
            u, v = np.rint([u_float, v_float]).astype(int)
            if not (2 <= u < 254 and 2 <= v < 254):
                continue
            interior_red.append(bool(red[v, u]))
            if not red[v, u]:
                continue
            patch = depth_frames[step, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            if len(valid) >= 3:
                z_errors.append(float(z_cad - np.median(valid)*.0001))
        rows.append({"step": step, "support_rmse_px_after_pose_fit":
                     float(np.sqrt(np.mean(error ** 2))),
                     "heldout_odd_direction_rmse_px":
                     float(np.sqrt(np.mean(partial_error[odd] ** 2))),
                     "interior_red_fraction": float(np.mean(interior_red)) if interior_red else None,
                     "median_abs_interior_sensor_Z_error_m":
                     float(np.median(np.abs(z_errors))) if z_errors else None,
                     "sensor_depth_samples": len(z_errors),
                     "pose": fit_pose.x.tolist()})
    return rows


def motion_check(k: np.ndarray, obs: dict[int, np.ndarray], source: dict) -> list[dict]:
    steps = (130, 135, 140)
    frames = source["obs/side_1"]
    poses = []
    for step in steps:
        target = obs.get(step, support(red_hull(frames[step])))
        pose0 = initial_pose(step, frames[step], k)
        fit_pose = least_squares(lambda x: support(project_model(x[:3], x[3:], k)) - target,
                                 pose0, loss="soft_l1", f_scale=2, max_nfev=120)
        poses.append(fit_pose.x)
    centers = np.asarray([Rotation.from_rotvec(p[:3]).as_matrix() @ EIGHT.mean(0) + p[3:]
                          for p in poses])
    tcp = source["obs/tcp_pose"]
    rows = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        obj_angle = float((Rotation.from_rotvec(poses[i][:3]).inv() *
                           Rotation.from_rotvec(poses[j][:3])).magnitude()*180/np.pi)
        tcp_angle = float((Rotation.from_quat(tcp[steps[i], 3:]).inv() *
                           Rotation.from_quat(tcp[steps[j], 3:])).magnitude()*180/np.pi)
        rows.append({"steps": [steps[i], steps[j]],
                     "object_center_shift_m": float(np.linalg.norm(centers[j]-centers[i])),
                     "tcp_shift_m": float(np.linalg.norm(tcp[steps[j], :3]-tcp[steps[i], :3])),
                     "object_rotation_deg": obj_angle, "tcp_rotation_deg": tcp_angle})
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
        seeds.append(k)
    fits = [fit(TRAIN, obs, frames, seed) for seed in seeds]
    best = min(fits, key=lambda x: x["cost"])
    k = np.asarray(best["K"])
    leave_one_out = []
    for excluded in TRAIN:
        subset = tuple(s for s in TRAIN if s != excluded)
        trial = fit(subset, obs, frames, k)
        leave_one_out.append({"excluded_step": excluded, "K": trial["K"],
                              "rmse_support_px": trial["rmse_support_px"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        perturbed = {s: support(hulls[s] + rng.normal(0, 2., hulls[s].shape)) for s in TRAIN}
        trial = fit(TRAIN, perturbed, frames, k, max_nfev=100)
        if trial["success"]:
            km = np.asarray(trial["K"])
            mc.append([float(km[0, 0]), float(km[1, 1]),
                       float(km[0, 2]), float(km[1, 2])])
    mc_arr = np.asarray(mc)
    depth = source["obs/side_1_depth"]
    result = {"status": "EXPLORATORY_EXACT_ROUNDED_CAD_SILHOUETTE_FIT",
              "train_frames": TRAIN, "heldout_frames": HOLDOUT,
              "cad_width_depth_height_radius_m": [W, D, H, RADIUS],
              "directions_rad": ANGLES.tolist(),
              "seed_fits": fits, "best_fit": best,
              "leave_one_train_frame_out": leave_one_out,
              "monte_carlo_silhouette_vertex_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K_fx_fy_cx_cy": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc_arr, [5, 50, 95], axis=0).tolist() if len(mc) else None,
              "heldout": validate(k, HOLDOUT, obs, frames, depth),
              "heldout_candidate_K_control": validate(K0, HOLDOUT, obs, frames, depth),
              "object_vs_tcp_motion": motion_check(k, obs, source),
              "limitations": ["Red mask hull is incomplete under gripper occlusion.",
                              "Held-out poses are fitted to their own silhouettes; support residual is not an independent CAD landmark check.",
                              "No sensor depth used in fitting K or poses."]}
    pose_by_step = {s: np.asarray(p) for s, p in zip(TRAIN, best["poses"])}
    pose_by_step.update({r["step"]: np.asarray(r["pose"]) for r in result["heldout"]})
    ordered_steps = TRAIN + HOLDOUT
    overlay = np.full((2*384, 4*384, 3), 255, np.uint8)
    for i, step in enumerate(ordered_steps):
        img = frames[step].copy()
        pose = pose_by_step[step]
        predicted = cv2.convexHull(np.rint(project_model(pose[:3], pose[3:], k)).astype(np.int32))
        measured = cv2.convexHull(hulls[step].astype(np.int32))
        cv2.polylines(img, [measured], True, (255, 0, 0), 1)
        cv2.polylines(img, [predicted], True, (0, 255, 0), 1)
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img, f"{step} {'train' if step in TRAIN else 'test'}", (5, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 2)
        overlay[(i//4)*384:(i//4+1)*384, (i%4)*384:(i%4+1)*384] = img
    cv2.imwrite(str(OUT / "effective_k_rounded_cad_silhouette_overlay.png"), overlay)
    (OUT / "effective_k_rounded_cad_silhouette.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"seed_K": [f["K"] for f in fits],
                      "seed_cost": [f["cost"] for f in fits],
                      "best_rmse": best["rmse_support_px"],
                      "leave_one_out_K": [r["K"] for r in leave_one_out],
                      "monte_carlo_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "monte_carlo_successes": len(mc),
                      "heldout": result["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
