"""Test whether published FMB RGB frames identify an effective 256px K.

This is an empirical probe, not factory calibration. Five silhouette vertices
are extracted per frame: four on the broad face and one on the top face.
Rounded CAD corners and gripper occlusion make these approximate landmarks.
Sensor depth is never used to estimate K or camera poses in the bundle fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import D, EIGHT, FRONT, K as K_CANDIDATE, OUT, SOURCE, W, H


TRAIN = (128, 132, 136, 140, 144)
HOLDOUT = (130, 138, 146)
STEPS = TRAIN + HOLDOUT
OBJECT = np.vstack([FRONT, [W / 2, D / 2, H]]).astype(np.float64)
INTERIOR = np.asarray([[(a - .5) * W, -D / 2, (1 - b) * H]
                       for a in (.2, .5, .8) for b in (.25, .5, .75)])


def landmarks(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
            cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    vertices = cv2.approxPolyDP(hull, .015 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    ymin, ymax = int(vertices[:, 1].min()), int(vertices[:, 1].max())
    top = vertices[vertices[:, 1] <= ymin + 20]
    bottom = vertices[vertices[:, 1] >= ymax - 10]
    if len(top) < 3 or len(bottom) < 2:
        raise ValueError(f"Could not identify top/bottom silhouette vertices: {vertices.tolist()}")
    top_front_left = top[np.argmin(top[:, 0])]
    top_front_right = top[np.argmax(top[:, 0])]
    other_top = top[(top[:, 0] != top_front_left[0]) & (top[:, 0] != top_front_right[0])]
    if len(other_top) != 1:
        raise ValueError(f"Ambiguous top face: {vertices.tolist()}")
    top_back_right = other_top[0]
    bottom_left = bottom[np.argmin(bottom[:, 0])]
    bottom_right = bottom[np.argmax(bottom[:, 0])]
    points = np.asarray([top_front_left, top_front_right, bottom_right,
                         bottom_left, top_back_right], dtype=np.float64)
    return points, vertices


def project(obj: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> np.ndarray:
    return cv2.projectPoints(obj, rvec, tvec, k, np.zeros(5))[0].reshape(-1, 2)


def pose_for_k(k: np.ndarray, uv: np.ndarray, use_top: bool = True) -> tuple[np.ndarray, np.ndarray]:
    obj = OBJECT if use_top else OBJECT[:4]
    pts = uv if use_top else uv[:4]
    if use_top:
        ok, rv, tv = cv2.solvePnP(obj, pts, k, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError("EPNP failed")
        fit = least_squares(lambda x: (project(obj, x[:3], x[3:], k) - pts).ravel(),
                            np.r_[rv.ravel(), tv.ravel()], max_nfev=120)
        return fit.x[:3], fit.x[3:]
    n, rvs, tvs, _ = cv2.solvePnPGeneric(obj, pts, k, None, flags=cv2.SOLVEPNP_IPPE)
    if n < 1:
        raise RuntimeError("IPPE failed")
    candidates = [(float(np.mean((project(obj, rv, tv, k) - pts) ** 2)), rv.ravel(), tv.ravel())
                  for rv, tv in zip(rvs, tvs)]
    _, rv, tv = min(candidates, key=lambda x: x[0])
    return rv, tv


def unpack_k(x: np.ndarray) -> np.ndarray:
    return np.asarray([[np.exp(x[0]), 0, x[2]], [0, np.exp(x[1]), x[3]], [0, 0, 1]])


def fit_bundle(observed: dict[int, np.ndarray], steps: tuple[int, ...],
               k_seed: np.ndarray, max_nfev: int = 130) -> dict:
    k0 = np.asarray(k_seed, dtype=np.float64)
    poses = [np.r_[*pose_for_k(k0, observed[s])] for s in steps]
    x0 = np.r_[np.log(k0[0, 0]), np.log(k0[1, 1]), k0[0, 2], k0[1, 2],
               np.asarray(poses).ravel()]

    def residual(x: np.ndarray) -> np.ndarray:
        k = unpack_k(x)
        return np.concatenate([(project(OBJECT, x[4+6*i:7+6*i],
                                        x[7+6*i:10+6*i], k) - observed[s]).ravel()
                               for i, s in enumerate(steps)])

    lower = np.r_[np.log(30), np.log(40), -128, -128,
                  np.tile([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, .02], len(steps))]
    upper = np.r_[np.log(800), np.log(1000), 384, 384,
                  np.tile([np.inf, np.inf, np.inf, np.inf, np.inf, 2.0], len(steps))]
    fit = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                        f_scale=2.0, max_nfev=max_nfev, x_scale="jac")
    k = unpack_k(fit.x)
    err = residual(fit.x).reshape(len(steps), 5, 2)
    poses_out = [fit.x[4+6*i:10+6*i].tolist() for i in range(len(steps))]
    return {"K": k.tolist(), "x": fit.x.tolist(), "poses": poses_out,
            "success": bool(fit.success), "message": fit.message,
            "rmse_coordinate_px": float(np.sqrt(np.mean(err ** 2))),
            "per_frame_rmse_coordinate_px": np.sqrt(np.mean(err ** 2, axis=(1, 2))).tolist(),
            "cost": float(fit.cost)}


def validate(k: np.ndarray, observed: dict[int, np.ndarray], source: dict) -> list[dict]:
    rows = []
    for step in HOLDOUT:
        uv = observed[step]
        _, branch_rvs, branch_tvs, _ = cv2.solvePnPGeneric(
            OBJECT[:4], uv[:4], k, None, flags=cv2.SOLVEPNP_IPPE)
        branch_checks = [{"four_corner_rmse_coordinate_px":
                          float(np.sqrt(np.mean((project(OBJECT[:4], rv, tv, k) - uv[:4]) ** 2))),
                          "fifth_point_error_px":
                          float(np.linalg.norm(project(OBJECT[4:], rv, tv, k)[0] - uv[4]))}
                         for rv, tv in zip(branch_rvs, branch_tvs)]
        rv4, tv4 = pose_for_k(k, uv, use_top=False)
        pred = project(OBJECT, rv4, tv4, k)
        rv5, tv5 = pose_for_k(k, uv, use_top=True)
        pred5 = project(OBJECT, rv5, tv5, k)
        depth = source["obs/side_1_depth"][step]
        checks = []
        for point in INTERIOR:
            u, v = np.rint(project(point[None], rv5, tv5, k)[0]).astype(int)
            if u < 2 or u > 253 or v < 2 or v > 253:
                continue
            patch = depth[v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            if len(valid) >= 3:
                z_cad = float((Rotation.from_rotvec(rv5).as_matrix() @ point + tv5)[2])
                checks.append(z_cad - float(np.median(valid) * .0001))
        rows.append({"step": step,
                     "IPPE_branch_checks": branch_checks,
                     "best_possible_fifth_point_error_px_across_branches":
                     min(b["fifth_point_error_px"] for b in branch_checks),
                     "fifth_point_error_px_after_four_point_IPPE": float(np.linalg.norm(pred[4] - uv[4])),
                     "all_five_reprojection_rmse_px": float(np.sqrt(np.mean((pred5 - uv) ** 2))),
                     "median_abs_interior_sensor_Z_error_m":
                     float(np.median(np.abs(checks))) if checks else None,
                     "sensor_depth_samples": len(checks)})
    return rows


def motion_rigidity(k: np.ndarray, source: dict) -> dict:
    steps = (130, 135, 140)
    poses = [pose_for_k(k, landmarks(source["obs/side_1"][s])[0]) for s in steps]
    rots = [Rotation.from_rotvec(rv) for rv, _ in poses]
    points = np.asarray([rot.as_matrix() @ EIGHT.T + tv[:, None]
                         for rot, (_, tv) in zip(rots, poses)]).transpose(0, 2, 1)
    cad_distances = np.linalg.norm(EIGHT[:, None] - EIGHT[None, :], axis=2)
    distance_error = float(np.max(np.abs(
        np.linalg.norm(points[:, :, None] - points[:, None, :], axis=3) - cad_distances)))
    tcp = source["obs/tcp_pose"]
    pairs = []
    for i, j in ((0, 1), (1, 2), (0, 2)):
        object_angle = float((rots[i].inv() * rots[j]).magnitude() * 180 / np.pi)
        tcp_i = Rotation.from_quat(tcp[steps[i], 3:])
        tcp_j = Rotation.from_quat(tcp[steps[j], 3:])
        tcp_angle = float((tcp_i.inv() * tcp_j).magnitude() * 180 / np.pi)
        object_shift = float(np.linalg.norm(points[j].mean(0) - points[i].mean(0)))
        tcp_shift = float(np.linalg.norm(tcp[steps[j], :3] - tcp[steps[i], :3]))
        pairs.append({"steps": [steps[i], steps[j]],
                      "object_rotation_deg": object_angle, "tcp_rotation_deg": tcp_angle,
                      "object_center_shift_m": object_shift, "tcp_shift_m": tcp_shift})
    return {"cad_pairwise_distance_max_error_m_by_construction": distance_error,
            "object_vs_tcp_pairs": pairs,
            "note": "CAD pairwise distances are invariant by construction; TCP motion is an external check if the grasp is rigid."}


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frames = source["obs/side_1"]
    observed, hulls = {}, {}
    for s in STEPS:
        observed[s], hulls[s] = landmarks(frames[s])
    overlay = np.full((len(STEPS) // 4 * 384, 4 * 384, 3), 255, np.uint8)
    for i, step in enumerate(STEPS):
        img = cv2.resize(frames[step], (384, 384), interpolation=cv2.INTER_NEAREST)
        for j, (u, v) in enumerate(observed[step]):
            xy = (round(u * 1.5), round(v * 1.5))
            cv2.circle(img, xy, 4, (0, 255, 0) if j < 4 else (255, 0, 0), -1)
            cv2.putText(img, str(j), (xy[0] + 5, xy[1] + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 0), 1)
        cv2.putText(img, str(step), (4, 376), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)
        overlay[(i // 4)*384:(i // 4+1)*384, (i % 4)*384:(i % 4+1)*384] = img
    cv2.imwrite(str(OUT / "effective_k_landmark_overlay.png"), overlay)

    seeds = []
    for factor in (.6, .8, 1., 1.2, 1.5):
        seed = K_CANDIDATE.copy()
        seed[0, 0] *= factor
        seed[1, 1] *= factor
        seeds.append(seed)
    fits = [fit_bundle(observed, TRAIN, k) for k in seeds]
    chosen = min(fits, key=lambda f: f["cost"])
    k = np.asarray(chosen["K"])
    leave_one_out = []
    for excluded in TRAIN:
        subset = tuple(s for s in TRAIN if s != excluded)
        fit = fit_bundle(observed, subset, k)
        leave_one_out.append({"excluded_step": excluded, "K": fit["K"],
                              "rmse_coordinate_px": fit["rmse_coordinate_px"],
                              "success": fit["success"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        noisy = {s: observed[s] + rng.normal(0, 2.0, observed[s].shape) for s in TRAIN}
        try:
            fit = fit_bundle(noisy, TRAIN, k, max_nfev=90)
            if fit["success"]:
                km = np.asarray(fit["K"])
                mc.append([float(km[0, 0]), float(km[1, 1]),
                           float(km[0, 2]), float(km[1, 2])])
        except (RuntimeError, ValueError):
            pass
    mc_array = np.asarray(mc)
    normals = np.asarray([Rotation.from_rotvec(p[:3]).as_matrix()[:, 1]
                          for p in chosen["poses"]])
    normal_angles = np.degrees(np.arccos(np.clip(normals @ normals[0], -1, 1)))
    result = {
        "status": "EXPLORATORY_EFFECTIVE_K_FROM_APPROXIMATE_SILHOUETTE_VERTICES",
        "source": str(SOURCE.relative_to(Path(__file__).resolve().parents[1])),
        "train_frames": TRAIN, "heldout_frames": HOLDOUT,
        "landmark_order": ["front_top_left", "front_top_right", "front_bottom_right",
                           "front_bottom_left", "back_top_right"],
        "cad_landmarks_m": OBJECT.tolist(),
        "annotations_uv": {str(s): observed[s].tolist() for s in STEPS},
        "convex_hull_vertices_uv": {str(s): hulls[s].tolist() for s in STEPS},
        "seed_fits": [{k: v for k, v in f.items() if k != "x"} for f in fits],
        "best_fit": {k: v for k, v in chosen.items() if k != "x"},
        "leave_one_train_frame_out": leave_one_out,
        "monte_carlo_annotation_sigma_px": 2.0,
        "monte_carlo_attempts": 100,
        "monte_carlo_successes": len(mc),
        "monte_carlo_K_fx_fy_cx_cy": mc,
        "monte_carlo_K_5_50_95_percentiles":
            np.percentile(mc_array, [5, 50, 95], axis=0).tolist() if len(mc) else None,
        "train_face_normal_change_deg_from_first": normal_angles.tolist(),
        "heldout": validate(k, observed, source),
        "heldout_candidate_K_control": validate(K_CANDIDATE, observed, source),
        "motion_rigidity": motion_rigidity(k, source),
        "depth_units_assumed_m_per_count": .0001,
        "limitations": ["Silhouette vertices are approximate because the CAD is rounded and gripper occludes edges.",
                        "Frames 128-146 occupy a narrow pose range.",
                        "Held-out poses are fitted from their own image; the fifth point is the independent reprojection check.",
                        "Sensor depth is a diagnostic only and retains registration/scale uncertainty."],
    }
    (OUT / "effective_k_bundle_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"seed_K": [f["K"] for f in fits],
                      "seed_cost": [f["cost"] for f in fits],
                      "best_K": chosen["K"], "best_rmse_px": chosen["rmse_coordinate_px"],
                      "leave_one_out_K": [f["K"] for f in leave_one_out],
                      "monte_carlo_K_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "monte_carlo_successes": len(mc),
                      "normal_change_deg": normal_angles.tolist(),
                      "heldout": result["heldout"]}, indent=2))


if __name__ == "__main__":
    main()
