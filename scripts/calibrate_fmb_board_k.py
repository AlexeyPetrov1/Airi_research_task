"""Estimate FMB side_1 effective K from the stationary metric assembly board.

The official STEP board has a 230 x 230 x 50 mm body and 13 distinct holes.
Hole centers are symmetry-derived *features* of the hole outlines, not physical
surface points. Four segmented outline corners are virtual intersections of
filleted CAD edges, while two front bottom tangencies are physical but only
pixel-accurate in the published 256px image. Sensor depth
and future MolmoMotion GT are excluded from calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT, SOURCE


BOARD = OUT / "peg_board_top_wires.json"
TRAIN = (0, 16, 32, 48)
HOLDOUT = (8, 24, 40, 56)
HOLDOUT_FEATURES = (8, 12, 16)


def cad_features(solid: int) -> tuple[np.ndarray, list[str], list[np.ndarray]]:
    data = json.loads(BOARD.read_text(encoding="utf8"))[solid]
    # The photographed hole face has the larger outer outline at z=-1.5 mm;
    # the opposite face at z=48.5 mm is inset by a 2 mm edge round. Both
    # planes have through-hole wires, so geometry alone does not name a top.
    face = next(f for f in data["top_faces"] if abs(f["z_mm"]+1.5) < .001)
    wires = face["wires"]
    base = np.array([[0., 35.], [280., 40.], [570., 40.]])[solid]
    upper = np.array([[0., 0., -1.5], [230., 0., -1.5],
                      [230., 230., -1.5], [0., 230., -1.5]])
    lower = np.array([[2., 228., 48.5], [228., 228., 48.5]])
    centers = np.array([np.array(w["bbox_center_mm"])-np.r_[base, 0.] for w in wires[1:]])
    points = np.vstack((upper, lower, centers))*.001
    names = ["top_back_left", "top_back_right", "top_front_right", "top_front_left",
             "bottom_front_left", "bottom_front_right"] + [f"hole_center_{i}" for i in range(1, 14)]
    bounds = []
    for w in wires[1:]:
        b = np.array(w["bounds_mm"])
        bounds.append(np.array([b[0]-base[0], b[1]-base[1], b[3]-base[0], b[4]-base[1]]))
    return points, names, bounds


def ordered_quad(contour: np.ndarray) -> np.ndarray:
    polygon = cv2.approxPolyDP(contour, .015*cv2.arcLength(contour, True), True).reshape(-1, 2)
    if len(polygon) != 4:
        raise ValueError(f"Expected four board top corners, got {polygon.tolist()}")
    top = polygon[np.argsort(polygon[:, 1])[:2]]
    bottom = polygon[np.argsort(polygon[:, 1])[2:]]
    return np.array([top[np.argmin(top[:, 0])], top[np.argmax(top[:, 0])],
                     bottom[np.argmax(bottom[:, 0])], bottom[np.argmin(bottom[:, 0])]],
                    dtype=float)


def extract(frame: np.ndarray, solid: int, top_v: int = 100,
            dark_v: int = 100, close_size: int = 5) -> tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    top_mask = cv2.inRange(hsv, (95, 90, top_v), (120, 255, 255))
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE,
                                np.ones((close_size, close_size), np.uint8))
    contours, _ = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    top_contour = max(contours, key=cv2.contourArea)
    # A front hole can interrupt the blue region's outer contour. The board's
    # actual outer outline is convex, so use its hull for the four corners.
    top_quad = ordered_quad(cv2.convexHull(top_contour))

    full_mask = cv2.inRange(hsv, (95, 80, 20), (125, 255, 255))
    full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_CLOSE,
                                 np.ones((close_size, close_size), np.uint8))
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    full_contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(full_contour, .01*cv2.arcLength(full_contour, True),
                               True).reshape(-1, 2)
    left, right = top_quad[3], top_quad[2]
    expected_top_y = left[1] + (polygon[:, 0]-left[0])*(right[1]-left[1])/(right[0]-left[0])
    front = polygon[polygon[:, 1] > expected_top_y+8]
    if len(front) < 2:
        raise ValueError(f"No board front bottom edge: {polygon.tolist()}")
    bottom = np.array([front[np.argmin(front[:, 0])], front[np.argmax(front[:, 0])]], dtype=float)

    polygon_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(polygon_mask, top_quad.astype(np.int32), 255)
    polygon_mask = cv2.erode(polygon_mask, np.ones((3, 3), np.uint8))
    dark = cv2.inRange(hsv, (90, 50, 0), (130, 255, dark_v)) & polygon_mask
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    observed = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            continue
        center = [moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]]
        observed.append({"uv": center, "area_px2": float(area),
                         "bbox_xywh": list(cv2.boundingRect(contour))})
    if len(observed) < 13:
        raise ValueError(f"Only {len(observed)} hole components")
    points, _, _ = cad_features(solid)
    cad_top = points[:4, :2]*1000
    matrix = cv2.getPerspectiveTransform(cad_top.astype(np.float32),
                                         top_quad.astype(np.float32))
    guess = cv2.perspectiveTransform((points[6:, :2]*1000).astype(np.float32)[None],
                                     matrix)[0]
    candidates = np.array([item["uv"] for item in observed])
    distance = np.linalg.norm(guess[:, None]-candidates[None], axis=2)
    rows, cols = linear_sum_assignment(distance)
    if len(rows) != 13 or np.max(distance[rows, cols]) > 8:
        raise ValueError(f"Hole matching failed: distances {distance[rows, cols]}")
    holes = np.zeros((13, 2))
    match_info = []
    for i, j in zip(rows, cols):
        holes[i] = candidates[j]
        match_info.append({"hole_id": int(i+1), "component_index": int(j),
                           "homography_match_distance_px": float(distance[i, j]),
                           "area_px2": observed[j]["area_px2"],
                           "bbox_xywh": observed[j]["bbox_xywh"]})
    uv = np.vstack((top_quad, bottom, holes))
    return uv, {"top_quad_px": top_quad.tolist(), "bottom_front_px": bottom.tolist(),
                "hole_matches": match_info, "dark_component_count": len(observed)}


def unpack(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]],
                     [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def project(points: np.ndarray, x: np.ndarray, distortion: np.ndarray | None = None) -> np.ndarray:
    uv, _ = cv2.projectPoints(points, x[4:7], x[7:10], unpack(x), distortion)
    return uv.reshape(-1, 2)


def score(points: np.ndarray, observed: np.ndarray, x: np.ndarray,
          feature_ids: np.ndarray) -> dict:
    pred = project(points, x)[feature_ids]
    error = np.linalg.norm(observed[:, feature_ids]-pred[None], axis=2)
    return {"mean_px": float(np.mean(error)),
            "median_px": float(np.median(error)),
            "rmse_px": float(np.sqrt(np.mean(error**2))),
            "p90_px": float(np.percentile(error, 90)),
            "max_px": float(np.max(error)),
            "per_feature_mean_px": np.mean(error, axis=0).tolist()}


def fit(points: np.ndarray, observations: np.ndarray, feature_ids: np.ndarray,
        x0: np.ndarray, fix_k: bool = False, max_nfev: int = 250) -> dict:
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-2.]*2, .02]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [2.]*2, 2.]
    if fix_k:
        frozen = x0[:4].copy()
        pack = lambda y: np.r_[frozen, y]
        initial, lo, hi = x0[4:], lower[4:], upper[4:]
    else:
        pack = lambda y: y
        initial, lo, hi = x0, lower, upper
    res = least_squares(lambda y: (project(points, pack(y))[feature_ids][None]-
                                    observations[:, feature_ids]).ravel(),
                        initial, bounds=(lo, hi), loss="soft_l1", f_scale=2.,
                        x_scale="jac", max_nfev=max_nfev)
    x = pack(res.x)
    return {"x": x.tolist(), "K": unpack(x).tolist(),
            "success": bool(res.success), "message": res.message,
            "cost": float(res.cost), "nfev": int(res.nfev),
            "train": score(points, observations, x, feature_ids),
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def depth_crosscheck(x: np.ndarray, frames: np.ndarray, depth: np.ndarray,
                     frame_ids: tuple[int, ...]) -> dict:
    # Only the interior of the photographed flat hole face, far from hole and
    # outer boundaries, is used. These samples never enter the K objective.
    grid = np.array([[u, v, -.0015] for u in np.linspace(.025, .205, 10)
                     for v in np.linspace(.025, .205, 10)], dtype=float)
    projected = project(grid, x)
    rot, _ = cv2.Rodrigues(x[4:7])
    cad_xyz = (rot @ grid.T).T + x[7:10]
    rows = []
    all_signed = []
    for s in frame_ids:
        hsv = cv2.cvtColor(frames[s], cv2.COLOR_BGR2HSV)
        bright_blue = cv2.inRange(hsv, (95, 90, 120), (120, 255, 255))
        safe = cv2.erode(bright_blue, np.ones((7, 7), np.uint8)) > 0
        signed = []
        for (uf, vf), xyz in zip(projected, cad_xyz):
            u, v = np.rint((uf, vf)).astype(int)
            if not (3 <= u < 253 and 3 <= v < 253 and safe[v, u]):
                continue
            patch = depth[s, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            if len(valid) < 10:
                continue
            signed.append(float(xyz[2]-np.median(valid)*.0001))
        all_signed.extend(signed)
        rows.append({"frame": s, "count": len(signed),
                     "median_abs_Z_error_mm": float(np.median(np.abs(signed))*1000)
                     if signed else None,
                     "median_signed_Z_error_mm": float(np.median(signed)*1000)
                     if signed else None})
    error = np.array(all_signed)
    return {"per_frame": rows, "count": len(error),
            "median_abs_Z_error_mm": float(np.median(np.abs(error))*1000)
            if len(error) else None,
            "p90_abs_Z_error_mm": float(np.percentile(np.abs(error), 90)*1000)
            if len(error) else None,
            "median_signed_Z_error_mm": float(np.median(error)*1000)
            if len(error) else None}


def main() -> None:
    src = np.load(SOURCE, allow_pickle=True).item()
    all_ids = TRAIN + HOLDOUT
    records = {}
    for i in all_ids:
        uv, meta = extract(src["obs/side_1"][i], 1)
        records[str(i)] = {"uv": uv.tolist(), **meta}
    target = OUT / "board_rgb_observations.json"
    target.write_text(json.dumps({"status": "AUTO_SEGMENTED_BOARD_FEATURES_UNVALIDATED",
                                  "source": str(SOURCE), "solid_index": 1,
                                  "train_frames": TRAIN, "holdout_frames": HOLDOUT,
                                  "records": records}, indent=2) + "\n", encoding="utf8")
    points, names, _ = cad_features(1)
    train_obs = np.array([records[str(i)]["uv"] for i in TRAIN])
    hold_obs = np.array([records[str(i)]["uv"] for i in HOLDOUT])
    all_obs = np.concatenate((train_obs, hold_obs))
    repeat_errors = np.linalg.norm(all_obs-all_obs.mean(axis=0)[None], axis=2)
    use = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    separate = np.array(HOLDOUT_FEATURES)
    seeds = []
    for factor in (.7, 1., 1.4):
        k = K_NOMINAL.copy()
        k[0, 0] *= factor
        k[1, 1] *= factor
        ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(points[use]),
                                  np.ascontiguousarray(train_obs.mean(axis=0)[use]),
                                  k, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError("Board PnP initialization failed")
        x0 = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2],
                   rv.ravel(), tv.ravel()]
        seeds.append(fit(points, train_obs, use, x0))
    best = min(seeds, key=lambda row: row["cost"])
    x = np.array(best["x"])
    fixed_x = x.copy()
    fixed_x[:4] = [np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                   K_NOMINAL[0, 2], K_NOMINAL[1, 2]]
    fixed_nominal = fit(points, train_obs, use, fixed_x, fix_k=True)
    subset = []
    for excluded in (4, 5, 7, 10, 13, 18):
        chosen = use[use != excluded]
        row = fit(points, train_obs, chosen, x)
        subset.append({"excluded_feature": names[excluded], "K": row["K"],
                       "success": row["success"],
                       "heldout_feature_error_px": score(points, hold_obs,
                                                          np.array(row["x"]), separate)["rmse_px"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        perturbed = train_obs + rng.normal(0, 2., train_obs.shape)
        row = fit(points, perturbed, use, x, max_nfev=130)
        if row["success"]:
            km = np.array(row["K"])
            mc.append([km[0, 0], km[1, 1], km[0, 2], km[1, 2]])
    result = {"status": "CANDIDATE_K_FROM_STATIC_METRIC_BOARD_NOT_YET_DEPTH_VALIDATED",
              "source_step": str(OUT / "peg_board_official.step"),
              "board_solid_index": 1,
              "solid_choice": "Medium board: projected hole-center layout matches RGB to mean 0.51 px versus 3.84 and 1.95 px for large/small solid.",
              "train_frames": TRAIN, "holdout_frames": HOLDOUT,
              "train_feature_ids": [names[i] for i in use],
              "holdout_feature_ids": [names[i] for i in separate],
              "cad_feature_xyz_m": {name: p.tolist() for name, p in zip(names, points)},
              "feature_semantics": "hole centers are derived from CAD wire bbox symmetry and segmented dark-region centroid; outer corners are virtual intersections of 2 mm rounded edges; bottom front tangencies are physical",
              "annotation_repeatability": {"method": "Repeated automatic extraction on 8 static RGB frames; not an independent manual marking",
                  "mean_px": float(np.mean(repeat_errors)),
                  "median_px": float(np.median(repeat_errors)),
                  "p90_px": float(np.percentile(repeat_errors, 90)),
                  "max_px": float(np.max(repeat_errors))},
              "seed_fits": seeds, "best_fit": best,
              "holdout_frames_train_features": score(points, hold_obs, x, use),
              "train_frames_heldout_features": score(points, train_obs, x, separate),
              "holdout_frames_heldout_features": score(points, hold_obs, x, separate),
              "nominal_fixed_K_fit": fixed_nominal,
              "nominal_holdout_frames": score(points, hold_obs,
                                                np.array(fixed_nominal["x"]),
                                                np.arange(len(names))),
              "board_depth_crosscheck_holdout": depth_crosscheck(
                  x, src["obs/side_1"], src["obs/side_1_depth"], HOLDOUT),
              "board_depth_crosscheck_nominal": depth_crosscheck(
                  np.array(fixed_nominal["x"]), src["obs/side_1"],
                  src["obs/side_1_depth"], HOLDOUT),
              "subset_fits": subset,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["Single stationary board pose provides no viewpoint diversity.",
                              "Outer mask corners are virtual intersections of rounded CAD edges at 256 px.",
                              "Hole dark-region centroids can shift with lighting and perspective.",
                              "Sensor depth and future GT were not used to fit K."]}
    target = OUT / "board_calibration_probe.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"K": best["K"], "train": best["train"],
                      "holdout_frames": result["holdout_frames_train_features"],
                      "holdout_features": result["holdout_frames_heldout_features"],
                      "nominal_holdout": result["nominal_holdout_frames"],
                      "repeatability": result["annotation_repeatability"],
                      "K_interval": result["monte_carlo_K_5_50_95_percentiles"],
                      "subset_K": [s["K"] for s in subset]}, indent=2))


if __name__ == "__main__":
    main()
