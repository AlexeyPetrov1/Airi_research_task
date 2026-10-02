"""Joint RGB-only side_1 K from several placements of the official FMB board."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibrate_fmb_board_k import (HOLDOUT, HOLDOUT_FEATURES, TRAIN,
                                   cad_features, depth_crosscheck, extract)
from calibrate_fmb_two_boards_k import fit_multi, score_multi
from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT, SOURCE


DATA = SOURCE.parents[3] / "data/fmb/single_object_manipulation_dataset"
EPISODES = (2, 3, 4, 5, 6)


def background_alignment(reference: np.ndarray, other: np.ndarray) -> dict:
    gray0 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gray1 = cv2.cvtColor(other, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    mask = np.zeros(reference.shape[:2], np.uint8)
    mask[:75] = 255
    mask[:75, 65:150] = 0  # moving peg and robot in all five episodes
    warp = np.eye(2, 3, dtype=np.float32)
    score, warp = cv2.findTransformECC(
        gray0, gray1, warp, cv2.MOTION_TRANSLATION,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), mask, 5)
    return {"ECC": float(score),
            "translation_px": warp[:, 2].tolist(),
            "mean_abs_static_gray_difference": float(
                np.mean(np.abs(gray0[mask > 0]-gray1[mask > 0])))}


def main() -> None:
    points, names, _ = cad_features(1)
    features = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    test_features = np.array(HOLDOUT_FEATURES)
    train_obs, hold_obs, frames, depths = [], [], [], []
    metadata = []
    correspondences = []
    frame_manifest = []
    reference = None
    for number in EPISODES:
        path = DATA / f"1_M_L_3_vertical_n_{number}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        source = np.load(path, allow_pickle=True).item()
        image = source["obs/side_1"]
        first = image[0].copy()
        if reference is None:
            reference = first
        alignment = background_alignment(reference, first)
        records = {s: extract(image[s], 1) for s in TRAIN + HOLDOUT}
        for s, (uv, extra) in records.items():
            split = "train" if s in TRAIN else "holdout"
            frame_manifest.append({"source_episode": number, "frame_id": s,
                                   "split": split,
                                   "primitive": str(source["primitive"][s]),
                                   "visibility": "board top, 13 holes and front side visible",
                                   "selection_reason": "Static, unobscured board before forecast window",
                                   "hole_match_max_px": max(
                                       row["homography_match_distance_px"]
                                       for row in extra["hole_matches"])})
            for j, (name, xyz, xy) in enumerate(zip(names, points, uv)):
                semantics = ("virtual_rounded_outline_intersection" if j < 4 else
                             "physical_front_bottom_tangency" if j < 6 else
                             "derived_hole_center_not_material_point")
                correspondences.append({"source_episode": number,
                                        "frame": s, "split": split,
                                        "cad_point_id": name,
                                        "xyz_cad_m": xyz.tolist(),
                                        "uv_rgb_px": xy.tolist(),
                                        "semantics": semantics,
                                        "used_for_K_fit": split == "train" and
                                        j in features})
        train_obs.append(np.array([records[s][0] for s in TRAIN]))
        hold_obs.append(np.array([records[s][0] for s in HOLDOUT]))
        frames.append(image[list(HOLDOUT)].copy())
        depths.append(source["obs/side_1_depth"][list(HOLDOUT)].copy())
        quad = np.array(records[0][1]["top_quad_px"])
        metadata.append({"episode": number, "source": str(path),
                         "frames": {"train": TRAIN, "holdout": HOLDOUT},
                         "object_info": source["object_info"],
                         "static_background": alignment,
                         "board_top_quad_frame0_px": quad.tolist(),
                         "board_center_frame0_px": quad.mean(axis=0).tolist(),
                         "board_back_edge_angle_deg": float(np.degrees(
                             np.arctan2(quad[1, 1]-quad[0, 1],
                                        quad[1, 0]-quad[0, 0]))),
                         "mean_hole_homography_match_px": float(np.mean(
                             [m["homography_match_distance_px"]
                              for m in records[0][1]["hole_matches"]]))})
        del source
    (OUT / "board_calibration_frames.json").write_text(json.dumps({
        "camera": "side_1", "image_size": [256, 256],
        "train_holdout_split_fixed_by_frame_id": True,
        "frames": frame_manifest}, indent=2) + "\n", encoding="utf8")
    (OUT / "board_cad_rgb_correspondences.json").write_text(json.dumps({
        "cad_source": str(OUT / "peg_board_official.step"),
        "board_solid_index": 1,
        "warning": "Hole centers and rounded-outline intersections are geometric features, not persistent material surface points.",
        "correspondences": correspondences}, indent=2) + "\n", encoding="utf8")
    train_obs = tuple(train_obs)
    hold_obs = tuple(hold_obs)
    if any(row["static_background"]["ECC"] < .99 or
           max(abs(x) for x in row["static_background"]["translation_px"]) > 1
           for row in metadata):
        raise ValueError("Cannot justify one fixed side_1 camera across episodes")
    seeds = []
    for factor in (.7, 1., 1.4):
        k = K_NOMINAL.copy()
        k[0, 0] *= factor
        k[1, 1] *= factor
        poses = []
        for obs in train_obs:
            ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(points[features]),
                                      np.ascontiguousarray(obs.mean(axis=0)[features]),
                                      k, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                raise RuntimeError("PnP seed failed")
            poses.extend([*rv.ravel(), *tv.ravel()])
        seed = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2], poses]
        seeds.append(fit_multi(points, train_obs, features, seed))
    best = min(seeds, key=lambda row: row["cost"])
    bx = np.array(best["x"])
    fixed = bx.copy()
    fixed[:4] = [np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                 K_NOMINAL[0, 2], K_NOMINAL[1, 2]]
    nominal = fit_multi(points, train_obs, features, fixed, fix_k=True)
    nx = np.array(nominal["x"])
    subset = []
    for excluded in (4, 5, 7, 10, 13, 18):
        ids = features[features != excluded]
        row = fit_multi(points, train_obs, ids, bx)
        subset.append({"excluded_feature": names[excluded],
                       "K": row["K"], "success": row["success"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        noisy = tuple(a+rng.normal(0, 2., a.shape) for a in train_obs)
        row = fit_multi(points, noisy, features, bx, max_nfev=120)
        if row["success"]:
            k = np.array(row["K"])
            mc.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    depth_new, depth_nom = [], []
    normals = []
    for i, (rgb, dep) in enumerate(zip(frames, depths)):
        xnew = np.r_[bx[:4], bx[4+6*i:10+6*i]]
        xnom = np.r_[nx[:4], nx[4+6*i:10+6*i]]
        depth_new.append(depth_crosscheck(xnew, rgb, dep, tuple(range(len(HOLDOUT)))))
        depth_nom.append(depth_crosscheck(xnom, rgb, dep, tuple(range(len(HOLDOUT)))))
        normals.append(Rotation.from_rotvec(bx[4+6*i:7+6*i]).apply([0, 0, 1]))
    normals = np.array(normals)
    normal_angles = np.degrees(np.arccos(np.clip(normals@normals.T, -1, 1)))
    result = {"status": "MULTI_BOARD_RGB_ONLY_K_CANDIDATE_PENDING_ACCEPTANCE",
              "official_board_cad": str(OUT / "peg_board_official.step"),
              "board_solid_index": 1,
              "camera_stream": "side_1 RGB 256x256",
              "episodes": metadata,
              "feature_ids_train": [names[i] for i in features],
              "feature_ids_holdout": [names[i] for i in test_features],
              "CAD_feature_xyz_m": {n: p.tolist() for n, p in zip(names, points)},
              "seed_fits": seeds, "best_fit": best,
              "heldout_frames_train_features": score_multi(points, hold_obs, bx, features),
              "heldout_frames_heldout_features": score_multi(points, hold_obs, bx, test_features),
              "fixed_nominal_fit": nominal,
              "fixed_nominal_heldout_frames": score_multi(points, hold_obs, nx,
                                                            np.arange(len(names))),
              "depth_crosscheck_new": depth_new,
              "depth_crosscheck_fixed_nominal": depth_nom,
              "board_normal_angle_matrix_deg": normal_angles.tolist(),
              "subset_fits": subset,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["All board poses are static within their source episode.",
                              "Outer corners are virtual intersections of rounded CAD edges.",
                              "Hole centers are shape-derived, not material points.",
                              "K was estimated without depth or future tracking GT."]}
    (OUT / "multi_board_bundle_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"K": best["K"],
                      "RGB_train_RMSE": [x["rmse_px"] for x in best["train"]],
                      "RGB_holdout_RMSE": [x["rmse_px"] for x in result["heldout_frames_train_features"]],
                      "depth_new_median_mm": [x["median_abs_Z_error_mm"] for x in depth_new],
                      "depth_nominal_median_mm": [x["median_abs_Z_error_mm"] for x in depth_nom],
                      "board_normal_angle_matrix_deg": normal_angles.tolist(),
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "subset_K": [s["K"] for s in subset]}, indent=2))


if __name__ == "__main__":
    main()
