"""RGB-only joint K from board CAD and an unoccluded peg lying on its table.

This exploratory route uses a contact constraint inferred from the scene, not
sensor depth. It is evaluated against held-out board features and Z16 only
after RGB optimization; a low RGB residual alone cannot validate K.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate_fmb_board_k import (HOLDOUT_FEATURES, cad_features,
                                   depth_crosscheck, extract, project)
from check_fmb_cross_orientation_depth import check as peg_depth_check
from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_cad_silhouette_k import project_model, support
from probe_fmb_cross_orientation_silhouette_k import observed_hull
from probe_fmb_table_contact_k import board_pose, peg_pose_on_table, solve_contact


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
SOURCE = ROOT / "data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy"


def unpack_k(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]], [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def fit_joint(board_points: np.ndarray, board_uv: np.ndarray,
              peg_support: np.ndarray, board_ids: np.ndarray,
              seed: np.ndarray, axis: str, fix_k: bool = False) -> dict:
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-2., -2., .02],
                  [-1., -1., -2*np.pi]]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [2., 2., 2.],
                  [1., 1., 2*np.pi]]
    if fix_k:
        frozen = seed[:4].copy()
        unpack = lambda y: np.r_[frozen, y]
        x0, lo, hi = seed[4:], lower[4:], upper[4:]
    else:
        unpack = lambda y: y
        x0, lo, hi = seed, lower, upper

    def residual(y: np.ndarray) -> np.ndarray:
        x = unpack(y)
        k = unpack_k(x)
        b = x[4:10]
        peg = peg_pose_on_table(x[10:13], b, axis)
        board_error = (project(board_points, x[:10])[board_ids]-
                       board_uv[board_ids]).ravel()
        peg_error = support(project_model(peg[:3], peg[3:], k))-peg_support
        return np.r_[board_error, peg_error]

    res = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1",
                        f_scale=2., x_scale="jac", max_nfev=260)
    x = unpack(res.x)
    k = unpack_k(x)
    peg = peg_pose_on_table(x[10:13], x[4:10], axis)
    board_error = np.linalg.norm(project(board_points, x[:10])-board_uv, axis=1)
    peg_error = support(project_model(peg[:3], peg[3:], k))-peg_support
    return {"K": k.tolist(), "board_pose": x[4:10].tolist(),
            "peg_pose": peg.tolist(), "peg_contact_params": x[10:13].tolist(),
            "contact_axis": axis, "cost": float(res.cost),
            "success": bool(res.success), "nfev": int(res.nfev),
            "board_train_rmse_px": float(np.sqrt(np.mean(board_error[board_ids]**2))),
            "board_holdout_rmse_px": float(np.sqrt(np.mean(
                board_error[np.asarray(HOLDOUT_FEATURES)]**2))),
            "peg_support_rmse_px": float(np.sqrt(np.mean(peg_error**2))),
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frames, depth = source["obs/side_1"], source["obs/side_1_depth"]
    frame = frames[0]
    board_points, board_names, _ = cad_features(1)
    board_uv, _ = extract(frame, 1)
    train_ids = np.array([i for i in range(len(board_names))
                          if i not in HOLDOUT_FEATURES])
    peg_hull = observed_hull(frame, True)
    peg_target = support(peg_hull)
    cross = json.loads((OUT / "cross_orientation_silhouette_k_probe.json").read_text())
    board_bundle = json.loads((OUT / "multi_board_bundle_calibration.json").read_text())
    cameras = {"nominal": K_NOMINAL,
               "cross_orientation": np.asarray(cross["best_fit"]["K"]),
               "board_RGB": np.asarray(board_bundle["best_fit"]["K"])}
    fitted = []
    fixed = None
    for name, k in cameras.items():
        bp, _ = board_pose(frame, k)
        for axis in ("y", "x"):
            seed_pose = solve_contact(peg_target, peg_hull, bp, k, axis)[0]
            seed = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2],
                         bp, seed_pose["table_xy_and_yaw"]]
            row = fit_joint(board_points, board_uv, peg_target, train_ids,
                            seed, axis)
            row["seed_camera"] = name
            fitted.append(row)
            if name == "nominal" and axis == "y":
                fixed = fit_joint(board_points, board_uv, peg_target,
                                  train_ids, seed, axis, fix_k=True)
    fitted.sort(key=lambda x: x["cost"])
    best = fitted[0]
    kbest, pbest = np.asarray(best["K"]), np.asarray(best["peg_pose"])
    best_seed = np.r_[np.log(kbest[0, 0]), np.log(kbest[1, 1]),
                      kbest[0, 2], kbest[1, 2], best["board_pose"],
                      best["peg_contact_params"]]
    rng = np.random.default_rng(5201)
    monte_carlo = []
    for _ in range(50):
        noisy_board = board_uv.copy()
        noisy_board[train_ids] += rng.normal(0, 2., noisy_board[train_ids].shape)
        noisy_peg = support(peg_hull+rng.normal(0, 2., peg_hull.shape))
        trial = fit_joint(board_points, noisy_board, noisy_peg,
                          train_ids, best_seed, best["contact_axis"])
        if trial["success"]:
            k = np.asarray(trial["K"])
            monte_carlo.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    leave_feature_out = []
    for feature in (0, 1, 2, 3, 4, 5, 7, 10, 13, 18):
        if feature not in train_ids:
            continue
        ids = train_ids[train_ids != feature]
        trial = fit_joint(board_points, board_uv, peg_target,
                          ids, best_seed, best["contact_axis"])
        leave_feature_out.append({"excluded_feature": board_names[feature],
                                  "K": trial["K"],
                                  "board_holdout_rmse_px": trial["board_holdout_rmse_px"],
                                  "success": trial["success"]})
    xboard = np.r_[np.log(kbest[0, 0]), np.log(kbest[1, 1]),
                   kbest[0, 2], kbest[1, 2], best["board_pose"]]
    result = {"status": "DIAGNOSTIC_RGB_ONLY_BOARD_PEG_CONTACT_K",
              "source": SOURCE.name, "train_board_feature_names": [board_names[i] for i in train_ids],
              "heldout_board_feature_names": [board_names[i] for i in HOLDOUT_FEATURES],
              "seed_fits": fitted, "best_fit": best,
              "fixed_nominal_K_RGB_fit": fixed,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 50,
              "monte_carlo_successes": len(monte_carlo),
              "monte_carlo_K_fx_fy_cx_cy": monte_carlo,
              "monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(monte_carlo, [5, 50, 95], axis=0).tolist()
                  if monte_carlo else None),
              "leave_one_feature_out": leave_feature_out,
              "best_fit_board_sensor_depth_check": depth_crosscheck(
                  xboard, frames, depth, (0,)),
              "best_fit_peg_sensor_depth_check": peg_depth_check(
                  frame, depth[0], kbest, pbest),
              "peg_neighbor_frame_RGB_support_rmse_px": {
                  str(step): float(np.sqrt(np.mean((support(project_model(
                      pbest[:3], pbest[3:], kbest))-
                      support(observed_hull(frames[step], True)))**2)))
                  for step in (5, 10)},
              "limitations": [
                  "Single unoccluded horizontal orientation, repeated in nearby frames.",
                  "Table contact, side-down identity and board table height are inferred, not recorded.",
                  "Board outer corners are virtual and can share systematic annotation bias.",
                  "K optimization excludes sensor depth and future trajectory GT."]}
    (OUT / "board_peg_table_contact_K_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"K": best["K"], "contact_axis": best["contact_axis"],
                      "RGB_train_board_px": best["board_train_rmse_px"],
                      "RGB_holdout_board_px": best["board_holdout_rmse_px"],
                      "RGB_train_peg_px": best["peg_support_rmse_px"],
                      "fixed_nominal_RGB_board_px": fixed["board_train_rmse_px"],
                      "fixed_nominal_RGB_peg_px": fixed["peg_support_rmse_px"],
                      "monte_carlo_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "board_sensor_depth": result["best_fit_board_sensor_depth_check"],
                      "peg_sensor_depth": result["best_fit_peg_sensor_depth_check"]},
                     indent=2))


if __name__ == "__main__":
    main()
