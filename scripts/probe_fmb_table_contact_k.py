"""RGB-only peg/table-contact diagnostic using FMB's metric board CAD.

The board pose is estimated from its RGB features for a fixed candidate K.
The peg is constrained to lie on the same table, with either long side down.
This supplies a physically motivated pose branch without using sensor depth.
It is still diagnostic: board edge features and the contact face are uncertain.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrate_fmb_board_k import cad_features, extract, fit as fit_board
from check_fmb_cross_orientation_depth import check as depth_check
from probe_fmb_cad_pnp import D, H, K as K_NOMINAL, W
from probe_fmb_cad_silhouette_k import project_model, support
from probe_fmb_cross_orientation_silhouette_k import observed_hull


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
SOURCE = ROOT / "data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy"
BOARD_BOTTOM_Z_M = .0485


def board_pose(frame: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, dict]:
    points, names, _ = cad_features(1)
    uv, _ = extract(frame, 1)
    ok, rv, tv = cv2.solvePnP(points, uv, k, None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("Board pose seed failed")
    x = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2],
              rv.ravel(), tv.ravel()]
    fitted = fit_board(points, uv[None], np.arange(len(names)), x, fix_k=True)
    return np.asarray(fitted["x"])[4:10], fitted["train"]


def peg_pose_on_table(params: np.ndarray, board: np.ndarray,
                      contact_axis: str) -> np.ndarray:
    board_r = Rotation.from_rotvec(board[:3]).as_matrix()
    n = board_r[:, 2]  # board CAD +z points through the board toward the table
    e1, e2 = board_r[:, 0], board_r[:, 1]
    xy = params[:2]
    major = np.cos(params[2])*e1 + np.sin(params[2])*e2
    if contact_axis == "y":
        ey = -n
        ex = np.cross(ey, major)
        contact_local = np.array([0., -D/2, H/2])
        rotation = np.column_stack((ex, ey, major))
    else:
        ex = -n
        ey = np.cross(major, ex)
        contact_local = np.array([-W/2, 0., H/2])
        rotation = np.column_stack((ex, ey, major))
    table_contact = board_r @ np.r_[xy, BOARD_BOTTOM_Z_M] + board[3:]
    translation = table_contact - rotation @ contact_local
    return np.r_[Rotation.from_matrix(rotation).as_rotvec(), translation]


def solve_contact(obs: np.ndarray, hull: np.ndarray, board: np.ndarray,
                  k: np.ndarray, axis: str) -> list[dict]:
    board_r = Rotation.from_rotvec(board[:3]).as_matrix()
    n = board_r[:, 2]
    center = np.mean([hull.min(axis=0), hull.max(axis=0)], axis=0)
    ray = np.linalg.inv(k) @ np.r_[center, 1.]
    half = D/2 if axis == "y" else W/2
    table_origin = board_r @ [0., 0., BOARD_BOTTOM_Z_M] + board[3:]
    scale = (n @ (table_origin-half*n)) / (n @ ray)
    center_camera = scale*ray
    contact_camera = center_camera+half*n
    xy_seed = (board_r.T @ (contact_camera-board[3:]))[:2]
    solutions = []
    for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
        p0 = np.r_[xy_seed, angle]

        def residual(p: np.ndarray) -> np.ndarray:
            pose = peg_pose_on_table(p, board, axis)
            return support(project_model(pose[:3], pose[3:], k))-obs

        result = least_squares(residual, p0,
                               bounds=([-1., -1., -2*np.pi],
                                       [1., 1., 2*np.pi]),
                               loss="soft_l1", f_scale=2., max_nfev=180)
        error = residual(result.x)
        solutions.append({"contact_face_axis": axis,
                          "table_xy_and_yaw": result.x.tolist(),
                          "pose": peg_pose_on_table(result.x, board, axis).tolist(),
                          "RGB_support_rmse_px": float(np.sqrt(np.mean(error**2))),
                          "cost": float(result.cost),
                          "success": bool(result.success)})
    solutions.sort(key=lambda row: row["cost"])
    return solutions


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frame, depth = source["obs/side_1"][0], source["obs/side_1_depth"][0]
    hull = observed_hull(frame, True)
    obs = support(hull)
    cross = json.loads((OUT / "cross_orientation_silhouette_k_probe.json").read_text())
    cameras = {"nominal_assumed_K": K_NOMINAL,
               "cross_orientation_RGB_K": np.asarray(cross["best_fit"]["K"])}
    result = {"source": SOURCE.name, "frame": 0,
              "board_bottom_z_m": BOARD_BOTTOM_Z_M,
              "K_estimation_uses_sensor_depth": False,
              "candidates": {}}
    for name, k in cameras.items():
        board, board_score = board_pose(frame, k)
        solutions = solve_contact(obs, hull, board, k, "y") + solve_contact(
            obs, hull, board, k, "x")
        solutions.sort(key=lambda row: row["cost"])
        best = solutions[0]
        result["candidates"][name] = {
            "K": k.tolist(), "board_pose": board.tolist(),
            "board_RGB_fit": board_score, "contact_solutions_best_three": solutions[:3],
            "best_solution_sensor_depth_independent_check": depth_check(
                frame, depth, k, np.asarray(best["pose"]))}
    result["limitations"] = [
        "Contact with the table and which peg side faces down are inferred from RGB; they are not measured metadata.",
        "The true table plane is approximated by the board CAD bottom plane.",
        "This diagnostic fixes each K; it does not estimate a new K."]
    (OUT / "horizontal_table_contact_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: {"best_RGB_rmse_px": value["contact_solutions_best_three"][0]["RGB_support_rmse_px"],
                            "best_contact_axis": value["contact_solutions_best_three"][0]["contact_face_axis"],
                            "sensor_depth": value["best_solution_sensor_depth_independent_check"]}
                      for key, value in result["candidates"].items()}, indent=2))


if __name__ == "__main__":
    main()
