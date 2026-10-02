"""Shared RGB-only K from three different tabletop poses of the same FMB peg.

Each episode has an independent board pose and a peg pose constrained to its
board's bottom/table plane. The contact face is selected by RGB fit at fixed
nominal K: n0=x, n4=y, n8=y. Z16 is reserved for independent checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate_fmb_board_k import (HOLDOUT_FEATURES, cad_features,
                                   depth_crosscheck, extract, fit as fit_board,
                                   project)
from check_fmb_cross_orientation_depth import check as peg_depth_check
from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_cad_silhouette_k import project_model, support
from probe_fmb_cross_orientation_silhouette_k import observed_hull
from probe_fmb_board_peg_contact_bundle import fit_joint as fit_one_episode
from probe_fmb_table_contact_k import peg_pose_on_table, solve_contact


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
DATA = ROOT / "data/fmb/single_object_manipulation_dataset"
EPISODES = (0, 4, 8)
CONTACT_AXIS = {0: "x", 4: "y", 8: "y"}


def k_from_x(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]], [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def seed_for_k(records: dict[int, dict], board_points: np.ndarray,
               train_ids: np.ndarray, k: np.ndarray) -> np.ndarray:
    entries = []
    for ep in EPISODES:
        uv = records[ep]["board_uv"]
        ok, rv, tv = cv2.solvePnP(
            np.ascontiguousarray(board_points[train_ids]),
            np.ascontiguousarray(uv[train_ids]), k, None,
            flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError(f"Board pose seed failed on episode {ep}")
        x = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2],
                  rv.ravel(), tv.ravel()]
        board = np.asarray(fit_board(board_points, uv[None], train_ids,
                                     x, fix_k=True)["x"])[4:10]
        contact = solve_contact(records[ep]["peg_support"],
                                records[ep]["peg_hull"], board, k,
                                CONTACT_AXIS[ep])[0]["table_xy_and_yaw"]
        entries.extend(np.r_[board, contact])
    return np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2], entries]


def fit_bundle(records: dict[int, dict], board_points: np.ndarray,
               train_ids: np.ndarray, episodes: tuple[int, ...],
               x0: np.ndarray, max_nfev: int = 260) -> dict:
    low = np.r_[np.log(40), np.log(40), 0., 0.,
                np.tile([-np.inf]*3+[-2., -2., .02, -1., -1., -2*np.pi],
                        len(episodes))]
    high = np.r_[np.log(800), np.log(800), 255., 255.,
                 np.tile([np.inf]*3+[2., 2., 2., 1., 1., 2*np.pi],
                         len(episodes))]

    def residual(x: np.ndarray) -> np.ndarray:
        k = k_from_x(x)
        parts = []
        for i, ep in enumerate(episodes):
            board = x[4+9*i:10+9*i]
            contact = x[10+9*i:13+9*i]
            peg = peg_pose_on_table(contact, board, CONTACT_AXIS[ep])
            bx = np.r_[x[:4], board]
            parts.extend(((project(board_points, bx)[train_ids]-
                           records[ep]["board_uv"][train_ids]).ravel(),
                          support(project_model(peg[:3], peg[3:], k))-
                          records[ep]["peg_support"]))
        return np.concatenate(parts)

    result = least_squares(residual, x0, bounds=(low, high), loss="soft_l1",
                           f_scale=2., x_scale="jac", max_nfev=max_nfev)
    x = result.x
    k = k_from_x(x)
    scores, poses = {}, {}
    for i, ep in enumerate(episodes):
        board = x[4+9*i:10+9*i]
        contact = x[10+9*i:13+9*i]
        peg = peg_pose_on_table(contact, board, CONTACT_AXIS[ep])
        err_board = np.linalg.norm(project(board_points, np.r_[x[:4], board])-
                                   records[ep]["board_uv"], axis=1)
        err_peg = (support(project_model(peg[:3], peg[3:], k))-
                   records[ep]["peg_support"])
        scores[str(ep)] = {
            "board_train_rmse_px": float(np.sqrt(np.mean(err_board[train_ids]**2))),
            "board_holdout_rmse_px": float(np.sqrt(np.mean(
                err_board[np.asarray(HOLDOUT_FEATURES)]**2))),
            "peg_support_rmse_px": float(np.sqrt(np.mean(err_peg**2)))}
        poses[str(ep)] = {"board_pose": board.tolist(), "peg_pose": peg.tolist(),
                          "peg_contact": contact.tolist(),
                          "contact_axis": CONTACT_AXIS[ep]}
    return {"K": k.tolist(), "x": x.tolist(), "cost": float(result.cost),
            "success": bool(result.success), "message": str(result.message),
            "nfev": int(result.nfev), "scores": scores, "poses": poses,
            "jacobian_singular_values": np.linalg.svd(result.jac,
                                                       compute_uv=False).tolist()}


def main() -> None:
    board_points, board_names, _ = cad_features(1)
    train_ids = np.array([i for i in range(len(board_names))
                          if i not in HOLDOUT_FEATURES])
    records, sources = {}, {}
    for ep in EPISODES:
        path = DATA / f"1_M_L_3_horizontal_n_{ep}.npy"
        src = np.load(path, allow_pickle=True).item()
        frame = src["obs/side_1"][0]
        board_uv, _ = extract(frame, 1)
        peg_hull = observed_hull(frame, True)
        records[ep] = {"board_uv": board_uv,
                       "peg_hull": peg_hull,
                       "peg_support": support(peg_hull)}
        sources[ep] = {"frame": frame.copy(),
                       "depth": src["obs/side_1_depth"][0].copy()}
        del src
    previous = json.loads((OUT / "board_peg_table_contact_K_probe.json").read_text())
    single_k = np.asarray(previous["best_fit"]["K"])
    board_k = np.asarray(json.loads((OUT / "multi_board_bundle_calibration.json").read_text())
                         ["best_fit"]["K"])
    seeds = []
    for name, k in (("nominal", K_NOMINAL), ("single_contact", single_k),
                    ("board_RGB", board_k)):
        trial = fit_bundle(records, board_points, train_ids, EPISODES,
                           seed_for_k(records, board_points, train_ids, k))
        trial["seed_name"] = name
        seeds.append(trial)
    seeds.sort(key=lambda row: row["cost"])
    best = seeds[0]
    kbest = np.asarray(best["K"])
    independent_checks = {}
    for ep in EPISODES:
        pose = best["poses"][str(ep)]
        xboard = np.r_[np.log(kbest[0, 0]), np.log(kbest[1, 1]),
                       kbest[0, 2], kbest[1, 2], pose["board_pose"]]
        independent_checks[str(ep)] = {
            "board_sensor_depth": depth_crosscheck(
                xboard, sources[ep]["frame"][None],
                sources[ep]["depth"][None], (0,)),
            "peg_sensor_depth": peg_depth_check(
                sources[ep]["frame"], sources[ep]["depth"], kbest,
                np.asarray(pose["peg_pose"]))}
    leave_episode_out = []
    best_x = np.asarray(best["x"])
    for excluded in EPISODES:
        kept = tuple(ep for ep in EPISODES if ep != excluded)
        packed = np.r_[best_x[:4], *[best_x[4+9*EPISODES.index(ep):
                                             13+9*EPISODES.index(ep)]
                                    for ep in kept]]
        trial = fit_bundle(records, board_points, train_ids, kept, packed)
        held_k = np.asarray(trial["K"])
        seed_all = seed_for_k(records, board_points, train_ids, held_k)
        index = EPISODES.index(excluded)
        seed_held = np.r_[seed_all[:4], seed_all[4+9*index:13+9*index]]
        held_score = fit_one_episode(
            board_points, records[excluded]["board_uv"],
            records[excluded]["peg_support"], train_ids, seed_held,
            CONTACT_AXIS[excluded], fix_k=True)
        leave_episode_out.append({"excluded_episode": excluded,
                                  "K": trial["K"], "success": trial["success"],
                                  "cost": trial["cost"],
                                  "heldout_episode_fixed_K_RGB_scores": {
                                      "board_train_rmse_px": held_score["board_train_rmse_px"],
                                      "board_holdout_features_rmse_px": held_score["board_holdout_rmse_px"],
                                      "peg_support_rmse_px": held_score["peg_support_rmse_px"]}})
    rng = np.random.default_rng(5201)
    monte_carlo = []
    for _ in range(50):
        noisy = {}
        for ep in EPISODES:
            board_uv = records[ep]["board_uv"].copy()
            board_uv[train_ids] += rng.normal(0, 2., board_uv[train_ids].shape)
            peg_support = support(records[ep]["peg_hull"]+
                                  rng.normal(0, 2., records[ep]["peg_hull"].shape))
            noisy[ep] = {**records[ep], "board_uv": board_uv,
                         "peg_support": peg_support}
        trial = fit_bundle(noisy, board_points, train_ids, EPISODES,
                           best_x, max_nfev=180)
        if trial["success"]:
            k = np.asarray(trial["K"])
            monte_carlo.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    realistic_monte_carlo = []
    realistic_shared_bias = []
    sigmas = np.array([2.]*6+[.5]*13)
    for use_shared, destination in ((False, realistic_monte_carlo),
                                    (True, realistic_shared_bias)):
        for _ in range(75):
            common_outer = rng.normal(0, 1., (6, 2)) if use_shared else np.zeros((6, 2))
            noisy = {}
            for ep in EPISODES:
                board_uv = records[ep]["board_uv"].copy()
                board_uv[train_ids] += (rng.normal(0, 1., (len(train_ids), 2))*
                                        sigmas[train_ids, None])
                board_uv[:6] += common_outer
                peg_support = support(records[ep]["peg_hull"]+
                                      rng.normal(0, 1., records[ep]["peg_hull"].shape))
                noisy[ep] = {**records[ep], "board_uv": board_uv,
                             "peg_support": peg_support}
            trial = fit_bundle(noisy, board_points, train_ids, EPISODES,
                               best_x, max_nfev=180)
            if trial["success"]:
                k = np.asarray(trial["K"])
                destination.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    result = {"status": "DIAGNOSTIC_RGB_ONLY_MULTI_HORIZONTAL_CONTACT_K",
              "source_episodes": [f"1_M_L_3_horizontal_n_{ep}.npy" for ep in EPISODES],
              "contact_axis_chosen_from_RGB_only": CONTACT_AXIS,
              "board_train_features": [board_names[i] for i in train_ids],
              "board_holdout_features": [board_names[i] for i in HOLDOUT_FEATURES],
              "seed_fits": seeds, "best_fit": best,
              "independent_sensor_depth_checks": independent_checks,
              "leave_one_episode_out": leave_episode_out,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 50,
              "monte_carlo_successes": len(monte_carlo),
              "monte_carlo_K_fx_fy_cx_cy": monte_carlo,
              "monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(monte_carlo, [5, 50, 95], axis=0).tolist()
                  if monte_carlo else None),
              "heteroscedastic_noise_assumptions_px": {
                  "board_outer_six": 2., "board_hole_centers": .5,
                  "peg_hull_vertices": 1.,
                  "common_outer_bias_extra_if_enabled": 1.},
              "heteroscedastic_monte_carlo_K": realistic_monte_carlo,
              "heteroscedastic_monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(realistic_monte_carlo, [5, 50, 95], axis=0).tolist()
                  if realistic_monte_carlo else None),
              "shared_bias_monte_carlo_K": realistic_shared_bias,
              "shared_bias_monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(realistic_shared_bias, [5, 50, 95], axis=0).tolist()
                  if realistic_shared_bias else None),
              "limitations": [
                  "All three peg poses lie on one table plane; yaw differs but out-of-plane tilt does not.",
                  "Table contact and which peg face is down are RGB-inferred assumptions.",
                  "Outer board corners are virtual, and segmentation can share systematic bias.",
                  "Sensor depth and future GT were excluded from RGB calibration."]}
    (OUT / "multi_horizontal_contact_K_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"K": best["K"], "scores": best["scores"],
                      "leave_episode_out_K": leave_episode_out,
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "hetero_MC_percentiles": result["heteroscedastic_monte_carlo_K_5_50_95_percentiles"],
                      "shared_bias_MC_percentiles": result["shared_bias_monte_carlo_K_5_50_95_percentiles"],
                      "sensor_checks": independent_checks}, indent=2))


if __name__ == "__main__":
    main()
