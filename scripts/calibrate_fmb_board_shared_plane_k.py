"""RGB-only K assuming the repositioned boards share one physical table plane."""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrate_fmb_board_k import (HOLDOUT, HOLDOUT_FEATURES, TRAIN,
                                   cad_features, depth_crosscheck, extract,
                                   project, score, unpack)
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT


def camera_pose(x: np.ndarray, i: int) -> np.ndarray:
    if i == 0:
        return x[:10]
    delta = x[10+3*(i-1):13+3*(i-1)]
    base_rotation = Rotation.from_rotvec(x[4:7])
    rotation = base_rotation * Rotation.from_rotvec([0., 0., delta[2]])
    translation = base_rotation.apply([delta[0], delta[1], 0.]) + x[7:10]
    return np.r_[x[:4], rotation.as_rotvec(), translation]


def fit(points: np.ndarray, observations: tuple[np.ndarray, ...],
        features: np.ndarray, seed: np.ndarray,
        fix_k: bool = False, max_nfev: int = 250) -> dict:
    count = len(observations)
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-2.]*2, .02,
                  np.tile([-.5, -.5, -np.pi], count-1)]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [2.]*2, 2.,
                  np.tile([.5, .5, np.pi], count-1)]
    if fix_k:
        k0 = seed[:4].copy()
        pack = lambda y: np.r_[k0, y]
        initial, lo, hi = seed[4:], lower[4:], upper[4:]
    else:
        pack = lambda y: y
        initial, lo, hi = seed, lower, upper
    def residual(y):
        x = pack(y)
        return np.concatenate([(project(points, camera_pose(x, i))[features][None]-
                                obs[:, features]).ravel()
                               for i, obs in enumerate(observations)])
    res = least_squares(residual, initial, bounds=(lo, hi), loss="soft_l1",
                        f_scale=2., x_scale="jac", max_nfev=max_nfev)
    x = pack(res.x)
    return {"x": x.tolist(), "K": unpack(x).tolist(),
            "success": bool(res.success), "message": res.message,
            "cost": float(res.cost), "nfev": int(res.nfev),
            "train": [score(points, obs, camera_pose(x, i), features)
                      for i, obs in enumerate(observations)],
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def multi_score(points, obs, x, features):
    return [score(points, a, camera_pose(x, i), features)
            for i, a in enumerate(obs)]


def main() -> None:
    points, names, _ = cad_features(1)
    features = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    held_features = np.array(HOLDOUT_FEATURES)
    train, hold, frames, depths = [], [], [], []
    for number in EPISODES:
        source = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        rgb = source["obs/side_1"]
        observations = {s: extract(rgb[s], 1)[0] for s in TRAIN+HOLDOUT}
        train.append(np.array([observations[s] for s in TRAIN]))
        hold.append(np.array([observations[s] for s in HOLDOUT]))
        frames.append(rgb[list(HOLDOUT)].copy())
        depths.append(source["obs/side_1_depth"][list(HOLDOUT)].copy())
        del source
    train, hold = tuple(train), tuple(hold)
    previous = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    old = np.array(previous["best_fit"]["x"])
    base_r = Rotation.from_rotvec(old[4:7])
    deltas = []
    for i in range(1, len(EPISODES)):
        ri = Rotation.from_rotvec(old[4+6*i:7+6*i])
        relative = base_r.inv()*ri
        t = base_r.inv().apply(old[7+6*i:10+6*i]-old[7:10])
        deltas.extend([t[0], t[1], relative.as_rotvec()[2]])
    x0 = np.r_[old[:10], deltas]
    seeds = []
    for factor in (.85, 1., 1.15):
        seed = x0.copy()
        seed[:2] += np.log(factor)
        seeds.append(fit(points, train, features, seed))
    best = min(seeds, key=lambda x: x["cost"])
    bx = np.array(best["x"])
    nominal_seed = bx.copy()
    nominal_seed[:4] = [np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                        K_NOMINAL[0, 2], K_NOMINAL[1, 2]]
    nominal = fit(points, train, features, nominal_seed, fix_k=True)
    nx = np.array(nominal["x"])
    subset = []
    for excluded in (4, 5, 7, 10, 13, 18):
        selected = features[features != excluded]
        row = fit(points, train, selected, bx)
        subset.append({"excluded_feature": names[excluded],
                       "K": row["K"], "success": row["success"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        noisy = tuple(a+rng.normal(0, 2., a.shape) for a in train)
        row = fit(points, noisy, features, bx, max_nfev=150)
        if row["success"]:
            k = np.array(row["K"])
            mc.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    depth_new = [depth_crosscheck(camera_pose(bx, i), frames[i], depths[i],
                                   tuple(range(len(HOLDOUT))))
                 for i in range(len(EPISODES))]
    depth_nom = [depth_crosscheck(camera_pose(nx, i), frames[i], depths[i],
                                   tuple(range(len(HOLDOUT))))
                 for i in range(len(EPISODES))]
    result = {"status": "SHARED_TABLE_PLANE_BOARD_K_CANDIDATE_PENDING_ACCEPTANCE",
              "hypothesis": "All five repositioned medium boards share exactly one normal and table height; each has free in-plane XY translation and rotation.",
              "source_episodes": EPISODES,
              "train_frames_each_episode": TRAIN,
              "holdout_frames_each_episode": HOLDOUT,
              "feature_ids_train": [names[i] for i in features],
              "feature_ids_holdout": [names[i] for i in held_features],
              "seed_fits": seeds, "best_fit": best,
              "heldout_frames_train_features": multi_score(points, hold, bx, features),
              "heldout_frames_heldout_features": multi_score(points, hold, bx, held_features),
              "nominal_fixed_K_fit": nominal,
              "nominal_heldout": multi_score(points, hold, nx, np.arange(len(names))),
              "depth_crosscheck_new": depth_new,
              "depth_crosscheck_nominal": depth_nom,
              "subset_fits": subset,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["Shared table-plane hypothesis is geometric, not verified by saved extrinsics.",
                              "RGB corner intersections and hole centers have systematic correspondence uncertainty.",
                              "Depth and future GT are validation only."]}
    (OUT / "shared_plane_board_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"K": best["K"],
                      "train_RMSE": [v["rmse_px"] for v in best["train"]],
                      "heldout_RMSE": [v["rmse_px"] for v in result["heldout_frames_train_features"]],
                      "depth_new_mm": [v["median_abs_Z_error_mm"] for v in depth_new],
                      "depth_nominal_mm": [v["median_abs_Z_error_mm"] for v in depth_nom],
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "subset_K": [v["K"] for v in subset]}, indent=2))


if __name__ == "__main__":
    main()
