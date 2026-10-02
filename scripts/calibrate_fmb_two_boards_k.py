"""Shared side_1 K from two positions of the same official FMB medium board.

The camera view and static background match between raw FMB trajectories
1_M_L_3_vertical_n_2 and _n_3, while the board is moved/rotated. Only RGB
and official board CAD enter the fit. Depth is reserved for validation.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate_fmb_board_k import (HOLDOUT, HOLDOUT_FEATURES, TRAIN,
                                   cad_features, depth_crosscheck, extract,
                                   project, score, unpack)
from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import OUT, SOURCE


SOURCES = (SOURCE, SOURCE.parents[3] / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy")


def fit_multi(points: np.ndarray, observations: tuple[np.ndarray, ...],
              features: np.ndarray, seed: np.ndarray,
              fix_k: bool = False, max_nfev: int = 200) -> dict:
    count = len(observations)
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  np.tile([-np.inf]*3+[-2.]*2+[.02], count)]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  np.tile([np.inf]*3+[2.]*2+[2.], count)]
    if fix_k:
        k0 = seed[:4].copy()
        pack = lambda p: np.r_[k0, p]
        initial, lo, hi = seed[4:], lower[4:], upper[4:]
    else:
        pack = lambda p: p
        initial, lo, hi = seed, lower, upper

    def residual(parameters: np.ndarray) -> np.ndarray:
        x = pack(parameters)
        return np.concatenate([(project(points, np.r_[x[:4], x[4+6*i:10+6*i]])[features][None]-
                                observed[:, features]).ravel()
                               for i, observed in enumerate(observations)])

    res = least_squares(residual, initial, bounds=(lo, hi), loss="soft_l1",
                        f_scale=2., x_scale="jac", max_nfev=max_nfev)
    x = pack(res.x)
    return {"x": x.tolist(), "K": unpack(x[:10]).tolist(),
            "success": bool(res.success), "message": res.message,
            "cost": float(res.cost), "nfev": int(res.nfev),
            "train": [score(points, obs, np.r_[x[:4], x[4+6*i:10+6*i]], features)
                      for i, obs in enumerate(observations)],
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def score_multi(points: np.ndarray, observations: tuple[np.ndarray, ...],
                x: np.ndarray, features: np.ndarray) -> list[dict]:
    return [score(points, obs, np.r_[x[:4], x[4+6*i:10+6*i]], features)
            for i, obs in enumerate(observations)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [np.load(path, allow_pickle=True).item() for path in SOURCES]
    points, names, _ = cad_features(1)
    train_features = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    test_features = np.array(HOLDOUT_FEATURES)
    observations = []
    details = {}
    for episode, src in zip(("n_2", "n_3"), sources):
        records = {i: extract(src["obs/side_1"][i], 1) for i in TRAIN + HOLDOUT}
        train = np.array([records[i][0] for i in TRAIN])
        hold = np.array([records[i][0] for i in HOLDOUT])
        observations.append((train, hold))
        details[episode] = {"train_frames": TRAIN, "holdout_frames": HOLDOUT,
                            "uv": {str(i): records[i][0].tolist() for i in records},
                            "top_quad_frame0": records[0][1]["top_quad_px"],
                            "bottom_front_frame0": records[0][1]["bottom_front_px"]}
    train_obs = tuple(x[0] for x in observations)
    held_obs = tuple(x[1] for x in observations)
    (OUT / "two_board_rgb_observations.json").write_text(json.dumps({
        "source_episodes": [str(x) for x in SOURCES],
        "cad_step": str(OUT / "peg_board_official.step"),
        "board_solid_index": 1,
        "landmark_names": names,
        "cad_feature_xyz_m": points.tolist(),
        "train_feature_ids": train_features.tolist(),
        "holdout_feature_ids": test_features.tolist(),
        "episodes": details}, indent=2) + "\n", encoding="utf8")
    starts = []
    for factor in (.7, 1., 1.4):
        k = K_NOMINAL.copy()
        k[0, 0] *= factor
        k[1, 1] *= factor
        poses = []
        for obs in train_obs:
            ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(points[train_features]),
                                      np.ascontiguousarray(obs.mean(axis=0)[train_features]),
                                      k, None, flags=cv2.SOLVEPNP_EPNP)
            if not ok:
                raise RuntimeError("Board PnP failed")
            poses.extend([*rv.ravel(), *tv.ravel()])
        x0 = np.r_[np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2], poses]
        starts.append(fit_multi(points, train_obs, train_features, x0))
    best = min(starts, key=lambda item: item["cost"])
    bx = np.array(best["x"])
    fixed_seed = bx.copy()
    fixed_seed[:4] = [np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                      K_NOMINAL[0, 2], K_NOMINAL[1, 2]]
    nominal = fit_multi(points, train_obs, train_features, fixed_seed, fix_k=True)
    subsets = []
    for excluded in (4, 5, 7, 10, 13, 18):
        chosen = train_features[train_features != excluded]
        fit = fit_multi(points, train_obs, chosen, bx)
        subsets.append({"excluded_feature": names[excluded], "K": fit["K"],
                        "success": fit["success"],
                        "heldout_feature_RMSE_px":
                            [r["rmse_px"] for r in score_multi(points, held_obs,
                                np.array(fit["x"]), test_features)]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(100):
        noisy = tuple(obs+rng.normal(0, 2., obs.shape) for obs in train_obs)
        fit = fit_multi(points, noisy, train_features, bx, max_nfev=120)
        if fit["success"]:
            k = np.array(fit["K"])
            mc.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    depth_new = []
    depth_nom = []
    nx = np.array(nominal["x"])
    for i, src in enumerate(sources):
        xnew = np.r_[bx[:4], bx[4+6*i:10+6*i]]
        xnom = np.r_[nx[:4], nx[4+6*i:10+6*i]]
        depth_new.append(depth_crosscheck(xnew, src["obs/side_1"],
                                          src["obs/side_1_depth"], HOLDOUT))
        depth_nom.append(depth_crosscheck(xnom, src["obs/side_1"],
                                          src["obs/side_1_depth"], HOLDOUT))
    result = {"status": "TWO_POSE_BOARD_RGB_K_CANDIDATE_PENDING_ACCEPTANCE",
              "source_episodes": [str(x) for x in SOURCES],
              "official_board_cad": str(OUT / "peg_board_official.step"),
              "solid_index": 1,
              "fixed_camera_hypothesis": "Static background edges align in both published side_1 views; board changed pose.",
              "same_K_both_episodes": True,
              "train_frames_each_episode": TRAIN,
              "holdout_frames_each_episode": HOLDOUT,
              "train_feature_ids": [names[i] for i in train_features],
              "holdout_feature_ids": [names[i] for i in test_features],
              "seed_fits": starts, "best_fit": best,
              "heldout_frames_train_features": score_multi(points, held_obs, bx, train_features),
              "heldout_frames_heldout_features": score_multi(points, held_obs, bx, test_features),
              "fixed_nominal_fit": nominal,
              "fixed_nominal_heldout": score_multi(points, held_obs, nx,
                                                     np.arange(len(names))),
              "depth_crosscheck_new": depth_new,
              "depth_crosscheck_fixed_nominal": depth_nom,
              "subset_fits": subsets,
              "monte_carlo_annotation_sigma_px": 2.,
              "monte_carlo_attempts": 100, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["Only two board poses, with modest rotation difference.",
                              "Outer quadrilateral corners are virtual intersections of rounded edges.",
                              "Hole centers are shape-derived features, not physical CAD surface points.",
                              "Frame holdout repeats the same pose within each episode.",
                              "RGB and depth use the same 256px grid but exact preprocessing is unknown."]}
    (OUT / "two_board_bundle_calibration.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"K": best["K"], "train": best["train"],
                      "heldout_rgb": result["heldout_frames_train_features"],
                      "heldout_features": result["heldout_frames_heldout_features"],
                      "depth_new": depth_new, "depth_nominal": depth_nom,
                      "nominal_rgb": result["fixed_nominal_heldout"],
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"],
                      "subset_K": [s["K"] for s in subsets]}, indent=2))


if __name__ == "__main__":
    main()
