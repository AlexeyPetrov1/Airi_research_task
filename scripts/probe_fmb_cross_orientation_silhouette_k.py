"""Diagnostic RGB-only K fit adding a horizontal peg view to vertical views.

This tests whether the newly available distinct orientation fixes the prior
rounded-CAD silhouette ambiguity. The horizontal peg is unobscured at t=0;
nearby t=5/10 are used only as a correlated holdout. Future GT and sensor
depth are excluded from fit and model selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import H, K as K_NOMINAL, SOURCE as VERTICAL_SOURCE
from probe_fmb_cad_silhouette_k import (DIRS, MODEL, TRAIN as VERTICAL_TRAIN,
                                        HOLDOUT as VERTICAL_HOLDOUT,
                                        project_model, support)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
HORIZONTAL_SOURCE = ROOT / "data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy"
HORIZONTAL_TRAIN = (0,)
HORIZONTAL_HOLDOUT = (5, 10)


def observed_hull(frame: np.ndarray, horizontal: bool) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = 25 if horizontal else 13
    saturation = 60 if horizontal else 75
    mask = (cv2.inRange(hsv, (0, saturation, 90), (hue, 255, 255)) |
            cv2.inRange(hsv, (170, saturation, 90), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(float)


def k_from_x(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]], [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def fit_horizontal_pose(obs: np.ndarray, hull: np.ndarray, k: np.ndarray) -> list[dict]:
    """RGB-only multistart pose seeds, retaining all near-best branches."""
    center = np.mean([hull.min(axis=0), hull.max(axis=0)], axis=0)
    rng = np.random.default_rng(5201)
    rotations = list(Rotation.random(36, random_state=rng))
    solutions = []
    for z0 in (.3, .5, .7):
        midpoint_camera = np.array([(center[0]-k[0, 2])*z0/k[0, 0],
                                    (center[1]-k[1, 2])*z0/k[1, 1], z0])
        for rotation in rotations:
            rv = rotation.as_rotvec()
            tv = midpoint_camera - rotation.apply([0., 0., H/2])
            seed = np.r_[rv, tv]
            res = least_squares(
                lambda p: support(project_model(p[:3], p[3:], k))-obs,
                seed, bounds=([-np.inf]*3 + [-2., -2., .05],
                              [np.inf]*3 + [2., 2., 2.]),
                loss="soft_l1", f_scale=2., max_nfev=100)
            error = support(project_model(res.x[:3], res.x[3:], k))-obs
            solutions.append({"pose": res.x.tolist(),
                              "rmse_px": float(np.sqrt(np.mean(error**2))),
                              "cost": float(res.cost), "success": bool(res.success)})
    solutions.sort(key=lambda x: x["cost"])
    return solutions


def fit_joint(obs: dict[str, np.ndarray], vpose: dict[int, np.ndarray],
              hpose: np.ndarray, kseed: np.ndarray) -> dict:
    keys = [f"v{s}" for s in VERTICAL_TRAIN] + ["h0"]
    poses = [vpose[s] for s in VERTICAL_TRAIN] + [hpose]
    x0 = np.r_[np.log(kseed[0, 0]), np.log(kseed[1, 1]),
               kseed[0, 2], kseed[1, 2], np.asarray(poses).ravel()]

    def residual(x: np.ndarray) -> np.ndarray:
        k = k_from_x(x)
        return np.concatenate([
            support(project_model(x[4+6*i:7+6*i], x[7+6*i:10+6*i], k))-obs[key]
            for i, key in enumerate(keys)])

    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  np.tile([-np.inf]*3+[-2., -2., .05], len(keys))]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  np.tile([np.inf]*3+[2., 2., 2.], len(keys))]
    res = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                        f_scale=2., x_scale="jac", max_nfev=180)
    err = residual(res.x).reshape(len(keys), len(DIRS))
    return {"K": k_from_x(res.x).tolist(), "poses": {
        key: res.x[4+6*i:10+6*i].tolist() for i, key in enumerate(keys)},
        "train_per_frame_rmse_px": {
            key: float(np.sqrt(np.mean(err[i]**2))) for i, key in enumerate(keys)},
        "train_rmse_px": float(np.sqrt(np.mean(err**2))),
        "cost": float(res.cost), "success": bool(res.success),
        "message": str(res.message)}


def heldout_error(hulls: dict[str, np.ndarray], k: np.ndarray,
                  pose_seeds: dict[str, np.ndarray]) -> dict:
    rows = []
    for key, hull in hulls.items():
        target = support(hull)
        pose0 = pose_seeds[key]
        even = np.arange(0, len(DIRS), 2)
        odd = np.arange(1, len(DIRS), 2)
        res = least_squares(lambda p: (support(project_model(p[:3], p[3:], k))-
                                       target)[even], pose0,
                            loss="soft_l1", f_scale=2., max_nfev=120)
        error = support(project_model(res.x[:3], res.x[3:], k))-target
        rows.append({"key": key,
                     "heldout_odd_direction_rmse_px": float(
                         np.sqrt(np.mean(error[odd]**2))),
                     "pose_fit_even_direction_rmse_px": float(
                         np.sqrt(np.mean(error[even]**2)))})
    return {"per_frame": rows,
            "mean_odd_direction_rmse_px": float(np.mean([
                r["heldout_odd_direction_rmse_px"] for r in rows]))}


def main() -> None:
    vertical = np.load(VERTICAL_SOURCE, allow_pickle=True).item()
    horizontal = np.load(HORIZONTAL_SOURCE, allow_pickle=True).item()
    vf, hf = vertical["obs/side_1"], horizontal["obs/side_1"]
    train_hulls = {f"v{s}": observed_hull(vf[s], False) for s in VERTICAL_TRAIN}
    train_hulls["h0"] = observed_hull(hf[0], True)
    test_hulls = {f"v{s}": observed_hull(vf[s], False) for s in VERTICAL_HOLDOUT}
    test_hulls.update({f"h{s}": observed_hull(hf[s], True)
                       for s in HORIZONTAL_HOLDOUT})
    targets = {key: support(hull) for key, hull in train_hulls.items()}
    previous = json.loads((ROOT / "molmo-motion/runs/sharerobot_fmb_episode_5201/"
                           "effective_k_rounded_cad_silhouette.json").read_text())
    prev_fit = previous["best_fit"]
    prev_k = np.asarray(prev_fit["K"])
    vpose = {s: np.asarray(p) for s, p in zip(VERTICAL_TRAIN, prev_fit["poses"])}
    hseeds = fit_horizontal_pose(targets["h0"], train_hulls["h0"], K_NOMINAL)
    fits = []
    for kseed in (K_NOMINAL, prev_k):
        for branch in hseeds[:3]:
            fits.append(fit_joint(targets, vpose, np.asarray(branch["pose"]), kseed))
    fits.sort(key=lambda x: x["cost"])
    best = fits[0]
    kbest = np.asarray(best["K"])
    hseed = np.asarray(hseeds[0]["pose"])
    test_seeds = {f"v{s}": vpose[min(VERTICAL_TRAIN, key=lambda t:abs(t-s))]
                  for s in VERTICAL_HOLDOUT}
    test_seeds.update({f"h{s}": hseed for s in HORIZONTAL_HOLDOUT})
    holdout = heldout_error(test_hulls, kbest, test_seeds)
    holdout_nominal = heldout_error(test_hulls, K_NOMINAL, test_seeds)
    rng = np.random.default_rng(5201)
    monte_carlo = []
    for _ in range(30):
        noisy = {key: support(hull+rng.normal(0, 2., hull.shape))
                 for key, hull in train_hulls.items()}
        trial = fit_joint(noisy, vpose, np.asarray(best["poses"]["h0"]), kbest)
        if trial["success"]:
            k = np.asarray(trial["K"])
            monte_carlo.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    result = {"status": "DIAGNOSTIC_RGB_ONLY_CROSS_ORIENTATION_K",
              "train": list(train_hulls), "holdout": list(test_hulls),
              "new_episode": HORIZONTAL_SOURCE.name,
              "horizontal_pose_seed_best_three": hseeds[:3],
              "joint_seed_fits": fits, "best_fit": best,
              "heldout": holdout, "heldout_nominal_K": holdout_nominal,
              "monte_carlo_sigma_px": 2., "monte_carlo_attempts": 30,
              "monte_carlo_successes": len(monte_carlo),
              "monte_carlo_K_fx_fy_cx_cy": monte_carlo,
              "monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(monte_carlo, [5, 50, 95], axis=0).tolist()
                  if monte_carlo else None),
              "limitations": [
                  "Horizontal unoccluded frames are nearly the same pose.",
                  "Horizontal t0 silhouette is only 42x28 pixels.",
                  "Vertical late silhouette has gripper occlusion; its hull is an incomplete surface observation.",
                  "RGB holdout frames near train poses cannot validate absolute K.",
                  "No sensor depth or future GT used to estimate or choose K."]}
    (OUT / "cross_orientation_silhouette_k_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"K": best["K"], "train_rmse_px": best["train_rmse_px"],
                      "heldout_rgb_px": holdout["mean_odd_direction_rmse_px"],
                      "heldout_nominal_rgb_px": holdout_nominal["mean_odd_direction_rmse_px"],
                      "monte_carlo_percentiles": result["monte_carlo_K_5_50_95_percentiles"]},
                     indent=2))


if __name__ == "__main__":
    main()
