"""Diagnostic RGB/TCP camera calibration from the visible end of a grasped peg.

Unlike the CAD-only silhouette fit, this fit assumes the peg end is rigidly
attached to the robot TCP over the selected interval. Neither sensor depth nor
future tracking GT is used in the objective. It is a diagnostic route, not a
claim that effective K has been identified.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import K as K_NOMINAL


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy"
OUT = ROOT / "runs/fmb_effective_k_256_calibration"
TRAIN = tuple(range(64, 89, 2)) + tuple(range(100, 121, 2))
HOLDOUT = tuple(range(65, 89, 2)) + tuple(range(101, 121, 2)) + tuple(range(121, 126))


def red_end(frame: np.ndarray) -> tuple[np.ndarray, dict]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
            cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    x, y, w, h = cv2.boundingRect(contour)
    if area < 150 or y + h >= 230:
        raise ValueError(f"No clean peg end: area={area}, box={(x, y, w, h)}")
    component = np.zeros_like(mask)
    cv2.drawContours(component, [contour], -1, 255, thickness=cv2.FILLED)
    bottom = y + h - 1
    xs = np.flatnonzero(np.any(component[max(y, bottom - 2):bottom + 1] > 0, axis=0))
    if len(xs) < 3:
        raise ValueError(f"Peg end too narrow: {len(xs)} pixels")
    uv = np.array([(float(xs.min()) + float(xs.max())) / 2, float(bottom)], dtype=float)
    return uv, {"bounding_box_xywh": [x, y, w, h], "area_px2": area,
                "bottom_band_width_px": int(xs.max() - xs.min() + 1)}


def k_from_x(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]],
                     [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def project(x: np.ndarray, poses: np.ndarray) -> np.ndarray:
    k = k_from_x(x)
    offset_base = Rotation.from_quat(poses[:, 3:]).apply(x[10:13])
    world = poses[:, :3] + offset_base
    cam = Rotation.from_rotvec(x[4:7]).apply(world) + x[7:10]
    if np.any(cam[:, 2] <= 0):
        # Preserve a finite residual through invalid optimizer steps.
        cam[:, 2] = np.maximum(cam[:, 2], 1e-4)
    uv = np.stack([k[0, 0] * cam[:, 0] / cam[:, 2] + k[0, 2],
                   k[1, 1] * cam[:, 1] / cam[:, 2] + k[1, 2]], axis=1)
    return uv


def fit(poses: np.ndarray, uv: np.ndarray, seed: np.ndarray,
        free_k: bool = True) -> dict:
    x0 = seed.copy()
    lower = np.array([np.log(40), np.log(40), 0., 0.,
                      -np.inf, -np.inf, -np.inf, -3., -3., -3., -.3, -.3, -.3])
    upper = np.array([np.log(800), np.log(800), 255., 255.,
                      np.inf, np.inf, np.inf, 3., 3., 3., .3, .3, .3])
    if not free_k:
        x0 = x0[4:]
        lower = lower[4:]
        upper = upper[4:]
        base = seed[:4].copy()
        residual = lambda y: (project(np.r_[base, y], poses) - uv).ravel()
    else:
        residual = lambda y: (project(y, poses) - uv).ravel()
    result = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=2., x_scale="jac", max_nfev=600)
    x = result.x if free_k else np.r_[base, result.x]
    error = project(x, poses) - uv
    norms = np.linalg.norm(error, axis=1)
    return {"x": x.tolist(), "K": k_from_x(x).tolist(),
            "success": bool(result.success), "message": result.message,
            "cost": float(result.cost), "nfev": int(result.nfev),
            "mean_px": float(np.mean(norms)), "rmse_px": float(np.sqrt(np.mean(norms**2))),
            "p90_px": float(np.percentile(norms, 90)), "max_px": float(np.max(norms)),
            "singular_values_jacobian": np.linalg.svd(result.jac, compute_uv=False).tolist()}


def score(x: np.ndarray, poses: np.ndarray, uv: np.ndarray) -> dict:
    errors = np.linalg.norm(project(x, poses) - uv, axis=1)
    return {"mean_px": float(np.mean(errors)),
            "median_px": float(np.median(errors)),
            "rmse_px": float(np.sqrt(np.mean(errors**2))),
            "p90_px": float(np.percentile(errors, 90)),
            "max_px": float(np.max(errors)), "per_frame_px": errors.tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE, allow_pickle=True).item()
    ids = TRAIN + HOLDOUT
    observation = {i: red_end(source["obs/side_1"][i]) for i in ids}
    image = np.full((4*256, 6*256, 3), 255, np.uint8)
    for j, i in enumerate(ids[:24]):
        tile = source["obs/side_1"][i].copy()
        u, v = observation[i][0]
        cv2.circle(tile, (round(u), round(v)), 3, (0, 255, 0), -1)
        cv2.putText(tile, str(i), (4, 20), cv2.FONT_HERSHEY_SIMPLEX, .6,
                    (0, 0, 0), 2)
        image[j//6*256:(j//6+1)*256, j%6*256:(j%6+1)*256] = tile
    cv2.imwrite(str(OUT / "tcp_tip_landmarks.png"), image)
    tcp = source["obs/tcp_pose"]
    train_pose = tcp[list(TRAIN)]
    hold_pose = tcp[list(HOLDOUT)]
    train_uv = np.array([observation[i][0] for i in TRAIN])
    hold_uv = np.array([observation[i][0] for i in HOLDOUT])
    success, rv, tv = cv2.solvePnP(np.ascontiguousarray(train_pose[:, :3]),
                                  np.ascontiguousarray(train_uv),
                                  K_NOMINAL, None, flags=cv2.SOLVEPNP_EPNP)
    if not success:
        raise RuntimeError("Initial PnP failed")
    seed = np.r_[np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                 K_NOMINAL[0, 2], K_NOMINAL[1, 2], rv.ravel(), tv.ravel(),
                 [0., 0., 0.]]
    starts = []
    for mult in (.7, 1., 1.4):
        for offset in (np.zeros(3), np.array([0., 0., -.10])):
            trial = seed.copy()
            trial[:2] += np.log(mult)
            trial[10:13] = offset
            starts.append(fit(train_pose, train_uv, trial))
    best = min(starts, key=lambda item: item["cost"])
    bx = np.array(best["x"])
    nominal = fit(train_pose, train_uv, seed, free_k=False)
    rng = np.random.default_rng(5201)
    perturb = []
    for _ in range(40):
        noisy = train_uv + rng.normal(0, 2., train_uv.shape)
        trial = fit(train_pose, noisy, bx)
        if trial["success"]:
            kval = np.array(trial["K"])
            perturb.append([kval[0, 0], kval[1, 1], kval[0, 2], kval[1, 2]])
    subsets = []
    for residue in range(4):
        chosen = [j for j in range(len(TRAIN)) if j % 4 != residue]
        trial = fit(train_pose[chosen], train_uv[chosen], bx)
        subsets.append({"excluded_modulo_4": residue, "K": trial["K"],
                        "rmse_px": trial["rmse_px"], "success": trial["success"]})
    observations = {str(i): {"uv_rgb_px": observation[i][0].tolist(),
                             **observation[i][1]} for i in ids}
    result = {"status": "DIAGNOSTIC_TCP_TIP_K_NOT_YET_VALIDATED",
              "assumption": "The same visible peg tip is rigidly attached to TCP in selected frames.",
              "train_frames": TRAIN, "holdout_frames": HOLDOUT,
              "landmark_extraction": "Center of bottom three red-mask rows; 256x256 RGB.",
              "observations": observations,
              "train_tcp_position_axis_range_m": np.ptp(train_pose[:, :3], axis=0).tolist(),
              "train_tcp_position_singular_values_m":
                  np.linalg.svd(train_pose[:, :3]-train_pose[:, :3].mean(0),
                                compute_uv=False).tolist(),
              "start_fits": starts, "best_fit": best,
              "best_holdout": score(bx, hold_pose, hold_uv),
              "nominal_fixed_K_fit": nominal,
              "nominal_fixed_K_holdout": score(np.array(nominal["x"]), hold_pose, hold_uv),
              "subset_fits": subsets,
              "annotation_perturbation_sigma_px": 2.,
              "annotation_perturbation_attempts": 40,
              "annotation_perturbation_K": perturb,
              "annotation_perturbation_K_5_50_95_percentiles":
                  np.percentile(perturb, (5, 50, 95), axis=0).tolist() if perturb else None,
              "limitations": ["TCP synchronization and rigid grasp are assumed.",
                              "The mask bottom can shift with occlusion or thresholding.",
                              "Camera K must be checked against independent sensor depth and CAD geometry."]}
    (OUT / "tcp_tip_k_probe.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf8")
    print(json.dumps({"K": best["K"], "train": best["rmse_px"],
                      "holdout": result["best_holdout"],
                      "nominal_holdout": result["nominal_fixed_K_holdout"],
                      "K_percentiles": result["annotation_perturbation_K_5_50_95_percentiles"],
                      "subset_K": [x["K"] for x in subsets]}, indent=2))


if __name__ == "__main__":
    main()
