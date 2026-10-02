"""Diagnostic effective-K fit from RGB peg tips and metric TCP trajectories.

After the second grasp, one distal peg tip is visible during insertion in
three horizontal FMB episodes. The fit shares K and camera/base extrinsics;
each episode gets a constant TCP->tip offset. No depth or future tracking GT
is used. This is an alternative physical anchor, subject to tip identity,
grasp rigidity and RGB/TCP synchronization checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import red_end


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/fmb/single_object_manipulation_dataset"
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
EPISODES = (0, 4, 8)
INTERVALS = {0: (210, 290), 4: (193, 235), 8: (222, 251)}


def k_from_x(x: np.ndarray) -> np.ndarray:
    return np.array([[np.exp(x[0]), 0., x[2]], [0., np.exp(x[1]), x[3]], [0., 0., 1.]])


def project(x: np.ndarray, poses: np.ndarray, episode_indexes: np.ndarray) -> np.ndarray:
    k = k_from_x(x)
    offsets = np.asarray([x[10+3*i:13+3*i] for i in episode_indexes])
    base = poses[:, :3] + Rotation.from_quat(poses[:, 3:]).apply(offsets)
    cam = Rotation.from_rotvec(x[4:7]).apply(base) + x[7:10]
    z = np.maximum(cam[:, 2], 1e-5)
    return np.column_stack((k[0, 0]*cam[:, 0]/z+k[0, 2],
                            k[1, 1]*cam[:, 1]/z+k[1, 2]))


def fit(poses: np.ndarray, indexes: np.ndarray, uv: np.ndarray,
        weights: np.ndarray, seed: np.ndarray, fix_k: bool = False,
        max_nfev: int = 500) -> dict:
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-3.]*3, np.tile([-.3]*3, len(EPISODES))]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [3.]*3, np.tile([.3]*3, len(EPISODES))]
    if fix_k:
        frozen = seed[:4].copy()
        pack = lambda y: np.r_[frozen, y]
        x0, lo, hi = seed[4:], lower[4:], upper[4:]
    else:
        pack = lambda y: y
        x0, lo, hi = seed, lower, upper

    def residual(y: np.ndarray) -> np.ndarray:
        return ((project(pack(y), poses, indexes)-uv)*weights[:, None]).ravel()

    result = least_squares(residual, x0, bounds=(lo, hi), loss="soft_l1",
                           f_scale=2., x_scale="jac", max_nfev=max_nfev)
    x = pack(result.x)
    error = np.linalg.norm(project(x, poses, indexes)-uv, axis=1)
    return {"x": x.tolist(), "K": k_from_x(x).tolist(),
            "success": bool(result.success), "cost": float(result.cost),
            "message": str(result.message), "nfev": int(result.nfev),
            "mean_px": float(np.mean(error)),
            "rmse_px": float(np.sqrt(np.mean(error**2))),
            "p90_px": float(np.percentile(error, 90)),
            "max_px": float(np.max(error)),
            "per_episode_rmse_px": {str(ep): float(np.sqrt(np.mean(
                error[indexes == i]**2))) for i, ep in enumerate(EPISODES)
                if np.any(indexes == i)},
            "jacobian_singular_values": np.linalg.svd(result.jac,
                                                        compute_uv=False).tolist()}


def score(x: np.ndarray, poses: np.ndarray, indexes: np.ndarray,
          uv: np.ndarray) -> dict:
    err = np.linalg.norm(project(x, poses, indexes)-uv, axis=1)
    return {"count": len(err), "mean_px": float(np.mean(err)),
            "median_px": float(np.median(err)),
            "rmse_px": float(np.sqrt(np.mean(err**2))),
            "p90_px": float(np.percentile(err, 90)),
            "max_px": float(np.max(err)),
            "per_episode_rmse_px": {str(ep): float(np.sqrt(np.mean(
                err[indexes == i]**2))) for i, ep in enumerate(EPISODES)
                if np.any(indexes == i)}}


def main() -> None:
    rows = []
    for i, ep in enumerate(EPISODES):
        source = np.load(DATA/f"1_M_L_3_horizontal_n_{ep}.npy",
                         allow_pickle=True).item()
        frames, tcp, primitive = (source["obs/side_1"], source["obs/tcp_pose"],
                                  source["primitive"])
        start, stop = INTERVALS[ep]
        for step in range(start, stop+1):
            if primitive[step] != "insert":
                raise ValueError(f"Episode {ep} frame {step} is not insert")
            try:
                uv, info = red_end(frames[step])
            except ValueError:
                continue
            x, y, w, h = info["bounding_box_xywh"]
            if h < 1.6*w:
                continue
            local = step-start
            train = local < int(.8*(stop-start+1)) and local % 2 == 0
            rows.append({"episode": ep, "episode_index": i, "step": step,
                         "split": "train" if train else "holdout",
                         "holdout_kind": ("temporal_extrapolation" if local >=
                                          int(.8*(stop-start+1)) else
                                          "neighbor_interpolation") if not train else None,
                         "tcp_pose": tcp[step].tolist(),
                         "uv_rgb_px": uv.tolist(), **info})
        del source
    train = [row for row in rows if row["split"] == "train"]
    hold = [row for row in rows if row["split"] == "holdout"]
    train_pose = np.asarray([row["tcp_pose"] for row in train])
    train_uv = np.asarray([row["uv_rgb_px"] for row in train])
    train_idx = np.asarray([row["episode_index"] for row in train])
    hold_pose = np.asarray([row["tcp_pose"] for row in hold])
    hold_uv = np.asarray([row["uv_rgb_px"] for row in hold])
    hold_idx = np.asarray([row["episode_index"] for row in hold])
    per_ep_counts = np.bincount(train_idx, minlength=len(EPISODES))
    weights = np.sqrt(np.mean(per_ep_counts)/per_ep_counts[train_idx])
    ok, rv, tv = cv2.solvePnP(np.ascontiguousarray(train_pose[:, :3]),
                              np.ascontiguousarray(train_uv), K_NOMINAL,
                              None, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        raise RuntimeError("Initial TCP-position PnP failed")
    initial = np.r_[np.log(K_NOMINAL[0, 0]), np.log(K_NOMINAL[1, 1]),
                    K_NOMINAL[0, 2], K_NOMINAL[1, 2], rv.ravel(), tv.ravel(),
                    np.zeros(3*len(EPISODES))]
    seeds = []
    for focal in (.7, 1., 1.4):
        for offset in (0., -.1):
            start = initial.copy()
            start[:2] += np.log(focal)
            start[10+2::3] = offset
            trial = fit(train_pose, train_idx, train_uv, weights, start)
            trial["seed_focal_factor"] = focal
            trial["seed_tcp_offset_z_m"] = offset
            seeds.append(trial)
    seeds.sort(key=lambda row: row["cost"])
    best = seeds[0]
    bx = np.asarray(best["x"])
    nominal = fit(train_pose, train_idx, train_uv, weights, initial,
                  fix_k=True)
    candidate_files = {
        "K_board_RGB_5_poses": ("multi_board_bundle_calibration.json", "best_fit"),
        "K_board_peg_contact_RGB": ("board_peg_table_contact_K_probe.json", "best_fit"),
        "K_multi_horizontal_contact_RGB": ("multi_horizontal_contact_K_probe.json", "best_fit"),
        "K_cross_orientation_RGB": ("cross_orientation_silhouette_k_probe.json", "best_fit"),
    }
    fixed_candidates = {"K_nominal": {
        "K": K_NOMINAL.tolist(), "train": nominal,
        "holdout": score(np.asarray(nominal["x"]), hold_pose, hold_idx, hold_uv)}}
    variants_file = (OUT.parent / "sharerobot_fmb_episode_5201/"
                     "quantitative_2d_sensor_t126/k_sensitivity_variants.json")
    variants = json.loads(variants_file.read_text())["variants"]
    for name in ("K_simple", "K_focal_0925"):
        k = np.asarray(variants[name]["K"])
        fixed_seed = bx.copy()
        fixed_seed[:4] = [np.log(k[0, 0]), np.log(k[1, 1]),
                          k[0, 2], k[1, 2]]
        trial = fit(train_pose, train_idx, train_uv, weights,
                    fixed_seed, fix_k=True)
        fixed_candidates[name] = {"K": k.tolist(), "train": trial,
                                  "holdout": score(np.asarray(trial["x"]),
                                                   hold_pose, hold_idx, hold_uv)}
    for name, (filename, key) in candidate_files.items():
        k = np.asarray(json.loads((OUT/filename).read_text())[key]["K"])
        fixed_seed = bx.copy()
        fixed_seed[:4] = [np.log(k[0, 0]), np.log(k[1, 1]),
                          k[0, 2], k[1, 2]]
        trial = fit(train_pose, train_idx, train_uv, weights,
                    fixed_seed, fix_k=True)
        fixed_candidates[name] = {"K": k.tolist(), "train": trial,
                                  "holdout": score(np.asarray(trial["x"]),
                                                   hold_pose, hold_idx, hold_uv)}
    result = {"status": "DIAGNOSTIC_MULTI_EPISODE_TCP_TIP_K",
              "camera_stream": "side_1 256x256",
              "source_episodes": [f"1_M_L_3_horizontal_n_{ep}.npy" for ep in EPISODES],
              "intervals": INTERVALS,
              "train_counts": {str(ep): int(per_ep_counts[i])
                               for i, ep in enumerate(EPISODES)},
              "holdout_counts": {str(ep): int(np.count_nonzero(hold_idx == i))
                                 for i, ep in enumerate(EPISODES)},
              "records": rows,
              "start_fits": seeds,
              "best_fit": best,
              "best_holdout": score(bx, hold_pose, hold_idx, hold_uv),
              "fixed_nominal_fit": nominal,
              "fixed_nominal_holdout": score(np.asarray(nominal["x"]),
                                             hold_pose, hold_idx, hold_uv),
              "fixed_RGB_only_K_candidate_TCP_checks": fixed_candidates,
              "limitations": [
                  "The bottom of the segmented peg is an image extremum, not a verified persistent CAD material point.",
                  "A rigid TCP-to-tip offset and exact RGB/TCP synchronization are assumed during insertion.",
                  "No sensor depth or future evaluation GT enters the fit or candidate selection."]}
    (OUT / "multi_episode_tcp_tip_K_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"K": best["K"], "train": best["per_episode_rmse_px"],
                      "heldout": result["best_holdout"],
                      "nominal_heldout": result["fixed_nominal_holdout"],
                      "seed_costs": [row["cost"] for row in seeds]}, indent=2))


if __name__ == "__main__":
    main()
