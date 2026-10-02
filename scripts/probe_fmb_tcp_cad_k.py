"""Probe effective K from RGB, a STEP-derived rounded envelope and TCP motion.

This route uses TCP as a calibration constraint because the preceding RGB-only
silhouette bundle was not identifiable. Depth and future 2D GT are held out.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_cad_silhouette_k import MODEL, red_hull
from probe_fmb_tcp_tip_k import OUT, SOURCE, k_from_x


ANGLES = np.arange(0, 2*np.pi, np.pi/6)
DIRS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))
TRAIN = (64, 66, 68, 70, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146)
HOLDOUT = (65, 67, 69, 129, 131, 133, 135, 137, 139, 141, 143, 145)


def projected_support(x: np.ndarray, tcp: np.ndarray) -> np.ndarray:
    k = k_from_x(x)
    rcb = Rotation.from_rotvec(x[4:7])
    rto = Rotation.from_rotvec(x[10:13])
    projected = []
    for state in tcp:
        rbt = Rotation.from_quat(state[3:])
        rco = rcb * rbt * rto
        tco = rcb.apply(state[:3] + rbt.apply(x[13:16])) + x[7:10]
        camera = rco.apply(MODEL) + tco
        z = np.maximum(camera[:, 2], 1e-4)
        uv = np.column_stack((k[0, 0]*camera[:, 0]/z + k[0, 2],
                              k[1, 1]*camera[:, 1]/z + k[1, 2]))
        projected.append(np.max(uv @ DIRS.T, axis=0))
    return np.asarray(projected)


def summary(x: np.ndarray, tcp: np.ndarray, target: np.ndarray) -> dict:
    err = projected_support(x, tcp)-target
    norms = np.sqrt(np.mean(err**2, axis=1))
    return {"rmse_support_px": float(np.sqrt(np.mean(err**2))),
            "median_frame_rmse_support_px": float(np.median(norms)),
            "p90_frame_rmse_support_px": float(np.percentile(norms, 90)),
            "max_frame_rmse_support_px": float(np.max(norms)),
            "per_frame_rmse_support_px": norms.tolist()}


def depth_crosscheck(x: np.ndarray, states: np.ndarray, frames: np.ndarray,
                     depths: np.ndarray, steps: tuple[int, ...]) -> dict:
    from probe_fmb_effective_k import INTERIOR

    k = k_from_x(x)
    rcb = Rotation.from_rotvec(x[4:7])
    rto = Rotation.from_rotvec(x[10:13])
    rows = []
    all_errors = []
    for s in steps:
        state = states[s]
        rbt = Rotation.from_quat(state[3:])
        rco = rcb * rbt * rto
        tco = rcb.apply(state[:3] + rbt.apply(x[13:16])) + x[7:10]
        camera = rco.apply(INTERIOR) + tco
        uv = np.column_stack((k[0, 0]*camera[:, 0]/camera[:, 2]+k[0, 2],
                              k[1, 1]*camera[:, 1]/camera[:, 2]+k[1, 2]))
        hsv = cv2.cvtColor(frames[s], cv2.COLOR_BGR2HSV)
        red = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
               cv2.inRange(hsv, (170, 75, 90), (179, 255, 255))) > 0
        errors = []
        for (uf, vf), z in zip(uv, camera[:, 2]):
            u, v = np.rint((uf, vf)).astype(int)
            if not (3 <= u < 253 and 3 <= v < 253 and red[v, u]):
                continue
            patch = depths[s, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            if len(valid) < 3:
                continue
            errors.append(float(z-np.median(valid)*.0001))
        all_errors.extend(errors)
        rows.append({"step": s, "valid_interior_points": len(errors),
                     "signed_error_m": errors,
                     "median_abs_error_mm": float(np.median(np.abs(errors))*1000)
                     if errors else None})
    values = np.asarray(all_errors)
    return {"per_frame": rows, "count": len(values),
            "median_abs_Z_error_mm": float(np.median(np.abs(values))*1000) if len(values) else None,
            "p90_abs_Z_error_mm": float(np.percentile(np.abs(values), 90)*1000) if len(values) else None,
            "signed_Z_bias_mm": float(np.median(values)*1000) if len(values) else None}


def fit(tcp: np.ndarray, target: np.ndarray, x0: np.ndarray,
        max_nfev: int = 140) -> dict:
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-3.]*3, [-np.inf]*3, [-.3]*3]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [3.]*3, [np.inf]*3, [.3]*3]
    res = least_squares(lambda x: (projected_support(x, tcp)-target).ravel(),
                        x0, bounds=(lower, upper), loss="soft_l1", f_scale=2.,
                        x_scale="jac", max_nfev=max_nfev)
    return {"x": res.x.tolist(), "K": k_from_x(res.x).tolist(),
            "success": bool(res.success), "message": res.message,
            "cost": float(res.cost), "nfev": int(res.nfev),
            "train": summary(res.x, tcp, target),
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def initial_vector(tcp: np.ndarray) -> np.ndarray:
    tip = json.loads((OUT / "tcp_tip_k_probe.json").read_text(encoding="utf8"))
    ex = np.array(tip["best_fit"]["x"])
    silhouette = json.loads((SOURCE.parent / "effective_k_rounded_cad_silhouette.json").read_text(
        encoding="utf8"))
    pose130 = next(row["pose"] for row in silhouette["heldout"] if row["step"] == 130)
    rco = Rotation.from_rotvec(pose130[:3])
    tco = np.array(pose130[3:])
    rcb = Rotation.from_rotvec(ex[4:7])
    rbt = Rotation.from_quat(tcp[130, 3:])
    rto = rbt.inv() * rcb.inv() * rco
    tto = rbt.inv().apply(rcb.inv().apply(tco-ex[7:10])-tcp[130, :3])
    return np.r_[ex[:10], rto.as_rotvec(), tto]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.load(SOURCE, allow_pickle=True).item()
    tcp = source["obs/tcp_pose"]
    frames = source["obs/side_1"]
    ids = TRAIN + HOLDOUT
    target = {s: np.max(red_hull(frames[s]) @ DIRS.T, axis=0) for s in ids}
    frame_audit = []
    for s in ids:
        x, y, w, h = cv2.boundingRect(red_hull(frames[s]).astype(np.int32))
        frame_audit.append({"frame_id": s,
                            "split": "train" if s in TRAIN else "holdout",
                            "primitive": str(source["primitive"][s]),
                            "gripper_pose": int(source["obs/gripper_pose"][s]),
                            "red_mask_bbox_xywh_px": [x, y, w, h],
                            "visibility": "narrow peg at grasp transition" if s < 100
                                          else "larger peg; gripper near upper end",
                            "selection_reason": "RGB/TCP diagnostic: peg red-mask silhouette visible",
                            "unique_physical_CAD_point_correspondences_verified": 0})
    (OUT / "calibration_frames.json").write_text(json.dumps({
        "purpose": "Frame audit for the diagnostic TCP-constrained CAD silhouette fit",
        "source_episode": "1_M_L_3_vertical_n_2.npy",
        "camera": "side_1", "image_size_px": [256, 256],
        "holdout_selected_before_K_fit": True,
        "point_bundle_feasibility": "No frame has six verified, uniquely identifiable physical CAD-to-RGB points; rounded featureless peg and gripper occlusion prevent the requested point bundle.",
        "frames": frame_audit}, indent=2) + "\n", encoding="utf8")
    (OUT / "cad_rgb_correspondences.json").write_text(json.dumps({
        "status": "NO_VERIFIED_SIX_POINT_CAD_RGB_SET",
        "correspondences": [],
        "reason": "Silhouette extrema and bounding-box corners are viewpoint-dependent or virtual on the rounded CAD; they cannot be assigned persistent physical cad_point_id values.",
        "alternative_observations": "Directional silhouette supports in tcp_cad_k_probe.json"},
        indent=2) + "\n", encoding="utf8")
    train_target = np.array([target[s] for s in TRAIN])
    hold_target = np.array([target[s] for s in HOLDOUT])
    x0 = initial_vector(tcp)
    seeds = []
    for kstart in (K_NOMINAL, np.array([[145.71, 0., 125.55],
                                        [0., 203.35, 116.17], [0., 0., 1.]])):
        seed = x0.copy()
        seed[:4] = [np.log(kstart[0, 0]), np.log(kstart[1, 1]),
                    kstart[0, 2], kstart[1, 2]]
        seeds.append(fit(tcp[list(TRAIN)], train_target, seed))
    best = min(seeds, key=lambda row: row["cost"])
    bx = np.array(best["x"])
    subsets = []
    for residue in range(3):
        keep = np.array([i for i in range(len(TRAIN)) if i % 3 != residue])
        f = fit(tcp[np.array(TRAIN)[keep]], train_target[keep], bx, max_nfev=100)
        subsets.append({"excluded_modulo_3": residue, "K": f["K"],
                        "train_rmse_support_px": f["train"]["rmse_support_px"],
                        "holdout": summary(np.array(f["x"]), tcp[list(HOLDOUT)],
                                           hold_target), "success": f["success"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(50):
        # Perturb measured support values without the positive max-selection
        # bias created by adding noise to every candidate contour vertex.
        noisy = train_target + rng.normal(0, 2., train_target.shape)
        f = fit(tcp[list(TRAIN)], noisy, bx, max_nfev=100)
        if f["success"]:
            k = np.array(f["K"])
            mc.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    result = {"status": "DIAGNOSTIC_TCP_CONSTRAINED_CAD_K_NOT_YET_VALIDATED",
              "assumptions": ["One fixed transform from TCP to CAD object for all fitted frames.",
                              "Red mask hull approximates the full rounded CAD silhouette.",
                              "TCP and RGB records are synchronized."],
              "train_frames": TRAIN, "holdout_frames": HOLDOUT,
              "cad_source": str(SOURCE.parent / "peg_official.step"),
              "cad_model": "Rounded cuboid envelope sampled from STEP width, depth, height and corner radius; it is not a full STEP surface projection.",
              "cad_dimensions_width_depth_height_radius_m": [.04032, .02592, .150, .00288],
              "support_angles_rad": ANGLES.tolist(),
              "observed_support_px": {str(s): target[s].tolist() for s in ids},
              "seed_fits": seeds, "best_fit": best,
              "holdout": summary(bx, tcp[list(HOLDOUT)], hold_target),
              "sensor_depth_crosscheck_holdout": depth_crosscheck(
                  bx, tcp, frames, source["obs/side_1_depth"], HOLDOUT),
              "subset_fits": subsets,
              "monte_carlo_support_sigma_px": 2.,
              "monte_carlo_attempts": 50, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["TCP motion is used in calibration; it is not an independent validation here.",
                              "Occlusion, rounded face correspondence and synchronization can bias K.",
                              "Sensor depth and future GT do not enter fitting or model selection."]}
    (OUT / "tcp_cad_k_probe.json").write_text(json.dumps(result, indent=2) + "\n",
                                             encoding="utf8")
    print(json.dumps({"K": best["K"], "train": best["train"],
                      "holdout": result["holdout"],
                      "subset_K": [s["K"] for s in subsets],
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"]}, indent=2))


if __name__ == "__main__":
    main()
