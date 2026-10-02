"""Calibrate from the peg's visible bottom cross-section across TCP motion.

The bottom cross-section remains visible when the peg top is outside the RGB
frame. This is a diagnostic RGB+CAD+TCP fit, subject to rigid-grasp and
correspondence assumptions; sensor depth remains a held-out check.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_silhouette_k import MODEL
from probe_fmb_tcp_cad_k import depth_crosscheck, initial_vector
from probe_fmb_tcp_tip_k import OUT, SOURCE, k_from_x, red_end


BOTTOM = MODEL[np.isclose(MODEL[:, 2], 0.)]
TRAIN = tuple(range(72, 89, 2)) + tuple(range(100, 121, 2))
HOLDOUT = tuple(range(73, 89, 2)) + tuple(range(101, 121, 2))
LATE = (129, 131, 133, 135, 137, 139, 141, 143, 145)


def observed(frame: np.ndarray) -> np.ndarray:
    uv, meta = red_end(frame)
    return np.r_[uv, meta["bottom_band_width_px"]]


def predict(x: np.ndarray, states: np.ndarray) -> np.ndarray:
    k = k_from_x(x)
    rcb = Rotation.from_rotvec(x[4:7])
    rto = Rotation.from_rotvec(x[10:13])
    out = []
    for state in states:
        rbt = Rotation.from_quat(state[3:])
        rco = rcb * rbt * rto
        tco = rcb.apply(state[:3] + rbt.apply(x[13:16])) + x[7:10]
        camera = rco.apply(BOTTOM) + tco
        z = np.maximum(camera[:, 2], 1e-4)
        u = k[0, 0]*camera[:, 0]/z + k[0, 2]
        v = k[1, 1]*camera[:, 1]/z + k[1, 2]
        out.append([(u.min()+u.max())/2, v.max(), u.max()-u.min()])
    return np.array(out)


def score(x: np.ndarray, tcp: np.ndarray, target: np.ndarray) -> dict:
    e = predict(x, tcp)-target
    return {"mean_abs_u_v_width_px": np.mean(np.abs(e), axis=0).tolist(),
            "rmse_u_v_width_px": np.sqrt(np.mean(e*e, axis=0)).tolist(),
            "overall_rmse_px": float(np.sqrt(np.mean(e*e))),
            "max_norm_px": float(np.max(np.linalg.norm(e, axis=1))),
            "per_frame_residual_u_v_width_px": e.tolist()}


def fit(tcp: np.ndarray, target: np.ndarray, x0: np.ndarray,
        max_nfev: int = 250) -> dict:
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  [-np.inf]*3, [-3.]*3, [-np.inf]*3, [-.3]*3]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  [np.inf]*3, [3.]*3, [np.inf]*3, [.3]*3]
    res = least_squares(lambda x: (predict(x, tcp)-target).ravel(), x0,
                        bounds=(lower, upper), loss="soft_l1", f_scale=2.,
                        x_scale="jac", max_nfev=max_nfev)
    return {"x": res.x.tolist(), "K": k_from_x(res.x).tolist(),
            "cost": float(res.cost), "success": bool(res.success),
            "message": res.message, "nfev": int(res.nfev),
            "train": score(res.x, tcp, target),
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = np.load(SOURCE, allow_pickle=True).item()
    frames = src["obs/side_1"]
    tcp = src["obs/tcp_pose"]
    target = {i: observed(frames[i]) for i in TRAIN + HOLDOUT + LATE}
    train = np.array([target[i] for i in TRAIN])
    hold = np.array([target[i] for i in HOLDOUT])
    late = np.array([target[i] for i in LATE])
    seed = initial_vector(tcp)
    prior = json.loads((OUT / "tcp_cad_k_probe.json").read_text(encoding="utf8"))
    seeds = []
    for x0 in (seed, np.array(prior["best_fit"]["x"])):
        for mult in (.8, 1., 1.2):
            x = x0.copy()
            x[:2] += np.log(mult)
            seeds.append(fit(tcp[list(TRAIN)], train, x))
    best = min(seeds, key=lambda row: row["cost"])
    bx = np.array(best["x"])
    subsets = []
    for residue in range(4):
        keep = np.array([i for i in range(len(TRAIN)) if i % 4 != residue])
        row = fit(tcp[np.array(TRAIN)[keep]], train[keep], bx)
        subsets.append({"excluded_modulo_4": residue, "K": row["K"],
                        "holdout_rmse_px": score(np.array(row["x"]),
                                                 tcp[list(HOLDOUT)], hold)["overall_rmse_px"]})
    rng = np.random.default_rng(5201)
    mc = []
    for _ in range(50):
        noisy = train + rng.normal(0, 2., train.shape)
        row = fit(tcp[list(TRAIN)], noisy, bx, max_nfev=150)
        if row["success"]:
            k = np.array(row["K"])
            mc.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    result = {"status": "DIAGNOSTIC_TCP_BOTTOM_CAD_K_NOT_YET_VALIDATED",
              "train_frames": TRAIN, "holdout_frames": HOLDOUT,
              "late_unfitted_frames": LATE,
              "observation_definition": "Center and bottom row of red mask, plus red width in final 3 rows; model is projected rounded CAD bottom cross-section envelope.",
              "observed_u_v_width_px": {str(i): target[i].tolist() for i in target},
              "seed_fits": seeds, "best_fit": best,
              "holdout": score(bx, tcp[list(HOLDOUT)], hold),
              "late_unfitted": score(bx, tcp[list(LATE)], late),
              "holdout_sensor_depth": depth_crosscheck(
                  bx, tcp, frames, src["obs/side_1_depth"], HOLDOUT),
              "late_sensor_depth": depth_crosscheck(
                  bx, tcp, frames, src["obs/side_1_depth"], LATE),
              "subset_fits": subsets,
              "monte_carlo_observation_sigma_px": 2.,
              "monte_carlo_attempts": 50, "monte_carlo_successes": len(mc),
              "monte_carlo_K": mc,
              "monte_carlo_K_5_50_95_percentiles":
                  np.percentile(mc, [5, 50, 95], axis=0).tolist() if mc else None,
              "limitations": ["The red lower envelope approximates the physical CAD bottom plane.",
                              "Rigid grasp and RGB/TCP synchronization are assumed.",
                              "No sensor depth or future GT in fitting."]}
    (OUT / "tcp_bottom_k_probe.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf8")
    print(json.dumps({"K": best["K"],
                      "train": best["train"]["overall_rmse_px"],
                      "holdout": result["holdout"]["overall_rmse_px"],
                      "late_unfitted": result["late_unfitted"]["overall_rmse_px"],
                      "holdout_depth": result["holdout_sensor_depth"],
                      "late_depth": result["late_sensor_depth"],
                      "subset_K": [x["K"] for x in subsets],
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"]}, indent=2))


if __name__ == "__main__":
    main()
