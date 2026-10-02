"""Compare zero-distortion and k1/k2 RGB board calibration on held-out data."""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, extract, project
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from probe_fmb_tcp_tip_k import OUT


def project_distorted(points: np.ndarray, x: np.ndarray, i: int) -> np.ndarray:
    k = np.array([[np.exp(x[0]), 0., x[2]], [0., np.exp(x[1]), x[3]], [0., 0., 1.]])
    distortion = np.array([x[-2], x[-1], 0., 0., 0.])
    uv, _ = cv2.projectPoints(points, x[4+6*i:7+6*i], x[7+6*i:10+6*i],
                              k, distortion)
    return uv.reshape(-1, 2)


def evaluate(points, observations, x, features, distorted):
    rows = []
    for i, obs in enumerate(observations):
        uv = project_distorted(points, x, i) if distorted else project(
            points, np.r_[x[:4], x[4+6*i:10+6*i]])
        e = np.linalg.norm(uv[features][None]-obs[:, features], axis=2)
        rows.append({"rmse_px": float(np.sqrt(np.mean(e*e))),
                     "p90_px": float(np.percentile(e, 90)),
                     "max_px": float(np.max(e))})
    return rows


def fit(points, observations, features, seed):
    count = len(observations)
    lower = np.r_[np.log(40), np.log(40), 0., 0.,
                  np.tile([-np.inf]*3+[-2.]*2+[.02], count), -2., -5.]
    upper = np.r_[np.log(800), np.log(800), 255., 255.,
                  np.tile([np.inf]*3+[2.]*2+[2.], count), 2., 5.]
    def residual(x):
        return np.concatenate([(project_distorted(points, x, i)[features][None]-
                                obs[:, features]).ravel()
                               for i, obs in enumerate(observations)])
    res = least_squares(residual, seed, bounds=(lower, upper), loss="soft_l1",
                        f_scale=2., x_scale="jac", max_nfev=200)
    return {"x": res.x.tolist(), "success": bool(res.success),
            "message": res.message, "cost": float(res.cost),
            "k1": float(res.x[-2]), "k2": float(res.x[-1]),
            "jacobian_singular_values": np.linalg.svd(res.jac, compute_uv=False).tolist()}


def main() -> None:
    points, names, _ = cad_features(1)
    train_features = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    hold_features = np.array(HOLDOUT_FEATURES)
    train, hold = [], []
    for number in EPISODES:
        src = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        frames = src["obs/side_1"]
        obs = {s: extract(frames[s], 1)[0] for s in TRAIN+HOLDOUT}
        train.append(np.array([obs[s] for s in TRAIN]))
        hold.append(np.array([obs[s] for s in HOLDOUT]))
        del src
    reference = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    pinhole = np.array(reference["best_fit"]["x"])
    result = fit(points, tuple(train), train_features, np.r_[pinhole, 0., 0.])
    x = np.array(result["x"])
    rows = {"status": "DISTORTION_CANDIDATE_REQUIRES_HOLDOUT_IMPROVEMENT_ABOVE_ANNOTATION_NOISE",
            "source_episodes": EPISODES,
            "train_feature_ids": [names[i] for i in train_features],
            "holdout_feature_ids": [names[i] for i in hold_features],
            "distorted_fit": result,
            "pinhole_heldout_frames_train_features": evaluate(
                points, hold, pinhole, train_features, False),
            "distorted_heldout_frames_train_features": evaluate(
                points, hold, x, train_features, True),
            "pinhole_heldout_frames_heldout_features": evaluate(
                points, hold, pinhole, hold_features, False),
            "distorted_heldout_frames_heldout_features": evaluate(
                points, hold, x, hold_features, True),
            "note": "No depth or future GT used to fit or compare the RGB camera models."}
    (OUT / "distortion_model_comparison.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"k1": result["k1"], "k2": result["k2"],
                      "pinhole_heldout": rows["pinhole_heldout_frames_train_features"],
                      "distorted_heldout": rows["distorted_heldout_frames_train_features"],
                      "pinhole_feature_heldout": rows["pinhole_heldout_frames_heldout_features"],
                      "distorted_feature_heldout": rows["distorted_heldout_frames_heldout_features"]}, indent=2))


if __name__ == "__main__":
    main()
