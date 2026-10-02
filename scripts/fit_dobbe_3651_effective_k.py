"""Exploratory shared K/registration fit from HoNY RGB-D and recorded poses.

This is a diagnostic with held-out frame pairs. It cannot validate the sensor
extrinsic, Record3D depth semantics, or ShareRobot pixel identity by itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_dobbe_3651_camera_geometry import DATA, OUT, PAIRS, load_frames, make_matches


ROOT = Path(__file__).resolve().parents[1]
TRAIN = {(0, 20), (20, 40), (40, 60), (180, 200)}
HOLDOUT = {(60, 80), (200, 220)}
C = np.asarray([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)


def pair_residual(pair: dict, depth: np.ndarray, pose_r: np.ndarray,
                  pose_t: np.ndarray, params: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy, du, dv = params
    i, j = pair["pair"]
    uv_i, uv_j = pair["uv_i"], pair["uv_j"]
    ud = uv_i[:, 0] + du
    vd = (uv_i[:, 1] + .5) * .75 - .5 + dv
    z = map_coordinates(depth[i], [vd, ud], order=1, mode="nearest")
    p_i_opt = np.column_stack(((uv_i[:, 0] - cx) / fx * z,
                               (uv_i[:, 1] - cy) / fy * z, z))
    p_world = (p_i_opt @ C) @ pose_r[i].T + pose_t[i]
    p_j_opt = ((p_world - pose_t[j]) @ pose_r[j]) @ C.T
    z_j = np.maximum(p_j_opt[:, 2], .01)
    projected = np.column_stack((fx * p_j_opt[:, 0] / z_j + cx,
                                 fy * p_j_opt[:, 1] / z_j + cy))
    return projected - uv_j


def summarize(pairs, depth, pose_r, pose_t, params):
    output = []
    for pair in pairs:
        error = np.linalg.norm(pair_residual(pair, depth, pose_r, pose_t, params),
                               axis=1)
        output.append({"pair": pair["pair"], "n": len(error),
                       "median_px": float(np.median(error)),
                       "p90_px": float(np.percentile(error, 90)),
                       "inlier_5px_fraction": float(np.mean(error < 5))})
    return output


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    depth = np.frombuffer(liblzfse.decompress(
        (DATA / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    pose_t = np.asarray([labels[str(i)]["xyz"] for i in range(len(labels))])
    pose_q = np.asarray([labels[str(i)]["quats"] for i in range(len(labels))])
    pose_r = Rotation.from_quat(pose_q).as_matrix()
    frames = load_frames({i for pair in PAIRS for i in pair})
    pairs = make_matches(frames)
    train = [p for p in pairs if tuple(p["pair"]) in TRAIN]
    holdout = [p for p in pairs if tuple(p["pair"]) in HOLDOUT]
    initial = np.asarray([200, 200, 128, 128, 0, 0], dtype=float)
    residual = lambda params: np.concatenate(
        [pair_residual(p, depth, pose_r, pose_t, params).ravel() for p in train])
    fit = least_squares(residual, initial,
                        bounds=([70, 70, 80, 80, -20, -20],
                                [600, 600, 176, 176, 20, 20]),
                        loss="soft_l1", f_scale=4, max_nfev=500)
    result = {"status": "EXPLORATORY_NOT_VALIDATED",
              "C_label_to_optical": C.astype(int).tolist(),
              "axis_selection_note": "C selected in an earlier exploratory scan including holdout pairs",
              "params": {name: float(value) for name, value in
                         zip(("fx", "fy", "cx", "cy", "depth_du", "depth_dv"), fit.x)},
              "optimizer_success": bool(fit.success), "optimizer_message": fit.message,
              "before_train": summarize(train, depth, pose_r, pose_t, initial),
              "before_holdout": summarize(holdout, depth, pose_r, pose_t, initial),
              "after_train": summarize(train, depth, pose_r, pose_t, fit.x),
              "after_holdout": summarize(holdout, depth, pose_r, pose_t, fit.x)}
    (OUT / "dobbe_3651_effective_k_fit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
