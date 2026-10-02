"""Stress-test a joint K, image warp and camera-offset hypothesis.

Large held-out errors or boundary solutions mean the fitted K is not a valid
substitute for the missing Record3D metadata.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from fit_dobbe_3651_effective_k import C, DATA, HOLDOUT, OUT, PAIRS, TRAIN, load_frames, make_matches


ROOT = Path(__file__).resolve().parents[1]


def residual(pair, depth, pose_r, pose_t, params):
    fx, fy, cx, cy, du, dv = params[:6]
    delta = params[6:9]
    baseline = params[9:12]
    c = Rotation.from_rotvec(delta).as_matrix() @ C
    i, j = pair["pair"]
    a, b = pair["uv_i"], pair["uv_j"]
    z = map_coordinates(depth[i],
                        [(a[:, 1] + .5) * .75 - .5 + dv, a[:, 0] + du],
                        order=1, mode="nearest")
    optical = np.column_stack(((a[:, 0] - cx) / fx * z,
                               (a[:, 1] - cy) / fy * z, z))
    world = (optical @ c + baseline) @ pose_r[i].T + pose_t[i]
    optical_j = (((world - pose_t[j]) @ pose_r[j]) - baseline) @ c.T
    zj = np.maximum(optical_j[:, 2], .01)
    pred = np.column_stack((fx * optical_j[:, 0] / zj + cx,
                            fy * optical_j[:, 1] / zj + cy))
    return pred - b


def metrics(pairs, depth, pose_r, pose_t, params):
    result = []
    for pair in pairs:
        e = np.linalg.norm(residual(pair, depth, pose_r, pose_t, params), axis=1)
        result.append({"pair": pair["pair"], "n": len(e),
                       "median_px": float(np.median(e)),
                       "p90_px": float(np.percentile(e, 90)),
                       "under_5px": float(np.mean(e < 5))})
    return result


def main():
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    depth = np.frombuffer(liblzfse.decompress(
        (DATA / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    pose_t = np.asarray([labels[str(i)]["xyz"] for i in range(len(labels))])
    pose_r = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(len(labels))]).as_matrix()
    pairs = make_matches(load_frames({i for pair in PAIRS for i in pair}))
    for pair in pairs:
        _, mask = cv2.findFundamentalMat(pair["uv_i"], pair["uv_j"],
                                         cv2.USAC_MAGSAC, 2.0, .99)
        if mask is not None:
            keep = mask.ravel().astype(bool)
            pair["uv_i"] = pair["uv_i"][keep]
            pair["uv_j"] = pair["uv_j"][keep]
    train = [p for p in pairs if tuple(p["pair"]) in TRAIN]
    holdout = [p for p in pairs if tuple(p["pair"]) in HOLDOUT]
    initial = np.asarray([200, 200, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0],
                         dtype=float)
    fun = lambda x: np.concatenate(
        [residual(p, depth, pose_r, pose_t, x).ravel() for p in train])
    fit = least_squares(fun, initial,
                        bounds=([70, 70, 80, 80, -12, -12,
                                 -.45, -.45, -.45, -.15, -.15, -.15],
                                [600, 600, 176, 176, 12, 12,
                                 .45, .45, .45, .15, .15, .15]),
                        loss="soft_l1", f_scale=3, max_nfev=700)
    names = ("fx", "fy", "cx", "cy", "depth_du", "depth_dv",
             "rx", "ry", "rz", "baseline_x", "baseline_y", "baseline_z")
    result = {"status": "EXPLORATORY_NOT_VALIDATED",
              "parameter_meaning": "K_rgb, RGB-to-depth translation, optical-axis correction, device-to-camera baseline",
              "axis_selection_note": "discrete axis choice preceded and used all pairs",
              "params": dict(zip(names, [float(v) for v in fit.x])),
              "success": bool(fit.success),
              "before_train": metrics(train, depth, pose_r, pose_t, initial),
              "before_holdout": metrics(holdout, depth, pose_r, pose_t, initial),
              "after_train": metrics(train, depth, pose_r, pose_t, fit.x),
              "after_holdout": metrics(holdout, depth, pose_r, pose_t, fit.x)}
    (OUT / "dobbe_3651_joint_calibration_stress.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
