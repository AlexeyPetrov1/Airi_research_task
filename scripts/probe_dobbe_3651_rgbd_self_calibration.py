"""Exploratory effective-K fit using RGB-D rigidity, independent of saved poses.

Depth is sampled with a 256x256 RGB to 192x256 depth row-scale hypothesis.
Unknown relative camera pose is fitted separately for each frame pair. An
accepted result would additionally need stable K across train subsets and low
held-out 3D residuals; the script reports these diagnostics, not approval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares

from probe_dobbe_3651_camera_geometry import DATA, OUT, load_frames


ROOT = Path(__file__).resolve().parents[1]
PAIRS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50),
         (100, 110), (110, 120), (180, 190), (190, 200), (200, 210), (210, 220)]
HOLDOUT = {(30, 40), (110, 120), (200, 210)}


def sample_z(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    v = (uv[:, 1] + .5) * .75 - .5
    return map_coordinates(depth, [v, uv[:, 0]], order=1, mode="nearest")


def xyz(uv: np.ndarray, z: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = k
    return np.column_stack(((uv[:, 0] - cx) * z / fx,
                            (uv[:, 1] - cy) * z / fy, z))


def rigid_align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    diag = np.diag([1., 1., np.linalg.det(u @ vt)])
    rot = u @ diag @ vt
    trans = bc - ac @ rot
    return rot, trans


def pair_residual(pair: dict, k: np.ndarray) -> np.ndarray:
    a = xyz(pair["uv_i"], pair["z_i"], k)
    b = xyz(pair["uv_j"], pair["z_j"], k)
    rot, trans = rigid_align(a, b)
    return (a @ rot + trans - b).ravel()


def stats(pair: dict, k: np.ndarray) -> dict:
    err = np.linalg.norm(pair_residual(pair, k).reshape(-1, 3), axis=1)
    return {"pair": pair["pair"], "n": len(err),
            "median_3d_m": float(np.median(err)),
            "p90_3d_m": float(np.percentile(err, 90))}


def prepare(depth: np.ndarray) -> list[dict]:
    frames = load_frames({i for p in PAIRS for i in p})
    sift = cv2.SIFT_create(nfeatures=2200)
    feats = {i: sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None)
             for i, im in frames.items()}
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    output = []
    k0 = np.array([200., 200., 128., 128.])
    rng = np.random.default_rng(11)
    for i, j in PAIRS:
        ki, di = feats[i]
        kj, dj = feats[j]
        if di is None or dj is None:
            continue
        matches = [m for m, n in matcher.knnMatch(di, dj, k=2)
                   if m.distance < .7 * n.distance]
        uv_i = np.asarray([ki[m.queryIdx].pt for m in matches], dtype=float)
        uv_j = np.asarray([kj[m.trainIdx].pt for m in matches], dtype=float)
        if len(uv_i) < 12:
            continue
        _, mask = cv2.findFundamentalMat(uv_i, uv_j, cv2.USAC_MAGSAC,
                                         1.5, .999, 10000)
        if mask is None:
            continue
        keep = mask.ravel().astype(bool)
        uv_i, uv_j = uv_i[keep], uv_j[keep]
        zi, zj = sample_z(depth[i], uv_i), sample_z(depth[j], uv_j)
        keep = ((zi > .1) & (zi < 2.) & (zj > .1) & (zj < 2.) &
                (uv_i[:, 0] > 3) & (uv_i[:, 0] < 252) &
                (uv_j[:, 0] > 3) & (uv_j[:, 0] < 252))
        uv_i, uv_j, zi, zj = uv_i[keep], uv_j[keep], zi[keep], zj[keep]
        if len(zi) < 10:
            continue
        ai, bj = xyz(uv_i, zi, k0), xyz(uv_j, zj, k0)
        # A coarse 3D RANSAC removes independently moving objects and wrong
        # feature matches before fitting K; no K is selected from holdouts.
        best = np.zeros(len(ai), dtype=bool)
        for _ in range(300):
            choice = rng.choice(len(ai), 3, replace=False)
            rot, trans = rigid_align(ai[choice], bj[choice])
            inliers = np.linalg.norm(ai @ rot + trans - bj, axis=1) < .025
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() < 9:
            continue
        output.append({"pair": [i, j], "uv_i": uv_i[best],
                       "uv_j": uv_j[best], "z_i": zi[best], "z_j": zj[best],
                       "initial_matches": len(matches),
                       "rigid_inliers": int(best.sum())})
    return output


def fit(pairs: list[dict]) -> np.ndarray:
    if not pairs:
        raise RuntimeError("No pairs to fit")
    objective = lambda k: np.concatenate([pair_residual(p, k) for p in pairs])
    result = least_squares(objective, [200, 200, 128, 128],
                           bounds=([90, 90, 64, 64], [600, 600, 192, 192]),
                           loss="soft_l1", f_scale=.01, max_nfev=200)
    return result.x


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse
    depth = np.frombuffer(liblzfse.decompress(
        (DATA / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    pairs = prepare(depth)
    train = [p for p in pairs if tuple(p["pair"]) not in HOLDOUT]
    holdout = [p for p in pairs if tuple(p["pair"]) in HOLDOUT]
    if len(train) < 4 or len(holdout) < 2:
        raise RuntimeError(f"Insufficient pairs: {len(train)} train, {len(holdout)} holdout")
    k0 = np.array([200., 200., 128., 128.])
    k = fit(train)
    subsets = [train[::2], train[1::2], train[:len(train)//2],
               train[len(train)//2:]]
    output = {
        "status": "EXPLORATORY_NOT_VALIDATED",
        "method": "RGB-D static-feature 3D rigidity; free relative pose per pair",
        "depth_row_mapping": "v_d=(v_rgb+.5)*.75-.5, u_d=u_rgb",
        "initial_K": k0.tolist(), "fit_K": k.tolist(),
        "subset_K": [fit(s).tolist() for s in subsets],
        "train_before": [stats(p, k0) for p in train],
        "train_after": [stats(p, k) for p in train],
        "holdout_before": [stats(p, k0) for p in holdout],
        "holdout_after": [stats(p, k) for p in holdout],
        "pair_counts": [{"pair": p["pair"], "initial_matches": p["initial_matches"],
                         "rigid_inliers": p["rigid_inliers"]} for p in pairs],
        "limitation": "Depth-dependent fit, not an independent camera calibration; depth registration and distortion unverified globally",
    }
    (OUT / "dobbe_3651_rgbd_self_calibration.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
