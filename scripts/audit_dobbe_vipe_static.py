"""Independent causal SIFT checks separate geometry from background-tracker errors."""
from __future__ import annotations

import argparse
import cv2
import numpy as np

from dobbe_vipe_v1 import scene_dir, RUN, read, write
from dobbe_vipe_geometry import lift, project, stats


def audit(scene, branch):
    dest = scene_dir(scene, branch)
    p = read(dest / "protocol.json")
    g = np.load(dest / "geometry_all.npz")
    frames = [cv2.imread(str(dest / "frames" / f"{i:05d}.png"), cv2.IMREAD_GRAYSCALE)
              for i in range(len(p["observed_raw_ids"]))]
    mask = cv2.imread(str(RUN / scene / "background_mask_t0.png"), cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=.02)
    k0, d0 = sift.detectAndCompute(frames[-1], mask)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    reports, all_errors = [], []
    # Include both broad prefix and the exact H3 intervals; no future is read.
    indices = sorted(set(np.linspace(0, len(frames)-4, 6).astype(int).tolist() + p["history_vipe_indices"][:-1]))
    for t in indices:
        kt, dt = sift.detectAndCompute(frames[t], None)
        if d0 is None or dt is None:
            reports.append({"to_raw": p["observed_raw_ids"][t], "error": "no descriptors"})
            continue
        forward = matcher.knnMatch(d0, dt, k=2)
        reverse = matcher.knnMatch(dt, d0, k=2)
        rev = {a.queryIdx: a.trainIdx for a, b in reverse if a.distance < .7*b.distance}
        good = [a for a, b in forward if a.distance < .7*b.distance and rev.get(a.trainIdx) == a.queryIdx]
        if len(good) < 8:
            reports.append({"to_raw": p["observed_raw_ids"][t], "mutual_ratio_matches": len(good)})
            continue
        src = np.array([k0[m.queryIdx].pt for m in good], np.float32)
        tgt = np.array([kt[m.trainIdx].pt for m in good], np.float32)
        cv2.setRNGSeed(0)
        f, inliers = cv2.findFundamentalMat(src, tgt, cv2.FM_RANSAC, 1.0, .999)
        if f is None or inliers is None:
            reports.append({"to_raw": p["observed_raw_ids"][t], "error": "no fundamental consensus"})
            continue
        src, tgt = src[inliers[:, 0] > 0], tgt[inliers[:, 0] > 0]
        world, visible, _ = lift(src[None], g["depths"][-1:], g["intrinsics"][-1:],
                                g["poses"][-1:], np.ones((1, len(src)), bool))
        pred = project(world[0], g["poses"][t], g["intrinsics"][t])
        valid = visible[0] & np.isfinite(pred).all(-1)
        errors = np.linalg.norm(pred[valid]-tgt[valid], axis=-1)
        reports.append({"to_raw": p["observed_raw_ids"][t],
                        "mutual_ratio_matches": len(good), "fundamental_inliers": len(src), **stats(errors)})
        all_errors.extend(errors)
        panel = cv2.cvtColor(frames[t], cv2.COLOR_GRAY2BGR)
        for a, b in zip(pred[valid], tgt[valid]):
            cv2.circle(panel, tuple(b.round().astype(int)), 2, (0, 255, 0), -1)
            if np.abs(a).max() < 1e5:
                cv2.line(panel, tuple(a.round().astype(int)), tuple(b.round().astype(int)), (0, 0, 255), 1)
        cv2.imwrite(str(dest / f"static_sift_{t:05d}.png"), panel)
        np.savez(dest / f"static_sift_{t:05d}.npz", t0_uv=src, earlier_uv=tgt,
                 predicted_uv=pred, valid=valid)
    write(dest / "independent_static_sift.json", {"method": "SIFT mutual Lowe .7; fundamental RANSAC 1px; t0 background mask; depth/c2w not used for match selection",
          "future_used": False, "aggregate_px": stats(all_errors), "pairs": reports,
          "gate_changed": False})
    print(scene, branch, stats(all_errors), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["A", "B", "C"], required=True)
    parser.add_argument("--branch", choices=["no_vda", "default", "rectified"], default="no_vda")
    args = parser.parse_args()
    audit(args.scene, args.branch)
