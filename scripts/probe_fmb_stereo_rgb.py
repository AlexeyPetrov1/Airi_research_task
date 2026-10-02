"""Test whether the two published FMB RGB views support stereo matching.

An uncalibrated fundamental matrix is only a correspondence diagnostic; it
cannot produce metric triangulation without effective intrinsics and scale.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "fmb_effective_k_256_calibration"
EP = ROOT / "runs" / "sharerobot_fmb_episode_5201" / "1_M_L_3_vertical_n_2.npy"
STEPS = (0, 37, 74, 110, 120, 128, 136, 144)


def red_properties(image: np.ndarray) -> dict:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0)
            | (cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)) > 0))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    if count <= 1:
        return {"pixels": 0, "center": None, "bbox": None}
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return {"pixels": int(stats[i, cv2.CC_STAT_AREA]),
            "center": centers[i].tolist(),
            "bbox": stats[i, :4].tolist()}


def sift_stereo(left: np.ndarray, right: np.ndarray) -> dict:
    sift = cv2.SIFT_create(nfeatures=1800)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    key1, des1 = sift.detectAndCompute(left_gray, None)
    key2, des2 = sift.detectAndCompute(right_gray, None)
    result = {"keypoints_side_1": len(key1), "keypoints_side_2": len(key2)}
    if des1 is None or des2 is None:
        return result | {"ratio_matches": 0}
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < .72 * n.distance]
    result["ratio_matches"] = len(good)
    if len(good) < 8:
        return result
    p1 = np.float32([key1[m.queryIdx].pt for m in good])
    p2 = np.float32([key2[m.trainIdx].pt for m in good])
    f, inliers_f = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, .99, 5000)
    h, inliers_h = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    result["fundamental_inliers"] = int(inliers_f.sum()) if inliers_f is not None else 0
    result["homography_inliers"] = int(inliers_h.sum()) if inliers_h is not None else 0
    if f is not None and f.shape == (3, 3):
        result["F"] = f.tolist()
        if inliers_f is not None:
            paired = np.column_stack([p1[inliers_f.ravel() > 0], p2[inliers_f.ravel() > 0]])
            result["inlier_pairs_xyxy"] = paired.tolist()
    if h is not None:
        result["H"] = h.tolist()
    return result


def main() -> None:
    data = np.load(EP, allow_pickle=True).item()
    frames = []
    for t in STEPS:
        side1 = data["obs/side_1"][t]
        side2 = data["obs/side_2"][t]
        frames.append({"step": t, "red_side_1": red_properties(side1),
                       "red_side_2": red_properties(side2),
                       "sift": sift_stereo(side1, side2)})
    result = {"source": str(EP), "frames": frames,
              "limitation": "SIFT/F/H only checks tentative stereo correspondences. There are no verified nonplanar 3D matches or recorded per-camera effective COLOR K/extrinsics, so metric stereo triangulation is not validated."}
    path = OUT / "stereo_rgb_probe.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"step": f["step"], "red_pixels": [f["red_side_1"]["pixels"],
                       f["red_side_2"]["pixels"]],
                       "sift": {k: v for k, v in f["sift"].items()
                                if k in ("keypoints_side_1", "keypoints_side_2", "ratio_matches",
                                         "fundamental_inliers", "homography_inliers")}}
                      for f in frames], indent=2))


if __name__ == "__main__":
    main()
