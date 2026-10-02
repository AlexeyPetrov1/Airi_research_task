"""Measure RGB silhouette sensitivity for a newly sampled horizontal FMB peg.

The check uses no depth and does not estimate K. It only tests whether the
additional pose family supplies a stable, fully visible CAD silhouette.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy"
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
STEPS = tuple(range(0, 50, 5))
HUE_MAX = (13, 20, 25, 30, 35)
ANGLES = np.arange(0, 2*np.pi, np.pi/8)
DIRS = np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))


def hull(frame: np.ndarray, hue_max: int) -> tuple[np.ndarray, int]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 60, 90), (hue_max, 255, 255)) |
            cv2.inRange(hsv, (170, 60, 90), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea)
    return cv2.convexHull(best).reshape(-1, 2).astype(float), int(cv2.contourArea(best))


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    frames = source["obs/side_1"]
    rows = []
    canvas = np.full((2*256, 5*256, 3), 255, np.uint8)
    for i, step in enumerate(STEPS):
        frame = frames[step]
        hulls, areas = {}, {}
        for threshold in HUE_MAX:
            points, area = hull(frame, threshold)
            hulls[threshold], areas[threshold] = points, area
        supports = np.stack([np.max(hulls[k] @ DIRS.T, axis=0) for k in HUE_MAX])
        changes = np.ptp(supports, axis=0)
        x, y, w, h = cv2.boundingRect(hulls[25].astype(np.int32))
        rows.append({"step": step, "hue_thresholds": HUE_MAX,
                     "contour_area_px2": {str(k): areas[k] for k in HUE_MAX},
                     "mask_bbox_h25_xywh_px": [x, y, w, h],
                     "support_shift_median_px": float(np.median(changes)),
                     "support_shift_p90_px": float(np.percentile(changes, 90)),
                     "support_shift_max_px": float(np.max(changes)),
                     "hull_touches_top": bool(y == 0),
                     "hull_touches_image_border": bool(x == 0 or y == 0 or
                                                        x+w >= 256 or y+h >= 256)})
        draw = frame.copy()
        cv2.polylines(draw, [hulls[13].astype(np.int32)], True, (255, 0, 0), 1)
        cv2.polylines(draw, [hulls[35].astype(np.int32)], True, (0, 255, 0), 1)
        cv2.putText(draw, f"t{step}", (4, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    .55, (0, 255, 255), 2)
        canvas[(i//5)*256:(i//5+1)*256, (i%5)*256:(i%5+1)*256] = draw
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "horizontal_peg_segmentation_thresholds.png"), canvas)
    (OUT / "horizontal_peg_segmentation_stability.json").write_text(json.dumps({
        "source": str(SOURCE), "method": "Hue-threshold variation in RGB; blue=h13, green=h35",
        "rows": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
