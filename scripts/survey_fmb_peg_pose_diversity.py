"""Inspect whether additional same-object FMB episodes provide diverse peg views.

Only RGB is used. This is a visibility and image-geometry survey, not a pose
estimate or camera calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from calibrate_fmb_multi_boards_k import background_alignment


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/fmb/single_object_manipulation_dataset"
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
EPISODES = (("vertical", 2), ("vertical", 3), ("vertical", 4),
            ("vertical", 5), ("vertical", 6), ("horizontal", 0),
            ("horizontal", 4), ("horizontal", 8))
FRACTIONS = (0, .125, .25, .375, .5, .625, .75, .875, 1)


def peg_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
            cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return np.zeros_like(mask)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (components == index).astype(np.uint8) * 255


def main() -> None:
    canvas = np.full((len(EPISODES) * 192, len(FRACTIONS) * 192, 3), 255, np.uint8)
    rows = []
    alignments = []
    reference = None
    for row, (angle, episode) in enumerate(EPISODES):
        path = DATA / f"1_M_L_3_{angle}_n_{episode}.npy"
        source = np.load(path, allow_pickle=True).item()
        frames = source["obs/side_1"]
        if reference is None:
            reference = frames[0].copy()
        alignments.append({"angle": angle, "episode": episode,
                           "to_vertical_n2_frame0": background_alignment(
                               reference, frames[0])})
        for col, fraction in enumerate(FRACTIONS):
            step = round(fraction * (len(frames) - 1))
            frame = frames[step].copy()
            mask = peg_mask(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
                area = cv2.contourArea(contour)
                touches_top = y <= 1
                touches_bottom = y + h >= 255
                rect = cv2.minAreaRect(contour)
                image_angle = float(rect[2])
            else:
                x = y = w = h = 0
                area = 0.0
                touches_top = touches_bottom = False
                image_angle = None
            cv2.putText(frame, f"{angle[0]}{episode} t{step}", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            canvas[row*192:(row+1)*192, col*192:(col+1)*192] = cv2.resize(
                frame, (192, 192), interpolation=cv2.INTER_AREA)
            rows.append({"angle": angle, "episode": episode,
                         "step": step, "fraction": fraction,
                         "peg_mask_area_px2": area,
                         "peg_bbox_xywh_px": [x, y, w, h],
                         "mask_touches_top": touches_top,
                         "mask_touches_bottom": touches_bottom,
                         "min_area_rect_angle_deg": image_angle})
        del source
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "peg_pose_diversity_contact.png"), canvas)
    (OUT / "peg_pose_diversity_survey.json").write_text(json.dumps({
        "method": "RGB-only HSV red-mask survey; minAreaRect angle is only a 2D diagnostic",
        "episodes": EPISODES, "sample_fractions": FRACTIONS,
        "static_background_alignment": alignments,
        "samples": rows}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows),
                      "contact_sheet": str(OUT / "peg_pose_diversity_contact.png")}, indent=2))


if __name__ == "__main__":
    main()
