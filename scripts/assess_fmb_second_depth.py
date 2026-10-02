"""Measure raw depth validity inside the yellow peg's visible interior."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_L_L_4_vertical_n_0.npy"
OUT = ROOT / "runs/fmb_second_scene"


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    result = {"source_file": SOURCE.name, "mask_method": "BGR to HSV; yellow H=5..38 S>=100 V>=80; 3x3 erosion; visual diagnostic only", "cameras": {}}
    for camera in ("side_1", "side_2"):
        rows = []
        for i, (bgr, depth) in enumerate(zip(data[f"obs/{camera}"], data[f"obs/{camera}_depth"])):
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (5, 100, 80), (38, 255, 255))
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
            positive = depth[mask] > 0
            rows.append({"frame": i, "primitive": str(data["primitive"][i]), "mask_pixels": int(mask.sum()),
                         "valid_depth_fraction_in_mask": float(positive.mean()) if positive.size else None})
            if i in (0, 56, 65, 94, 103, 112, 122, 131, 141):
                overlay = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
                overlay[mask & (depth == 0)] = [255, 0, 255]
                overlay[mask & (depth > 0)] = [255, 255, 0]
                Image.fromarray(overlay).save(OUT / "candidate_previews" / f"depth_mask_{camera}_{i:03}.png")
        fractions = [r["valid_depth_fraction_in_mask"] for r in rows if r["mask_pixels"] >= 30]
        result["cameras"][camera] = {
            "frames_with_30plus_mask_pixels": len(fractions),
            "median_valid_fraction": float(np.median(fractions)) if fractions else None,
            "p05_valid_fraction": float(np.percentile(fractions, 5)) if fractions else None,
            "per_frame": rows,
        }
    (OUT / "candidate_depth_roi.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for camera, stats in result["cameras"].items():
        print(camera, stats["frames_with_30plus_mask_pixels"], stats["median_valid_fraction"], stats["p05_valid_fraction"])


if __name__ == "__main__":
    main()
