"""Measure red-cup RGB side-edge agreement with depth discontinuities."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dobbe_oxe" / "target_raw"
OUT = ROOT / "runs" / "plex_dobbe_preflight"
FRAMES = (0, 20, 40, 60, 80, 100, 180, 220)


def red_cup_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170)) &
            (hsv[:, :, 1] >= 75) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        raise RuntimeError("No red object component")
    candidate = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    if stats[candidate, cv2.CC_STAT_AREA] < 150:
        raise RuntimeError("Red component too small")
    return labels == candidate


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    depth = np.frombuffer(liblzfse.decompress(
        (DATA / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    capture = cv2.VideoCapture(str(DATA / "compressed_video_h264.mp4"))
    results = []
    for index in FRAMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"RGB frame {index} missing")
        mask = red_cup_mask(frame)
        y_values, x_values = np.where(mask)
        y_min, y_max = int(y_values.min()), int(y_values.max())
        offsets_left, offsets_right = [], []
        strengths_left, strengths_right = [], []
        side_rows = []
        for y in range(y_min + 8, y_max - 8, 3):
            xs = np.flatnonzero(mask[y])
            if len(xs) < 18:
                continue
            left, right = int(xs[0]), int(xs[-1])
            yd = int(np.clip(round((y + .5) * .75 - .5), 2, 189))
            row = cv2.GaussianBlur(depth[index, yd:yd + 1], (5, 1), 0)[0]
            gradient = np.roll(row, -2) - np.roll(row, 2)
            side_rows.append((left, right, gradient))
            for edge, sign, offsets, strengths in (
                (left, -1, offsets_left, strengths_left),
                (right, 1, offsets_right, strengths_right),
            ):
                lo, hi = max(edge - 18, 3), min(edge + 19, 253)
                window = sign * gradient[lo:hi]
                peak = int(np.argmax(window)) + lo
                offsets.append(peak - edge)
                strengths.append(float(window[peak - lo]))
        shifts = {}
        for shift in range(-15, 16):
            scores = [-gradient[left + shift] + gradient[right + shift]
                      for left, right, gradient in side_rows
                      if 3 <= left + shift < 253 and 3 <= right + shift < 253]
            shifts[shift] = float(np.mean(scores)) if scores else float("-inf")
        best_shift = max(shifts, key=shifts.get)
        result = {"frame": index, "rgb_cup_bbox": [int(x_values.min()), y_min,
                                                      int(x_values.max()), y_max],
                  "rows": len(offsets_left),
                  "best_horizontal_shift_px": best_shift,
                  "edge_score_at_zero": shifts[0],
                  "edge_score_at_plus_10": shifts[10],
                  "left_edge_median_offset_px": float(np.median(offsets_left)),
                  "right_edge_median_offset_px": float(np.median(offsets_right)),
                  "left_edge_p90_abs_offset_px": float(np.percentile(np.abs(offsets_left), 90)),
                  "right_edge_p90_abs_offset_px": float(np.percentile(np.abs(offsets_right), 90)),
                  "left_edge_median_depth_step": float(np.median(strengths_left)),
                  "right_edge_median_depth_step": float(np.median(strengths_right))}
        results.append(result)
    capture.release()
    output = {"mapping_hypothesis": "u_depth=u_rgb, v_depth=(v_rgb+.5)*192/256-.5",
              "measurement": "signed horizontal depth-gradient peak within 18 px of RGB red side edge",
              "frames": results,
              "warning": "Cup mask and nearby unrelated depth edges can bias this diagnostic."}
    (OUT / "dobbe_3651_cup_rgb_depth_alignment.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
