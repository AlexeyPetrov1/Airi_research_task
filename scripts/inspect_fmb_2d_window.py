"""Print red-object visibility and depth statistics for the quantitative FMB window."""

from __future__ import annotations

import cv2
import numpy as np

from probe_fmb_cad_pnp import SOURCE


def red_component(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int, int]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) |
            cv2.inRange(hsv, (170, 75, 90), (179, 255, 255))) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
    index = max(range(1, count), key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return labels == index, tuple(map(int, stats[index]))


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    for step in (90, 100, 110, 115, 120, 123, 125, 126, 127, 130, 135, 140, 145, 147):
        mask, box = red_component(data["obs/side_1"][step])
        raw = data["obs/side_1_depth"][step][mask]
        valid = raw[raw > 0]
        print(step, str(data["primitive"][step]), "xywha", box,
              "valid_fraction", round(len(valid) / box[4], 3),
              "median_raw", float(np.median(valid)) if len(valid) else None)


if __name__ == "__main__":
    main()
