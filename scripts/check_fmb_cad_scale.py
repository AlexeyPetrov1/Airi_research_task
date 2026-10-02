"""Compare the known 150 mm FMB peg with its RGB size and sensor depth."""

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"
STEPS = [0, 5, 10, 20, 37, 74, 110, 120, 125, 130, 135, 140, 145]
FY_256_CANDIDATE = 380.209 * 256 / 480


def main() -> None:
    source = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    rows = []
    for step in STEPS:
        bgr = source["obs/side_1"][step]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = ((cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0) |
                (cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)) > 0))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"))
        components = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                      if stats[i, cv2.CC_STAT_AREA] >= 30]
        if not components:
            continue
        _, i = max(components)
        x, y, w, h, area = map(int, stats[i])
        eroded = cv2.erode((labels == i).astype("uint8"), np.ones((5, 5), "uint8")) > 0
        d = source["obs/side_1_depth"][step][eroded]
        d = d[d > 0]
        rows.append({"step": step, "rgb_bbox_xywh": [x, y, w, h],
                     "rgb_area_px": area, "sensor_depth_interior_median_m":
                     float(np.median(d) * 0.0001) if len(d) else None,
                     "depth_valid_interior": int(len(d)),
                     "naive_height_depth_m": FY_256_CANDIDATE * .150 / h,
                     "caveat": "Height formula assumes the peg axis is parallel to the image plane."})
    result = {"official_cad_long_peg_height_m": .150,
              "K_rgb_candidate_fy_px": FY_256_CANDIDATE,
              "sensor_scale_m_per_count_inferred": .0001,
              "frames": rows}
    (OUT / "cad_scale_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for r in rows:
        print(r["step"], r["rgb_bbox_xywh"],
              round(r["naive_height_depth_m"], 3),
              r["sensor_depth_interior_median_m"])


if __name__ == "__main__":
    main()
