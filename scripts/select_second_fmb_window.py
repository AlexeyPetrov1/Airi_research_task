"""Search pre-inference candidate windows/inner points for second FMB trial."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component
from probe_fmb_future_tracking import track


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy"
OUT = ROOT / "runs/fmb_second_example_1_M_L_3_vertical_n_3"


def candidates(data: dict, t0: int) -> dict:
    frames, depth = data["obs/side_1"], data["obs/side_1_depth"]
    red0, box = red_component(frames[t0])
    rows = range(max(20, box[1]+18), min(125, box[1]+box[3]-12), 8)
    grid = []
    for y in rows:
        xs = np.flatnonzero(red0[y])
        if len(xs) < 12:
            continue
        left, right = float(xs.min()), float(xs.max())
        for alpha in (.2, .3, .4, .5, .6, .7, .8):
            grid.append({"row": y, "alpha": alpha,
                         "uv_t0": [left+alpha*(right-left), float(y)]})
    uv = np.asarray([entry["uv_t0"] for entry in grid], dtype="float32")
    tracked = track(frames, t0, t0-2, uv)
    good = []
    for j, entry in enumerate(grid):
        per_frame = []
        for step in (t0-2, t0-1, t0):
            u, v = tracked[step][j]
            if not np.isfinite([u, v]).all():
                per_frame.append({"step": step, "valid": False})
                continue
            x, y = np.rint([u, v]).astype(int)
            red, _ = red_component(frames[step])
            if not (2 <= x < 254 and 2 <= y < 254):
                per_frame.append({"step": step, "valid": False})
                continue
            distance = cv2.distanceTransform(red.astype("uint8"), cv2.DIST_L2, 3)[y, x]
            raw = depth[step, y-2:y+3, x-2:x+3]
            valid = raw[raw > 0]
            per_frame.append({"step": step, "uv": [float(u), float(v)],
                              "red": bool(red[y, x]), "red_boundary_distance_px": float(distance),
                              "valid_count": len(valid),
                              "median_raw": float(np.median(valid)) if len(valid) else None,
                              "mad_raw": float(np.median(np.abs(valid-np.median(valid))))
                              if len(valid) else None})
        entry["history"] = per_frame
        if all(f.get("red") and f.get("red_boundary_distance_px", 0) >= 2.5 and
               f.get("valid_count", 0) >= 15 for f in per_frame):
            good.append(entry)
    tcp = data["obs/tcp_pose"]
    return {"t0": t0, "future_20_available": t0+20 < len(frames),
            "primitive_at_t0": str(data["primitive"][t0]),
            "primitive_constant_through_future": len(set(map(str, data["primitive"][t0:t0+21]))) == 1,
            "red_bbox_at_t0": box, "candidate_count": len(grid), "valid_candidate_count": len(good),
            "tcp_motion_20_steps_mm": float(np.linalg.norm(tcp[t0+20, :3]-tcp[t0, :3])*1000),
            "valid_candidates": good}


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    result = {"source": SOURCE.name, "selection_before_model_inference": True,
              "candidates": [candidates(data, t0) for t0 in range(122, 136)]}
    (OUT / "window_point_search.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print([(row["t0"], row["valid_candidate_count"],
            round(row["tcp_motion_20_steps_mm"],1),
            row["primitive_constant_through_future"])
           for row in result["candidates"]])


if __name__ == "__main__":
    main()
