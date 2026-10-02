"""Exploratory point and visibility probe before fixing the quantitative run."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import SOURCE


OUT = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"
T0 = 126


def initial_points(frame: np.ndarray) -> np.ndarray:
    mask, _ = red_component(frame)
    points = []
    for y, alpha in ((35, .35), (35, .55), (45, .25), (45, .45),
                     (55, .35), (55, .75), (65, .25), (65, .55)):
        xs = np.flatnonzero(mask[y])
        left, right = float(xs.min()), float(xs.max())
        points.append([left + alpha * (right - left), float(y)])
    return np.asarray(points, dtype="float32")


def candidate_grid(frames: np.ndarray, depth_frames: np.ndarray) -> list[dict]:
    mask, _ = red_component(frames[T0])
    xy = []
    labels = []
    for y in (25, 35, 45, 55, 65, 75, 85):
        xs = np.flatnonzero(mask[y])
        left, right = float(xs.min()), float(xs.max())
        for alpha in (.25, .35, .45, .55, .65, .75):
            xy.append([left + alpha * (right - left), float(y)])
            labels.append((y, alpha))
    tracked = track(frames, T0, 124, np.asarray(xy, dtype="float32"))
    rows = []
    for j, (y0, alpha) in enumerate(labels):
        probe = {"y_at_t0": y0, "alpha": alpha, "uv_t0": xy[j], "frames": []}
        for step in (124, 125, 126):
            if not np.isfinite(tracked[step][j]).all():
                probe["frames"].append({"step": step, "uv": None,
                                        "red": False, "valid_count": 0,
                                        "median_raw": None, "mad_raw": None})
                continue
            u, v = np.rint(tracked[step][j]).astype(int)
            if not (2 <= u < 254 and 2 <= v < 254):
                probe["frames"].append({"step": step, "uv": [int(u), int(v)],
                                        "red": False, "valid_count": 0,
                                        "median_raw": None, "mad_raw": None})
                continue
            red, _ = red_component(frames[step])
            patch = depth_frames[step, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            probe["frames"].append({"step": step, "uv": [int(u), int(v)],
                                    "red": bool(red[v, u]),
                                    "valid_count": int(len(valid)),
                                    "median_raw": float(np.median(valid)) if len(valid) else None,
                                    "mad_raw": float(np.median(np.abs(valid-np.median(valid))))
                                    if len(valid) else None})
        rows.append(probe)
    return rows


def track(frames: np.ndarray, start: int, end: int, p0: np.ndarray) -> dict[int, np.ndarray]:
    result = {start: p0.copy()}
    position = p0.copy()
    direction = 1 if end > start else -1
    for step in range(start + direction, end + direction, direction):
        a = cv2.cvtColor(frames[step - direction], cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(frames[step], cv2.COLOR_BGR2GRAY)
        new, status, error = cv2.calcOpticalFlowPyrLK(a, b, position.reshape(-1, 1, 2), None,
                                                       winSize=(21, 21), maxLevel=2,
                                                       criteria=(cv2.TERM_CRITERIA_EPS |
                                                                 cv2.TERM_CRITERIA_COUNT, 30, .01))
        position = new.reshape(-1, 2)
        position[status.ravel() == 0] = np.nan
        result[step] = position.copy()
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True).item()
    frames = data["obs/side_1"]
    p0 = initial_points(frames[T0])
    grid = candidate_grid(frames, data["obs/side_1_depth"])
    (OUT / "history_point_candidates.json").write_text(json.dumps(grid, indent=2), encoding="utf8")
    for row in grid:
        if min(f["valid_count"] for f in row["frames"]) >= 15 and all(f["red"] for f in row["frames"]):
            print("CANDIDATE", row["y_at_t0"], row["alpha"],
                  "min_valid", min(f["valid_count"] for f in row["frames"]),
                  "max_mad", max(f["mad_raw"] for f in row["frames"]))
    positions = track(frames, T0, 124, p0) | track(frames, T0, 146, p0)
    checks = {}
    canvas = np.full((2 * 512, 4 * 512, 3), 255, np.uint8)
    show = (124, 125, 126, 130, 135, 140, 145, 146)
    for step in range(124, 147):
        mask, _ = red_component(frames[step])
        depth = data["obs/side_1_depth"][step]
        rows = []
        for j, (uf, vf) in enumerate(positions[step]):
            if not np.isfinite([uf, vf]).all():
                rows.append({"id": j, "uv": None, "red": False, "valid_count_5x5": 0})
                continue
            u, v = np.rint([uf, vf]).astype(int)
            valid = depth[v-2:v+3, u-2:u+3]
            valid = valid[valid > 0]
            rows.append({"id": j, "uv": [float(uf), float(vf)],
                         "red": bool(mask[v, u]) if 0 <= u < 256 and 0 <= v < 256 else False,
                         "valid_count_5x5": len(valid),
                         "median_raw": float(np.median(valid)) if len(valid) else None})
        checks[str(step)] = rows
        if step in show:
            tile = frames[step].copy()
            for row in rows:
                if row["uv"] is None:
                    continue
                u, v = np.rint(row["uv"]).astype(int)
                cv2.circle(tile, (u, v), 2, (0, 255, 0) if row["red"] else (0, 0, 0), -1)
                cv2.putText(tile, str(row["id"]), (u + 2, v - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.putText(tile, str(step), (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        .6, (255, 255, 255), 2)
            i = show.index(step)
            canvas[i//4*512:(i//4+1)*512, i%4*512:(i%4+1)*512] = cv2.resize(
                tile, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "tracking_probe.png"), canvas)
    (OUT / "tracking_probe.json").write_text(json.dumps({"t0": T0,
        "initial_2d": p0.tolist(), "checks": checks}, indent=2), encoding="utf8")
    for step in show:
        print(step, "red", sum(x["red"] for x in checks[str(step)]),
              "depth5x5", [x["valid_count_5x5"] for x in checks[str(step)]])


if __name__ == "__main__":
    main()
