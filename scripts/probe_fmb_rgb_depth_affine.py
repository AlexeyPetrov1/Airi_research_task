"""Cross-validate a restricted RGB-to-depth affine from physical silhouettes.

This is a diagnostic: RGB color boundaries and depth discontinuities need not
be the same physical contour, especially at occlusions and board sidewalls.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "fmb_effective_k_256_calibration"
EP = ROOT / "runs" / "sharerobot_fmb_episode_5201"


def fit(pairs: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(pairs[:, 0], pairs[:, 1], 1)
    return float(slope), float(intercept)


def error(pairs: np.ndarray, slope: float, intercept: float) -> dict:
    if len(pairs) == 0:
        return {"n": 0, "median_abs_px": None, "p90_abs_px": None}
    residual = np.abs(pairs[:, 1] - (slope * pairs[:, 0] + intercept))
    return {"n": len(pairs), "median_abs_px": float(np.median(residual)),
            "p90_abs_px": float(np.percentile(residual, 90))}


def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return ((cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0)
            | (cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)) > 0))


def vertical_edges(bgr: np.ndarray, depth: np.ndarray) -> list[dict]:
    mask = red_mask(bgr)
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return []
    xlo, xhi = np.percentile(xs, [25, 75]).astype(int)
    output = []
    smoothed = cv2.GaussianBlur(cv2.medianBlur(depth, 5).astype(np.float32),
                                (3, 3), .7)
    for x in np.unique(np.linspace(xlo, xhi, 7, dtype=int)):
        red_y = np.flatnonzero(mask[:, x])
        if len(red_y) < 8:
            continue
        top, bottom = int(red_y.min()), int(red_y.max())
        if top < 15 or bottom > 233 or bottom - top < 9:
            continue
        profile = smoothed[:, x]
        inside = profile[top + 3:bottom - 2]
        if len(inside) < 4 or np.mean(inside > 0) < .8:
            continue
        foreground = float(np.median(inside[inside > 0]))
        for side, boundary, outside in (("top", top - .5, profile[max(0, top - 15):max(0, top - 5)]),
                                        ("bottom", bottom + .5, profile[min(256, bottom + 6):min(256, bottom + 16)])):
            outside = outside[outside > 0]
            if len(outside) < 4:
                continue
            background = float(np.median(outside))
            if background - foreground < 300:
                continue
            target = (background + foreground) / 2
            lo, hi = max(0, int(boundary) - 12), min(254, int(boundary) + 12)
            if side == "top":
                candidates = [y + .5 for y in range(lo, hi + 1)
                              if profile[y] > target >= profile[y + 1]]
            else:
                candidates = [y + .5 for y in range(lo, hi + 1)
                              if profile[y] <= target < profile[y + 1]]
            if candidates:
                output.append({"side": side, "rgb": boundary,
                               "depth": min(candidates, key=lambda y: abs(y - boundary)),
                               "x": int(x)})
    return output


def main() -> None:
    existing = json.loads((EP / "rgb_depth_alignment_check.json").read_text())
    source = np.load(EP / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    x_edges, x_centers, y_edges = [], [], []
    vertical_by_frame = {}
    for frame in existing["frames"]:
        t = frame["step"]
        for row in frame["rows"]:
            sides = row["sides"]
            for item in sides.values():
                x_edges.append((t, item["rgb_boundary_x"], item["depth_boundary_x"]))
            if "left" in sides and "right" in sides:
                x_centers.append((t, (sides["left"]["rgb_boundary_x"] + sides["right"]["rgb_boundary_x"]) / 2,
                                  (sides["left"]["depth_boundary_x"] + sides["right"]["depth_boundary_x"]) / 2))
        found = vertical_edges(source["obs/side_1"][t], source["obs/side_1_depth"][t])
        vertical_by_frame[str(t)] = found
        y_edges.extend((t, item["rgb"], item["depth"]) for item in found)
    x_edges, x_centers, y_edges = (np.asarray(v, dtype=float).reshape(-1, 3)
                                   for v in (x_edges, x_centers, y_edges))
    train = (0, 37, 74, 110, 120)
    holdout = (130, 140)

    def analyze(samples: np.ndarray) -> dict:
        tr = samples[np.isin(samples[:, 0], train), 1:]
        te = samples[np.isin(samples[:, 0], holdout), 1:]
        if len(tr) < 4:
            return {"train_count": len(tr), "holdout_count": len(te), "fit": None}
        slope, intercept = fit(tr)
        return {"train_count": len(tr), "holdout_count": len(te),
                "fit": {"scale": slope, "shift_px": intercept},
                "identity_train": error(tr, 1, 0), "affine_train": error(tr, slope, intercept),
                "identity_holdout": error(te, 1, 0), "affine_holdout": error(te, slope, intercept)}

    x_model = analyze(x_centers)
    y_model = analyze(y_edges)
    board = np.array([(r["step"], np.mean(r["rgb_edge_x"]),
                       np.mean(r["depth_edge_x"]))
                      for r in existing["fixed_board_control"]["rows"]], dtype=float)
    board_check = {}
    if len(board) and x_model["fit"]:
        board_check = {"identity": error(board[:, 1:], 1, 0),
                       "peg_center_affine": error(board[:, 1:], x_model["fit"]["scale"],
                                                  x_model["fit"]["shift_px"])}
    result = {"source": str(EP / "1_M_L_3_vertical_n_2.npy"),
              "train_steps": train, "holdout_steps": holdout,
              "x_peg_centers": x_model, "x_peg_edges": analyze(x_edges),
              "y_peg_edges": y_model, "board_independent_control": board_check,
              "vertical_edges_by_frame": vertical_by_frame,
              "caveat": "The red RGB boundary and depth discontinuity can be different physical contours; fitted values are not a validated pixel mapping."}
    path = OUT / "rgb_depth_affine_probe.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("x_peg_centers", "x_peg_edges", "y_peg_edges", "board_independent_control")}, indent=2))


if __name__ == "__main__":
    main()
