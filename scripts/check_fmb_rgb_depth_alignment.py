"""Check RGB/depth boundary agreement for the red peg in episode_5201.

This is a pixel-grid diagnostic, not proof of an exact intrinsic matrix. It
uses several positions of one readily segmented object and rejects rows with
little foreground/background depth contrast.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_fmb_episode_5201"
STEPS = (0, 37, 74, 110, 120, 130, 140)


def median_finite(values: np.ndarray) -> float | None:
    valid = values[np.isfinite(values)]
    return float(np.median(valid)) if len(valid) else None


def crossing(profile: np.ndarray, edge_rgb: float, threshold: float,
             direction: str) -> float | None:
    lo = max(0, int(edge_rgb) - 12)
    hi = min(254, int(edge_rgb) + 12)
    if direction == "left":
        candidates = [u + .5 for u in range(lo, hi + 1)
                      if profile[u] > threshold >= profile[u + 1]]
    else:
        candidates = [u + .5 for u in range(lo, hi + 1)
                      if profile[u] <= threshold < profile[u + 1]]
    return min(candidates, key=lambda p: abs(p - edge_rgb)) if candidates else None


def main() -> None:
    source = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    frames = []
    overlays = []
    for step in STEPS:
        bgr = source["obs/side_1"][step]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = source["obs/side_1_depth"][step]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        peg = ((cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0) |
               (cv2.inRange(hsv, (170, 75, 90), (179, 255, 255)) > 0))
        candidate_rows = np.where(peg.sum(axis=1) > 10)[0]
        candidate_rows = candidate_rows[(candidate_rows >= 12) & (candidate_rows <= 170)]
        if not len(candidate_rows):
            raise ValueError(f"No red peg rows at step {step}")
        ylo, yhi = np.percentile(candidate_rows, [20, 80]).astype(int)
        rows = sorted(set(np.linspace(ylo, yhi, 7, dtype=int).tolist()))
        smoothed = cv2.medianBlur(depth, 5).astype(np.float64)
        row_results = []
        overlay = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        for y in rows:
            red_x = np.where(peg[y])[0]
            if len(red_x) < 8:
                continue
            left_rgb, right_rgb = int(red_x.min()), int(red_x.max())
            if right_rgb - left_rgb > 60 or right_rgb - left_rgb < 8:
                continue
            profile = smoothed[y]
            valid = profile > 0
            if int(valid.sum()) < 100:
                continue
            profile = np.interp(np.arange(256), np.where(valid)[0], profile[valid])
            profile = cv2.GaussianBlur(profile[None, :], (5, 1), .8)[0]
            inside = profile[left_rgb + 3:right_rgb - 2]
            if len(inside) < 4:
                continue
            foreground = float(np.median(inside))
            rgb_boundaries = {"left": left_rgb - .5, "right": right_rgb + .5}
            backgrounds = {
                "left": (max(0, left_rgb - 15), max(0, left_rgb - 5)),
                "right": (min(256, right_rgb + 6), min(256, right_rgb + 16)),
            }
            sides = {}
            for side in ("left", "right"):
                lo, hi = backgrounds[side]
                if hi - lo < 4:
                    continue
                background = float(np.median(profile[lo:hi]))
                contrast = background - foreground
                if contrast < 300:
                    continue
                edge = rgb_boundaries[side]
                d_edge = crossing(profile, edge, (foreground + background) / 2, side)
                if d_edge is None:
                    continue
                sides[side] = {
                    "rgb_boundary_x": edge,
                    "depth_boundary_x": d_edge,
                    "depth_minus_rgb_px": d_edge - edge,
                    "foreground_raw": foreground,
                    "background_raw": background,
                    "contrast_raw": contrast,
                    "zero_fraction_near_edge": float(np.mean(depth[y, max(0, int(edge)-5):min(256, int(edge)+6)] == 0)),
                }
                draw.line((edge, y - 3, edge, y + 3), fill="lime", width=2)
                draw.line((d_edge, y - 3, d_edge, y + 3), fill="magenta", width=2)
            if sides:
                center_u = (left_rgb + right_rgb) // 2
                center_z = int(depth[y, center_u])
                background_nearest = min(side_result["background_raw"] for side_result in sides.values())
                row_results.append({"y": y, "rgb_peg_x": [left_rgb, right_rgb],
                                    "rgb_interior_center_x": center_u,
                                    "raw_depth_at_center": center_z,
                                    "center_depth_valid": center_z > 0,
                                    "center_depth_nearer_than_background": bool(
                                        center_z > 0 and center_z < background_nearest - 300),
                                    "sides": sides})

        pairs = [r for r in row_results if "left" in r["sides"] and "right" in r["sides"]]
        center_offsets = np.array([
            (r["sides"]["left"]["depth_boundary_x"] + r["sides"]["right"]["depth_boundary_x"]
             - r["sides"]["left"]["rgb_boundary_x"] - r["sides"]["right"]["rgb_boundary_x"]) / 2
            for r in pairs], dtype=float)
        width_excess = np.array([
            r["sides"]["right"]["depth_boundary_x"] - r["sides"]["left"]["depth_boundary_x"]
            - (r["sides"]["right"]["rgb_boundary_x"] - r["sides"]["left"]["rgb_boundary_x"])
            for r in pairs], dtype=float)
        frame = {
            "step": step,
            "candidate_rows": rows,
            "rows_with_both_edges": len(pairs),
            "median_left_offset_px": median_finite(np.array([r["sides"]["left"]["depth_minus_rgb_px"] for r in pairs])),
            "median_right_offset_px": median_finite(np.array([r["sides"]["right"]["depth_minus_rgb_px"] for r in pairs])),
            "median_center_offset_px": median_finite(center_offsets),
            "median_depth_silhouette_width_excess_px": median_finite(width_excess),
            "red_pixel_depth_valid_fraction": float(np.mean(depth[peg] > 0)),
            "sampled_rgb_interior_centers": len(row_results),
            "sampled_centers_with_valid_depth": sum(r["center_depth_valid"] for r in row_results),
            "sampled_centers_nearer_than_background": sum(r["center_depth_nearer_than_background"] for r in row_results),
            "rows": row_results,
        }
        frames.append(frame)
        draw.rectangle((0, 0, 95, 15), fill="black")
        draw.text((4, 2), f"step {step}", fill="white")
        overlays.append(overlay)

    # A second, wide fixed object checks for a camera-wide translation. Its
    # blue color mask covers the visible face; depth edges may include dark
    # side walls, so width disagreement is expected and is not calibration.
    board_control = []
    for step in STEPS[:5]:
        bgr = source["obs/side_1"][step]
        depth = source["obs/side_1_depth"][step]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        blue = cv2.inRange(hsv, (85, 40, 40), (140, 255, 255)) > 0
        filtered = cv2.medianBlur(depth, 3)
        for y in (110, 125, 140, 155, 170):
            rgb_x = np.where(blue[y])[0]
            if len(rgb_x) < 50:
                continue
            left_rgb, right_rgb = int(rgb_x.min()), int(rgb_x.max())
            if left_rgb < 40 or right_rgb > 240:
                continue
            profile = filtered[y].astype(float)
            edges = []
            for rgb_boundary in (left_rgb - .5, right_rgb + .5):
                lo = max(0, int(rgb_boundary) - 12)
                hi = min(254, int(rgb_boundary) + 12)
                depth_boundary = max(range(lo, hi + 1),
                                     key=lambda u: abs(profile[u + 1] - profile[u])) + .5
                edges.append(depth_boundary)
            board_control.append({
                "step": step, "row_y": y,
                "rgb_edge_x": [left_rgb - .5, right_rgb + .5],
                "depth_edge_x": edges,
                "center_offset_px": (edges[0] + edges[1] - left_rgb - right_rgb) / 2,
                "depth_width_excess_px": edges[1] - edges[0] - (right_rgb - left_rgb + 1),
            })

    summary = {
        "source_file": "1_M_L_3_vertical_n_2.npy",
        "camera": "side_1",
        "steps": list(STEPS),
        "object": "red peg, color mask in source BGR converted to HSV",
        "edge_method": "For selected rows, compare RGB color-mask boundaries with mid-contrast crossings in median-smoothed raw depth; positive offset means depth edge lies right of RGB edge.",
        "depth_zero_handling": "Zeros interpolated along row solely for edge detection; raw zero fraction at each edge is recorded.",
        "frames": frames,
        "all_frame_median_center_offset_px": median_finite(np.array([f["median_center_offset_px"] for f in frames if f["median_center_offset_px"] is not None])),
        "all_frame_median_width_excess_px": median_finite(np.array([f["median_depth_silhouette_width_excess_px"] for f in frames if f["median_depth_silhouette_width_excess_px"] is not None])),
        "interior_centers_total": sum(f["sampled_rgb_interior_centers"] for f in frames),
        "interior_centers_valid": sum(f["sampled_centers_with_valid_depth"] for f in frames),
        "interior_centers_nearer_than_background": sum(f["sampled_centers_nearer_than_background"] for f in frames),
        "fixed_board_control": {
            "method": "Blue face extent versus strongest local depth discontinuity on five rows of each of the first five sampled frames; board sidewalls can widen depth footprint.",
            "rows": board_control,
            "median_center_offset_px": median_finite(np.array([r["center_offset_px"] for r in board_control])),
            "median_width_excess_px": median_finite(np.array([r["depth_width_excess_px"] for r in board_control])),
        },
        "interpretation": "A silhouette center near the RGB center supports approximate shared pixel geometry. Edge disagreement and invalid depth do not verify per-pixel correspondence or the exact 256x256 intrinsic matrix.",
    }
    (OUT / "rgb_depth_alignment_check.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    sheet = Image.new("RGB", (4 * 256, 2 * 256), "white")
    for i, panel in enumerate(overlays):
        sheet.paste(panel, ((i % 4) * 256, (i // 4) * 256))
    sheet.save(OUT / "rgb_depth_alignment_edges.png")
    print(json.dumps({"steps": list(STEPS),
                      "frames": [{k: v for k, v in f.items() if k in (
                          "step", "rows_with_both_edges", "median_left_offset_px",
                          "median_right_offset_px", "median_center_offset_px",
                          "median_depth_silhouette_width_excess_px",
                          "red_pixel_depth_valid_fraction",
                          "sampled_rgb_interior_centers", "sampled_centers_with_valid_depth",
                          "sampled_centers_nearer_than_background")}
                          for f in frames],
                      "median_center_offset_px": summary["all_frame_median_center_offset_px"],
                      "median_width_excess_px": summary["all_frame_median_width_excess_px"],
                      "interior_centers_total": summary["interior_centers_total"],
                      "interior_centers_valid": summary["interior_centers_valid"],
                      "interior_centers_nearer_than_background": summary["interior_centers_nearer_than_background"],
                      "board_control_median_center_offset_px": summary["fixed_board_control"]["median_center_offset_px"],
                      "board_control_median_width_excess_px": summary["fixed_board_control"]["median_width_excess_px"]}, indent=2))


if __name__ == "__main__":
    main()
