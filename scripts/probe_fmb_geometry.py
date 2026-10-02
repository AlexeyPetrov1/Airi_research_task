"""Independent physical checks for the FMB side_1 raw depth units.

The chosen red peg is rigidly held after grasp. Its lowest visible interior
pixels give a repeatable physical tip during the board approach. This probe
compares their depth-derived motion to the robot's metric TCP positions.
It is diagnostic only: it does not produce MolmoMotion inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_fmb_episode_5201"
SOURCE = OUT / "1_M_L_3_vertical_n_2.npy"


def similarity_fit(camera_raw: np.ndarray, tcp_m: np.ndarray) -> dict:
    """Least-squares rigid camera-to-robot fit with a positive global scale."""
    x_center = camera_raw.mean(axis=0)
    y_center = tcp_m.mean(axis=0)
    x = camera_raw - x_center
    y = tcp_m - y_center
    u, _, vt = np.linalg.svd(x.T @ y)
    sign = np.eye(3)
    sign[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ sign @ vt
    denom = float(np.sum(x * x))
    scale = float(np.sum((x @ rotation) * y) / denom)
    predicted = scale * x @ rotation + y_center
    residual = np.linalg.norm(predicted - tcp_m, axis=1)
    return {
        "estimated_m_per_raw_unit": scale,
        "residual_median_m": float(np.median(residual)),
        "residual_p95_m": float(np.percentile(residual, 95)),
        "residual_max_m": float(residual.max()),
        "sample_count": len(x),
        "tcp_span_m": float(np.linalg.norm(np.ptp(tcp_m, axis=0))),
        "camera_raw_span": float(np.linalg.norm(np.ptp(camera_raw, axis=0))),
        "rotation": rotation.tolist(),
    }


def fixed_scale_residual(camera_raw: np.ndarray, tcp_m: np.ndarray,
                         scale: float, rotation: np.ndarray) -> dict:
    x = camera_raw - camera_raw.mean(axis=0)
    y = tcp_m - tcp_m.mean(axis=0)
    residual = np.linalg.norm(scale * x @ rotation - y, axis=1)
    return {"median_m": float(np.median(residual)),
            "p95_m": float(np.percentile(residual, 95)),
            "rmse_m": float(np.sqrt(np.mean(residual ** 2)))}


def main() -> None:
    raw = np.load(SOURCE, allow_pickle=True).item()
    rgb_bgr = raw["obs/side_1"]
    depth = raw["obs/side_1_depth"]
    tcp = raw["obs/tcp_pose"][:, :3]
    calibration = json.loads((OUT / "side_1_intrinsics_official").read_text(encoding="utf8"))
    fx = float(calibration["rectified.2.fx"]) * 256 / 640
    fy = float(calibration["rectified.2.fy"]) * 256 / 480
    cx = (float(calibration["rectified.2.ppx"]) + .5) * 256 / 640 - .5
    cy = (float(calibration["rectified.2.ppy"]) + .5) * 256 / 480 - .5

    samples = []
    panels = []
    for step in range(106, 141, 2):
        rgb = cv2.cvtColor(rgb_bgr[step], cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(rgb_bgr[step], cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 75, 90), (13, 255, 255)) > 0
        # The bottom of the red peg is the tracked tip; use interior pixels
        # above the boundary to avoid mixed-pixel depth and the blue board.
        yy, xx = np.where(mask)
        if len(xx) < 100:
            continue
        y_bottom = int(yy.max())
        inner = mask & (np.indices(mask.shape)[0] >= y_bottom - 12) & (np.indices(mask.shape)[0] <= y_bottom - 5)
        yi, xi = np.where(inner)
        valid = depth[step, yi, xi] > 0
        if int(valid.sum()) < 10:
            continue
        xi, yi = xi[valid], yi[valid]
        # Median raw Z is robust to isolated edge/bad-depth pixels.
        z = float(np.median(depth[step, yi, xi]))
        u, v = float(np.median(xi)), float(np.median(yi))
        p = [z * (u - cx) / fx, z * (v - cy) / fy, z]
        samples.append({"step": step, "pixel_uv": [u, v], "raw_z": z,
                        "camera_xyz_raw": p, "tcp_xyz_m": tcp[step].tolist(),
                        "interior_pixel_count": int(valid.sum()), "bottom_y": y_bottom})
        if step in (106, 112, 120, 128, 136, 140):
            im = Image.fromarray(rgb)
            dr = ImageDraw.Draw(im)
            dr.ellipse((u - 4, v - 4, u + 4, v + 4), outline="yellow", width=2)
            dr.text((5, 5), f"step {step} Zraw={z:.0f}", fill="yellow")
            panels.append(im)

    point_raw = np.array([r["camera_xyz_raw"] for r in samples])
    tcp_m = np.array([r["tcp_xyz_m"] for r in samples])
    all_fit = similarity_fit(point_raw, tcp_m)
    # Inspecting the diagnostic image showed that step 106 samples background
    # Z (4704 raw units) at the tip. The contiguous approach from 112 to 140
    # keeps the tip visible and depth around 2394..2682 raw units.
    stable = [row for row in samples if 112 <= row["step"] <= 140]
    stable_raw = np.array([r["camera_xyz_raw"] for r in stable])
    stable_tcp = np.array([r["tcp_xyz_m"] for r in stable])
    fit = similarity_fit(stable_raw, stable_tcp)
    rotation = np.array(fit["rotation"])
    # An independent constraint is the D405's documented 7-50 cm ideal range.
    positive = depth[0][depth[0] > 0]
    candidates = []
    for scale in (0.01, 0.001, 0.0001, 0.00005):
        nominal = positive * scale
        candidates.append({
            "scale_m_per_unit": scale,
            "frame0_median_depth_m": float(np.median(nominal)),
            "frame0_p05_depth_m": float(np.percentile(nominal, 5)),
            "frame0_p95_depth_m": float(np.percentile(nominal, 95)),
            "frame0_fraction_in_d405_ideal_7_to_50_cm": float(np.mean((nominal >= .07) & (nominal <= .5))),
            "tcp_motion_fixed_scale_residual": fixed_scale_residual(stable_raw, stable_tcp, scale, rotation),
            "ratio_to_tcp_motion_estimate": float(scale / fit["estimated_m_per_raw_unit"]),
        })
    result = {"method": "bottom interior pixels of rigid red peg compared with metric robot TCP trajectory",
              "intrinsics_status": "CANDIDATE_FULL_FRAME_RESIZE_NOT_YET_VALIDATED",
              "K_256_candidate": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
              "fit": fit, "all_samples_fit": all_fit,
              "stable_window": [112, 140],
              "stable_window_subfits": {f"{lo}_140": similarity_fit(
                  np.array([r["camera_xyz_raw"] for r in stable if r["step"] >= lo]),
                  np.array([r["tcp_xyz_m"] for r in stable if r["step"] >= lo]))
                  for lo in (118, 122, 128)},
              "candidates": candidates, "samples": samples,
              "limitations": ["Tip extraction is automated and may move across physical surface as orientation changes.",
                              "TCP is the gripper tool center, not the peg tip; object orientation changes slightly.",
                              "Depth-derived 3D uses candidate full-frame K resize; scale inference depends on it."]}
    (OUT / "geometry_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    if panels:
        sheet = Image.new("RGB", (len(panels) * 256, 256), "white")
        for i, panel in enumerate(panels):
            sheet.paste(panel, (i * 256, 0))
        sheet.save(OUT / "peg_tip_motion_probe.png")
    print(json.dumps({"fit": fit, "candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
