"""Independent Z16 check of the RGB-only horizontal-peg silhouette fit.

Candidate K and object poses are frozen from the RGB-only probe. Dense CAD
surface samples are z-buffered, then compared at interior color-mask pixels.
The check is diagnostic because published RGB/depth alignment is approximate.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import D, H, K as K_NOMINAL, W


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
SOURCE = ROOT / "data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy"
RADIUS = .00288


def surface_points() -> np.ndarray:
    faces = []
    x = np.linspace(-W/2+RADIUS, W/2-RADIUS, 50)
    y = np.linspace(-D/2+RADIUS, D/2-RADIUS, 40)
    z = np.linspace(0., H, 240)
    for yy in (-D/2, D/2):
        xx, zz = np.meshgrid(x, z)
        faces.append(np.column_stack((xx.ravel(), np.full(xx.size, yy), zz.ravel())))
    for xx in (-W/2, W/2):
        yy, zz = np.meshgrid(y, z)
        faces.append(np.column_stack((np.full(yy.size, xx), yy.ravel(), zz.ravel())))
    for zz in (0., H):
        xx, yy = np.meshgrid(x, y)
        faces.append(np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, zz))))
    return np.vstack(faces).astype(np.float64)


def check(frame: np.ndarray, depth: np.ndarray, k: np.ndarray,
          pose: np.ndarray) -> dict:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (cv2.inRange(hsv, (0, 60, 90), (25, 255, 255)) |
           cv2.inRange(hsv, (170, 60, 90), (179, 255, 255)))
    red = cv2.erode(red, np.ones((3, 3), np.uint8)) > 0
    model = surface_points()
    camera = Rotation.from_rotvec(pose[:3]).apply(model) + pose[3:]
    uv = cv2.projectPoints(model, pose[:3], pose[3:], k, None)[0].reshape(-1, 2)
    pixel = np.rint(uv).astype(int)
    valid = ((camera[:, 2] > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < 256) &
             (pixel[:, 1] >= 0) & (pixel[:, 1] < 256))
    pixel, camera = pixel[valid], camera[valid]
    zbuffer = np.full((256, 256), np.inf)
    np.minimum.at(zbuffer, (pixel[:, 1], pixel[:, 0]), camera[:, 2])
    selected = red & (depth > 0) & np.isfinite(zbuffer)
    error_mm = (zbuffer[selected]-depth[selected]*.0001)*1000
    return {"interior_comparable_pixels": int(np.count_nonzero(selected)),
            "median_abs_Z_error_mm": float(np.median(np.abs(error_mm))),
            "p90_abs_Z_error_mm": float(np.percentile(np.abs(error_mm), 90)),
            "median_signed_Z_error_mm": float(np.median(error_mm)),
            "mean_signed_Z_error_mm": float(np.mean(error_mm)),
            "CAD_Z_median_m": float(np.median(zbuffer[selected])),
            "sensor_Z_median_m": float(np.median(depth[selected])*.0001)}


def main() -> None:
    probe = json.loads((OUT / "cross_orientation_silhouette_k_probe.json").read_text())
    source = np.load(SOURCE, allow_pickle=True).item()
    frame, depth = source["obs/side_1"][0], source["obs/side_1_depth"][0]
    kbest = np.asarray(probe["best_fit"]["K"])
    pbest = np.asarray(probe["best_fit"]["poses"]["h0"])
    pnom = np.asarray(probe["horizontal_pose_seed_best_three"][0]["pose"])
    result = {"source_episode": SOURCE.name, "frame": 0,
              "depth_scale_m_per_raw": .0001,
              "candidate_RGB_only": {"K": kbest.tolist(), "pose": pbest.tolist(),
                                     **check(frame, depth, kbest, pbest)},
              "nominal_assumed_K_RGB_pose": {
                  "K": K_NOMINAL.tolist(), "pose": pnom.tolist(),
                  **check(frame, depth, K_NOMINAL, pnom)},
              "warning": "Metric comparison is independent of RGB fit, but RGB/depth registration and Z16 scale were not recorded in the episode."}
    (OUT / "cross_orientation_sensor_depth_check.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: {name: value for name, value in row.items()
                            if name not in ("K", "pose")}
                      for key, row in result.items() if isinstance(row, dict)}, indent=2))


if __name__ == "__main__":
    main()
