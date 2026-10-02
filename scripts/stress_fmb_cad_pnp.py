"""Probe local sensitivity of the candidate three-frame planar PnP fit.

Independent Gaussian pixel jitter is a sensitivity probe, not a calibrated
annotation-error distribution. A focal sweep is likewise hypothetical because
the active color intrinsics are unknown. Sensor depth is never used in fitting.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import ANNOTATIONS, EIGHT, FRONT, K, OUT, SOURCE


def quantiles(a: np.ndarray) -> dict[str, float]:
    return {key: float(val) for key, val in zip(
        ("q05", "median", "q95"), np.quantile(a, (.05, .5, .95)))}


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    steps = list(ANNOTATIONS)
    corners = np.asarray([ANNOTATIONS[s] for s in steps], dtype=np.float64)
    tcp = source["obs/tcp_pose"]
    tcp_distance = float(np.linalg.norm(tcp[140, :3] - tcp[130, :3]))
    prior = json.loads((OUT / "cad_pnp_probe.json").read_text(encoding="utf8"))
    joint = prior["joint_constant_orientation_fit"]
    rv0 = Rotation.from_matrix(joint["shared_R"]).as_rotvec()
    t0 = np.asarray([f["t_m"] for f in joint["frames"]]).ravel()
    x0 = np.r_[rv0, t0]
    rng = np.random.default_rng(5201)
    records = []

    for focal_factor in (.9, 1.0, 1.1):
        k = K.copy()
        k[0, 0] *= focal_factor
        k[1, 1] *= focal_factor

        def residual(x: np.ndarray, observed: np.ndarray) -> np.ndarray:
            return np.concatenate([
                (cv2.projectPoints(FRONT, x[:3], x[3+3*i:6+3*i], k, None)[0]
                 .reshape(4, 2) - observed[i]).ravel()
                for i in range(len(steps))])

        for jitter_sigma_px, count in ((0.0, 1), (1.0, 200), (2.0, 200)):
            rows = []
            for _ in range(count):
                observed = corners + rng.normal(0, jitter_sigma_px, corners.shape)
                fit = least_squares(residual, x0, args=(observed,), method="lm")
                rot = Rotation.from_rotvec(fit.x[:3]).as_matrix()
                trans = fit.x[3:].reshape(len(steps), 3)
                points = np.asarray([(rot @ EIGHT.T).T + t for t in trans])
                if not fit.success or np.any(points[:, :, 2] <= 0):
                    continue
                centers = points.mean(axis=1)
                rows.append({
                    "median_point_Z_m": float(np.median(points[:, :, 2])),
                    "center_130_to_140_displacement_m": float(
                        np.linalg.norm(centers[2] - centers[0])),
                    "corner_rmse_px": float(np.sqrt(np.mean(residual(fit.x, observed)**2))),
                })
            if not rows:
                raise RuntimeError(f"No valid fits at factor={focal_factor}, sigma={jitter_sigma_px}")
            records.append({
                "focal_factor": focal_factor,
                "independent_corner_jitter_sigma_px": jitter_sigma_px,
                "attempted": count,
                "valid_positive_Z": len(rows),
                "median_point_Z_m": quantiles(np.asarray([r["median_point_Z_m"] for r in rows])),
                "center_130_to_140_displacement_m": quantiles(np.asarray([
                    r["center_130_to_140_displacement_m"] for r in rows])),
                "difference_from_tcp_displacement_m": quantiles(np.asarray([
                    r["center_130_to_140_displacement_m"] - tcp_distance for r in rows])),
                "corner_rmse_px": quantiles(np.asarray([r["corner_rmse_px"] for r in rows])),
            })

    result = {
        "status": "LOCAL_SENSITIVITY_ONLY_NOT_INTRINSIC_CALIBRATION",
        "source": "manual planar RGB corners at FMB steps 130,135,140",
        "focal_change": "fx and fy scaled jointly; cx and cy fixed",
        "jitter": "independent zero-mean Gaussian noise per annotated coordinate",
        "tcp_130_to_140_displacement_m": tcp_distance,
        "sensor_depth_used_in_fit": False,
        "limitations": [
            "Pixel jitter does not model systematic corner identification error or occlusion.",
            "A good TCP displacement match alone does not identify the color intrinsic matrix or absolute Z.",
            "The four landmarks on each frame are coplanar.",
        ],
        "records": records,
    }
    target = OUT / "cad_pnp_local_sensitivity.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
