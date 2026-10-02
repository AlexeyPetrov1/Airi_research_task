"""Annotate future peg surface coordinates from RGB only, blind to predictions.

The peg is nearly textureless. A sequential affine registration of its red mask
defines the primary image-plane correspondence; a direct registration repeat and
dense flow provide uncertainty diagnostics. Points close to mask boundaries or
with weak registration are excluded by one common, predeclared validity rule.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import SOURCE as DEFAULT_SOURCE
from probe_fmb_dense_flow import follow as dense_follow
from probe_fmb_mask_ecc import follow as ecc_follow, segment


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))
SOURCE = Path(os.environ.get("FMB_QUANT_SOURCE", DEFAULT_SOURCE))
T0 = int(os.environ.get("FMB_QUANT_T0", "126"))
FUTURE = tuple(range(T0+1, T0+21))
REPEAT_FRAMES = tuple(T0+offset for offset in (1, 6, 11, 16, 20))
REPEAT_POINTS = (0, 7)


def check_frozen() -> None:
    freeze = json.loads((RUN / "input_freeze.json").read_text(encoding="utf8"))
    for name, expected in freeze["sha256"].items():
        if hashlib.sha256((RUN / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen input changed: {name}")


def direct_registration(frame0: np.ndarray, frame1: np.ndarray, p0: np.ndarray) -> tuple[np.ndarray, dict]:
    reference = segment(frame0)
    target = segment(frame1)
    m0, box0 = red_component(frame0)
    m1, box1 = red_component(frame1)
    c0 = np.array(np.where(m0)).mean(axis=1)[::-1]
    c1 = np.array(np.where(m1)).mean(axis=1)[::-1]
    sx, sy = box1[2]/box0[2], box1[3]/box0[3]
    warp = np.float32([[sx, 0, c1[0]-sx*c0[0]], [0, sy, c1[1]-sy*c0[1]]])
    try:
        score, warp = cv2.findTransformECC(reference, target, warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                            200, 1e-6))
        diagnostic = {"success": True, "score": float(score), "warp": warp.tolist()}
    except cv2.error as error:
        diagnostic = {"success": False, "error": str(error), "warp": warp.tolist()}
    return (warp @ np.c_[p0, np.ones(len(p0))].T).T, diagnostic


def main() -> None:
    check_frozen()
    if (RUN / "gt_2d.npy").exists():
        raise FileExistsError("Future GT already frozen; refusing to overwrite")
    source = np.load(SOURCE, allow_pickle=True).item()
    frames = source["obs/side_1"]
    points = np.load(RUN / "points_2d_at_t0.npy").astype("float64")
    assert points.shape == (8, 2)
    tracked, registration = ecc_follow(frames, T0, T0+20, points)
    dense = dense_follow(frames, T0, T0+20, points)
    gt = np.stack([tracked[step] for step in FUTURE], axis=1)
    alt = np.stack([dense[step] for step in FUTURE], axis=1)
    reason = np.full((8, 20), "visible", dtype="<U16")
    mask = np.ones((8, 20), dtype=bool)
    diagnostics = []
    for ti, step in enumerate(FUTURE):
        red, _ = red_component(frames[step])
        distance = cv2.distanceTransform(red.astype("uint8"), cv2.DIST_L2, 3)
        score = registration[step].get("score", -1)
        for j, (u, v) in enumerate(gt[:, ti]):
            x, y = np.rint([u, v]).astype(int)
            if not (0 <= x < 256 and 0 <= y < 256):
                code, boundary_distance = "outside_frame", 0.
            elif not red[y, x]:
                code, boundary_distance = "occluded", 0.
            elif score < .95:
                code, boundary_distance = "tracking_failed", float(distance[y, x])
            elif distance[y, x] < 2.0:
                code, boundary_distance = "ambiguous", float(distance[y, x])
            else:
                code, boundary_distance = "visible", float(distance[y, x])
            reason[j, ti] = code
            mask[j, ti] = code == "visible"
            diagnostics.append({"step": step, "point_id": j, "uv": [float(u), float(v)],
                                "reason": code, "distance_to_red_boundary_px": boundary_distance,
                                "ECC_score": float(score),
                                "dense_flow_disagreement_px": float(np.linalg.norm(alt[j, ti]-gt[j, ti]))})
    direct_repeat = []
    for step in REPEAT_FRAMES:
        repeated, check = direct_registration(frames[T0], frames[step], points)
        for point_id in REPEAT_POINTS:
            direct_repeat.append({"step": step, "point_id": point_id,
                                  "primary_uv": tracked[step][point_id].tolist(),
                                  "repeat_uv": repeated[point_id].tolist(),
                                  "difference_px": float(np.linalg.norm(
                                      repeated[point_id]-tracked[step][point_id])),
                                  "direct_registration": check})
    differences = np.asarray([x["difference_px"] for x in direct_repeat])
    repeatability = {"method": "BLIND_ALGORITHMIC_REPEAT_DIRECT_MASK_ECC_VS_SEQUENTIAL_MASK_ECC",
                     "note": "Second algorithm reads source RGB and frozen t0 points only, not primary GT or prediction; this is not an independent human annotation.",
                     "observations": direct_repeat, "count": len(direct_repeat),
                     "mean_px": float(np.mean(differences)),
                     "median_px": float(np.median(differences)),
                     "p90_px": float(np.percentile(differences, 90)),
                     "max_px": float(np.max(differences))}
    np.save(RUN / "gt_2d.npy", gt.astype("float32"))
    np.save(RUN / "validity_mask.npy", mask)
    np.save(RUN / "reason_mask.npy", reason)
    np.save(RUN / "gt_dense_flow_alternative.npy", alt.astype("float32"))
    (RUN / "annotation_repeatability.json").write_text(
        json.dumps(repeatability, indent=2) + "\n", encoding="utf8")
    (RUN / "gt_tracking_diagnostics.json").write_text(json.dumps({
        "method": "RGB_RED_MASK_SEQUENTIAL_AFFINE_ECC",
        "prediction_read": False, "calibration_read": False,
        "same_physical_point_assumption": "surface coordinates transported by a 2D affine warp of the visible peg mask; textureless face makes identity approximate",
        "validity_rule": "inside principal red component, distance >=2px from boundary, ECC score >=0.95",
        "point_frame_diagnostics": diagnostics,
        "registration": registration,
        "valid_pairs": int(mask.sum()), "total_pairs": 160,
        "FDE_valid_points": int(mask[:, -1].sum())}, indent=2) + "\n", encoding="utf8")
    canvas = np.full((4*256, 5*256, 3), 255, np.uint8)
    for ti, step in enumerate(FUTURE):
        tile = frames[step].copy()
        for j, (u, v) in enumerate(gt[:, ti]):
            xy = tuple(np.rint([u, v]).astype(int))
            color = (0, 255, 0) if mask[j, ti] else (0, 0, 0)
            cv2.circle(tile, xy, 2, color, -1)
            cv2.putText(tile, str(j), (xy[0]+2, xy[1]-2),
                        cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
        cv2.putText(tile, str(step), (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 2)
        canvas[ti//5*256:(ti//5+1)*256, ti%5*256:(ti%5+1)*256] = tile
    cv2.imwrite(str(RUN / "gt_all_future_frames.png"), canvas)
    coverage = np.full((8*30, 20*30, 3), 255, np.uint8)
    for j in range(8):
        for ti in range(20):
            color = (44, 175, 64) if mask[j, ti] else (55, 55, 180)
            cv2.rectangle(coverage, (ti*30, j*30), ((ti+1)*30-1, (j+1)*30-1), color, -1)
    cv2.imwrite(str(RUN / "coverage.png"), coverage)
    print("GT", gt.shape, "valid", int(mask.sum()), "/160", "FDE points",
          int(mask[:, -1].sum()), "repeat median", round(repeatability["median_px"], 2),
          "repeat p90", round(repeatability["p90_px"], 2))


if __name__ == "__main__":
    main()
