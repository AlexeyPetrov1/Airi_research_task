"""RGB-only approximate rounded-CAD poses for the history 124--126.

Only visible row spans and the lower end of the red object are fitted. This is
an independent diagnostic of sensor Z, not a precise point-correspondence PnP.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import D, K, SOURCE
from probe_fmb_cad_silhouette_k import MODEL
from probe_fmb_effective_k import landmarks, pose_for_k


OUT = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"


def polygon_row_span(polygon: np.ndarray, y: float) -> tuple[float, float]:
    previous = np.roll(polygon, 1, axis=0)
    good = ((polygon[:, 1] <= y) & (previous[:, 1] >= y)) | ((previous[:, 1] <= y) & (polygon[:, 1] >= y))
    if not np.any(good):
        return float("nan"), float("nan")
    a, b = previous[good], polygon[good]
    denom = b[:, 1] - a[:, 1]
    same_y = np.abs(denom) < 1e-9
    xs = np.where(same_y, a[:, 0], a[:, 0] + (y - a[:, 1]) / np.where(same_y, 1, denom) * (b[:, 0] - a[:, 0]))
    return float(np.min(xs)), float(np.max(xs))


def model_polygon(pose: np.ndarray) -> np.ndarray:
    pts = cv2.projectPoints(MODEL, pose[:3], pose[3:], K, None)[0].reshape(-1, 2)
    return cv2.convexHull(pts.astype("float32")).reshape(-1, 2)


def observations(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    mask, box = red_component(frame)
    bottom = int(box[1] + box[3] - 1)
    ys = np.arange(30, bottom - 8, 10, dtype=float)
    spans = np.asarray([(np.flatnonzero(mask[int(y)]).min(),
                         np.flatnonzero(mask[int(y)]).max()) for y in ys], dtype=float)
    return ys, spans, bottom


def fit_pose(frame: np.ndarray, seeds: list[np.ndarray]) -> dict:
    ys, spans, bottom = observations(frame)

    def residual(pose: np.ndarray) -> np.ndarray:
        poly = model_polygon(pose)
        spans_model = np.asarray([polygon_row_span(poly, y) for y in ys])
        if not np.isfinite(spans_model).all():
            return np.full(2 * len(ys) + 1, 100.0)
        return np.r_[(spans_model - spans).ravel(), (poly[:, 1].max() - bottom) * 2]

    bounds = ([-np.inf]*5 + [.05], [np.inf]*5 + [1.0])
    fits = [least_squares(residual, seed, bounds=bounds, loss="soft_l1",
                          f_scale=3., max_nfev=150) for seed in seeds]
    best = min(fits, key=lambda fit: fit.cost)
    return {"pose": best.x.tolist(), "cost": float(best.cost),
            "row_span_rmse_px": float(np.sqrt(np.mean(residual(best.x)[:-1]**2))),
            "bottom_error_px": float(residual(best.x)[-1] / 2),
            "observed_rows": ys.tolist(), "observed_left_right": spans.tolist(),
            "all_seed_costs": [float(fit.cost) for fit in fits],
            "success": bool(best.success)}


def fit_shared_orientation(frames: np.ndarray, individual: dict,
                           tcp: np.ndarray | None = None,
                           motion_weight_px_per_mm: float = 0.,
                           steps: tuple[int, int, int] = (124, 125, 126)) -> dict:
    obs = {step: observations(frames[step]) for step in steps}
    base = np.asarray([individual[str(step)]["pose"] for step in steps])
    x0 = np.r_[base[2, :3], base[:, 3:].ravel()]

    def residual(x: np.ndarray) -> np.ndarray:
        rows = []
        for i, step in enumerate(steps):
            ys, spans, bottom = obs[step]
            pose = np.r_[x[:3], x[3+3*i:6+3*i]]
            poly = model_polygon(pose)
            fitted = np.asarray([polygon_row_span(poly, y) for y in ys])
            if not np.isfinite(fitted).all():
                rows.append(np.full(2*len(ys)+1, 100.))
            else:
                rows.append(np.r_[(fitted-spans).ravel(), 2*(poly[:, 1].max()-bottom)])
        if tcp is not None and motion_weight_px_per_mm:
            t = x[3:].reshape(3, 3)
            for i, j in ((0, 1), (1, 2), (0, 2)):
                object_distance = np.linalg.norm(t[j] - t[i])
                robot_distance = np.linalg.norm(tcp[steps[j], :3] - tcp[steps[i], :3])
                rows.append(np.asarray([(object_distance-robot_distance) * 1000 *
                                        motion_weight_px_per_mm]))
        return np.concatenate(rows)

    lower = np.r_[[-np.inf]*3, np.tile([-np.inf, -np.inf, .05], 3)]
    upper = np.r_[[np.inf]*3, np.tile([np.inf, np.inf, 1.], 3)]
    fitted = least_squares(residual, x0, bounds=(lower, upper), loss="soft_l1",
                           f_scale=3., max_nfev=300)
    return {"cost": float(fitted.cost), "success": bool(fitted.success),
            "motion_weight_px_per_mm": motion_weight_px_per_mm,
            "shared_rotation_vector": fitted.x[:3].tolist(),
            "poses": {str(step): np.r_[fitted.x[:3], fitted.x[3+3*i:6+3*i]].tolist()
                      for i, step in enumerate(steps)},
            "per_frame_span_rmse_px": {str(step): float(np.sqrt(np.mean(residual(fitted.x)[
                sum(2*len(obs[s][0])+1 for s in steps[:i]):
                sum(2*len(obs[s][0])+1 for s in steps[:i+1])-1]**2)))
                for i, step in enumerate(steps)}}


def main() -> None:
    data = np.load(SOURCE, allow_pickle=True).item()
    frames = data["obs/side_1"]
    guess_uv, _ = landmarks(frames[126])
    rv, tv = pose_for_k(K, guess_uv)
    seed = np.r_[rv, tv]
    base_seeds = []
    for z in (.18, .22, .28):
        p = seed.copy()
        p[3:5] *= z / p[5]
        p[5] = z
        base_seeds.append(p)
    result = {"method": "RGB_ROUNDED_CAD_VISIBLE_ROW_SPANS_DIAGNOSTIC",
              "K": K.tolist(), "history_steps": [124, 125, 126], "poses": {}}
    for step in (126, 125, 124):
        trial = fit_pose(frames[step], base_seeds)
        result["poses"][str(step)] = trial
        fit = np.asarray(trial["pose"])
        base_seeds = [fit] + base_seeds
        print(step, "Z", round(fit[5], 4), "cost", round(trial["cost"], 2),
              "RMSE", round(trial["row_span_rmse_px"], 2))
    result["shared_orientation_fit"] = fit_shared_orientation(frames, result["poses"])
    result["shared_orientation_tcp_motion_fit"] = fit_shared_orientation(
        frames, result["poses"], data["obs/tcp_pose"], motion_weight_px_per_mm=2.)
    print("shared orientation", result["shared_orientation_fit"]["cost"],
          result["shared_orientation_fit"]["per_frame_span_rmse_px"])
    print("TCP-constrained", result["shared_orientation_tcp_motion_fit"]["cost"],
          result["shared_orientation_tcp_motion_fit"]["per_frame_span_rmse_px"])
    candidates = json.loads((OUT / "history_point_candidates.json").read_text(encoding="utf8"))
    pose = np.asarray(result["shared_orientation_tcp_motion_fit"]["poses"]["126"])
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    normal = rotation[:, 1]
    origin = pose[3:] + rotation @ np.asarray([0., -D / 2, 0.])
    inv_k = np.linalg.inv(K)
    chosen = ((35, .35), (35, .55), (45, .25), (45, .45),
              (55, .35), (55, .75), (65, .25), (65, .55))
    points = []
    for y, alpha in chosen:
        candidate = next(row for row in candidates if row["y_at_t0"] == y and row["alpha"] == alpha)
        u, v = candidate["uv_t0"]
        ray = inv_k @ np.asarray([u, v, 1.])
        xyz = (normal @ origin) / (normal @ ray) * ray
        local = rotation.T @ (xyz - pose[3:])
        points.append({"y": y, "alpha": alpha, "CAD_local_xyz_m": local.tolist(),
                       "CAD_camera_Z_m": float(xyz[2]),
                       "sensor_Z_m": candidate["frames"][2]["median_raw"] * .0001})
    result["ray_front_face_checks"] = points
    for item in points:
        print("ray", item["y"], item["alpha"], "local", np.round(item["CAD_local_xyz_m"], 3),
              "Zcad", round(item["CAD_camera_Z_m"], 3),
              "Zsensor", round(item["sensor_Z_m"], 3))
    cad_points = np.asarray([item["CAD_local_xyz_m"] for item in points])
    for step in (124, 125, 126):
        pose = np.asarray(result["shared_orientation_tcp_motion_fit"]["poses"][str(step)])
        rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
        xyz = (rotation @ cad_points.T).T + pose[3:]
        projected = cv2.projectPoints(cad_points, pose[:3], pose[3:], K, None)[0].reshape(-1, 2)
        uv = np.asarray([next(row for row in candidates if row["y_at_t0"] == y and row["alpha"] == alpha)
                         ["frames"][step - 124]["uv"] for y, alpha in chosen], dtype=float)
        sensor_z = np.asarray([next(row for row in candidates if row["y_at_t0"] == y and row["alpha"] == alpha)
                               ["frames"][step - 124]["median_raw"] * .0001
                               for y, alpha in chosen])
        print("crosscheck", step, "UV median px", round(float(np.median(np.linalg.norm(projected-uv, axis=1))), 2),
              "Z median mm", round(float(np.median(np.abs(xyz[:, 2]-sensor_z))*1000), 1),
              "Z range", np.round(xyz[:, 2], 3).tolist())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "history_silhouette_pose_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf8")
    canvas = np.full((3*512, 512, 3), 255, np.uint8)
    for i, step in enumerate((124, 125, 126)):
        im = frames[step].copy()
        poly = model_polygon(np.asarray(result["shared_orientation_tcp_motion_fit"]["poses"][str(step)]))
        cv2.polylines(im, [np.rint(poly).astype("int32")], True, (0, 255, 0), 1)
        canvas[i*512:(i+1)*512] = cv2.resize(im, (512, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(OUT / "history_silhouette_pose_probe.png"), canvas)


if __name__ == "__main__":
    main()
