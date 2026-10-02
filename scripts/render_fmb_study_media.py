"""Render FMB figures and videos only from saved frames, tracks and forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from evaluate_fmb_quantitative_2d import interpolate, project
from run_fmb_ablation_suite import ROOT, RUNS, write_json


BLUE = (255, 100, 25)     # BGR: observed GT
ORANGE = (0, 145, 255)   # BGR: MolmoMotion
GRAY = (150, 150, 150)   # BGR: stationary baseline
GREEN = (50, 190, 60)
WHITE = (245, 245, 245)
BLACK = (15, 15, 15)
COLOR = {"sensor": ORANGE, "pnp": GREEN, "cad": (210, 90, 190)}


def header(panel: np.ndarray, title: str, subtitle: str = "", bottom: bool = False) -> np.ndarray:
    start = panel.shape[0] - 36 if bottom else 0
    cv2.rectangle(panel, (0, start), (panel.shape[1], start + 36), BLACK, -1)
    cv2.putText(panel, title, (7, start + 14), cv2.FONT_HERSHEY_SIMPLEX, .39, WHITE, 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(panel, subtitle, (7, start + 29), cv2.FONT_HERSHEY_SIMPLEX, .31, WHITE, 1, cv2.LINE_AA)
    return panel


def point(panel: np.ndarray, uv: np.ndarray, idx: int, color: tuple[int, int, int], marker: str) -> bool:
    if not np.isfinite(uv).all():
        return False
    x, y = np.rint(uv).astype(int)
    if not (0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]):
        return False
    if marker == "cross":
        cv2.drawMarker(panel, (x, y), BLACK, cv2.MARKER_CROSS, 10, 3)
        cv2.drawMarker(panel, (x, y), color, cv2.MARKER_CROSS, 8, 2)
    else:
        cv2.circle(panel, (x, y), 5, BLACK, -1)
        cv2.circle(panel, (x, y), 3, color, -1)
    cv2.putText(panel, str(idx), (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                .31, WHITE, 2, cv2.LINE_AA)
    cv2.putText(panel, str(idx), (x + 5, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                .31, BLACK, 1, cv2.LINE_AA)
    return True


def make_real_panel(frame: np.ndarray, gt: np.ndarray, valid: np.ndarray, t: float) -> np.ndarray:
    out = frame.copy()
    for i in range(8):
        if valid[i]:
            point(out, gt[i], i, BLUE, "circle")
    return header(out, "Observed future", f"t = {t:.1f} s; blue = RGB-derived 2D GT", bottom=True)


def make_pred_panel(frame: np.ndarray, pred: np.ndarray, stationary: np.ndarray,
                    t: float, name: str = "MolmoMotion") -> np.ndarray:
    out = frame.copy()
    outside = 0
    for i in range(8):
        point(out, stationary[i], i, GRAY, "circle")
        outside += not point(out, pred[i], i, ORANGE, "cross")
    return header(out, name, f"orange = forecast; gray = static; outside {outside}/8", bottom=True)


def make_error_panel(frame: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                     valid: np.ndarray, t: float) -> np.ndarray:
    out = np.full_like(frame, 250)
    # One pixel-coordinate plane and one scale for every frame and episode.
    # It includes forecasts that leave the original 256x256 image.
    def map_xy(value: np.ndarray) -> tuple[int, int]:
        u, v = value
        return round(26 + (u + 80) * 205 / 416), round(39 + (v + 150) * 205 / 416)

    image_ul, image_lr = map_xy(np.array([0, 0])), map_xy(np.array([256, 256]))
    cv2.rectangle(out, image_ul, image_lr, (110, 110, 110), 1)
    cv2.putText(out, "image 0..256", (image_ul[0] + 2, image_lr[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX, .28, BLACK, 1)
    mean_error = float(np.linalg.norm(pred[valid] - gt[valid], axis=1).mean()) if valid.any() else float("nan")
    for i in range(8):
        if not valid[i]:
            continue
        a, b = map_xy(gt[i]), map_xy(pred[i])
        cv2.line(out, a, b, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.circle(out, a, 4, BLUE, -1)
        cv2.drawMarker(out, b, ORANGE, cv2.MARKER_CROSS, 8, 2)
        cv2.putText(out, str(i), (b[0] + 3, b[1] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, .29, BLACK, 1)
    return header(out, "Extended pixel plane", f"mean error {mean_error:.1f} px; box = image")


def make_chart_panel(error: np.ndarray, baselines: dict[str, np.ndarray], step: int) -> np.ndarray:
    panel = np.full((256, 256, 3), 250, np.uint8)
    all_errors = np.concatenate((error, *baselines.values()))
    maximum = max(10., float(np.nanmax(all_errors)) * 1.05)
    # Shared y range over the full episode, fixed across all video frames.
    left, right, top, bottom = 39, 246, 48, 211
    cv2.line(panel, (left, top), (left, bottom), BLACK, 1)
    cv2.line(panel, (left, bottom), (right, bottom), BLACK, 1)
    for value in (0., maximum / 2, maximum):
        y = round(bottom - (value / maximum) * (bottom - top))
        cv2.line(panel, (left, y), (right, y), (220, 220, 220), 1)
        cv2.putText(panel, f"{value:.0f}", (2, y + 3), cv2.FONT_HERSHEY_SIMPLEX, .28, BLACK, 1)
    lines = [(error, ORANGE), (baselines["stationary"], GRAY),
             (baselines["constant_velocity"], (170, 90, 70))]
    for values, color in lines:
        points = []
        for j in range(step + 1):
            x = round(left + j * (right - left) / 19)
            y = round(bottom - values[j] * (bottom - top) / maximum)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(panel, [np.asarray(points, np.int32)], False, color, 2)
        cv2.circle(panel, points[-1], 3, color, -1)
    cursor = round(left + step * (right - left) / 19)
    cv2.line(panel, (cursor, top), (cursor, bottom), BLACK, 1)
    cv2.putText(panel, "0", (left - 2, 226), cv2.FONT_HERSHEY_SIMPLEX, .28, BLACK, 1)
    cv2.putText(panel, "2.0 s", (right - 33, 226), cv2.FONT_HERSHEY_SIMPLEX, .28, BLACK, 1)
    return header(panel, "Mean 2D error over time", "orange model; gray static; brown velocity")


def video(path: Path, frames: list[np.ndarray], fps: int = 10, repeat: int = 3) -> None:
    if not frames:
        raise ValueError("No frames")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (h, w):
                raise ValueError("Video frame dimensions changed")
            for _ in range(repeat):
                writer.write(frame)
    finally:
        writer.release()
    cap = cv2.VideoCapture(str(path))
    decoded = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if decoded != len(frames) * repeat:
        raise RuntimeError(f"Video decode length mismatch: {path}: {decoded}")


def geometry_cloud_panel(sensor: np.ndarray, cad: np.ndarray) -> np.ndarray:
    """XY and XZ views, both at 1 px/mm with fixed camera-coordinate axes."""
    out = np.full((256, 256, 3), 250, np.uint8)
    cv2.line(out, (35, 116), (225, 116), (140, 140, 140), 1)
    cv2.line(out, (35, 242), (225, 242), (140, 140, 140), 1)
    for points, color, mark in ((sensor, BLUE, "circle"), (cad, GREEN, "cross")):
        for xyz in points:
            x, y, z = xyz * 1000
            xpos = round(35 + (x - 30))
            ypos = round(114 - (y + 110))
            zpos = round(242 - (z - 150))
            for position in ((xpos, ypos), (xpos, zpos)):
                if mark == "circle":
                    cv2.circle(out, position, 3, color, -1)
                else:
                    cv2.drawMarker(out, position, color, cv2.MARKER_CROSS, 7, 1)
    cv2.putText(out, "XY (camera mm)", (7, 50), cv2.FONT_HERSHEY_SIMPLEX, .33, BLACK, 1)
    cv2.putText(out, "XZ (camera mm)", (7, 150), cv2.FONT_HERSHEY_SIMPLEX, .33, BLACK, 1)
    return header(out, "Historical 3D points", "blue sensor; green CAD; 1 px/mm")


def render_one(name: str, run: Path) -> dict:
    config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
    raw = ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"]
    source = np.load(raw, allow_pickle=True).item()
    rgb = source["obs/side_1"]  # FMB arrays are BGR.
    depth = source["obs/side_1_depth"]
    k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
    gt = np.load(run / "gt_2d.npy")
    valid = np.load(run / "validity_mask.npy")
    nominal = project(interpolate(np.load(run / "prediction_15hz.npy")), k)
    stationary = np.load(run / "baseline_stationary_2d.npy")
    constant = np.load(run / "baseline_constant_velocity_2d.npy")
    histories = np.load(run / "history_points_2d.npy")
    sensor_xyz = np.load(run / "history_sensor_3d.npy")
    cad_xyz = np.load(run / "history_cad_silhouette_tcp_3d.npy")
    methods = {"sensor": nominal,
               "pnp": project(interpolate(np.load(run / "variants/pnp/prediction_15hz.npy")), k),
               "cad": project(interpolate(np.load(run / "variants/cad_silhouette/prediction_15hz.npy")), k)}
    dest = run / "study_media_v1"
    dest.mkdir(parents=True, exist_ok=True)
    method_errors = {key: np.linalg.norm(value - gt, axis=2).mean(axis=0)
                     for key, value in methods.items()}
    baselines = {"stationary": np.linalg.norm(stationary - gt, axis=2).mean(axis=0),
                 "constant_velocity": np.linalg.norm(constant - gt, axis=2).mean(axis=0)}
    truth_frames, compare_frames = [], []
    for ti, step in enumerate(config["future_steps"]):
        frame = rgb[step]
        t = (ti + 1) / 10
        truth = make_real_panel(frame, gt[:, ti], valid[:, ti], t)
        prediction = make_pred_panel(frame, nominal[:, ti], stationary[:, ti], t)
        error = make_error_panel(frame, gt[:, ti], nominal[:, ti], valid[:, ti], t)
        chart = make_chart_panel(method_errors["sensor"], baselines, ti)
        truth_frames.append(np.vstack((np.hstack((truth, prediction)),
                                       np.hstack((error, chart)))))
        panels = [truth]
        for method, title in (("sensor", "Sensor depth"), ("pnp", "Planar PnP"),
                              ("cad", "CAD silhouette + TCP")):
            panel = frame.copy()
            outside = 0
            for idx in range(8):
                outside += not point(panel, methods[method][idx, ti], idx, COLOR[method], "cross")
            panels.append(header(panel, title, f"K candidate; outside {outside}/8", bottom=True))
        compare_frames.append(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    video(dest / "forecast_vs_observed.mp4", truth_frames)
    video(dest / "geometry_methods.mp4", compare_frames)
    selected = [4, 9, 14, 19]
    cv2.imwrite(str(dest / "forecast_vs_observed_overview.png"),
                np.hstack([truth_frames[j] for j in selected]))
    cv2.imwrite(str(dest / "geometry_methods_overview.png"),
                np.hstack([compare_frames[j] for j in selected]))
    # History video: RGB points and depth on one fixed 0.14--0.30 m scale.
    geometry_frames = []
    for hi, step in enumerate(config["history_steps"]):
        left = rgb[step].copy()
        for idx in range(8):
            point(left, histories[hi, idx], idx, BLUE, "circle")
        header(left, "Historical RGB", f"step {step}; point IDs 0..7", bottom=True)
        meters = depth[step].astype(float) * 1e-4
        normalized = np.clip((meters - .14) / .16, 0, 1)
        colored = cv2.applyColorMap(np.uint8(normalized * 255), cv2.COLORMAP_TURBO)
        colored[depth[step] == 0] = 0
        header(colored, "Sensor Z16 depth", "fixed color scale 0.14..0.30 m", bottom=True)
        cloud = geometry_cloud_panel(sensor_xyz[hi], cad_xyz[hi])
        geometry_frames.append(np.hstack((left, colored, cloud)))
    video(dest / "input_and_geometry.mp4", geometry_frames, fps=6, repeat=6)
    input_overview = np.vstack(geometry_frames)
    footer = np.full((47, input_overview.shape[1], 3), 18, np.uint8)
    cv2.putText(footer, f"t0 = step {config['t0']}; action:", (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, .42, WHITE, 1, cv2.LINE_AA)
    cv2.putText(footer, config["action_text_exact_passed_to_model"], (8, 37),
                cv2.FONT_HERSHEY_SIMPLEX, .37, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(str(dest / "input_and_geometry_overview.png"), np.vstack((input_overview, footer)))
    report = {"episode": name, "source_file": config["source_file"],
              "source_camera": "side_1", "source_RGB_channel_order": "BGR",
              "GT_source": "saved RGB-derived 2D proxy", "K_status": "SUPPORTED_NOT_EXACT",
              "action_text": config["action_text_exact_passed_to_model"],
              "nominal_time_s": [(j + 1) / 10 for j in range(20)],
              "visuals_generated_from_saved_predictions_only": True,
              "future_images_used_for_display_and_evaluation_only": True,
              "videos": ["input_and_geometry.mp4", "forecast_vs_observed.mp4", "geometry_methods.mp4"],
              "still_overviews": ["input_and_geometry_overview.png", "forecast_vs_observed_overview.png", "geometry_methods_overview.png"],
              "forecast_outside_image_count": int(((nominal[..., 0] < 0) | (nominal[..., 0] >= 256) |
                                                   (nominal[..., 1] < 0) | (nominal[..., 1] >= 256)).sum()),
              "fixed_depth_scale_m": [.14, .30]}
    write_json(dest / "media_manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    args = parser.parse_args()
    for name in args.episodes:
        print(json.dumps(render_one(name, RUNS[name]), indent=2))


if __name__ == "__main__":
    main()
