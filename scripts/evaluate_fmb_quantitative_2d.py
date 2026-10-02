"""Evaluate frozen MolmoMotion predictions on native FMB 10 Hz RGB GT."""

from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probe_fmb_cad_pnp import SOURCE as DEFAULT_SOURCE


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))
BASE = RUN.parent
SOURCE = Path(os.environ.get("FMB_QUANT_SOURCE", DEFAULT_SOURCE))
PRED_TIMES = np.arange(1, 31, dtype=float)/15
GT_TIMES = np.arange(1, 21, dtype=float)/10
DIAGONAL = float(np.hypot(256, 256))


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.stack((k[0, 0]*xyz[..., 0]/xyz[..., 2]+k[0, 2],
                     k[1, 1]*xyz[..., 1]/xyz[..., 2]+k[1, 2]), axis=-1)


def interpolate(future: np.ndarray) -> np.ndarray:
    assert future.shape == (8, 30, 3)
    return np.stack([[np.interp(GT_TIMES, PRED_TIMES, future[j, :, axis])
                      for axis in range(3)] for j in range(8)], axis=0).transpose(0, 2, 1)


def measure(prediction: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[dict, np.ndarray]:
    error = np.linalg.norm(prediction-gt, axis=2)
    finite = np.isfinite(error)
    if not finite[mask].all():
        raise ValueError("Non-finite prediction at a valid GT observation")
    valid = error[mask]
    final = error[:, -1][mask[:, -1]]
    values = {"ADE_2D_px": float(valid.mean()) if len(valid) else None,
              "FDE_2D_px": float(final.mean()) if len(final) else None,
              "ADE_2D_norm": float(valid.mean()/DIAGONAL) if len(valid) else None,
              "FDE_2D_norm": float(final.mean()/DIAGONAL) if len(final) else None,
              "valid_pairs": int(mask.sum()), "coverage": float(mask.mean()),
              "FDE_valid_points": int(mask[:, -1].sum())}
    for t in (.5, 1., 1.5, 2.):
        index = int(round(t*10))-1
        sample = error[:, index][mask[:, index]]
        values[f"mean_error_{t:.1f}s_px"] = float(sample.mean()) if len(sample) else None
        values[f"valid_points_{t:.1f}s"] = int(mask[:, index].sum())
    return values, error


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw_points(frame: np.ndarray, points: np.ndarray, label: str,
                color: tuple[int, int, int], valid: np.ndarray | None = None) -> np.ndarray:
    for j, (u, v) in enumerate(points):
        if valid is not None and not valid[j]:
            continue
        if not np.isfinite([u, v]).all():
            continue
        x, y = np.rint([u, v]).astype(int)
        if 0 <= x < 256 and 0 <= y < 256:
            cv2.circle(frame, (x, y), 2, color, -1)
            cv2.putText(frame, str(j), (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX,
                        .27, color, 1)
    cv2.putText(frame, label, (4, 242), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    return frame


def visuals(source: dict, gt: np.ndarray, mask: np.ndarray, predictions: dict[str, np.ndarray],
            error_by_time: list[dict], k: np.ndarray) -> None:
    frames = source["obs/side_1"]
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    history_steps = config["history_steps"]
    t0 = config["t0"]
    history_uv = np.load(RUN / "history_points_2d.npy")
    depth = json.loads((RUN / "history_depth_probe.json").read_text(encoding="utf8"))
    history = np.full((256, 3*256, 3), 255, np.uint8)
    quality = np.full((256, 3*256, 3), 255, np.uint8)
    for i, step in enumerate(history_steps):
        panel = frames[step].copy()
        qpanel = frames[step].copy()
        for j, (u, v) in enumerate(history_uv[i]):
            x, y = np.rint([u, v]).astype(int)
            cv2.circle(panel, (x, y), 2, (0, 255, 0), -1)
            cv2.putText(panel, str(j), (x+2, y-2), cv2.FONT_HERSHEY_SIMPLEX,
                        .3, (255, 255, 255), 1)
            count = depth["frames"][i]["points"][j]["valid_count"]
            cv2.rectangle(qpanel, (x-2, y-2), (x+2, y+2), (0, 255, 0), 1)
            cv2.putText(qpanel, f"{j}:{count}", (x+3, y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, .25, (255, 255, 255), 1)
        cv2.putText(panel, f"step {step}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 255, 255), 2)
        cv2.putText(qpanel, f"step {step} (valid / 25)", (4, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 2)
        history[:, i*256:(i+1)*256] = panel
        quality[:, i*256:(i+1)*256] = qpanel
    cv2.imwrite(str(RUN / "history_8_points.png"), history)
    cv2.imwrite(str(RUN / "history_depth_quality.png"), quality)
    sensor = np.load(RUN / "history_sensor_3d.npy")
    pnp = np.load(RUN / "history_pnp_3d.npy")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, step in enumerate(history_steps):
        color = plt.cm.viridis(i/2)
        ax.scatter(*(sensor[i].T*1000), c=[color], label=f"sensor {step}")
        ax.scatter(*(pnp[i].T*1000), c=[color], marker="x", label=f"PnP {step}")
        for a, b in zip(sensor[i], pnp[i]):
            ax.plot(*np.stack([a, b]).T*1000, color=color, alpha=.4)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RUN / "sensor_vs_pnp_3d.png", dpi=160)
    plt.close(fig)
    montage = np.full((4*256, 4*256, 3), 255, np.uint8)
    for row, step in enumerate((t0+5, t0+10, t0+15, t0+20)):
        ti = step-t0-1
        for col, name in enumerate(("GT", "MolmoMotion", "stationary", "constant_velocity")):
            panel = frames[step].copy()
            points = gt[:, ti] if name == "GT" else predictions[name][:, ti]
            color = {"GT": (0, 255, 0), "MolmoMotion": (0, 0, 255),
                     "stationary": (255, 80, 0), "constant_velocity": (200, 0, 200)}[name]
            draw_points(panel, points, f"{name} step {step}", color, mask[:, ti])
            montage[row*256:(row+1)*256, col*256:(col+1)*256] = panel
    cv2.imwrite(str(RUN / "future_frames_predictions_gt.png"), montage)
    overlay = frames[t0].copy()
    for name, color in (("GT", (0, 255, 0)), ("MolmoMotion", (0, 0, 255)),
                        ("stationary", (255, 80, 0)),
                        ("constant_velocity", (200, 0, 200))):
        points = gt if name == "GT" else predictions[name]
        for j in range(8):
            polyline = np.rint(points[j]).astype("int32")
            cv2.polylines(overlay, [polyline], False, color, 1)
            x, y = polyline[-1]
            if 0 <= x < 256 and 0 <= y < 256:
                cv2.circle(overlay, (x, y), 2, color, -1)
    cv2.imwrite(str(RUN / "future_trajectory_overlay.png"),
                cv2.resize(overlay, (768, 768), interpolation=cv2.INTER_NEAREST))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cv2.cvtColor(frames[t0], cv2.COLOR_BGR2RGB), extent=(0, 256, 256, 0))
    ax.plot([0, 256, 256, 0, 0], [0, 0, 256, 256, 0],
            color="black", linewidth=1, label="image boundary")
    colors = {"GT": "green", "MolmoMotion": "red",
              "stationary": "tab:blue", "constant_velocity": "purple"}
    for name in ("GT", "MolmoMotion", "stationary", "constant_velocity"):
        centers = (gt if name == "GT" else predictions[name]).mean(axis=0)
        ax.plot(centers[:, 0], centers[:, 1], color=colors[name],
                linewidth=2, label=f"{name} point centroid")
        ax.scatter(centers[-1, 0], centers[-1, 1], color=colors[name], s=45)
    ax.set(xlabel="u (px)", ylabel="v (px)", xlim=(-30, 440), ylim=(500, -230),
           title="Forecast outside image retained in pixel-error metric")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(RUN / "future_trajectory_extended_plane.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    times = np.asarray([row["time_s"] for row in error_by_time])
    for name, col in (("MolmoMotion", "MolmoMotion_mean_error_px"),
                      ("stationary", "stationary_mean_error_px"),
                      ("constant_velocity", "constant_velocity_mean_error_px")):
        ax.plot(times, [row[col] for row in error_by_time], marker=".", label=name)
    ax.set(xlabel="Future time (s, nominal)", ylabel="Mean 2D error (px)", xlim=(0, 2))
    ax.grid(alpha=.3)
    second = ax.twinx()
    second.plot(times, [row["valid_points"] for row in error_by_time],
                color="black", linestyle="--", alpha=.6, label="valid points")
    second.set(ylabel="Valid points", ylim=(0, 9))
    ax.legend(loc="upper left")
    second.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RUN / "error_vs_time.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    from matplotlib.colors import ListedColormap
    ax.imshow(mask.astype(int), aspect="auto", interpolation="nearest",
              cmap=ListedColormap(["#d62728", "#2ca02c"]), vmin=0, vmax=1)
    ax.set(xticks=[0, 4, 9, 14, 19],
           xticklabels=["0.1", "0.5", "1.0", "1.5", "2.0"],
           yticks=np.arange(8), yticklabels=[str(i) for i in range(8)],
           xlabel="Future time (s, nominal)", ylabel="Point ID",
           title=f"GT validity: {int(mask.sum())}/160 pairs; "
                 f"FDE: {int(mask[:, -1].sum())}/8 points")
    ax.set_xticks(np.arange(-.5, 20, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(RUN / "coverage.png", dpi=160)
    plt.close(fig)
    if (BASE / "mapping_contact_sheet.png").exists():
        shutil.copy2(BASE / "mapping_contact_sheet.png", RUN / "share_to_fmb_mapping.png")


def main() -> None:
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    model_run = json.loads((RUN / "model_run.json").read_text(encoding="utf8"))
    assert model_run["success"] and model_run["prediction_completeness"] == 1.0
    gt = np.load(RUN / "gt_2d.npy").astype(float)
    mask = np.load(RUN / "validity_mask.npy")
    assert gt.shape == (8, 20, 2) and mask.shape == (8, 20)
    variants = json.loads((RUN / "k_sensitivity_variants.json").read_text(encoding="utf8"))["variants"]
    k = np.asarray(variants["K_nominal"]["K"])
    original = np.load(RUN / "prediction_15hz.npy").astype(float)
    pred_10 = interpolate(original)
    assert pred_10.shape == (8, 20, 3)
    np.save(RUN / "prediction_10hz.npy", pred_10.astype("float32"))
    np.save(RUN / "prediction_time_15hz_s.npy", PRED_TIMES)
    np.save(RUN / "evaluation_time_10hz_s.npy", GT_TIMES)
    if not (np.isfinite(pred_10).all() and (pred_10[..., 2] > 0).all()):
        raise ValueError("Interpolated prediction has invalid or nonpositive Z")
    pred_2d = project(pred_10, k)
    np.save(RUN / "prediction_2d.npy", pred_2d.astype("float32"))
    outside = ((pred_2d[..., 0] < 0) | (pred_2d[..., 0] >= 256) |
               (pred_2d[..., 1] < 0) | (pred_2d[..., 1] >= 256))
    (RUN / "projection_check.json").write_text(json.dumps({
        "finite": bool(np.isfinite(pred_2d).all()), "all_positive_Z": True,
        "outside_image_pairs": int(outside.sum()),
        "outside_image_by_time": outside.sum(axis=0).tolist(),
        "coordinates_clipped_for_metric": False}, indent=2) + "\n", encoding="utf8")
    history = np.load(RUN / "history_points_2d.npy").astype(float)
    stationary = np.repeat(history[-1, :, None, :], 20, axis=1)
    times_history = np.asarray([-.2, -.1, 0.])
    centered = times_history-times_history.mean()
    velocity = np.sum(centered[:, None, None]*(history-history.mean(axis=0)), axis=0)/np.sum(centered**2)
    constant = history[-1, :, None, :] + velocity[:, None, :]*GT_TIMES[None, :, None]
    np.save(RUN / "baseline_stationary_2d.npy", stationary.astype("float32"))
    np.save(RUN / "baseline_constant_velocity_2d.npy", constant.astype("float32"))
    predictions = {"MolmoMotion": pred_2d, "stationary": stationary,
                   "constant_velocity": constant}
    metrics = {}
    errors = {}
    rows = []
    for name, prediction in predictions.items():
        metrics[name], errors[name] = measure(prediction, gt, mask)
        rows.append({"method": name, **metrics[name]})
    write_csv(RUN / "metrics.csv", rows)
    per_point = []
    for j in range(8):
        row = {"point_id": j, "valid_future_frames": int(mask[j].sum())}
        for name, error in errors.items():
            row[name+"_ADE_px"] = float(error[j, mask[j]].mean()) if mask[j].any() else None
            row[name+"_FDE_px"] = float(error[j, -1]) if mask[j, -1] else None
        per_point.append(row)
    write_csv(RUN / "per_point_metrics.csv", per_point)
    error_by_time = []
    for ti, t in enumerate(GT_TIMES):
        row = {"time_s": float(t), "source_step": config["t0"]+1+ti,
               "valid_points": int(mask[:, ti].sum())}
        for name in predictions:
            sample = errors[name][:, ti][mask[:, ti]]
            row[name+"_mean_error_px"] = float(sample.mean()) if len(sample) else None
        error_by_time.append(row)
    write_csv(RUN / "error_by_time.csv", error_by_time)
    alternative_gt = np.load(RUN / "gt_dense_flow_alternative.npy").astype(float)
    alt_disagreement = np.linalg.norm(alternative_gt-gt, axis=2)
    alternative_metrics = {name: measure(prediction, alternative_gt, mask)[0]
                           for name, prediction in predictions.items()}
    result = {"status": "QUANTITATIVE_2D_RESULT_WITH_CALIBRATION_AND_TRACKING_UNCERTAINTY",
              "source_episode": config.get("source_identity", f"FMB {config['source_file']}"),
              "t0": config["t0"], "history_steps": config["history_steps"],
              "future_steps": config["future_steps"],
              "prediction_completeness_finite_pairs": 240,
              "prediction_total_pairs": 240,
              "GT_valid_pairs": int(mask.sum()), "GT_total_pairs": 160,
              "GT_coverage": float(mask.mean()), "image_diagonal_px": DIAGONAL,
              "metrics": metrics,
              "annotation_repeatability": json.loads((RUN / "annotation_repeatability.json").read_text(encoding="utf8")),
              "alternative_dense_flow_2D_disagreement_px": {
                  "median": float(np.median(alt_disagreement[mask])),
                  "p90": float(np.percentile(alt_disagreement[mask], 90)),
                  "max": float(np.max(alt_disagreement[mask]))},
              "metrics_using_dense_flow_alternative_GT": alternative_metrics,
              "limitations": ["GT tracks textureless surface coordinates by RGB mask affine registration; it is an image-derived proxy, not individually visible fiducials.",
                              "Camera K is a direct-resize hypothesis; exact active COLOR K and depth preprocessing are unrecorded.",
                              "Nominal 10 Hz FMB history differs from model typical 15 Hz timing; no source hardware timestamps.",
                              "One episode does not establish general model quality."]}
    (RUN / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    source = np.load(SOURCE, allow_pickle=True).item()
    visuals(source, gt, mask, predictions, error_by_time, k)
    sensitivity = []
    for name, info in variants.items():
        matrix = np.asarray(info["K"])
        if name == "K_nominal":
            mode, estimated = "full_inference", metrics["MolmoMotion"]
        else:
            label = {"K_simple": "simple", "K_focal_0925": "focal"}[name]
            variant_file = RUN / "variants" / label / "prediction_15hz.npy"
            if variant_file.exists():
                prediction3 = interpolate(np.load(variant_file).astype(float))
                np.save(RUN / "variants" / label / "prediction_10hz.npy",
                        prediction3.astype("float32"))
                estimated, _ = measure(project(prediction3, matrix), gt, mask)
                mode = "full_inference"
            else:
                estimated, _ = measure(project(pred_10, matrix), gt, mask)
                mode = "projection_only_same_nominal_3D_prediction"
        sensitivity.append({"K_variant": name,
                            "median_input_XYZ_shift_mm": info["median_input_XYZ_shift_mm"],
                            "p90_input_XYZ_shift_mm": info["p90_input_XYZ_shift_mm"],
                            "max_input_XYZ_shift_mm": info["max_input_XYZ_shift_mm"],
                            "median_abs_delta_Z_mm": info["median_abs_delta_Z_mm"],
                            "evaluation_mode": mode,
                            "ADE_2D_px": estimated["ADE_2D_px"],
                            "FDE_2D_px": estimated["FDE_2D_px"],
                            "coverage": estimated["coverage"]})
    write_csv(RUN / "calibration_sensitivity.csv", sensitivity)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = [row["K_variant"] for row in sensitivity]
    x = np.arange(len(labels))
    ax.bar(x-.18, [row["ADE_2D_px"] for row in sensitivity], width=.36, label="ADE")
    ax.bar(x+.18, [row["FDE_2D_px"] for row in sensitivity], width=.36, label="FDE")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Pixel error", title="Calibration sensitivity (modes shown in CSV)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RUN / "calibration_sensitivity.png", dpi=160)
    plt.close(fig)
    print(json.dumps({"primary": metrics["MolmoMotion"],
                      "stationary": metrics["stationary"],
                      "constant_velocity": metrics["constant_velocity"],
                      "dense_GT_alternative": alternative_metrics,
                      "calibration": sensitivity}, indent=2))


if __name__ == "__main__":
    main()
