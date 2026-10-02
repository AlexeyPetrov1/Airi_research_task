"""Freeze independent sensor-depth input for a second raw FMB trial.

This uses the same model, K hypothesis, depth rule, action, timelines, and 2D
evaluation protocol as the episode_5201 experiment. No future RGB/depth enters
the historical 3D points or CAD pose fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import prepare_fmb_quantitative_2d as previous


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy"
BASE = ROOT / "runs/fmb_second_example_1_M_L_3_vertical_n_3"
RUN = BASE / "quantitative_2d_sensor_t130"
T0 = 130
STEPS = (128, 129, 130)
FUTURE = list(range(131, 151))
ACTION = "Insert the red rectangular peg into the matching hole on the blue board."
SELECTED = ((28, .3), (28, .7), (44, .5), (44, .8),
            (60, .2), (60, .7), (76, .3), (76, .8))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(name: str, value: dict | list) -> None:
    (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf8")


def selected_history(data: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    search = json.loads((BASE / "window_point_search.json").read_text(encoding="utf8"))
    window = next(row for row in search["candidates"] if row["t0"] == T0)
    lookup = {(row["row"], row["alpha"]): row for row in window["valid_candidates"]}
    picked = [lookup[key] for key in SELECTED]
    uv = np.asarray([[row["history"][i]["uv"] for row in picked]
                     for i in range(3)], dtype="float64")
    frames = []
    depths = np.empty((3, 8), dtype="float64")
    for i, step in enumerate(STEPS):
        entries = []
        for j, (u, v) in enumerate(uv[i]):
            x, y = np.rint([u, v]).astype(int)
            patch = data["obs/side_1_depth"][step, y-2:y+3, x-2:x+3]
            valid = patch[patch > 0]
            if len(valid) < 15:
                raise ValueError(f"Insufficient local depth at step={step}, point={j}")
            median = float(np.median(valid))
            depths[i, j] = median * 1e-4
            entries.append({"point_id": j, "uv": [float(u), float(v)],
                            "pixel_rounded": [int(x), int(y)],
                            "patch_size": [5, 5], "valid_count": len(valid),
                            "valid_fraction": len(valid)/25,
                            "raw_valid_values": valid.astype(int).tolist(),
                            "median_raw": median, "median_Z_m": median*1e-4,
                            "mad_raw": float(np.median(np.abs(valid.astype(float)-median))),
                            "min_raw": int(valid.min()), "max_raw": int(valid.max()),
                            "inside_red_mask": True})
        frames.append({"step": step, "points": entries})
    return uv, depths, frames


def main() -> None:
    if (RUN / "input_freeze.json").exists():
        raise FileExistsError("Frozen run already exists")
    RUN.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True).item()
    source_sha = sha256(SOURCE)
    probe = json.loads((BASE / "window_probe.json").read_text(encoding="utf8"))
    if source_sha != probe["source_sha256"]:
        raise ValueError("Source changed after window inspection")
    if data["obs/side_1"].shape != (165, 256, 256, 3):
        raise ValueError("Unexpected second episode RGB shape")
    if not all(str(data["primitive"][step]) == "insert" for step in range(128, 151)):
        raise ValueError("Instruction changes in selected window")
    if len(set(map(float, data["obs/gripper_pose"][128:151]))) != 1:
        raise ValueError("Gripper state changes in selected window")
    previous.STEPS = STEPS
    previous.T0 = T0
    variants = previous.k_values()
    uv, depth_m, depth_probe = selected_history(data)
    sensor = previous.deproject(uv, depth_m, variants["K_nominal"])
    if not (sensor.shape == (3, 8, 3) and np.isfinite(sensor).all() and
            (sensor[..., 2] > 0).all()):
        raise ValueError("Invalid historical sensor XYZ")
    local, cad_fit = previous.cad_poses_and_local_points(data, uv, variants["K_nominal"])
    pnp, pnp_details = previous.pnp_history(local, uv, variants["K_nominal"])
    silhouette = np.stack([
        (Rotation.from_rotvec(cad_fit["joint_RGB_TCP_fit"]["poses"][str(step)][:3])
         .as_matrix() @ local.T).T +
        np.asarray(cad_fit["joint_RGB_TCP_fit"]["poses"][str(step)][3:])
        for step in STEPS])
    rigid = previous.rigidity(sensor)
    cad_comparison = previous.compare_xyz(sensor, silhouette)
    decision = "PASS_WITH_CALIBRATION_UNCERTAINTY" if (
        max(row["rms_residual_mm"] for row in rigid["alignments"]) <= 10 and
        cad_comparison["median_XYZ_disagreement_mm"] <= 30) else "BLOCKED_GEOMETRY_CHECK"
    if decision != "PASS_WITH_CALIBRATION_UNCERTAINTY":
        save_json("preflight_probe_failed.json", {"decision": decision,
            "rigidity": rigid, "sensor_vs_CAD_silhouette": cad_comparison})
        raise ValueError("History geometry did not pass predeclared checks")
    deltas = {}
    for name, k in variants.items():
        adjusted = previous.deproject(uv, depth_m, k)
        shift = np.linalg.norm(adjusted-sensor, axis=2)*1000
        deltas[name] = {"K": k.tolist(),
                        "median_input_XYZ_shift_mm": float(np.median(shift)),
                        "p90_input_XYZ_shift_mm": float(np.percentile(shift, 90)),
                        "max_input_XYZ_shift_mm": float(np.max(shift)),
                        "median_abs_delta_Z_mm": float(np.median(np.abs(adjusted[..., 2]-sensor[..., 2]))*1000)}
        np.save(RUN / f"history_sensor_3d_{name}.npy", adjusted.astype("float32"))
    np.save(RUN / "history_points_2d.npy", uv.astype("float32"))
    np.save(RUN / "points_2d_at_t0.npy", uv[-1].astype("float32"))
    np.save(RUN / "history_sensor_3d.npy", sensor.astype("float32"))
    np.save(RUN / "history_pnp_3d.npy", pnp.astype("float32"))
    np.save(RUN / "history_cad_silhouette_tcp_3d.npy", silhouette.astype("float32"))
    for step in STEPS:
        cv2.imwrite(str(RUN / f"frame_{step}.png"), data["obs/side_1"][step])
    save_json("window_selection.json", {"candidate_probe": "../window_point_search.json",
        "final_t0": T0, "history_steps": list(STEPS), "future_steps": FUTURE,
        "selection_before_prediction": True,
        "rule": "Choose a stable insert window with 20 future steps and many valid interior depth candidates; t0=130 has 75 such candidates and 55 mm TCP motion.",
        "selected_row_alpha": SELECTED})
    save_json("experiment_config.json", {"source_file": SOURCE.name,
        "source_sha256": source_sha, "source_identity": f"FMB raw {SOURCE.name}, independent trial; no verified ShareRobot mapping",
        "huggingface_dataset_revision": probe["huggingface_dataset_revision"],
        "camera": "side_1", "image_size": [256, 256],
        "source_hz_nominal": 10, "history_dt_nominal_s": .1,
        "model_prediction_hz_assumed": 15, "model_future_steps": 30,
        "history_steps": list(STEPS), "t0": T0, "future_steps": FUTURE,
        "point_selection": "8 fixed row/fraction inner red-face positions from pre-inference depth-valid candidate grid",
        "selected_row_alpha": SELECTED,
        "source_primitive_exact": "insert", "source_object_info": data["object_info"],
        "action_text_exact_passed_to_model": ACTION,
        "action_text_derivation": "Same frozen English paraphrase of FMB insert primitive and visual object as first experiment; not verbatim metadata",
        "depth_scale_m_per_unit": 1e-4, "geometry_primary": "sensor Z16 5x5 positive median with K_nominal",
        "calibration_status": "SUPPORTED_NOT_EXACT", "future_not_used_for_3D_or_K": True})
    save_json("k_nominal.json", {"status": "SUPPORTED_NOT_EXACT",
        "source_K_640": previous.SOURCE_K.tolist(),
        "source_profile": "official side_1 rectified.2, active color K unverified",
        "hypothesis": "direct OpenCV cv2.resize 640x480 to 256x256, pixel-center convention",
        "K_nominal": variants["K_nominal"].tolist()})
    save_json("k_sensitivity_variants.json", {"primary": "K_nominal",
        "variants": deltas, "K_focal_0925_note": "depth-selected factor from prior trial; diagnostic only"})
    save_json("history_depth_probe.json", {"depth_scale_m_per_unit": 1e-4,
        "rule": "median of positive Z16 values in 5x5, require >=15 valid",
        "frames": depth_probe})
    save_json("cad_pose_fit.json", cad_fit)
    save_json("pnp_correspondences.json", pnp_details)
    save_json("sensor_vs_pnp.json", previous.compare_xyz(sensor, pnp))
    save_json("sensor_vs_cad_silhouette_tcp.json", cad_comparison)
    save_json("rigidity_check.json", rigid)
    cad_dist = np.linalg.norm(local[:, None]-local[None, :], axis=2)
    measured = np.linalg.norm(sensor[:, :, None]-sensor[:, None, :], axis=3)
    pairs = np.triu_indices(8, 1)
    relative = np.abs(measured[:, pairs[0], pairs[1]]-cad_dist[pairs])/cad_dist[pairs]
    save_json("cad_dimension_check.json", {"full_CAD_dimensions_m": [previous.W, previous.D, previous.H],
        "status": "PARTIAL_SPAN_ONLY", "partial_pair_relative_error": relative.tolist(),
        "partial_pair_median_relative_error": float(np.median(relative)),
        "partial_pair_max_relative_error": float(np.max(relative)),
        "warning": "No independently identifiable full-dimension endpoint pair on uniform red face."})
    save_json("preflight.json", {"decision": decision, "source_file": SOURCE.name,
        "camera": "side_1", "depth_scale_m_per_unit": 1e-4,
        "source_hz": 10, "K_nominal": variants["K_nominal"].tolist(),
        "K_status": "SUPPORTED_NOT_EXACT", "depth_rgb_alignment": "APPROXIMATE",
        "exact_pixel_registration": False,
        "geometry_method_primary": "SENSOR_DEPTH_K_NOMINAL",
        "history_points_valid": 24, "history_points_total": 24,
        "sensor_kabsch_max_rms_mm": max(row["rms_residual_mm"] for row in rigid["alignments"]),
        "sensor_vs_cad_silhouette_tcp_median_XYZ_mm": cad_comparison["median_XYZ_disagreement_mm"],
        "remaining_uncertainties": ["active COLOR K and exact depth preprocessing unknown",
            "object surface lacks distinctive fiducials", "nominal 10 Hz versus model 15 Hz physical time assumption"]})
    frozen = ("window_selection.json", "experiment_config.json", "k_nominal.json",
              "k_sensitivity_variants.json", "history_points_2d.npy",
              "points_2d_at_t0.npy", "history_sensor_3d.npy",
              "history_depth_probe.json", "preflight.json")
    save_json("input_freeze.json", {"created_before_primary_prediction": True,
        "sha256": {name: sha256(RUN / name) for name in frozen}})
    print(json.dumps({"run": str(RUN), "preflight": decision,
                      "sensor_vs_CAD_median_mm": cad_comparison["median_XYZ_disagreement_mm"],
                      "sensor_vs_PnP_median_mm": previous.compare_xyz(sensor, pnp)["median_XYZ_disagreement_mm"],
                      "Kabsch_max_rms_mm": max(row["rms_residual_mm"] for row in rigid["alignments"]),
                      "depth_range_m": [float(depth_m.min()), float(depth_m.max())]}, indent=2))


if __name__ == "__main__":
    main()
