"""Reconcile recovered FMB calibration evidence into episode_5201 preflight.

Run audit_fmb_episode_5201.py and probe_fmb_geometry.py first. This script
keeps the source linkage, then applies the narrower geometry gate from the
follow-up goal. It intentionally does not generate a model input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from audit_fmb_episode_5201 import camera_check


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_fmb_episode_5201"
INTRINSICS_URL = "https://functional-manipulation-benchmark.github.io/static/files/side_1"
FMB_CODE = "https://github.com/rail-berkeley/fmb/tree/d4da6ce044a9806f41e58bf7423b5a3c05289925"


def save_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def main() -> None:
    original = json.loads((OUT / "preflight.json").read_text(encoding="utf8"))
    probe = json.loads((OUT / "geometry_probe.json").read_text(encoding="utf8"))
    calibration_path = OUT / "side_1_intrinsics_official"
    calibration_bytes = calibration_path.read_bytes()
    calibration = json.loads(calibration_bytes)
    np_load_error = None
    try:
        np.load(calibration_path, allow_pickle=False)
    except ValueError as error:
        np_load_error = str(error)
    profile = {key.removeprefix("rectified.2."): float(value)
               for key, value in calibration.items() if key.startswith("rectified.2.")}
    k_source = [[profile["fx"], 0, profile["ppx"]],
                [0, profile["fy"], profile["ppy"]], [0, 0, 1]]
    # Only RGB resize is explicit in official code. This is a candidate for
    # depth because its 640x480 -> 256x256 preprocessing is unpublished.
    k_candidate = probe["K_256_candidate"]
    intrinsics = {
        "source_url": INTRINSICS_URL,
        "file_path": str(calibration_path.relative_to(ROOT)),
        "file_command_result": "JSON text data",
        "byte_size": len(calibration_bytes),
        "first_16_bytes_hex": calibration_bytes[:16].hex(),
        "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "np_load_allow_pickle_false_error": np_load_error,
        "format": "UTF-8 JSON; all numeric values encoded as strings",
        "key_count": len(calibration),
        "profile_used": "rectified.2",
        "intrinsics_resolution_wh": [int(profile["width"]), int(profile["height"])],
        "K_source": k_source,
        "distortion_coefficients": None,
        "distortion_note": "No distortion coefficients in JSON; rectified profile suggests a pinhole image but does not identify the active RGB stream.",
        "stream_metadata": None,
        "camera_serial_in_calibration": None,
        "camera_serial_in_official_env_code": "128422270679",
        "rgb_preprocessing_in_official_env_code": "cv2.resize(rgb, (256, 256)) from 640x480 RealSense color stream",
        "depth_preprocessing_in_official_env_code": "Aligned to color at 640x480, then stored as depth without a resize call",
        "published_depth_resolution_wh": [256, 256],
        "K_256_if_same_full_frame_cv2_resize": k_candidate,
        "K_256_status": "CANDIDATE_ONLY_UNDOCUMENTED_DEPTH_TRANSFORM",
        "blocking_gap": "Official capture code has no transformation taking aligned 640x480 depth to the published 256x256 depth; the JSON has no active color profile/stream metadata. Exact depth-pixel K cannot be reproduced from published code.",
    }
    save_json("intrinsics_decode.json", intrinsics)

    raw = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    rgb = raw["obs/side_1"][:, :, :, ::-1]
    camera_pairs = [(0, 37), (37, 74), (74, 110), (110, 147), (0, 147)]
    camera = [camera_check(rgb[a], rgb[b], a, b) for a, b in camera_pairs]
    save_json("camera_motion_check_extended.json", camera)

    depth = raw["obs/side_1_depth"]
    positive = depth[depth > 0]
    depth_check = json.loads((OUT / "depth_calibration_check.json").read_text(encoding="utf8"))
    depth_check.update({
        "nonzero_raw_percentiles_0_5_50_95_100": np.percentile(positive, [0, 5, 50, 95, 100]).tolist(),
        "zero_count_all_steps": int(np.count_nonzero(depth == 0)),
        "depth_representation": "uint16 count domain derived from RealSense Z16; unpublished 256x256 resampling",
        "depth_unit": "meters_per_count_inferred_geometrically",
        "depth_scale_m_per_unit": 0.0001,
        "depth_scale_evidence": "INFERRED_GEOMETRIC",
        "scale_candidate_table": probe["candidates"],
        "robot_motion_fit": probe["fit"],
        "K_transform": "If depth used same full-frame cv2.resize as RGB: f'=s*f, c'=s*(c+0.5)-0.5; unverified",
        "K_status": "BLOCKED_UNDOCUMENTED_DEPTH_RESIZE_AND_COLOR_PROFILE",
        "note": "Scale is independently supported by D405 range and a metric TCP-motion fit. Exact K_256 is still unverified; no metric 3D history is constructed.",
    })
    save_json("depth_calibration_check.json", depth_check)

    temporal = json.loads((OUT / "temporal_alignment.json").read_text(encoding="utf8"))
    temporal.update({
        "camera_fps": 15,
        "environment_hz": 10,
        "nominal_dt_s": 0.1,
        "timestamp_source": "FMB environment code nominal control frequency; not hardware timestamps",
        "exact_timestamps_available": False,
        "nominal_step_time_rule": "i*0.1 s only for nominal reporting; extra gripper sleeps are not represented",
        "gripper_state_change_steps": [23, 147],
        "extra_sleep_note": "set_gripper waits up to 1 s between commands and another 1.2 s on close / 0.6 s on open; do not assume globally uniform measured time.",
        "share_sampling_rule_status": "UNKNOWN_NOT_REQUIRED_AFTER_SOURCE_RECOVERY",
        "status": "PASS_NOMINAL_LOCAL_SEGMENTS_WITH_PAUSE_EXCEPTIONS",
    })
    save_json("temporal_alignment.json", temporal)

    preflight = {
        "share_robot_id": original["share_robot_id"],
        "source_dataset": "FMB",
        "source_file": "1_M_L_3_vertical_n_2.npy",
        "source_episode": original["source_episode"],
        "source_npy_sha256": original["source_npy_sha256"],
        "source_rlds_record_sha256": original["source_rlds_record_sha256"],
        "source_step_count": 148,
        "source_mapping_status": "PASS_SOURCE_EPISODE_RECOVERED",
        "source_mapping_evidence": "RLDS ordinal 5201 points to NPY path; two ShareRobot RGBs match steps 0 and 74; all 148 RLDS depth maps equal NPY depth after float32 cast",
        "share_frames_individually_matched": 2,
        "share_frames_total_listed": 29,
        "share_robot_sampling_rule": "unknown_not_required",
        "mapping_pairs": original["mapping_pairs"],
        "camera": "side_1",
        "camera_model": "RealSense D405",
        "camera_serial": "128422270679",
        "intrinsics_status": "BLOCKED_EXACT_WORKING_K",
        "intrinsics_source": INTRINSICS_URL,
        "intrinsics_file_sha256": intrinsics["sha256"],
        "intrinsics_profile": "rectified.2",
        "intrinsics_resolution": [640, 480],
        "working_resolution": [256, 256],
        "K_source": k_source,
        "image_transform_rgb": "full-frame cv2.resize from 640x480 to 256x256 in official environment code",
        "image_transform_depth": "unknown in published code; capture leaves aligned depth at 640x480 but released NPY is 256x256",
        "K_working": None,
        "K_working_if_same_resize": k_candidate,
        "depth_representation": "Z16-derived uint16 count domain; 256x256 preprocessing unknown",
        "depth_scale": 0.0001,
        "depth_scale_status": "PASS_INFERRED_GEOMETRIC",
        "depth_scale_evidence": "INFERRED_GEOMETRIC",
        "depth_scale_fit_m_per_unit": probe["fit"]["estimated_m_per_raw_unit"],
        "depth_scale_candidate_table": probe["candidates"],
        "camera_fps": 15,
        "environment_hz": 10,
        "nominal_dt": 0.1,
        "timestamp_source": temporal["timestamp_source"],
        "exact_timestamps_available": False,
        "timing_status": temporal["status"],
        "timing_pause_exception_steps": [23, 147],
        "camera_motion_status": "STATIC_SUPPORTED",
        "camera_motion_checks": camera,
        "history_3d_status": "NOT_CONSTRUCTED_EXACT_K_BLOCKED",
        "history_points_total": 24,
        "history_points_validated": 0,
        "leakage_check": "MODEL_NOT_RUN; robot-motion scale diagnostic uses episode steps 112..140, so independent calibration must be frozen before a later prediction window",
        "decision": "BLOCKED",
        "outcome": "VERIFIED_BLOCKER",
        "blocking_reasons": [
            "The exact 256x256 depth-pixel intrinsic matrix cannot be reproduced: the official 640x480 capture code aligns depth to color but has no depth resize, while the published NPY depth is 256x256. The calibration JSON does not identify an active color stream/profile or describe the missing transformation. Full-frame resize K is a plausible candidate, not a verified input K."
        ],
        "non_blocking_limitations": [
            "Only two ShareRobot PNGs are individually matched; source episode is recovered and full ShareRobot sampling is not needed.",
            "No actual timestamps; nominal 10 Hz has gripper-command pause exceptions.",
            "Scale inference uses the target episode's later TCP trajectory and is diagnostic; an unbiased future test should confirm/freeze calibration independently."
        ],
        "model_generation_calls": 0,
        "metric_pairs": 0,
        "coverage_valid_pairs": 0,
        "coverage_denominator": 240,
        "next_minimal_unblock": "Obtain the original FMB 640x480 -> 256x256 depth preprocessing/alignment transform and active color intrinsics for side_1 serial 128422270679, or a calibrated 256x256 aligned RGB-D camera profile for that recording. Confirm depth scale independently before target-window prediction.",
    }
    save_json("preflight.json", preflight)
    print(json.dumps({"decision": preflight["decision"], "outcome": preflight["outcome"],
                      "blocking_reasons": preflight["blocking_reasons"],
                      "depth_scale_fit_m_per_unit": preflight["depth_scale_fit_m_per_unit"]}, indent=2))


if __name__ == "__main__":
    main()
