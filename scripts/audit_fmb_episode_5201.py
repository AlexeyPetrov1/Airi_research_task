"""Audit the ShareRobot FMB episode against the selected public RLDS record.

Run after extracting `planning_rows.json`, the two ShareRobot PNGs, the RLDS
TFRecord payload, and the single source NPY identified by its RLDS file_path.
No model input or future point labels are created here.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import mmap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from probe_fmb_rlds_record import find_feature_values


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_fmb_episode_5201"
OLD = ROOT / "runs" / "sharerobot_transfer_berkeley_ur5_episode_26"
REV = "3266d92902b038ce40e7fb8aac5bfd9287eb3e45"
GCS_BASE = "https://storage.googleapis.com/gresearch/robotics/fmb/0.0.1"
ZIP_URL = "https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip"
SOURCE_NAME = "1_M_L_3_vertical_n_2.npy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coeff = cv2.dct(small.astype(np.float32))[:8, :8].flatten()
    return coeff > np.median(coeff[1:])


def ssim(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Mean channelwise SSIM, Gaussian 11x11 sigma=1.5, image range 255."""
    scores = []
    for c in range(3):
        x = rgb_a[:, :, c].astype(np.float64)
        y = rgb_b[:, :, c].astype(np.float64)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        var_x = cv2.GaussianBlur(x*x, (11, 11), 1.5) - mu_x*mu_x
        var_y = cv2.GaussianBlur(y*y, (11, 11), 1.5) - mu_y*mu_y
        cov = cv2.GaussianBlur(x*y, (11, 11), 1.5) - mu_x*mu_y
        c1, c2 = (0.01*255)**2, (0.03*255)**2
        score = ((2*mu_x*mu_y+c1)*(2*cov+c2)) / ((mu_x*mu_x+mu_y*mu_y+c1)*(var_x+var_y+c2))
        scores.append(float(score[5:-5, 5:-5].mean()))
    return float(np.mean(scores))


def camera_check(a: np.ndarray, b: np.ndarray, step_a: int, step_b: int) -> dict:
    gray_a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    # The board and bin floor are fixed. Exclude the upper object/gripper area.
    mask = np.zeros(gray_a.shape, np.uint8)
    mask[83:245, 20:235] = 255
    orb = cv2.ORB_create(nfeatures=1500, fastThreshold=5)
    kp_a, des_a = orb.detectAndCompute(gray_a, mask)
    kp_b, des_b = orb.detectAndCompute(gray_b, mask)
    if des_a is None or des_b is None:
        return {"source_steps": [step_a, step_b], "background_correspondences": 0,
                "ransac_inliers": 0, "median_background_displacement_px": None,
                "displacement_over_diagonal": None, "reason": "No ORB descriptors"}
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_a, des_b, k=2)
    good = [m for m, n in matches if m.distance < 0.75*n.distance]
    if len(good) < 4:
        return {"source_steps": [step_a, step_b], "background_correspondences": len(good),
                "ransac_inliers": 0, "median_background_displacement_px": None,
                "displacement_over_diagonal": None, "reason": "Too few matches for RANSAC"}
    p_a = np.float32([kp_a[m.queryIdx].pt for m in good])
    p_b = np.float32([kp_b[m.trainIdx].pt for m in good])
    transform, inliers = cv2.estimateAffinePartial2D(p_a, p_b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    inlier_mask = inliers.ravel().astype(bool) if inliers is not None else np.zeros(len(good), bool)
    displacement = np.linalg.norm(p_a[inlier_mask]-p_b[inlier_mask], axis=1)
    median = float(np.median(displacement)) if len(displacement) else None
    p95 = float(np.percentile(displacement, 95)) if len(displacement) else None
    selected = [m for m, keep in zip(good, inlier_mask) if keep]
    drawn = cv2.drawMatches(cv2.cvtColor(a, cv2.COLOR_RGB2BGR), kp_a,
                            cv2.cvtColor(b, cv2.COLOR_RGB2BGR), kp_b,
                            selected[:60], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    name = f"camera_motion_{step_a}_{step_b}.png"
    cv2.imwrite(str(OUT/name), drawn)
    return {"source_steps": [step_a, step_b], "background_correspondences": len(good),
            "ransac_inliers": int(inlier_mask.sum()), "median_background_displacement_px": median,
            "p95_background_displacement_px": p95,
            "displacement_over_diagonal": median/float(np.hypot(*a.shape[:2])) if median is not None else None,
            "affine_partial_2d": transform.tolist() if transform is not None else None,
            "visualization": name, "mask": "x=20..234, y=83..244; upper object/gripper excluded"}


def main() -> None:
    rows = json.loads((OUT/"planning_rows.json").read_text(encoding="utf8"))
    share_paths = sorted({p for row in rows for p in row["image"]},
                         key=lambda p: int(p.rsplit("frame_", 1)[1].split(".")[0]))
    assert len(rows) == 20 and len(share_paths) == 29
    raw = np.load(OUT/SOURCE_NAME, allow_pickle=True).item()
    assert raw["obs/side_1"].shape == (148, 256, 256, 3)
    source_step_count = 148
    record_path = OUT/"candidate_rlds_5201_record.bin"
    with record_path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as record:
        features = dict(find_feature_values(record))
        source_jpegs = features["steps/observation/image_side_1"]
        assert len(source_jpegs) == source_step_count
        file_path = bytes(record[slice(*features["episode_metadata/file_path"][0])]).decode()
        assert file_path.endswith("/"+SOURCE_NAME)
        depth_ranges = features["steps/observation/image_side_1_depth"]
        assert len(depth_ranges) == 1
        depth_start, depth_end = depth_ranges[0]
        rlds_depth = np.frombuffer(record[depth_start:depth_end], dtype="<f4").reshape(148, 256, 256)
        depth_equal_all_steps = bool(np.array_equal(rlds_depth, raw["obs/side_1_depth"].astype(np.float32)))
        assert depth_equal_all_steps

        def rgb_at(step: int) -> np.ndarray:
            start, end = source_jpegs[step]
            return np.asarray(Image.open(io.BytesIO(record[start:end])).convert("RGB"))

        known = []
        for share_step in (0, 15):
            share_file = OLD/f"reserve_fmb_episode_5201_frame_{share_step}.png"
            share = np.asarray(Image.open(share_file).convert("RGB"))
            # Search all steps; the ShareRobot index is not used to choose the match.
            distances = [(float(np.abs(rgb_at(i).astype(np.int16)-share.astype(np.int16)).mean()), i)
                         for i in range(source_step_count)]
            best_mae, best_step = min(distances)
            candidate = rgb_at(best_step)
            source_file = f"source_side_1_step_{best_step:03d}.png"
            Image.fromarray(candidate).save(OUT/source_file)
            known.append({"share_frame_id": f"frame_{share_step}", "share_section": "Trajectory",
                          "share_file": str(share_file.relative_to(ROOT)),
                          "share_sha256": sha256(share_file), "source_episode": file_path,
                          "source_step": best_step, "timestamp": None,
                          "share_resolution": [256, 256], "source_resolution": [256, 256],
                          "documented_image_transform": "JPEG decode only; no resize/crop/color transform",
                          "source_file": source_file, "pixel_mae": best_mae,
                          "second_best_pixel_mae": sorted(distances)[1][0],
                          "phash_distance": int(np.count_nonzero(phash(share) != phash(candidate))),
                          "ssim": ssim(share, candidate)})

        # The pattern below fits the two verified pairs and the author's
        # description of even 30-frame extraction. It is not a verified map.
        with (OUT/"share_to_source_mapping.csv").open("w", newline="", encoding="utf8") as file:
            writer = csv.DictWriter(file, fieldnames=["share_frame_id", "share_image_path",
                                                     "candidate_source_step", "verified_source_step",
                                                     "source_timestamp_s", "share_resolution",
                                                     "source_resolution", "phash_distance", "ssim",
                                                     "mapping_status", "nominal_source_time_s"])
            writer.writeheader()
            verified = {0: known[0]["source_step"], 15: known[1]["source_step"]}
            pair_metrics = {0: known[0], 15: known[1]}
            for path in share_paths:
                i = int(path.rsplit("frame_", 1)[1].split(".")[0])
                candidate = int(np.floor(source_step_count*i/30))
                writer.writerow({"share_frame_id": f"frame_{i}", "share_image_path": path,
                                 "candidate_source_step": candidate,
                                 "verified_source_step": verified.get(i, ""),
                                 "source_timestamp_s": "",
                                 "share_resolution": "256x256" if i in verified else "",
                                 "source_resolution": "256x256" if i in verified else "",
                                 "phash_distance": pair_metrics[i]["phash_distance"] if i in verified else "",
                                 "ssim": pair_metrics[i]["ssim"] if i in verified else "",
                                 "mapping_status": "VISUAL_MATCH" if i in verified else "INFERRED_ONLY",
                                 "nominal_source_time_s": round(candidate/10, 3)})

        sheet = Image.new("RGB", (600, 2*290), "white")
        draw = ImageDraw.Draw(sheet)
        for j, pair in enumerate(known):
            share = Image.open(ROOT/pair["share_file"]).convert("RGB")
            source = Image.open(OUT/pair["source_file"]).convert("RGB")
            sheet.paste(share, (20, j*290+24))
            sheet.paste(source, (320, j*290+24))
            draw.text((20, j*290+5), f"ShareRobot {pair['share_frame_id']}", fill="black")
            draw.text((320, j*290+5), f"FMB side_1 step {pair['source_step']}", fill="black")
        sheet.save(OUT/"mapping_contact_sheet.png")

        camera = [camera_check(rgb_at(a), rgb_at(b), a, b) for a, b in ((0, 37), (37, 74))]

    depth = raw["obs/side_1_depth"]
    depth_observed = {}
    for step in (0, 37, 74):
        d = depth[step]
        depth_observed[str(step)] = {"min_raw": int(d.min()), "median_raw": float(np.median(d)),
                                     "max_raw": int(d.max()), "nonpositive": int(np.count_nonzero(d <= 0)),
                                     "nan": 0, "inf": 0}
    calibration = json.loads((OLD/"reserve_fmb_side1_intrinsics").read_text(encoding="utf8"))
    candidate_k = [[float(calibration["rectified.2.fx"])*256/640, 0,
                    (float(calibration["rectified.2.ppx"])+0.5)*256/640-0.5],
                   [0, float(calibration["rectified.2.fy"])*256/480,
                    (float(calibration["rectified.2.ppy"])+0.5)*256/480-0.5], [0, 0, 1]]
    depth_check = {"source_depth_key": "obs/side_1_depth", "source_depth_shape": list(depth.shape),
                   "source_depth_dtype": str(depth.dtype), "raw_depth_diagnostics": depth_observed,
                   "rlds_depth_key": "steps/observation/image_side_1_depth",
                   "rlds_depth_exactly_equals_raw_float32_cast_all_148_steps": depth_equal_all_steps,
                   "depth_unit": None, "depth_scale_m_per_unit": None,
                   "history_points_total": 24, "history_depth_valid": 0,
                   "history_depth_coverage": 0.0, "selected_history_points": 0,
                   "history_depth_min_m": None, "history_depth_median_m": None,
                   "history_depth_max_m": None, "history_nan": None, "history_inf": None,
                   "history_nonpositive": None, "rgb_resolution": [256, 256],
                   "calibration_resolution": [640, 480],
                   "calibration_source": "https://functional-manipulation-benchmark.github.io/static/files/side_1",
                   "candidate_K_after_full_frame_resize": candidate_k,
                   "K_transform": "OpenCV resize pixel-center convention: f'=s*f, c'=s*(c+0.5)-0.5; candidate only",
                   "K_status": "UNVERIFIED_COLOR_STREAM_AND_DEPTH_RESIZE",
                   "distortion": None, "reprojection_error_px": None,
                   "note": "No 2D→3D→2D test: metric depth scale and active color K are unverified. Even a small reprojection error would test code, not depth accuracy."}
    (OUT/"depth_calibration_check.json").write_text(json.dumps(depth_check, indent=2)+"\n", encoding="utf8")

    temporal = {"source_step_count": source_step_count, "source_timestamps": None,
                "source_nominal_control_hz": 10, "source_nominal_dt_s": 0.1,
                "source_dt_median": None, "source_dt_median_status": "UNMEASURED_NO_TIMESTAMPS",
                "share_sampling_rule_candidate": "floor(148 * frame_index / 30)",
                "share_sampling_rule_status": "FITS_TWO_PAIRS_NOT_VERIFIED_FOR_THIRD",
                "verified_pair_sampling_residual_steps": [0, 0],
                "verified_pair_nominal_time_residual_s": [0.0, 0.0],
                "history_times": None, "future_target_times": None,
                "matched_source_times": None, "mean_abs_time_error": None,
                "max_abs_time_error": None,
                "note": "FMB collection environment defaults to 10 Hz; RealSense capture defaults to 15 FPS. Neither records actual timestamps in this NPY/RLDS episode."}
    (OUT/"temporal_alignment.json").write_text(json.dumps(temporal, indent=2)+"\n", encoding="utf8")
    (OUT/"camera_motion_check.json").write_text(json.dumps(camera, indent=2)+"\n", encoding="utf8")

    monotonic = sum(a["source_step"] >= b["source_step"] for a, b in zip(known, known[1:]))
    preflight = {
        "share_robot_id": "57_fmb#episode_5201", "share_robot_section": ["Planning", "Affordance", "Trajectory"],
        "source_dataset": "FMB", "source_episode": file_path,
        "source_release": "Open X-Embodiment FMB TFDS 0.0.1 and FMB single_object_manipulation.zip",
        "source_rlds_ordinal_candidate": 5201, "source_rlds_shard": 1228,
        "source_rlds_record_in_shard": 0, "source_rlds_record_sha256": sha256(record_path),
        "source_npy_sha256": sha256(OUT/SOURCE_NAME), "source_step_count": source_step_count,
        "share_frames_total": len(share_paths), "share_frames_claimed_by_card": 30,
        "source_frames_matched": len(known), "mapping_coverage": len(known)/len(share_paths),
        "unique_source_steps": len({p["source_step"] for p in known}),
        "monotonicity_violations": monotonic,
        "mapping_status": "PARTIAL",
        "mapping_status_detail": "TWO_INDEPENDENT_VISUAL_PAIRS_THIRD_UNAVAILABLE",
        "mapping_pairs": known,
        "temporal_alignment_status": "UNVERIFIED",
        "temporal_alignment_detail": "NO_ACTUAL_TIMESTAMPS_OR_THIRD_SAMPLING_PAIR",
        "source_dt_median": None, "history_times": None, "future_target_times": None,
        "matched_source_times": None, "mean_abs_time_error": None, "max_abs_time_error": None,
        "depth_status": "UNVERIFIED",
        "depth_status_detail": "RAW_UINT16_AVAILABLE_METRIC_SCALE_UNVERIFIED",
        "rlds_depth_exactly_equals_raw_all_148_steps": depth_equal_all_steps,
        "calibration_status": "UNVERIFIED",
        "calibration_status_detail": "SIDE_1_FILE_AVAILABLE_ACTIVE_COLOR_K_UNVERIFIED",
        "camera_motion_status": "PARTIAL",
        "camera_motion_status_detail": "SOURCE_BACKGROUND_CHECK_ONLY_HISTORY_NOT_SELECTED",
        "camera_motion_checks": camera,
        "history_3d_status": "NOT_CONSTRUCTED", "history_points_total": 24,
        "history_depth_valid": 0, "history_depth_coverage": 0.0,
        "future_2d_available": True, "future_2d_gt_status": "NOT_ANNOTATED",
        "future_2d_note": "148 source RGB steps exist; no independent 8-point GT or validity mask has been made.",
        "future_3d_available": False,
        "leakage_check": "PASS_NO_MODEL_INPUT_CREATED_CAMERA_DIAGNOSTIC_USES_STEPS_0_TO_74_ONLY",
        "decision": "BLOCKED",
        "blocking_reasons": [
            "Only two unique ShareRobot RGBs are individually retrievable; a third independent early/middle/late image from Planning is in a 64-part 343,174,473,847-byte split solid gzip archive. Direct frame_28 URL returns 404. The required 3-pair mapping gate is unmet.",
            "The episode has no actual timestamps. The nominal 10 Hz control rate does not establish measured frame times or the 30-frame sampling rule beyond two matched pairs.",
            "Raw uint16 depth is present but the camera-specific meters-per-unit scale was not recorded or published for this episode.",
            "The published side_1 calibration file has 640x480 rectified entries, but the active 256x256 color K and depth resize procedure for this episode are not documented; 24 metric history points cannot be validated.",
            "No eight persistent distinguishable physical points and independent 30-step future validity mask have been selected; 2D/3D metrics are unavailable."
        ],
        "checked_sources": [
            {"name": "ShareRobot pinned Planning JSONs", "url": f"https://huggingface.co/datasets/BAAI/ShareRobot/tree/{REV}/planning/jsons", "finding": "20 records; 29 distinct listed RGB paths, frame_0..frame_28"},
            {"name": "ShareRobot Planning image archive", "url": f"https://huggingface.co/datasets/BAAI/ShareRobot/tree/{REV}/planning/images", "finding": "64 consecutive parts of one gzip stream, 343174473847 bytes; direct frame_28 URL 404"},
            {"name": "ShareRobot pinned Trajectory", "url": f"https://huggingface.co/datasets/BAAI/ShareRobot/blob/{REV}/trajectory/trajectory.json", "finding": "frame_0 and frame_15 individually retrievable"},
            {"name": "ShareRobot pinned Affordance", "url": f"https://huggingface.co/datasets/BAAI/ShareRobot/blob/{REV}/affordance/affordance.json", "finding": "one annotation on frame_0; no third RGB"},
            {"name": "ShareRobot author mapping statement", "url": "https://github.com/FlagOpen/ShareRobot/issues/4#issuecomment-3094378733", "finding": "episode ID is original dataset episode ordinal; 30 frames evenly sampled"},
            {"name": "FMB TFDS 0.0.1 dataset_info", "url": GCS_BASE+"/dataset_info.json", "finding": "8611 train episodes, shardLengths place ordinal 5201 in shard 1228 record 0"},
            {"name": "FMB TFDS 0.0.1 features", "url": GCS_BASE+"/features.json", "finding": "side_1 RGB/depth and episode_metadata/file_path"},
            {"name": "FMB TFDS shard 1228", "url": GCS_BASE+"/fmb-train.tfrecord-01228-of-02017", "finding": file_path},
            {"name": "FMB raw ZIP central directory and selected NPY", "url": ZIP_URL, "finding": SOURCE_NAME},
            {"name": "FMB dataset documentation", "url": "https://functional-manipulation-benchmark.github.io/dataset/index.html", "finding": "raw BGR and depth schemas, side_1 intrinsics; no depth scale/timestamps"},
            {"name": "FMB robot/camera code", "url": "https://github.com/rail-berkeley/fmb/tree/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra", "finding": "10 Hz default control, 15 FPS camera, aligned Z16 depth, RGB resize to 256"},
            {"name": "RealSense Z16 depth documentation", "url": "https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0", "finding": "uint16 Z16 requires camera-specific depth_scale to convert to meters"}
        ],
        "candidates": [
            {"source_episode": file_path, "status": "STRONG_CANDIDATE_NOT_FULLY_VERIFIED", "reason": "Two unique ordered visual pairs match; third inaccessible."},
            {"source_episode": "insert_only_1_M_L_3_vertical_n_2.npy", "status": "REJECTED_FOR_FULL_EPISODE", "reason": "Different insert-only trajectory; RLDS file_path explicitly names the non-insert-only NPY."},
            {"source_episode": "lerobot/fmb episode_5201", "status": "REJECTED_AS_INDEX_MAPPING", "reason": "Derivative release lists only 1804 episodes; index 5201 cannot identify its episode."}
        ],
        "model_generation_calls": 0, "metric_pairs": 0,
        "coverage_valid_pairs": 0, "coverage_denominator": 240,
        "ade_3d_m": None, "fde_3d_m": None, "ade_2d_px": None, "fde_2d_px": None,
        "next_minimal_unblock": "Provide or extract one Planning RGB such as frame_28 for a third visual pair, plus side_1's recorded depth_units (meters/raw unit) and active aligned-color K/256x256 depth resize metadata."
    }
    (OUT/"preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    unavailable = {"ADE_3D_m": None, "FDE_3D_m": None,
                   "ADE_2D_px": None, "FDE_2D_px": None,
                   "ADE_2D_norm": None, "FDE_2D_norm": None,
                   "mean_error_step_5": None, "mean_error_step_15": None,
                   "mean_error_step_30": None}
    metrics = {"status": "NOT_EVALUATED_PREFLIGHT_BLOCKED", "valid_pairs": 0,
               "coverage": "0/240", "visible_pairs": None, "occluded_pairs": None,
               "untrackable_pairs": None, "validity_mask": None,
               "molmo_motion": unavailable, "stationary": unavailable,
               "constant_velocity": unavailable}
    (OUT/"metrics.json").write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf8")
    print(json.dumps({"decision": preflight["decision"], "share_frames": len(share_paths),
                      "matches": known, "camera": camera}, indent=2))


if __name__ == "__main__":
    main()
