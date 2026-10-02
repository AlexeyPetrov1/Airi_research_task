"""Freeze a sensor-depth MolmoMotion input before viewing any new prediction.

All geometry here uses FMB source steps 124--126 only. Future frames are used
only for the predeclared window motion/coverage check, never for XYZ or K.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from inspect_fmb_2d_window import red_component
from probe_fmb_cad_pnp import D, H, K, SOURCE, W
from probe_fmb_future_tracking import initial_points, track
from probe_fmb_history_geometry import kabsch
from probe_fmb_history_silhouette_pose import fit_pose, fit_shared_orientation
from probe_fmb_effective_k import landmarks, pose_for_k


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/sharerobot_fmb_episode_5201"
RUN = BASE / "quantitative_2d_sensor_t126"
T0 = 126
STEPS = (124, 125, 126)
ACTION = "Insert the red rectangular peg into the matching hole on the blue board."
DEPTH_SCALE = 0.0001
SOURCE_K = np.asarray([[380.209, 0, 311.477], [0, 380.209, 242.870], [0, 0, 1]])


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deproject(uv: np.ndarray, depth_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.stack(((uv[..., 0]-k[0, 2])/k[0, 0]*depth_m,
                     (uv[..., 1]-k[1, 2])/k[1, 1]*depth_m,
                     depth_m), axis=-1)


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.stack((k[0, 0]*xyz[..., 0]/xyz[..., 2]+k[0, 2],
                     k[1, 1]*xyz[..., 1]/xyz[..., 2]+k[1, 2]), axis=-1)


def k_values() -> dict[str, np.ndarray]:
    sx, sy = 256/640, 256/480
    nominal = np.asarray([[sx*SOURCE_K[0, 0], 0, sx*(SOURCE_K[0, 2]+.5)-.5],
                          [0, sy*SOURCE_K[1, 1], sy*(SOURCE_K[1, 2]+.5)-.5],
                          [0, 0, 1]])
    simple = nominal.copy()
    simple[0, 2], simple[1, 2] = sx*SOURCE_K[0, 2], sy*SOURCE_K[1, 2]
    focal = nominal.copy()
    focal[0, 0] *= .925
    focal[1, 1] *= .925
    assert np.allclose(nominal, K)
    return {"K_nominal": nominal, "K_simple": simple, "K_focal_0925": focal}


def history_uv_and_depth(data: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    frames = data["obs/side_1"]
    first = initial_points(frames[T0])
    tracked = track(frames, T0, 124, first)
    uv = np.stack([tracked[s] for s in STEPS]).astype("float64")
    raw_medians = np.empty((3, 8), dtype=float)
    probes = []
    for i, step in enumerate(STEPS):
        red, _ = red_component(frames[step])
        frame_rows = []
        for j, (uf, vf) in enumerate(uv[i]):
            u, v = np.rint([uf, vf]).astype(int)
            assert 2 <= u < 254 and 2 <= v < 254 and red[v, u]
            patch = data["obs/side_1_depth"][step, v-2:v+3, u-2:u+3]
            valid = patch[patch > 0]
            assert len(valid) >= 15, (step, j, len(valid))
            median = float(np.median(valid))
            raw_medians[i, j] = median
            frame_rows.append({"point_id": j, "uv": [float(uf), float(vf)],
                               "pixel_rounded": [int(u), int(v)],
                               "patch_size": [5, 5], "valid_count": int(len(valid)),
                               "valid_fraction": float(len(valid)/25),
                               "raw_valid_values": valid.astype(int).tolist(),
                               "median_raw": median, "median_Z_m": median*DEPTH_SCALE,
                               "mad_raw": float(np.median(np.abs(valid.astype(float)-median))),
                               "min_raw": int(valid.min()), "max_raw": int(valid.max()),
                               "inside_red_mask": True})
        probes.append({"step": step, "points": frame_rows})
    return uv, raw_medians*DEPTH_SCALE, probes


def cad_poses_and_local_points(data: dict, uv: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, dict]:
    frames = data["obs/side_1"]
    landmark_uv, _ = landmarks(frames[T0])
    rv, tv = pose_for_k(k, landmark_uv)
    seed = np.r_[rv, tv]
    seeds = []
    for z in (.18, .22, .28):
        p = seed.copy()
        p[3:5] *= z/p[5]
        p[5] = z
        seeds.append(p)
    individual = {}
    for step in reversed(STEPS):
        individual[str(step)] = fit_pose(frames[step], seeds)
        seeds = [np.asarray(individual[str(step)]["pose"])] + seeds
    shared = fit_shared_orientation(frames, individual, data["obs/tcp_pose"],
                                    motion_weight_px_per_mm=2., steps=STEPS)
    pose0 = np.asarray(shared["poses"][str(T0)])
    rotation = Rotation.from_rotvec(pose0[:3]).as_matrix()
    normal = rotation[:, 1]
    origin = pose0[3:] + rotation @ np.asarray([0., -D/2, 0.])
    rays = (np.linalg.inv(k) @ np.c_[uv[-1], np.ones(8)].T).T
    camera = ((normal @ origin)/(rays @ normal))[:, None]*rays
    local = ((rotation.T @ (camera-pose0[3:]).T).T).astype("float64")
    assert np.max(np.abs(local[:, 1]+D/2)) < 1e-6
    assert np.all(np.abs(local[:, 0]) < W/2) and np.all((local[:, 2] > 0) & (local[:, 2] < H))
    details = {"method": "RGB_ROUNDED_CAD_VISIBLE_ROW_SPANS_WITH_TCP_MOTION_PRIOR",
               "warning": "Approximate diagnostic pose fit; not sensor ground truth or exact PnP.",
               "CAD_dimensions_m": [W, D, H], "CAD_corner_radius_m": .00288,
               "fitted_individual_RGB_poses": individual,
               "joint_RGB_TCP_fit": shared, "CAD_local_point_coordinates_m": local.tolist(),
               "sensor_depth_used_in_fit": False, "future_frames_used_in_fit": False}
    return local, details


def pnp_history(local: np.ndarray, uv: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, dict]:
    history = []
    records = []
    for i, step in enumerate(STEPS):
        n, rvs, tvs, _ = cv2.solvePnPGeneric(local, uv[i], k, None,
                                               flags=cv2.SOLVEPNP_IPPE)
        assert n >= 1
        branches = []
        for rv, tv in zip(rvs, tvs):
            fit = least_squares(lambda p: (cv2.projectPoints(local, p[:3], p[3:], k, None)[0]
                                           .reshape(-1, 2)-uv[i]).ravel(),
                                np.r_[rv.ravel(), tv.ravel()], loss="soft_l1", f_scale=2.,
                                max_nfev=150)
            pose = fit.x
            points = (Rotation.from_rotvec(pose[:3]).as_matrix() @ local.T).T + pose[3:]
            reproj = project(points, k)
            error = np.linalg.norm(reproj-uv[i], axis=1)
            branches.append({"R": Rotation.from_rotvec(pose[:3]).as_matrix().tolist(),
                             "t_m": pose[3:].tolist(), "rotation_vector": pose[:3].tolist(),
                             "reprojection_error_px_each": error.tolist(),
                             "reprojection_rmse_px": float(np.sqrt(np.mean(error**2))),
                             "inlier_indices_at_5px": np.flatnonzero(error <= 5).tolist(),
                             "camera_points_m": points.tolist(),
                             "positive_Z": bool((points[:, 2] > 0).all())})
        selected = min(range(len(branches)),
                       key=lambda j: branches[j]["reprojection_rmse_px"] if branches[j]["positive_Z"]
                       else float("inf"))
        history.append(branches[selected]["camera_points_m"])
        records.append({"step": step, "observed_uv": uv[i].tolist(),
                        "object_points_CAD_m": local.tolist(), "branches": branches,
                        "selected_branch": selected})
    return np.asarray(history), {"method": "PLANAR_IPPE_ON_RGB_TRACKED_INTERIOR_CAD_POINTS",
                                 "warning": "CAD local point IDs estimated from RGB silhouette; KLT point drift and planar pose ambiguity can invalidate this cross-check.",
                                 "sensor_depth_used_in_fit": False,
                                 "future_frames_used_in_fit": False,
                                 "frames": records}


def compare_xyz(a: np.ndarray, b: np.ndarray) -> dict:
    difference = b-a
    norm = np.linalg.norm(difference, axis=2)*1000
    z = np.abs(difference[:, :, 2])*1000
    return {"delta_sensor_to_comparison_m": difference.tolist(),
            "euclidean_distance_mm_each": norm.tolist(),
            "median_XYZ_disagreement_mm": float(np.median(norm)),
            "p90_XYZ_disagreement_mm": float(np.percentile(norm, 90)),
            "max_XYZ_disagreement_mm": float(np.max(norm)),
            "median_abs_Z_disagreement_mm": float(np.median(z))}


def rigidity(xyz: np.ndarray) -> dict:
    distance = np.linalg.norm(xyz[:, :, None]-xyz[:, None, :], axis=-1)
    pairs = np.triu_indices(8, 1)
    values = distance[:, pairs[0], pairs[1]]
    per_pair = []
    for j, (a, b) in enumerate(zip(*pairs)):
        row = values[:, j]
        per_pair.append({"points": [int(a), int(b)], "distances_m": row.tolist(),
                         "mean_m": float(np.mean(row)), "std_m": float(np.std(row)),
                         "coefficient_of_variation": float(np.std(row)/np.mean(row)),
                         "max_minus_min_m": float(np.ptp(row))})
    aligned = []
    diagonal = float(np.linalg.norm([W, D, H]))
    for i in (0, 1):
        r, t, error = kabsch(xyz[i], xyz[2])
        aligned.append({"from_step": STEPS[i], "to_step": T0, "R": r.tolist(),
                        "translation_m": t.tolist(), "rms_residual_mm": error*1000,
                        "residual_over_CAD_diagonal": error/diagonal})
    return {"pairwise": per_pair, "alignments": aligned,
            "median_pairwise_std_mm": float(np.median(np.std(values, axis=0))*1000),
            "max_pairwise_range_mm": float(np.max(np.ptp(values, axis=0))*1000),
            "CAD_full_object_diagonal_m": diagonal}


def main() -> None:
    if (RUN / "input_freeze.json").exists():
        raise FileExistsError(f"Frozen run exists; refusing to overwrite: {RUN}")
    RUN.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True).item()
    assert data["obs/side_1"].shape == (148, 256, 256, 3)
    assert data["obs/side_1_depth"].shape == (148, 256, 256)
    assert all(data["primitive"][s] == "insert" for s in range(124, 147))
    variants = k_values()
    uv, z, depth_probe = history_uv_and_depth(data)
    sensor = deproject(uv, z, variants["K_nominal"])
    assert sensor.shape == (3, 8, 3) and np.isfinite(sensor).all() and (sensor[:, :, 2] > 0).all()
    local, cad_fit = cad_poses_and_local_points(data, uv, variants["K_nominal"])
    pnp, pnp_detail = pnp_history(local, uv, variants["K_nominal"])
    silhouette = np.stack([(Rotation.from_rotvec(cad_fit["joint_RGB_TCP_fit"]["poses"][str(s)][:3])
                            .as_matrix() @ local.T).T +
                           np.asarray(cad_fit["joint_RGB_TCP_fit"]["poses"][str(s)][3:])
                           for s in STEPS])
    deltas = {}
    for name, matrix in variants.items():
        transformed = deproject(uv, z, matrix)
        shift = np.linalg.norm(transformed-sensor, axis=2)*1000
        deltas[name] = {"K": matrix.tolist(),
                        "median_input_XYZ_shift_mm": float(np.median(shift)),
                        "p90_input_XYZ_shift_mm": float(np.percentile(shift, 90)),
                        "max_input_XYZ_shift_mm": float(np.max(shift)),
                        "median_abs_delta_Z_mm": float(np.median(np.abs(transformed[:, :, 2]-sensor[:, :, 2]))*1000)}
        np.save(RUN / f"history_sensor_3d_{name}.npy", transformed.astype("float32"))
    np.save(RUN / "history_points_2d.npy", uv.astype("float32"))
    np.save(RUN / "points_2d_at_t0.npy", uv[-1].astype("float32"))
    np.save(RUN / "history_sensor_3d.npy", sensor.astype("float32"))
    np.save(RUN / "history_pnp_3d.npy", pnp.astype("float32"))
    np.save(RUN / "history_cad_silhouette_tcp_3d.npy", silhouette.astype("float32"))
    for step in STEPS:
        cv2.imwrite(str(RUN / f"frame_{step}.png"), data["obs/side_1"][step])
    write_json(RUN / "k_nominal.json", {"status": "SUPPORTED_NOT_EXACT",
        "source_K_640": SOURCE_K.tolist(), "source_profile": "official side_1 rectified.2",
        "source_profile_is_active_color_K": "UNVERIFIED",
        "hypothesis": "direct OpenCV cv2.resize 640x480 -> 256x256",
        "pixel_center_convention": "cx'=sx*(cx+0.5)-0.5; cy'=sy*(cy+0.5)-0.5",
        "K_nominal": variants["K_nominal"].tolist()})
    write_json(RUN / "k_sensitivity_variants.json", {"primary": "K_nominal",
        "variants": deltas, "K_focal_0925_note": "factor chosen previously using sensor depth; diagnostic only"})
    write_json(RUN / "history_depth_probe.json", {"depth_scale_m_per_unit": DEPTH_SCALE,
        "rule": "median of positive raw Z16 values in a 5x5 patch at rounded RGB pixel; require >=15 positive",
        "frames": depth_probe})
    write_json(RUN / "cad_pose_fit.json", cad_fit)
    write_json(RUN / "pnp_correspondences.json", pnp_detail)
    write_json(RUN / "sensor_vs_pnp.json", compare_xyz(sensor, pnp))
    write_json(RUN / "sensor_vs_cad_silhouette_tcp.json", compare_xyz(sensor, silhouette))
    write_json(RUN / "rigidity_check.json", rigidity(sensor))
    cad_dist = np.linalg.norm(local[:, None]-local[None, :], axis=2)
    measured = np.linalg.norm(sensor[:, :, None]-sensor[:, None, :], axis=3)
    pairs = np.triu_indices(8, 1)
    relative = np.abs(measured[:, pairs[0], pairs[1]]-cad_dist[pairs])/cad_dist[pairs]
    write_json(RUN / "cad_dimension_check.json", {
        "full_CAD_dimensions_m": [W, D, H],
        "status": "PARTIAL_SPAN_CHECK_ONLY_FULL_DIMENSIONS_NOT_DIRECTLY_IDENTIFIABLE",
        "reason": "Only an upper interior patch of the uniform red face is observed as eight points; no full CAD width/depth/height endpoint pair is independently identified.",
        "local_CAD_points_from_RGB_silhouette_m": local.tolist(),
        "partial_pair_relative_error": relative.tolist(),
        "partial_pair_median_relative_error": float(np.median(relative)),
        "partial_pair_max_relative_error": float(np.max(relative)),
        "warning": "CAD local point coordinates are approximate RGB-silhouette inferences, not manually identifiable features."})
    old = {name: json.loads((BASE / name).read_text(encoding="utf8")) for name in
           ("preflight.json", "geometry_probe.json", "four_camera_sampling_check.json",
            "rgb_depth_alignment_check.json", "cad_pnp_local_sensitivity.json",
            "cad_pnp_focal_depth_sweep.json", "metric_geometry_comparison.json",
            "cad_history_candidate/molmo_candidate_validation.json")}
    write_json(RUN / "current_state.json", {
        "source_file": SOURCE.name, "source_sha256": sha256(SOURCE),
        "source_shape": list(data["obs/side_1"].shape),
        "previous_preflight_decision": old["preflight.json"]["decision"],
        "previous_preflight_is_for_old_geometry": True,
        "sampling_comparisons": old["four_camera_sampling_check.json"]["total_comparisons"],
        "interior_depth_valid_previous": [old["rgb_depth_alignment_check.json"]["interior_centers_valid"],
                                          old["rgb_depth_alignment_check.json"]["interior_centers_total"]],
        "scale_fit_m_per_unit": old["geometry_probe.json"]["fit"]["estimated_m_per_raw_unit"],
        "previous_model_parse_status": old["cad_history_candidate/molmo_candidate_validation.json"]["status"],
        "prior_focal_factor_selected_using_depth": old["cad_pnp_focal_depth_sweep.json"][
            "sensor_depth_used_to_select_focal_factor"],
        "depth_model_comparison_status": old["metric_geometry_comparison.json"]["status"],
        "prior_silhouette_K_probe": "effective_k_rounded_cad_silhouette.json",
        "contradiction_note": "Prior 5-point PnP correspondences used virtual rounded-CAD corners; not exact ground truth."})
    tcp = data["obs/tcp_pose"]
    candidates = []
    for t in (120, 124, 126, 127):
        future = min(t+20, 147)
        candidates.append({"t0": t, "history_steps": [t-2, t-1, t],
                           "future_20_available": bool(t+20 <= 147),
                           "tcp_motion_20_steps_mm": float(np.linalg.norm(tcp[future, :3]-tcp[t, :3])*1000),
                           "reason": {120: "peg upper body partly outside image; 126 has more interior area",
                                      124: "less visible interior area than 126",
                                      126: "selected: 24 positive local depth patches, visible upper face, 20-step future ending before gripper change",
                                      127: "last future step 147 coincides with gripper-state change and possible timing pause"}[t]})
    write_json(RUN / "window_selection.json", {"candidates": candidates,
        "final_t0": T0, "history_steps": list(STEPS), "future_steps": list(range(127, 147)),
        "window_selection_before_prediction": True,
        "rule": "latest window with 24 usable interior depth points, visible peg, substantial insertion motion, and 20 native future steps before gripper-state change at 147"})
    write_json(RUN / "experiment_config.json", {"source_file": SOURCE.name,
        "source_sha256": sha256(SOURCE), "camera": "side_1", "camera_serial": "128422270679",
        "image_size": [256, 256], "source_hz_nominal": 10, "history_dt_nominal_s": .1,
        "model_prediction_hz_assumed": 15, "model_future_steps": 30,
        "history_steps": list(STEPS), "t0": T0, "future_steps": list(range(127, 147)),
        "history_point_selection": "fixed RGB mask row/fraction grid at t0; optical flow backward to history; selected by history local depth only",
        "history_point_ids": list(range(8)),
        "source_primitive_exact": str(data["primitive"][T0]),
        "source_object_info": data["object_info"],
        "action_text_exact_passed_to_model": ACTION,
        "action_text_derivation": "English description of source primitive insert, object_info rectangle/Jeans Red, and visible blue board; not a verbatim FMB metadata field",
        "geometry_primary": "sensor Z16 5x5 positive median, 0.0001 m/raw, K_nominal",
        "geometry_crosscheck": "planar CAD PnP plus RGB rounded-CAD silhouette/TCP diagnostic",
        "calibration_status": "SUPPORTED_NOT_EXACT",
        "future_not_used_for_3D_or_K": True})
    comparison = compare_xyz(sensor, silhouette)
    rigid = rigidity(sensor)
    decision = "PASS_WITH_CALIBRATION_UNCERTAINTY" if (
        np.isfinite(sensor).all() and (sensor[:, :, 2] > 0).all() and
        max(row["rms_residual_mm"] for row in rigid["alignments"]) <= 10 and
        comparison["median_XYZ_disagreement_mm"] <= 30) else "BLOCKED"
    write_json(RUN / "preflight.json", {"source_file": SOURCE.name,
        "camera": "side_1", "camera_serial": "128422270679",
        "depth_scale_m_per_unit": DEPTH_SCALE, "depth_scale_geometry_probe": 9.804e-5,
        "source_hz": 10, "source_dt_s": .1,
        "K_nominal": variants["K_nominal"].tolist(), "K_status": "SUPPORTED_NOT_EXACT",
        "depth_rgb_alignment": "APPROXIMATE", "exact_pixel_registration": False,
        "geometry_method_primary": "SENSOR_DEPTH_K_NOMINAL",
        "geometry_crosscheck": "CAD_PNP_AND_RGB_CAD_SILHOUETTE_TCP",
        "history_points_valid": int(np.isfinite(sensor).all(axis=2).sum()),
        "history_points_total": 24,
        "sensor_kabsch_max_rms_mm": max(row["rms_residual_mm"] for row in rigid["alignments"]),
        "sensor_vs_cad_silhouette_tcp_median_XYZ_mm": comparison["median_XYZ_disagreement_mm"],
        "pnp_warning": "Planar KLT correspondences may drift; inspect sensor_vs_pnp separately.",
        "decision": decision,
        "remaining_uncertainties": ["active color K and distortion unrecorded",
            "published depth resize/registration unverified", "featureless object surface limits point identity",
            "source timestamps nominal; MolmoMotion temporal domain shift 15 Hz vs 10 Hz"]})
    frozen = ["window_selection.json", "experiment_config.json", "k_nominal.json",
              "k_sensitivity_variants.json", "history_points_2d.npy", "points_2d_at_t0.npy",
              "history_sensor_3d.npy", "history_depth_probe.json", "preflight.json"]
    write_json(RUN / "input_freeze.json", {"created_before_primary_prediction": True,
        "sha256": {name: sha256(RUN / name) for name in frozen}})
    print("RUN", RUN)
    print("preflight", decision, "sensor vs CAD silhouette median mm",
          round(comparison["median_XYZ_disagreement_mm"], 2),
          "sensor vs pure PnP median mm", round(compare_xyz(sensor, pnp)["median_XYZ_disagreement_mm"], 2),
          "Kabsch max mm", round(max(row["rms_residual_mm"] for row in rigid["alignments"]), 2))


if __name__ == "__main__":
    main()
