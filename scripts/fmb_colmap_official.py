"""Official RGB-only COLMAP controls on two FMB episodes and two separate views."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sqlite3

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dobbe_colmap_official import camera_candidate as radial_candidate, csv_write, database_audit, read, write
from dobbe_colmap_clean import summarize
from calibrate_fmb_board_k import cad_features, extract, TRAIN, HOLDOUT, HOLDOUT_FEATURES
from probe_fmb_cad_pnp import K as K_NOMINAL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/fmb_colmap_official_v1"
EPISODES = {
    "episode2": {"source": ROOT / "runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy", "t0": 126},
    "episode3": {"source": ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy", "t0": 130},
}
VIEWS = ("side_1", "wrist_1")
SCOPES = ("full", "causal")
MODEL = "SIMPLE_RADIAL"


def camera_candidate(camera):
    if camera.model == pycolmap.CameraModelId.PINHOLE:
        fx, fy, cx, cy = camera.params
        return {"model": "OPENCV", "params": [float(fx), float(fy), float(cx-.5), float(cy-.5), 0., 0., 0., 0.],
                "source_model": "PINHOLE", "pixel_convention": "COLMAP principal point minus 0.5 for OpenCV"}
    return radial_candidate(camera)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    prior = read(OUT / "protocol.json") if (OUT / "protocol.json").exists() else None
    protocol = {"episodes": {}, "runs": {}, "settings": {
        "data_type": "video", "camera_model": MODEL, "quality": "high", "single_camera": 1,
        "sparse": 1, "dense": 0, "mask": None, "manual_intrinsics": None, "manual_mapper_thresholds": None},
        "color": "Published BGR uint8 arrays saved using cv2.imwrite; no resizing or video re-encoding",
        "scope": ("Two existing FMB episodes; wrist_1 causal prefixes, additional PINHOLE control"
                  if MODEL == "PINHOLE" else "Two existing FMB episodes; separate side_1 and wrist_1 cameras, full and prefix to existing t0"),
        "oracle_rule": "Full models cannot initialize or select causal models; wrist K cannot be reused as side K",
        "coverage_pass": ">=80% input in one model, positive number of 3D points; does not certify K",
        "depth_role": "Sensor depth, CAD and TCP are post-reconstruction diagnostics only; depth scale 0.0001 m/count remains conditional",
        "tcp_role": "TCP poses describe end effector, not camera center; unknown camera-to-TCP mount prevents direct ground-truth trajectory comparison"}
    contact_rows = len(EPISODES)*len(VIEWS)
    contact_fig, contact_axes = plt.subplots(contact_rows, 4, figsize=(11, 2.5*contact_rows), constrained_layout=True)
    row = 0
    for episode, info in EPISODES.items():
        source = info["source"]
        source_sha = digest(source)
        if prior:
            assert prior["episodes"][episode]["source_sha256"] == source_sha
        data = np.load(source, allow_pickle=True).item()
        count = len(data["obs/side_1"])
        assert count > info["t0"] and data["obs/tcp_pose"].shape == (count, 7)
        protocol["episodes"][episode] = {"source": str(source), "source_sha256": source_sha,
            "frames": count, "t0": info["t0"], "keys": sorted(data),
            "array_shapes": {k: list(v.shape) for k, v in data.items() if isinstance(v, np.ndarray)}}
        for view in VIEWS:
            rgb = data[f"obs/{view}"]
            assert rgb.shape == (count, 256, 256, 3) and rgb.dtype == np.uint8
            assert data[f"obs/{view}_depth"].shape == (count, 256, 256)
            base = OUT / "png" / episode / view
            base.mkdir(parents=True, exist_ok=True)
            hashes = []
            for frame, bgr in enumerate(rgb):
                path = base / f"{frame:06d}.png"
                if not path.exists():
                    assert cv2.imwrite(str(path), bgr)
                assert np.array_equal(cv2.imread(str(path)), bgr)
                hashes.append(digest(path))
            protocol["episodes"][episode][view] = {"png_sha256": hashes, "RGB_only_COLMAP_input": True}
            for scope, frames in (("full", list(range(count))), ("causal", list(range(info["t0"]+1)))):
                if scope not in SCOPES:
                    continue
                slug = f"{episode}_{view}_{scope}"
                run = OUT / slug / "images"
                run.mkdir(parents=True, exist_ok=True)
                for frame in frames:
                    source_png = base / f"{frame:06d}.png"
                    target = run / source_png.name
                    if not target.exists():
                        shutil.copyfile(source_png, target)
                    assert digest(target) == hashes[frame]
                assert sorted(int(p.stem) for p in run.glob("*.png")) == frames
                protocol["runs"][slug] = {"episode": episode, "view": view, "scope": scope, "input_frames": frames,
                                         "eligible_for_causal_K": scope == "causal"}
            for ax, frame in zip(contact_axes[row], (0, count//3, info["t0"], count-1)):
                ax.imshow(cv2.cvtColor(rgb[frame], cv2.COLOR_BGR2RGB))
                ax.set(title=f"{episode} {view}, frame {frame}", xticks=[], yticks=[])
            row += 1
        del data
    if prior:
        assert prior == protocol
    else:
        write(OUT / "protocol.json", protocol)
    contact_fig.savefig(OUT / "input_contact_sheet.jpg", dpi=130)
    plt.close(contact_fig)
    print({"prepared": True, "episodes": {k: v["frames"] for k, v in protocol["episodes"].items()}, "runs": len(protocol["runs"])}, flush=True)


def board_check(data, camera):
    points, names, _ = cad_features(1)
    feature_ids = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    records = {}
    for frame in TRAIN+HOLDOUT:
        try:
            uv, _ = extract(data["obs/side_1"][frame], 1)
            records[frame] = uv
        except (ValueError, IndexError, cv2.error) as error:
            return {"status": "BOARD_EXTRACTION_FAILED", "frame": frame, "message": str(error)}
    train = np.array([records[f] for f in TRAIN])
    hold = np.array([records[f] for f in HOLDOUT])
    if camera is None:
        k, distortion = K_NOMINAL.copy(), np.zeros(4)
    else:
        params = camera_candidate(camera)["params"]
        k = np.array([[params[0], 0, params[2]], [0, params[1], params[3]], [0, 0, 1.]])
        distortion = np.array(params[4:])
    ok, rv, tv = cv2.solvePnP(points[feature_ids], train.mean(0)[feature_ids], k, distortion, flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return {"status": "BOARD_PNP_FAILED"}
    def project(p, pose):
        return cv2.projectPoints(p, pose[:3], pose[3:], k, distortion)[0].reshape(-1, 2)
    fit = least_squares(lambda pose: (project(points, pose)[feature_ids][None]-train[:, feature_ids]).ravel(),
                        np.r_[rv.ravel(), tv.ravel()], loss="soft_l1", f_scale=2., max_nfev=250)
    prediction = project(points, fit.x)
    error = np.linalg.norm(prediction[None]-hold, axis=2)
    grid = np.array([[u, v, -.0015] for u in np.linspace(.025, .205, 10) for v in np.linspace(.025, .205, 10)])
    uv = project(grid, fit.x)
    rotation = Rotation.from_rotvec(fit.x[:3]).as_matrix()
    xyz = grid@rotation.T+fit.x[3:]
    residuals = []
    for frame in HOLDOUT:
        hsv = cv2.cvtColor(data["obs/side_1"][frame], cv2.COLOR_BGR2HSV)
        safe = cv2.erode(cv2.inRange(hsv, (95, 90, 120), (120, 255, 255)), np.ones((7, 7), np.uint8)) > 0
        depth = data["obs/side_1_depth"][frame]
        for (u, v), p in zip(np.rint(uv).astype(int), xyz):
            if 3 <= u < 253 and 3 <= v < 253 and safe[v, u] and p[2] > 0:
                positive = depth[v-2:v+3, u-2:u+3]
                positive = positive[positive > 0]
                if len(positive) >= 10:
                    residuals.append(p[2]-float(np.median(positive))*.0001)
    return {"status": "DIAGNOSTIC_FIXED_K_RGB_POSE_FIT", "K_OpenCV": k.tolist(), "distortion": distortion.tolist(),
            "pose_only_RGB_fit_success": bool(fit.success), "rvec_tvec": fit.x.tolist(),
            "train_frames": list(TRAIN), "heldout_frames": list(HOLDOUT), "heldout_features": list(HOLDOUT_FEATURES),
            "holdout_all_feature_rmse_px": float(np.sqrt(np.mean(error**2))),
            "holdout_separate_feature_rmse_px": float(np.sqrt(np.mean(error[:, HOLDOUT_FEATURES]**2))),
            "depth_samples": len(residuals),
            "median_abs_Z_error_mm": float(np.median(np.abs(residuals))*1000) if residuals else None,
            "median_signed_Z_error_mm": float(np.median(residuals)*1000) if residuals else None,
            "limitations": "Approximate segmented CAD outline/hole features; depth registration, scale and color intrinsics not certified; K is frozen, only board pose fits RGB"}


def depth_check(rec, data, view):
    observations = []
    for point in rec.points3D.values():
        for el in point.track.elements:
            image = rec.images[el.image_id]
            frame = int(Path(image.name).stem)
            u, v = np.rint(image.points2D[el.point2D_idx].xy-.5).astype(int)
            if not 2 <= u < 254 or not 2 <= v < 254:
                continue
            patch = data[f"obs/{view}_depth"][frame, v-1:v+2, u-1:u+2]
            values = patch[patch > 0]*.0001
            if len(values) < 7 or np.ptp(values) > .02:
                continue
            predicted = float((image.cam_from_world() * point.xyz)[2])
            if predicted > 1e-8:
                observations.append((frame, predicted, float(np.median(values))))
    lengths = [p.track.length() for p in rec.points3D.values()]
    if not observations:
        return {"status": "NO_VALID_DEPTH_OBSERVATIONS", "observations": 0}
    a = np.array(observations)
    scale = float(np.median(a[:, 2]/a[:, 1]))
    relative = np.abs(scale*a[:, 1]-a[:, 2])/a[:, 2]
    rows = []
    for frame in sorted(set(a[:, 0].astype(int))):
        selected = a[a[:, 0] == frame]
        rows.append({"frame": int(frame), "n": len(selected),
                     "median_scale": float(np.median(selected[:, 2]/selected[:, 1])),
                     "median_relative_residual": float(np.median(np.abs(scale*selected[:, 1]-selected[:, 2])/selected[:, 2]))})
    return {"status": "DIAGNOSTIC_ONE_GLOBAL_SCALE_TO_SENSOR_Z", "observations": len(a),
            "track_length_median": float(np.median(lengths)), "median_depth_to_sfm_scale": scale,
            "median_relative_depth_residual": float(np.median(relative)), "p90_relative_depth_residual": float(np.percentile(relative, 90)),
            "per_frame": rows, "limitations": "Scale fits these same samples; moving robot/object points included; unverified RGB-depth registration and units; not K certification"}


def motion_check(rec, models, view):
    trajectory = models["trajectory"]
    centers = np.array([t["camera_center"] for t in trajectory])
    rotations = np.array([t["c2w_colmap"] for t in trajectory])[:, :3, :3]
    angle = Rotation.from_matrix(rotations[0].T@rotations).magnitude()*180/np.pi
    depths = [float((rec.images[el.image_id].cam_from_world()*p.xyz)[2])
              for p in rec.points3D.values() for el in p.track.elements]
    typical = float(np.median([z for z in depths if z > 0])) if any(z > 0 for z in depths) else None
    span = float(np.linalg.norm(np.ptp(centers, axis=0)))
    return {"center_span_sfm_units": span, "median_point_Z_sfm_units": typical,
            "span_over_median_scene_depth": span/typical if typical and typical > 0 else None,
            "max_rotation_from_first_deg": float(angle.max()),
            "known_mount": "Fixed environment camera" if view == "side_1" else "Camera attached to moving wrist",
            "limitation": "A nonconstant side-camera trajectory cannot be interpreted as physical camera motion; wrist TCP-to-camera transform is unknown"}


def pair_motion(run):
    with sqlite3.connect(f"file:{run / 'database.db'}?mode=ro", uri=True) as db:
        names = dict(db.execute("SELECT image_id,name FROM images"))
        keypoints = {i: np.frombuffer(b, np.float32).reshape(n, c)[:, :2].copy() if n else np.empty((0, 2))
                     for i, n, c, b in db.execute("SELECT image_id,rows,cols,data FROM keypoints")}
        rows = []
        for pid, n, blob in db.execute("SELECT pair_id,rows,data FROM two_view_geometries WHERE rows>0"):
            a, b = divmod(pid, 2147483647)
            matches = np.frombuffer(blob, np.uint32).reshape(n, 2)
            flow = np.linalg.norm(keypoints[a][matches[:, 0]]-keypoints[b][matches[:, 1]], axis=1)
            rows.append({"frame_a": int(Path(names[a]).stem), "frame_b": int(Path(names[b]).stem),
                         "n": n, "median_displacement_px": float(np.median(flow)),
                         "fraction_displacement_lt1px": float(np.mean(flow < 1))})
    csv_write(run / "verified_pair_displacements.csv", rows)
    return {"nonzero_pairs": len(rows), "median_pair_displacement_px": float(np.median([r["median_displacement_px"] for r in rows])) if rows else None,
            "median_pair_fraction_lt1px": float(np.median([r["fraction_displacement_lt1px"] for r in rows])) if rows else None,
            "limitation": "Image displacement is not triangulation angle; planar/static matches can pass verification"}


def analyze():
    protocol = read(OUT / "protocol.json")
    write(OUT / "validation_sources.json", {
        "board_CAD_features": "runs/fmb_effective_k_256_calibration/peg_board_top_wires.json",
        "board_CAD_sha256": digest(ROOT / "runs/fmb_effective_k_256_calibration/peg_board_top_wires.json"),
        "board_feature_extractor": "scripts/calibrate_fmb_board_k.py",
        "board_feature_extractor_sha256": digest(ROOT / "scripts/calibrate_fmb_board_k.py"),
        "source_array_hashes": {name: info["source_sha256"] for name, info in protocol["episodes"].items()},
        "depth_scale_m_per_count": .0001, "depth_semantics": "Z hypothesis, matching published pixel grids, registration not certified",
        "side_1_nominal_K_OpenCV": K_NOMINAL.tolist(), "nominal_K_status": "Historical profile-resize hypothesis, not active color ground truth",
        "RGB_board_pose_fit": "Fixed candidate K and k; fit board extrinsics only on frames 0,16,32,48, exclude features 8,12,16",
        "RGB_board_validation": "Frames 8,24,40,56 and excluded features; all within causal prefixes",
        "scope": "Diagnostics only, neither source depth/TCP nor nominal K is passed to COLMAP"})
    rows, graphs, all_models = [], [], []
    for episode, info in EPISODES.items():
        data = np.load(info["source"], allow_pickle=True).item()
        if "side_1" in VIEWS:
            nominal_board = board_check(data, None)
            write(OUT / f"{episode}_nominal_board_check.json", nominal_board)
        for slug, config in protocol["runs"].items():
            if config["episode"] != episode or not (OUT / slug / "execution.json").exists():
                continue
            run = OUT / slug
            receipt = read(run / "execution.json")
            assert receipt["exit_code"] == 0
            frames = config["input_frames"]
            graph, pairs = database_audit(run, frames)
            recs = {int(p.name): pycolmap.Reconstruction(p) for p in (run / "sparse").glob("*")
                    if p.is_dir() and (p / "cameras.bin").exists()}
            models = summarize(recs)
            for model in models:
                rec = recs[model["id"]]
                model["motion"] = motion_check(rec, model, config["view"])
                model["depth_validation"] = depth_check(rec, data, config["view"])
                if config["view"] == "side_1":
                    model["board_validation"] = board_check(data, next(iter(rec.cameras.values())))
                csv_write(run / "sparse_txt" / str(model["id"]) / "trajectory.csv", [
                    {"frame": t["frame"], "center_x": t["camera_center"][0], "center_y": t["camera_center"][1],
                     "center_z": t["camera_center"][2]} for t in model["trajectory"]])
                all_models.append((slug, model))
            best = models[0]["registered"] if models else 0
            log = (run / "colmap.log").read_text(encoding="utf-8", errors="replace")
            summary = {**config, "run": slug, "models": models, "graph": graph, "execution": receipt,
                "best_registered": best, "unique_registered": len({t["frame"] for m in models for t in m["trajectory"]}),
                "coverage_pass": bool(models and best >= .8*len(frames) and models[0]["points3D"] > 0),
                "verified_pair_motion": pair_motion(run),
                "log_diagnostics": {"linear_solver_failure_warnings": log.count("Linear solver failure"),
                    "covariant_sift_cpu": "Creating Covariant SIFT CPU feature extractor" in log,
                    "sift_gpu_matching": "Creating SIFT GPU feature matcher" in log,
                    "vocabulary_loop_pairing": "Generating image pairs with vocabulary tree" in log}}
            write(run / "summary.json", summary)
            rows.append({"run": slug, "view": config["view"], "scope": config["scope"], "images": len(frames),
                "models": len(models), "largest_model": best, "unique_registered": summary["unique_registered"],
                "largest_points3D": models[0]["points3D"] if models else 0, "coverage_pass": summary["coverage_pass"],
                "keypoints_median": graph["keypoints_median"], "pairs_ge100": graph["by_threshold"]["100"]["pairs"],
                "largest_component_ge100": graph["by_threshold"]["100"]["largest_component"]})
            graphs.append((slug, frames, pairs))
        del data
    csv_write(OUT / "experiment_results.csv", rows)
    camera_rows = []
    for slug, model in all_models:
        cam = model["cameras"][0]
        depth = model["depth_validation"]
        board = model.get("board_validation", {})
        camera_rows.append({"run": slug, "model_id": model["id"], "registered": model["registered"],
            "points3D": model["points3D"], "camera_model": MODEL,
            "f_px": cam["params"][0], "fy_px": cam["fy"], "k": cam["params"][3] if MODEL == "SIMPLE_RADIAL" else None,
            "internal_reprojection_px": model["mean_reprojection_px"],
            "camera_span_over_scene_depth": model["motion"]["span_over_median_scene_depth"],
            "max_rotation_deg": model["motion"]["max_rotation_from_first_deg"],
            "depth_median_relative_error": depth.get("median_relative_depth_residual"),
            "board_holdout_all_rmse_px": board.get("holdout_all_feature_rmse_px"),
            "board_median_abs_Z_error_mm": board.get("median_abs_Z_error_mm")})
    csv_write(OUT / "camera_diagnostics.csv", camera_rows)
    write(OUT / "decision.json", {"status": "COMPLETE" if len(rows) == len(protocol["runs"]) else "IN_PROGRESS", "runs": rows,
        "conditional_stride_required": False, "validated_side_1_K": None, "validated_wrist_1_K": None,
        "scope": f"{len(protocol['runs'])} unmasked {MODEL} automatic video controls, two episodes; not a dataset-wide benchmark",
        "limitations": "Fixed side camera has no camera-translation parallax; wrist and side K are camera specific; depth registration/scale and active color calibration are not independently certified"})
    if graphs:
        plot_rows = max(1, (len(graphs)+1)//2)
        fig, axes = plt.subplots(plot_rows, 2, figsize=(10, 4*plot_rows), constrained_layout=True, squeeze=False)
        for ax, (slug, frames, pairs) in zip(axes.flat, graphs):
            matrix = np.zeros((len(frames), len(frames)))
            for p in pairs:
                matrix[p["frame_a"], p["frame_b"]] = matrix[p["frame_b"], p["frame_a"]] = p["verified_inliers"]
            im = ax.imshow(matrix, vmin=0, vmax=250, cmap="viridis")
            ax.set(title=slug, xlabel="Source frame", ylabel="Source frame")
        for ax in axes.flat[len(graphs):]:
            ax.set_visible(False)
        fig.colorbar(im, ax=list(axes.flat), label="Verified inliers (clipped at 250)", shrink=.5)
        fig.savefig(OUT / "verified_match_graphs.png", dpi=120)
        plt.close(fig)
    if all_models:
        fig, ax = plt.subplots(figsize=(11, max(4, .3*len(all_models))), constrained_layout=True)
        labels = []
        for i, (slug, model) in enumerate(all_models):
            ax.scatter([t["frame"] for t in model["trajectory"]], np.full(model["registered"], i), s=8)
            labels.append(f"{slug}: {model['id']} ({model['registered']} frames)")
        ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Source frame", title="Registered frames by independent model")
        ax.invert_yaxis()
        fig.savefig(OUT / "registered_frame_coverage.png", dpi=140)
        plt.close(fig)
    print({"results": rows}, flush=True)


def audit():
    protocol, decision = read(OUT / "protocol.json"), read(OUT / "decision.json")
    assert decision["status"] == "COMPLETE" and len(decision["runs"]) == len(protocol["runs"])
    for episode, info in EPISODES.items():
        assert digest(info["source"]) == protocol["episodes"][episode]["source_sha256"]
    checks = []
    for slug, config in protocol["runs"].items():
        run = OUT / slug
        summary = read(run / "summary.json")
        frames = config["input_frames"]
        assert sorted(int(p.stem) for p in (run / "images").glob("*.png")) == frames
        hashes = protocol["episodes"][config["episode"]][config["view"]]["png_sha256"]
        assert all(digest(run / "images" / f"{f:06d}.png") == hashes[f] for f in frames)
        options = dict(zip(summary["execution"]["argv"][1::2], summary["execution"]["argv"][2::2]))
        assert options["--single_camera"] == "1" and options["--data_type"] == "video"
        assert options["--camera_model"] == MODEL and options["--quality"] == "high"
        assert options["--dense"] == "0" and options["--sparse"] == "1"
        assert set(options) == {"--workspace_path", "--image_path", "--data_type", "--single_camera", "--camera_model", "--quality", "--sparse", "--dense", "--use_gpu"}
        assert summary["eligible_for_causal_K"] == (config["scope"] == "causal")
        graph, _ = database_audit(run, frames)
        assert graph == summary["graph"]
        assert not (run / "dense").exists()
        projection_error = 0.
        for model in summary["models"]:
            rec = pycolmap.Reconstruction(run / "sparse" / str(model["id"]))
            assert rec.num_reg_images() == model["registered"] and rec.num_points3D() == model["points3D"]
            assert {t["frame"] for t in model["trajectory"]} <= set(frames)
            text_path = run / "sparse_txt" / str(model["id"])
            assert all((text_path / f"{name}.txt").exists()
                       for name in ("cameras", "images", "points3D", "rigs", "frames"))
            text_rec = pycolmap.Reconstruction(text_path)
            assert text_rec.num_reg_images() == rec.num_reg_images()
            assert text_rec.num_points3D() == rec.num_points3D()
            assert len(rec.cameras) == len(text_rec.cameras) == 1
            for camera_id, camera in rec.cameras.items():
                assert camera.model == getattr(pycolmap.CameraModelId, MODEL)
                assert np.allclose(camera.params, text_rec.cameras[camera_id].params, rtol=1e-12, atol=1e-10)
                p = camera_candidate(camera)["params"]
                k = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.]])
                xyz = np.array([[.01, .02, 1.], [-.04, .03, 1.5], [.08, -.06, 2.]])
                pixels = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), k, np.array(p[4:]))[0].reshape(-1, 2)
                expected = camera.img_from_cam(xyz)-.5
                assert np.allclose(pixels, expected, rtol=1e-12, atol=1e-8)
                projection_error = max(projection_error, float(np.max(np.abs(pixels-expected))))
            trajectory = {t["frame"]: t for t in model["trajectory"]}
            for image_id in rec.reg_image_ids():
                image, text_image = rec.images[image_id], text_rec.images[image_id]
                assert image.name == text_image.name
                assert np.allclose(image.cam_from_world().matrix(), text_image.cam_from_world().matrix(), atol=1e-10)
                t = trajectory[int(Path(image.name).stem)]
                assert np.allclose(image.projection_center(), t["camera_center"], atol=1e-10)
                assert np.allclose(image.cam_from_world().inverse().matrix(), np.array(t["c2w_colmap"])[:3], atol=1e-10)
            assert set(rec.points3D) == set(text_rec.points3D)
            for point_id, point in rec.points3D.items():
                assert np.allclose(point.xyz, text_rec.points3D[point_id].xyz, rtol=1e-12, atol=1e-10)
        checks.append({"run": slug, "hashes_and_subsets": True, "RGB_only_default_CLI": True,
                       "SQL_counts_and_no_focal_prior": True, "native_models_and_TXT": True,
                       "binary_TXT_and_trajectory_values": True,
                       "COLMAP_OpenCV_projection_max_error_px": projection_error})
    write(OUT / "audit.json", {"status": "PASS", "checks": checks, "validated_K_claimed": False})
    print({"audit": "PASS", "runs": len(checks)}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "analyze", "audit"])
    p.add_argument("--profile", choices=["baseline", "wrist-pinhole"], default="baseline")
    args = p.parse_args()
    if args.profile == "wrist-pinhole":
        OUT = ROOT / "runs/fmb_colmap_official_pinhole_v1"
        VIEWS, SCOPES, MODEL = ("wrist_1",), ("causal",), "PINHOLE"
    {"prepare": prepare, "analyze": analyze, "audit": audit}[args.mode]()
