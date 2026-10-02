"""Prepare and inspect official automatic_reconstructor outputs, without fitting K."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from dobbe_colmap_clean import components, summarize
from inspect_hony_scenes import liblzfse
from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_colmap_official_v1"
SOURCE = ROOT / "data/dobbe_oxe/target_raw"
PAIR_BASE = 2147483647
RUNS = {"full": list(range(243)), "causal": list(range(96)),
        "causal_stride5": list(range(0, 96, 5))}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, rows[0].keys(), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    video = SOURCE / "compressed_video_h264.mp4"
    images = OUT / "decoded_png"
    images.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    argv = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(video),
            "-vsync", "0", "-start_number", "0", str(images / "%06d.png")]
    if not (OUT / "protocol.json").exists():
        if list(images.iterdir()):
            raise RuntimeError("Incomplete decode exists; use a fresh output directory")
        subprocess.run(argv, check=True)
    pngs = sorted(images.glob("*.png"))
    assert [p.name for p in pngs] == [f"{i:06d}.png" for i in range(243)]
    assert all(cv2.imread(str(p)).shape == (256, 256, 3) for p in pngs)
    protocol = {"source": str(video.relative_to(ROOT)), "source_sha256": sha(video),
                "decoded_images": [{"frame": i, "name": p.name, "sha256": sha(p)}
                                   for i, p in enumerate(pngs)],
                "decode_argv": argv, "frame_count": 243, "image_size": [256, 256],
                "data_type": "video", "single_camera": True,
                "camera_model": "SIMPLE_RADIAL", "quality": "high",
                "sparse": True, "dense": False, "mask": None,
                "manual_camera_params": None, "manual_mapper_options": None,
                "oracle_rule": "Full output cannot initialize or select causal reconstruction",
                "conditional_stride_rule": "Only if full largest model >=80% input and causal largest <80% input",
                "pass_definition": "At least 80% of input in one model; diagnostic coverage only, not validation of K",
                "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]}
    if (OUT / "protocol.json").exists():
        assert read(OUT / "protocol.json") == protocol
    else:
        write(OUT / "protocol.json", protocol)
    for slug in ("full", "causal"):
        prepare_subset(slug)
    print(json.dumps({"prepared": True, "source_sha256": protocol["source_sha256"]}), flush=True)


def prepare_subset(slug):
    images = OUT / slug / "images"
    images.mkdir(parents=True, exist_ok=True)
    for i in RUNS[slug]:
        source = OUT / "decoded_png" / f"{i:06d}.png"
        target = images / source.name
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha(source) == sha(target)
    assert sorted(int(p.stem) for p in images.glob("*.png")) == RUNS[slug]


def database_audit(run, frames):
    with sqlite3.connect(f"file:{run / 'database.db'}?mode=ro", uri=True) as db:
        ids = dict(db.execute("SELECT image_id,name FROM images"))
        raw = dict(db.execute("SELECT pair_id,rows FROM matches"))
        verified = {pid: (n, cfg) for pid, n, cfg in db.execute(
            "SELECT pair_id,rows,config FROM two_view_geometries")}
        camera_rows = [{"camera_id": i, "model_id": m, "width": w, "height": h,
                        "params": np.frombuffer(p, np.float64).tolist(),
                        "has_prior_focal_length": bool(prior)} for i, m, w, h, p, prior in db.execute(
            "SELECT camera_id,model,width,height,params,prior_focal_length FROM cameras")]
        keypoints = dict(db.execute("SELECT image_id,rows FROM keypoints"))
        descriptors = dict(db.execute("SELECT image_id,rows FROM descriptors"))
    assert sorted(int(Path(name).stem) for name in ids.values()) == frames
    assert len(camera_rows) == 1 and not camera_rows[0]["has_prior_focal_length"]
    pairs = []
    for pid in sorted(raw.keys() | verified.keys()):
        a, b = divmod(pid, PAIR_BASE)
        n, cfg = verified.get(pid, (0, 0))
        pairs.append({"image_id_a": a, "image_id_b": b,
                      "frame_a": int(Path(ids[a]).stem), "frame_b": int(Path(ids[b]).stem),
                      "raw_matches": raw.get(pid, 0), "verified_inliers": n, "geometry_config": cfg})
    image_rows = []
    for i, name in sorted(ids.items()):
        relevant = [p for p in pairs if i in (p["image_id_a"], p["image_id_b"])]
        image_rows.append({"frame": int(Path(name).stem), "keypoints": keypoints.get(i, 0),
                           "descriptors": descriptors.get(i, 0),
                           "connected_images_ge15": sum(p["verified_inliers"] >= 15 for p in relevant),
                           "connected_images_ge30": sum(p["verified_inliers"] >= 30 for p in relevant)})
    csv_write(run / "match_pairs.csv", pairs)
    csv_write(run / "match_images.csv", image_rows)
    result = {"images": len(ids), "attempted_pairs": len(pairs),
              "possible_exhaustive_pairs": len(frames) * (len(frames)-1) // 2,
              "keypoints_median": float(np.median(list(keypoints.values()))),
              "keypoints_total": sum(keypoints.values()),
              "raw_matches_total": sum(raw.values()),
              "verified_inliers_total": sum(n for n, _ in verified.values()),
              "nonzero_verified_pairs": sum(n > 0 for n, _ in verified.values()),
              "by_threshold": {str(t): components(ids, pairs, t) for t in (15, 30, 50, 100)},
              "database_cameras": camera_rows}
    write(run / "match_graph.json", result)
    return result, pairs


def load_measurements():
    labels = read(SOURCE / "labels.json")
    assert len(labels) == 243 and sorted(map(int, labels)) == list(range(243))
    poses = np.repeat(np.eye(4)[None], 243, axis=0)
    poses[:, :3, :3] = Rotation.from_quat([labels[str(i)]["quats"] for i in range(243)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(243)]
    depth = np.frombuffer(liblzfse.decompress((SOURCE / "compressed_np_depth_float32.bin").read_bytes()),
                          np.float32).reshape(-1, 192, 256)
    assert len(depth) == 243
    return poses, depth


def sim3(trajectory, poses):
    frames = [t["frame"] for t in trajectory]
    x = np.array([t["camera_center"] for t in trajectory])
    y = poses[frames, :3, 3]
    xc, yc = x-x.mean(0), y-y.mean(0)
    covariance = yc.T @ xc / len(x)
    u, sv, vt = np.linalg.svd(covariance)
    sign = np.diag([1., 1., np.linalg.det(u @ vt)])
    r = u @ sign @ vt
    variance = np.sum(xc**2) / len(x)
    if variance < 1e-15:
        return {"status": "DEGENERATE_CAMERA_CENTERS", "n": len(x)}
    scale = float(np.sum(sv * np.diag(sign)) / variance)
    translation = y.mean(0)-scale*r@x.mean(0)
    aligned = scale*x@r.T+translation
    residual = np.linalg.norm(aligned-y, axis=1)
    label_optical = poses[frames, :3, :3] @ CAMERA_TO_LABEL
    rgb_rotation = np.array([t["c2w_colmap"] for t in trajectory])[:, :3, :3]
    angle = Rotation.from_matrix(label_optical.transpose(0, 2, 1) @ r @ rgb_rotation).magnitude()*180/np.pi
    extent = float(np.linalg.norm(yc, axis=1).max())
    return {"status": "DIAGNOSTIC_IN_SAMPLE_ALIGNMENT", "n": len(x), "frame_ids": frames,
            "scale": scale, "rotation": r.tolist(), "translation": translation.tolist(),
            "covariance_singular_values": sv.tolist(),
            "label_path_extent_from_mean_m": extent,
            "ate_rmse_m": float(np.sqrt(np.mean(residual**2))),
            "ate_median_m": float(np.median(residual)), "ate_p90_m": float(np.percentile(residual, 90)),
            "rotation_error_median_deg": float(np.median(angle)),
            "rotation_error_p90_deg": float(np.percentile(angle, 90)),
            "aligned_centers": aligned.tolist(), "label_centers": y.tolist(),
            "limitations": "Fitted on these same frames; short or nearly collinear paths constrain Sim(3) poorly; no calibration fitted"}


def camera_candidate(camera):
    f, cx, cy, radial = camera.params
    return {"model": "OPENCV", "params": [float(f), float(f), float(cx-.5), float(cy-.5), float(radial), 0., 0., 0.],
            "source_model": "SIMPLE_RADIAL", "pixel_convention": "COLMAP principal point minus 0.5 for OpenCV"}


def track_diagnostics(rec, poses, depth):
    masks = ROOT / "runs/dobbe_colmap_clean_rerun_v1/masks/main"
    mask_cache = {int(Path(im.name).stem): cv2.imread(str(masks / f"{int(Path(im.name).stem):04d}.png.png"), 0)
                  for im in rec.images.values()}
    samples = {"all": [], "frozen_background_mask": []}
    lengths = []
    for pid, point in sorted(rec.points3D.items()):
        lengths.append(point.track.length())
        for el in point.track.elements:
            image = rec.images[el.image_id]
            frame = int(Path(image.name).stem)
            uv = np.asarray(image.points2D[el.point2D_idx].xy)-.5
            z, spread = depth_at(depth[frame], *uv)
            if not np.isfinite(z) or spread > .08:
                continue
            camera = rec.cameras[image.camera_id]
            ray_xy = camera.cam_from_img(uv+.5)
            if ray_xy is None:
                continue
            ray_xy = np.asarray(ray_xy)
            if not np.all(np.isfinite(ray_xy)):
                continue
            ray = np.r_[ray_xy, 1.]
            sfm_z = float((image.cam_from_world() * point.xyz)[2])
            sample = (pid, frame, ray, z, sfm_z)
            samples["all"].append(sample)
            mask = mask_cache[frame]
            xy = np.rint(uv).astype(int)
            if mask is not None and 0 <= xy[0] < 256 and 0 <= xy[1] < 256 and mask[xy[1], xy[0]] > 0:
                samples["frozen_background_mask"].append(sample)
    diagnostics = {"points": len(lengths), "track_length_median": float(np.median(lengths)) if lengths else None,
                   "observations": sum(lengths), "groups": {},
                   "mask_role": "Post-reconstruction diagnostic only; no masks were provided to COLMAP"}
    for name, observations in samples.items():
        variants = {}
        for kind in ("z", "ray"):
            worlds, ratios, predicted, measured = {}, [], [], []
            for pid, frame, ray, z, sfm_z in observations:
                optical = ray*z if kind == "z" else ray/np.linalg.norm(ray)*z
                world = poses[frame, :3, :3] @ CAMERA_TO_LABEL @ optical + poses[frame, :3, 3]
                worlds.setdefault(pid, []).append(world)
                predicted_depth = sfm_z if kind == "z" else sfm_z*np.linalg.norm(ray)
                if predicted_depth > 1e-8:
                    ratios.append(z/predicted_depth)
                    predicted.append(predicted_depth)
                    measured.append(z)
            scatters = [np.linalg.norm(np.array(w)-np.median(w, axis=0), axis=1)
                        for w in worlds.values() if len(w) >= 3]
            all_scatter = np.concatenate(scatters) if scatters else np.array([])
            scale = float(np.median(ratios)) if ratios else None
            rel = np.abs(scale*np.array(predicted)-measured)/measured if scale is not None else np.array([])
            variants[kind] = {"observations_valid_depth": len(observations), "tracks_ge3_observations": len(scatters),
                              "median_world_scatter_m": float(np.median(all_scatter)) if len(all_scatter) else None,
                              "p90_world_scatter_m": float(np.percentile(all_scatter, 90)) if len(all_scatter) else None,
                              "median_depth_to_sfm_scale": scale,
                              "median_relative_depth_residual": float(np.median(rel)) if len(rel) else None}
        diagnostics["groups"][name] = variants
    return diagnostics


def analyze():
    poses, depth = load_measurements()
    steps = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    rotations = Rotation.from_matrix(poses[:-1, :3, :3].transpose(0, 2, 1) @ poses[1:, :3, :3]).magnitude()*180/np.pi
    jump_frames = (np.flatnonzero(steps > .1)+1).tolist()
    write(OUT / "label_motion_audit.json", {
        "median_step_m": float(np.median(steps)), "maximum_step_m": float(steps.max()),
        "largest_steps": [{"a": int(i), "b": int(i+1), "translation_m": float(steps[i]),
                           "rotation_deg": float(rotations[i])} for i in np.argsort(steps)[-10:][::-1]],
        "jump_destination_frames_gt_0_1m": jump_frames,
        "rule": "Post-hoc diagnostic prompted by plotted labels discontinuity; drop jump destination frames, align stable segments separately",
        "cause": "Not established; discontinuity may affect full trajectory and label-based track diagnostics"})
    points_path = ROOT / "runs/dobbe_colmap_clean_rerun_v1/causal_validation_correspondences.json"
    points = read(points_path)
    mask_policy = ROOT / "runs/dobbe_colmap_clean_rerun_v1/mask_policy_frozen.json"
    write(OUT / "validation_sources.json", {
        "labels_sha256": sha(SOURCE / "labels.json"),
        "measured_depth_sha256": sha(SOURCE / "compressed_np_depth_float32.bin"),
        "correspondences_source": str(points_path.relative_to(ROOT)),
        "correspondences_sha256": sha(points_path), "point_counts": [len(p["selected"]) for p in points],
        "mask_policy_sha256": sha(mask_policy),
        "optical_to_label": CAMERA_TO_LABEL.tolist(),
        "labels_pose_convention": "c2w, quaternion xyzw; previously audited DobbE exporter basis",
        "rgb_depth_pixel_mapping": "u_d=u_cv; v_d=(v_cv+0.5)*192/256-0.5",
        "role": "Independent diagnostics only; no depth, labels, masks or correspondences passed to COLMAP",
        "static_sample_limit": "22 automatic RGB matches, not manually certified static landmarks; Z/ray unresolved"})
    results, graphs, trajectories = [], [], []
    for slug, frames in RUNS.items():
        run = OUT / slug
        if not (run / "execution.json").exists():
            continue
        receipt = read(run / "execution.json")
        assert receipt["exit_code"] == 0, receipt
        graph, pairs = database_audit(run, frames)
        recs = {int(p.name): pycolmap.Reconstruction(p) for p in sorted((run / "sparse").glob("*"))
                if p.is_dir() and (p / "cameras.bin").exists()}
        models = summarize(recs)
        for model in models:
            rec = recs[model["id"]]
            txt = run / "sparse_txt" / str(model["id"])
            assert all((txt / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt"))
            model["trajectory_validation"] = sim3(model["trajectory"], poses)
            chunks, current = [], []
            for item in model["trajectory"]:
                if item["frame"] in jump_frames:
                    if current:
                        chunks.append(current)
                    current = []
                else:
                    if current and any(current[-1]["frame"] < f < item["frame"] for f in jump_frames):
                        chunks.append(current)
                        current = []
                    current.append(item)
            if current:
                chunks.append(current)
            registered_frames = [t["frame"] for t in model["trajectory"]]
            if any(min(registered_frames) <= f <= max(registered_frames) for f in jump_frames):
                model["trajectory_validation_stable_label_segments"] = [sim3(chunk, poses) for chunk in chunks if len(chunk) >= 10]
            model["track_depth_validation"] = track_diagnostics(rec, poses, depth)
            camera = next(iter(rec.cameras.values()))
            candidate = camera_candidate(camera)
            model["static_K_validation"] = {"camera": candidate,
                "oracle_K": slug == "full", "correspondences_sha256": sha(points_path),
                "point_counts": [len(p["selected"]) for p in points],
                "variants": {kind: evaluate(points, poses, candidate, "c2w", kind) for kind in ("z", "ray")}}
            csv_write(txt / "trajectory.csv", [{"frame": t["frame"],
                "center_x": t["camera_center"][0], "center_y": t["camera_center"][1],
                "center_z": t["camera_center"][2]} for t in model["trajectory"]])
        best = max((m["registered"] for m in models), default=0)
        summary = {"run": slug, "input_frames": frames, "models": models, "model_count": len(models),
                   "best_registered": best, "unique_registered": len({t["frame"] for m in models for t in m["trajectory"]}),
                   "coverage_pass": best >= .8*len(frames), "eligible_for_causal_K": slug != "full",
                   "execution": receipt, "graph": graph,
                   "baseline_static_K_z": evaluate(points, poses, {"model": "PINHOLE", "params": [200, 200, 128, 128]}, "c2w", "z")}
        log = (run / "colmap.log").read_text(encoding="utf-8", errors="replace")
        summary["log_diagnostics"] = {
            "linear_solver_failure_warnings": log.count("Linear solver failure"),
            "covariant_sift_cpu": "Creating Covariant SIFT CPU feature extractor" in log,
            "sift_gpu_matching": "Creating SIFT GPU feature matcher" in log,
            "vocabulary_loop_pairing": "Generating image pairs with vocabulary tree" in log}
        write(run / "summary.json", summary)
        results.append({"run": slug, "images": len(frames), "models": len(models),
                        "largest_model": best, "unique_registered": summary["unique_registered"],
                        "coverage_pass": summary["coverage_pass"], "keypoints_median": graph["keypoints_median"],
                        "attempted_pairs": graph["attempted_pairs"],
                        **{f"pairs_ge{t}": graph["by_threshold"][str(t)]["pairs"] for t in (15, 30, 50, 100)}})
        graphs.append((slug, frames, pairs))
        if models:
            trajectories.append((slug, models[0]["trajectory_validation"]))
    csv_write(OUT / "experiment_results.csv", results)
    by_run = {r["run"]: r for r in results}
    do_stride = by_run.get("full", {}).get("coverage_pass", False) and not by_run.get("causal", {}).get("coverage_pass", False)
    write(OUT / "decision.json", {"status": "OFFICIAL_BASELINE_COMPLETE" if len(results) >= 2 else "IN_PROGRESS",
          "runs": results, "conditional_stride_required": do_stride,
          "conditional_stride_completed": "causal_stride5" in by_run,
          "validated_causal_K": None,
          "causal_K_external_validation": "Not accepted: largest causal model reprojection on fixed 22 measured-depth/labels correspondences is worse than diagnostic baseline",
          "full_label_motion_caveat": "See label_motion_audit.json and post-hoc stable-segment Sim(3) results in full summary",
          "pass_definition": ">=80% input in one reconstruction; K additionally needs external validation",
          "limitations": "Fragmentation alone does not prove COLMAP is impossible; all-points SfM may follow apparatus; RGB is processed 256x256"})
    if graphs:
        fig, axes = plt.subplots(1, len(graphs), figsize=(5*len(graphs), 4), squeeze=False, constrained_layout=True)
        for ax, (slug, frames, pairs) in zip(axes.flat, graphs):
            index = {f: i for i, f in enumerate(frames)}
            matrix = np.zeros((len(frames), len(frames)))
            for p in pairs:
                a, b = index[p["frame_a"]], index[p["frame_b"]]
                matrix[a, b] = matrix[b, a] = p["verified_inliers"]
            im = ax.imshow(matrix, vmin=0, vmax=150, cmap="viridis")
            ax.set(title=slug, xlabel="Selected frame index", ylabel="Selected frame index")
        fig.colorbar(im, ax=list(axes.flat), label="Verified inliers (clipped at 150)")
        fig.savefig(OUT / "verified_match_graphs.png", dpi=160)
        plt.close(fig)
    if trajectories:
        fig, axes = plt.subplots(1, len(trajectories), figsize=(6*len(trajectories), 4), squeeze=False, constrained_layout=True)
        for ax, (slug, alignment) in zip(axes.flat, trajectories):
            if "aligned_centers" not in alignment:
                continue
            rgb, labels = np.array(alignment["aligned_centers"]), np.array(alignment["label_centers"])
            for j, axis in enumerate("xyz"):
                ax.plot(alignment["frame_ids"], labels[:, j], label=f"labels {axis}")
                ax.plot(alignment["frame_ids"], rgb[:, j], "--", label=f"COLMAP {axis}")
            ax.set(title=f"{slug}: largest model, Sim(3)", xlabel="Source frame", ylabel="Position (m)")
            for frame in jump_frames:
                if min(alignment["frame_ids"]) <= frame <= max(alignment["frame_ids"]):
                    ax.axvline(frame, color="grey", alpha=.5, linewidth=.8)
            ax.legend(fontsize=7)
        fig.savefig(OUT / "trajectory_alignment.png", dpi=160)
        plt.close(fig)
    if results:
        fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
        labels, row = [], 0
        for result in results:
            summary = read(OUT / result["run"] / "summary.json")
            for model in summary["models"]:
                frame_ids = [t["frame"] for t in model["trajectory"]]
                ax.scatter(frame_ids, np.full(len(frame_ids), row), s=13)
                labels.append(f"{result['run']} model {model['id']} ({len(frame_ids)} frames)")
                row += 1
        ax.set(yticks=range(row), yticklabels=labels, xlabel="Source frame index",
               xlim=(-2, 244), title="Registered frames in each independent model")
        ax.axvline(95.5, color="grey", linestyle="--", linewidth=1)
        ax.invert_yaxis()
        fig.savefig(OUT / "registered_frame_coverage.png", dpi=160)
        plt.close(fig)
    causal = OUT / "causal" / "summary.json"
    if causal.exists() and read(causal)["models"]:
        model_id = read(causal)["models"][0]["id"]
        rec = pycolmap.Reconstruction(OUT / "causal" / "sparse" / str(model_id))
        by_frame = {int(Path(im.name).stem): im for im in rec.images.values()}
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)
        for ax, frame in zip(axes, (0, 32, 64, 95)):
            rgb = cv2.imread(str(OUT / "decoded_png" / f"{frame:06d}.png"))
            ax.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            image = by_frame.get(frame)
            if image:
                uv = np.array([p.xy-.5 for p in image.points2D if p.point3D_id in rec.points3D])
                mask = cv2.imread(str(ROOT / f"runs/dobbe_colmap_clean_rerun_v1/masks/main/{frame:04d}.png.png"), 0)
                if len(uv):
                    xy = np.rint(uv).astype(int).clip(0, 255)
                    background = mask[xy[:, 1], xy[:, 0]] > 0
                    ax.scatter(*uv[~background].T, s=9, c="#ff3e56", alpha=.8, label="Excluded region")
                    ax.scatter(*uv[background].T, s=9, c="#22ddff", alpha=.8, label="Background mask")
            ax.set(title=f"Frame {frame}", xticks=[], yticks=[])
        axes[0].legend(fontsize=6, loc="lower left")
        fig.savefig(OUT / "causal_sparse_observations.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"results": results, "conditional_stride_required": do_stride}), flush=True)


def audit():
    protocol = read(OUT / "protocol.json")
    assert protocol["source_sha256"] == sha(SOURCE / "compressed_video_h264.mp4")
    decision = read(OUT / "decision.json")
    validation_sources = read(OUT / "validation_sources.json")
    assert validation_sources["labels_sha256"] == sha(SOURCE / "labels.json")
    assert validation_sources["measured_depth_sha256"] == sha(SOURCE / "compressed_np_depth_float32.bin")
    assert decision["status"] == "OFFICIAL_BASELINE_COMPLETE"
    assert not decision["conditional_stride_required"] or decision["conditional_stride_completed"]
    checks = []
    for row in decision["runs"]:
        slug = row["run"]
        run = OUT / slug
        summary = read(run / "summary.json")
        frames = RUNS[slug]
        assert summary["input_frames"] == frames
        expected = {p["frame"]: p["sha256"] for p in protocol["decoded_images"]}
        assert sorted(int(p.stem) for p in (run / "images").glob("*.png")) == frames
        assert all(sha(run / "images" / f"{f:06d}.png") == expected[f] for f in frames)
        argv = summary["execution"]["argv"]
        options = dict(zip(argv[1::2], argv[2::2]))
        assert options["--data_type"] == "video" and options["--camera_model"] == "SIMPLE_RADIAL"
        assert options["--quality"] == "high" and options["--single_camera"] == "1"
        assert options["--sparse"] == "1" and options["--dense"] == "0"
        assert set(options) <= {"--workspace_path", "--image_path", "--data_type", "--single_camera",
                               "--camera_model", "--quality", "--sparse", "--dense", "--use_gpu", "--num_threads"}
        assert not (run / "dense").exists()
        db_graph, _ = database_audit(run, frames)
        assert db_graph == summary["graph"]
        assert summary["eligible_for_causal_K"] == (slug != "full")
        for model in summary["models"]:
            rec = pycolmap.Reconstruction(run / "sparse" / str(model["id"]))
            assert rec.num_reg_images() == model["registered"]
            assert rec.num_points3D() == model["points3D"]
            assert set(t["frame"] for t in model["trajectory"]) <= set(frames)
            assert np.isclose(rec.compute_mean_reprojection_error(), model["mean_reprojection_px"])
            assert all((run / "sparse_txt" / str(model["id"]) / f"{name}.txt").exists()
                       for name in ("cameras", "images", "points3D", "rigs", "frames"))
        checks.append({"run": slug, "source_and_subset_hashes": True, "rgb_only_cli": True,
                       "database_actual_counts": True, "sparse_counts_and_text_exports": True})
    write(OUT / "audit.json", {"status": "PASS", "checks": checks,
                               "no_validated_causal_K_claim": decision["validated_causal_K"] is None})
    print(json.dumps({"audit": "PASS", "runs": len(checks)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "analyze", "prepare-stride", "audit"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "prepare-stride":
        assert read(OUT / "decision.json")["conditional_stride_required"]
        prepare_subset("causal_stride5")
    elif args.mode == "analyze":
        analyze()
    else:
        audit()
