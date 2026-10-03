"""Sealed inference, processor equivalence and measured-depth ablation."""
from __future__ import annotations

import time
import resource
import traceback

import cv2
import numpy as np

from dobbe_vipe_v1 import ROOT, RUN, CHECKPOINT, GATE, read, write, sha, freeze, scene_dir
from dobbe_vipe_geometry import lift, rigid_error, project, stats, independent_sift_geometry


def export_variants(scene, branch="no_vda"):
    dest = scene_dir(scene, branch)
    p = read(dest / "protocol.json")
    g = np.load(dest / "geometry_all.npz")
    tracks = np.load(dest / "tracks.npz")
    base = read(dest / "geometry_gate.json")
    h = p["history_vipe_indices"]
    poses, k = g["poses"], g["intrinsics"]
    if scene == "A" and tracks["tracks"].shape[1]-int(tracks["background_end"]) == 8:
        ids = np.arange(int(tracks["background_end"]), tracks["tracks"].shape[1])
        world = g["points_world"][h][:, ids]
        good = np.isfinite(world).all() and g["valid"][h][:, ids].all()
        rigidity = rigid_error(world) if good else None
        checks = dict(base["checks"])
        checks["complete_history_eight"] = bool(good)
        checks["rigid_history"] = rigidity is not None and rigidity["relative_to_t0_median_pair_distance"]["median"] <= GATE["rigid_relative_median_max"]
        save_variant(dest, "smoke_vipe", world, tracks["tracks"][-1, ids], poses[-1],
                     checks, {"legacy_eight_query_source": "runs/dobbe_rgbd_study/approx_history/points_2d_at_t0_candidate.npy",
                              "rigid_history_unsmoothed": rigidity, "smoothing": False})
    if base["selected_ids"]:
        ids = np.array(base["selected_ids"])
        save_variant(dest, "pure_vipe", g["object_smoothed"][h][:, ids],
            tracks["tracks"][-1, ids], poses[-1], base["checks"],
            {"selected_ids": ids.tolist(), "depth_source": "ViPE", "smoothing": True})
        if (dest / "paired_selection_protocol.json").exists():
            # The sensor-supported paired selection has its own frozen inputs.
            return
        measured = np.load(dest / "measured_depth_prefix.npy")
        measured = np.stack([cv2.resize(d, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT) for d in measured])
        hybrid_world, hybrid_valid, _ = lift(tracks["tracks"], measured, k, poses, tracks["visibility"])
        hybrid_history = hybrid_world[h][:, ids]
        hybrid_good = hybrid_valid[h][:, ids].all() and np.isfinite(hybrid_history).all()
        rigidity = rigid_error(hybrid_history) if hybrid_good else None
        checks = dict(base["checks"])
        checks["complete_history_eight"] = bool(hybrid_good)
        checks["rigid_history"] = rigidity is not None and rigidity["relative_to_t0_median_pair_distance"]["median"] <= GATE["rigid_relative_median_max"]
        review = dest / "hybrid_alignment_review.json"
        checks["rgb_depth_alignment_reviewed"] = review.exists() and read(review).get("spatial_alignment_supported") is True
        # Hybrid must pass its own static reprojection, using measured t0 depth.
        bg = np.arange(int(tracks["object_count"]), int(tracks.get("background_end", len(tracks["tracks"][0]))))
        errors, pair_counts = [], []
        for t in np.linspace(0, len(poses)-4, 6).astype(int):
            pred = project(hybrid_world[-1, bg], poses[t], k[t])
            valid = hybrid_valid[-1, bg] & tracks["visibility"][t, bg] & np.isfinite(pred).all(-1)
            observed = tracks["tracks"][t, bg]
            valid &= np.isfinite(observed).all(-1) & ((observed >= 0) & (observed <= 255)).all(-1)
            pair_counts.append(int(valid.sum()))
            errors.extend(np.linalg.norm(pred[valid]-tracks["tracks"][t, bg][valid], axis=-1))
        static = stats(errors)
        if base.get("static_validation") == "independent_sift":
            static = independent_sift_geometry(dest, measured, k, poses)
            pair_counts = [pair["count"] for pair in static["pairs"]]
        checks["static_evidence"] = static["count"] >= GATE["static_observations_min"] and sum(n >= 5 for n in pair_counts) >= GATE["static_pairs_min"]
        checks["static_median"] = static["median"] is not None and static["median"] <= GATE["static_median_px_max"]
        checks["static_p90"] = static["p90"] is not None and static["p90"] <= GATE["static_p90_px_max"]
        save_variant(dest, "hybrid", hybrid_history, tracks["tracks"][-1, ids], poses[-1], checks,
            {"selected_ids": ids.tolist(), "depth_source": "measured DobbE optical-Z hypothesis",
             "static_cross_frame_px": static, "rigid_history_unsmoothed": rigidity,
             "translation_scale": "ViPE c2w retained unchanged; no fitted rescaling",
             "smoothing": False, "caveat": "Compare with pure_vipe_unsmoothed for a strict depth-only ablation."})
        # Strict depth-only comparison: same tracks, no smoothing in either branch.
        save_variant(dest, "pure_vipe_unsmoothed", g["points_world"][h][:, ids],
            tracks["tracks"][-1, ids], poses[-1], base["checks"],
            {"selected_ids": ids.tolist(), "depth_source": "ViPE", "smoothing": False})


def save_variant(dest, name, world, uv, pose, checks, details):
    folder = dest / name
    folder.mkdir(exist_ok=True)
    for file, array in [("points_3d_world.npy", world), ("points_2d_t0.npy", uv), ("c2w_t0.npy", pose)]:
        np.save(folder / file, np.asarray(array, dtype=np.float32))
    source_gate = read(dest / "geometry_gate.json")
    if source_gate.get("static_validation") == "independent_sift":
        details = {**details, "static_validation": "independent_sift",
            "validation_scope": source_gate["validation_scope"],
            "selection_caveat": source_gate["selection_caveat"],
            "source_geometry_gate_sha256": sha(dest / "geometry_gate.json")}
    write(folder / "geometry_gate.json", {"molmo_ready": all(checks.values()),
        "status": "PASS_ESTIMATED_GEOMETRY" if all(checks.values()) else "REJECT_ESTIMATED_GEOMETRY",
        "checks": checks, "metric_ground_truth": False, "future_used": False, **details})


def inputs(scene, variant, branch="no_vda"):
    import torch
    from PIL import Image
    from molmo_motion import MolmoMotionProcessor
    dest = scene_dir(scene, branch)
    folder = dest / variant
    p = read(dest / "protocol.json")
    world = torch.from_numpy(np.load(folder / "points_3d_world.npy"))
    uv = torch.from_numpy(np.load(folder / "points_2d_t0.npy"))
    pose = torch.from_numpy(np.load(folder / "c2w_t0.npy"))
    if world.shape != (3, 8, 3) or uv.shape != (8, 2) or not torch.isfinite(world).all():
        raise ValueError("Invalid model input")
    frames = [Image.open(dest / "frames" / f"{i:05d}.png").convert("RGB") for i in p["history_vipe_indices"]]
    processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
    args = dict(history_frames=frames, points_2d_at_t0=uv, action=p["action"], future_horizon=30)
    a = processor(points_3d_history=world, c2w_at_t0=pose, **args)
    inv = torch.linalg.inv(pose)
    camera = world @ inv[:3, :3].T + inv[:3, 3]
    b = processor(points_3d_history=camera, **args)
    equal = {key: torch.equal(a[key], b[key]) for key in a if torch.is_tensor(a[key])}
    if not all(equal.values()):
        raise RuntimeError(f"Processor frame equivalence failed: {equal}")
    write(folder / "processor_equivalence.json", {"success": True, "tensor_equal": equal,
        "serialized_input_ids_equal": equal["input_ids"], "anchor_equal": equal["anchor_3d"],
        "input_ids_sha256": __import__("hashlib").sha256(a["input_ids"].numpy().tobytes()).hexdigest(),
        "world_to_camera_applied_once": True})
    np.save(folder / "points_3d_camera_t0_diagnostic.npy", camera.numpy())
    return a


def infer(scene, variant, branch="no_vda", reuse=None):
    import torch
    from molmo_motion import MolmoMotion
    dest = scene_dir(scene, branch)
    folder = dest / variant
    gate = read(folder / "geometry_gate.json")
    if not gate["molmo_ready"]:
        write(folder / "model_run.json", {"status": "SKIPPED_GEOMETRY_GATE", "success": False,
                                         "failed_checks": [key for key, value in gate["checks"].items() if not value]})
        print(scene, variant, "skipped: geometry gate", flush=True)
        return
    result_path = folder / "prediction_15hz.npy"
    if result_path.exists():
        previous = read(folder / "model_run.json")
        if not previous.get("success"):
            raise FileExistsError("Invalid existing prediction requires separate experiment folder")
        frozen = read(folder / "input_freeze.json")
        if sha(result_path) != previous["prediction_sha256"] or sha(folder / "input_freeze.json") != previous["input_freeze_sha256"] or sha(dest / "protocol.json") != frozen["protocol_sha256"]:
            raise RuntimeError("Cached forecast or sealed protocol changed")
        if any(sha(dest / name) != digest for name, digest in frozen["input_sha256"].items()):
            raise RuntimeError("Cached forecast inputs changed")
        return
    batch = inputs(scene, variant, branch)
    p = read(dest / "protocol.json")
    frozen_files = [folder / name for name in ["points_3d_world.npy", "points_2d_t0.npy", "c2w_t0.npy", "geometry_gate.json"]]
    frozen_files += [dest / "frames" / f"{i:05d}.png" for i in p["history_vipe_indices"]]
    checkpoint_path = CHECKPOINT / "model.pt"
    identity = (checkpoint_path.stat().st_size, checkpoint_path.stat().st_mtime_ns)
    if reuse is not None and "model" in reuse and reuse.get("checkpoint_identity") != identity:
        raise RuntimeError("Checkpoint changed while a loaded model is reused")
    if reuse is not None and reuse.get("checkpoint_identity") == identity:
        checkpoint_digest = reuse["checkpoint_sha256"]
    else:
        checkpoint_digest = sha(checkpoint_path)
        if reuse is not None:
            reuse.update(checkpoint_identity=identity, checkpoint_sha256=checkpoint_digest)
    freeze(folder / "input_freeze.json", {"scene": scene, "variant": variant,
        "action": p["action"], "history_raw_ids": p["history_raw_ids"],
        "source_fps": 30, "model_history_fps": 15, "future_horizon": 30,
        "input_sha256": {str(path.relative_to(dest)): sha(path) for path in frozen_files},
        "protocol_sha256": sha(dest / "protocol.json"), "seed": 0, "dtype": "bfloat16",
        "checkpoint_config_sha256": sha(CHECKPOINT / "config.yaml"),
        "checkpoint_model_sha256": checkpoint_digest,
        "geometry_status": "ViPE-estimated; not physical ground truth", "future_used": False})
    started = time.monotonic()
    status = {"status": "RUNNING", "success": False, "variant": variant, "action": p["action"],
              "geometry_branch": branch, "model_reused": bool(reuse is not None and "model" in reuse),
              "ram_peak_scope": "whole process, including earlier variants if model is reused",
              "torch_version": torch.__version__, "input_freeze_sha256": sha(folder / "input_freeze.json")}
    def update(stage):
        status.update(stage=stage, elapsed_seconds=time.monotonic()-started,
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        write(folder / "model_run.json", status)
        print(scene, variant, stage, flush=True)
    try:
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        update("load_model")
        if reuse is not None and "model" in reuse:
            model = reuse["model"]
        else:
            prior_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.bfloat16)
            try:
                model = MolmoMotion.from_pretrained(str(CHECKPOINT))
            finally:
                torch.set_default_dtype(prior_dtype)
            model._internal = model._internal.cuda().eval()
            if reuse is not None:
                reuse["model"] = model
        batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
        update("predict")
        torch.cuda.synchronize()
        prediction_start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.predict_trajectory(**batch)
        torch.cuda.synchronize()
        future = out.future_3d.detach().cpu().numpy()
        from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array
        parsed = parse_tracks_text(out.future_text)
        parsed_vis = np.zeros((8, 30), dtype=bool) if parsed is None else tracks_to_array(
            parsed, 8, 30, start_timestamp=3.0)[1]
        np.save(folder / "prediction_parsed_visibility.npy", parsed_vis)
        status["prediction_seconds"] = time.monotonic()-prediction_start
        np.save(result_path, future)
        (folder / "raw_model_output.txt").write_text(out.future_text, encoding="utf-8")
        status.update(prediction_shape=list(future.shape), prediction_finite=bool(np.isfinite(future).all()),
            parsed_point_frames=int(parsed_vis.sum()), expected_point_frames=240,
            success=bool(future.shape == (8, 30, 3) and np.isfinite(future).all() and parsed_vis.all() and not np.all(future == 0)),
            prediction_sha256=sha(result_path))
        status["status"] = "COMPLETE" if status["success"] else "INVALID_OUTPUT"
        update("complete")
        if status["success"]:
            render_forecast(folder, p)
    except Exception as exc:
        status.update(status="FAILED", error=str(exc), traceback=traceback.format_exc())
        update("failed")
        raise


def render_forecast(folder, protocol):
    """Export a shareable forecast plot without using any future observation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    future = np.load(folder / "prediction_15hz.npy")
    history = np.load(folder / "points_3d_camera_t0_diagnostic.npy")
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for i in range(8):
        color = plt.cm.tab10(i)
        ax.plot(*history[:, i].T, "o--", color=color, alpha=.7)
        ax.plot(*np.vstack([history[-1, i], future[i]]).T, color=color)
    ax.set_title(f"{protocol['scene']}: history and forecast in camera t0")
    ax.set_xlabel("X, estimated units"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax = fig.add_subplot(1, 2, 2)
    t = np.arange(1, 31)/15
    displacement = np.linalg.norm(future-history[-1, :, None], axis=-1)
    for i in range(8):
        ax.plot(t, displacement[i], color=plt.cm.tab10(i), label=f"point {i+1}")
    ax.set_xlabel("seconds after t0"); ax.set_ylabel("predicted displacement, estimated units")
    ax.set_title(protocol["action"]); ax.legend(ncol=2, fontsize=8)
    fig.tight_layout(); fig.savefig(folder / "forecast_3d.png", dpi=160); plt.close(fig)
