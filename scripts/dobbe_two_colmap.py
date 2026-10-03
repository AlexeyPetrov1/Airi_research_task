"""A sealed K-only MolmoMotion comparison, using two RGB-only COLMAP pipelines.

Run prepare, the official Windows runner, inputs, infer, then the renderer.
COLMAP extrinsics are exported as diagnostics and never enter MolmoMotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation

from dobbe_colmap_clean import summarize
from inspect_hony_scenes import liblzfse
from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dobbe_two_colmap_v1"
RAW = ROOT / "data/dobbe_oxe/target_raw"
OLD = ROOT / "runs/dobbe_rgbd_study/approx_history"
A_RUN = ROOT / "runs/dobbe_colmap_clean_rerun_v1/main_causal_pinhole_masked_stride1_all_dsp_affine_guided"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
HISTORY = [96, 97, 98]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    """Freeze method selection before seeing the new B reconstruction/forecasts."""
    RUN.mkdir(parents=True, exist_ok=True)
    target = RUN / "official/images"
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    # Use precisely the decoded RGB supplied to A, including the same prefix.
    for i in range(96):
        src = A_RUN.parent / f"images/main/{i:04d}.png"
        dst = target / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
        assert sha(src) == sha(dst)
        hashes[str(i)] = sha(dst)
    assert len(list(target.glob("*.png"))) == 96
    protocol = {
        "episode": "dobbe#episode_3651",
        "hony_recording": "Pick_and_Place/Home15/Env1/2023-04-25--02-05-30",
        "calibration_frames": list(range(96)), "history_frames": HISTORY,
        "method_A": str(A_RUN.relative_to(ROOT)),
        "method_A_profile": "clean masked PINHOLE; DSP-SIFT, affine, guided; default mapper",
        "method_B": "official COLMAP 4.2.1 automatic_reconstructor; video/high, PINHOLE, single camera, no mask, no manual K/initial pair/mapper thresholds",
        "model_selection": "largest registered component with nonzero sparse points, separately for each method; ties resolved by model id; no future or forecast quality used",
        "only_A_B_variable": "OpenCV K (COLMAP cx,cy minus 0.5); zero distortion in both branches",
        "extrinsics_rule": "same published Dobb-E c2w labels and CAMERA_TO_LABEL in both; COLMAP poses diagnostic only",
        "depth_semantics": "same measured HoNY depth, treated as optical camera Z (hypothesis)",
        "history_point_source": str((OLD / "manifest.json").relative_to(ROOT)),
        "history_point_policy": "reuse the eight existing history-only RGB IDs and their measured depth; no new selection by K or forecast",
        "rgb_sha256": hashes, "video_sha256": sha(RAW / "compressed_video_h264.mp4"),
        "labels_sha256": sha(RAW / "labels.json"),
        "depth_sha256": sha(RAW / "compressed_np_depth_float32.bin"),
        "A_config_sha256": sha(A_RUN / "config.json"),
        "B_clean_workspace": True,
        "projection_timing": "30 forecast steps at assumed 15 Hz; interpolate XYZ to the 60 real 30 Hz future frames 99..158; no synthetic RGB",
        "temporal_limit": "processor has no physical timestamp input; H3 is sampled at source 30 Hz",
        "forecast_status": "exploratory; neither camera accepted as ground-truth calibration",
    }
    if (RUN / "protocol.json").exists():
        assert read(RUN / "protocol.json") == protocol
    else:
        assert not (RUN / "official/database.db").exists(), "Need a clean DB"
        write(RUN / "protocol.json", protocol)
    print("Prepared clean official prefix: 96 identical decoded RGB frames", flush=True)


def reconstruction_summary(folder, dest):
    recs = {int(p.name): pycolmap.Reconstruction(p) for p in (folder / "sparse").iterdir() if p.is_dir()}
    models = summarize(recs)
    usable = [m for m in models if m["registered"] > 0 and m["points3D"] > 0]
    assert usable, "No real reconstruction: do not fabricate K"
    best = sorted(usable, key=lambda m: (-m["registered"], m["id"]))[0]
    for mid, rec in recs.items():
        out = dest / "sparse_txt" / str(mid)
        out.mkdir(parents=True, exist_ok=True)
        rec.write_text(out)
    rec = recs[best["id"]]
    cam = next(iter(rec.cameras.values()))
    assert len(rec.cameras) == 1 and cam.model_name == "PINHOLE"
    k = cam.calibration_matrix()
    k[0, 2] -= .5
    k[1, 2] -= .5
    points = list(rec.points3D.values())
    stats = {"source": str(folder.relative_to(ROOT)), "model_id": best["id"],
             "models": models, "registered": best["registered"], "input_frames": 96,
             "unique_registered": len({t["frame"] for m in models for t in m["trajectory"]}),
             "points3D": best["points3D"], "observations": rec.compute_num_observations(),
             "mean_reprojection_px": best["mean_reprojection_px"],
             "median_track_length": float(np.median([p.track.length() for p in points])),
             "camera_model": cam.model_name, "params_colmap": cam.params.tolist(),
             "K_opencv": k.tolist(), "distortion": [0, 0, 0, 0],
             "trajectory_usage": "diagnostic only; arbitrary SfM scale; NOT the MolmoMotion pose source",
             "calibration_validation": "NOT_VALIDATED; exploration only",
             "native_sparse_sha256": {p.name: sha(p) for p in (folder / "sparse" / str(best["id"])).glob("*.bin")}}
    np.save(dest / "K.npy", k)
    write(dest / "reconstruction.json", stats)
    return k, stats


def build_inputs():
    protocol = read(RUN / "protocol.json")
    execution = read(RUN / "official/execution.json")
    assert execution["exit_code"] == 0 and "PINHOLE" in execution["argv"]
    with sqlite3.connect(f"file:{RUN / 'official/database.db'}?mode=ro", uri=True) as db:
        cameras = list(db.execute("SELECT model,width,height,params,prior_focal_length FROM cameras"))
        names = sorted(n for (n,) in db.execute("SELECT name FROM images"))
    assert names == [f"{i:04d}.png" for i in range(96)]
    assert len(cameras) == 1 and cameras[0][0] == 1 and not cameras[0][4]
    initial = np.frombuffer(cameras[0][3], np.float64).tolist()
    write(RUN / "official/database_audit.json", {"images": len(names), "camera_model_id": cameras[0][0],
          "initial_params": initial, "has_prior_focal_length": bool(cameras[0][4])})
    old = read(OLD / "manifest.json")
    uv = np.array([p["rgb_uv"] for p in old["points"]], np.float64).transpose(1, 0, 2)
    z = np.array([p["depth_m"] for p in old["points"]], np.float64).T
    assert uv.shape == (3, 8, 2) and z.shape == (3, 8)
    raw_depth = np.frombuffer(liblzfse.decompress((RAW / "compressed_np_depth_float32.bin").read_bytes()), np.float32).reshape(-1, 192, 256)
    measured = np.array([[depth_at(raw_depth[i], *pt)[0] for pt in uvt] for i, uvt in zip(HISTORY, uv)])
    np.testing.assert_allclose(measured, z, atol=1e-8, rtol=0)
    labels = read(RAW / "labels.json")
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, :3] = Rotation.from_quat([labels[str(i)]["quats"] for i in HISTORY]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in HISTORY]
    np.testing.assert_allclose(poses, old["poses_c2w"], atol=1e-12)
    shared = RUN / "shared"
    shared.mkdir(exist_ok=True)
    np.save(shared / "history_uv.npy", uv)
    np.save(shared / "history_depth_m.npy", z)
    np.save(shared / "poses_c2w.npy", poses)
    np.save(shared / "points_2d_at_t0.npy", uv[-1].astype(np.float32))
    for i in HISTORY:
        shutil.copyfile(OLD / f"rgb_{i:04d}.png", shared / f"rgb_{i:04d}.png")
    histories = []
    summaries = {}
    for branch, folder in (("A", A_RUN), ("B", RUN / "official")):
        dest = RUN / branch
        dest.mkdir(exist_ok=True)
        k, stats = reconstruction_summary(folder, dest)
        optical = np.stack(((uv[..., 0] - k[0, 2]) * z / k[0, 0],
                            (uv[..., 1] - k[1, 2]) * z / k[1, 1], z), axis=-1)
        label = optical @ CAMERA_TO_LABEL.T
        world = label @ poses[:, :3, :3].transpose(0, 2, 1) + poses[:, None, :3, 3]
        history = ((world - poses[-1, :3, 3]) @ poses[-1, :3, :3]) @ CAMERA_TO_LABEL
        assert history.shape == (3, 8, 3) and np.isfinite(history).all() and (history[..., 2] > 0).all()
        np.save(dest / "history.npy", history.astype(np.float32))
        np.save(dest / "history_world.npy", world)
        projected = np.stack((k[0, 0] * history[-1, :, 0] / history[-1, :, 2] + k[0, 2],
                              k[1, 1] * history[-1, :, 1] / history[-1, :, 2] + k[1, 2]), axis=-1)
        np.testing.assert_allclose(projected, uv[-1], atol=1e-8)
        histories.append(history)
        summaries[branch] = {key: stats[key] for key in ("registered", "points3D", "K_opencv")}
    freeze_old = read(OLD / "input_freeze.json")
    assert sha(CHECKPOINT / "config.yaml") == freeze_old["checkpoint_config_sha256"]
    freeze = {"status": "FROZEN_K_ONLY_COMPARISON", "history_frames": HISTORY,
              "point_ids": old["point_ids"], "action_text": "Pick up the red cup.",
              "model_id": freeze_old["model_id"], "model_revision": freeze_old["model_revision"],
              "checkpoint_config_sha256": freeze_old["checkpoint_config_sha256"],
              "checkpoint_model_pt_sha256": freeze_old["checkpoint_model_pt_sha256"],
              "future_horizon": 30, "seed": 0, "dtype": "bfloat16",
              "optical_to_label": CAMERA_TO_LABEL.tolist(), "poses_source": "Dobb-E labels only",
              "depth_semantics": "measured camera Z hypothesis", "distortion_both": [0, 0, 0, 0],
              "sha256": {str(p.relative_to(RUN)): sha(p) for p in sorted(shared.iterdir())},
              "geometry_sha256": {b: sha(RUN / b / "history.npy") for b in ("A", "B")},
              "K_sha256": {b: sha(RUN / b / "K.npy") for b in ("A", "B")},
              "protocol_sha256": sha(RUN / "protocol.json"),
              "delta_history_mean_m": float(np.linalg.norm(histories[0] - histories[1], axis=-1).mean()),
              "reconstructions": summaries}
    if (RUN / "input_freeze.json").exists():
        assert read(RUN / "input_freeze.json") == freeze, "Frozen comparison already differs"
    else:
        write(RUN / "input_freeze.json", freeze)
    print(json.dumps({"histories": [list(h.shape) for h in histories], "summary": summaries,
                      "delta_history_m": freeze["delta_history_mean_m"]}), flush=True)


def verify_freeze():
    f = read(RUN / "input_freeze.json")
    assert f["status"] == "FROZEN_K_ONLY_COMPARISON"
    for name, expected in f["sha256"].items():
        assert sha(RUN / name) == expected, name
    for branch in ("A", "B"):
        assert sha(RUN / branch / "history.npy") == f["geometry_sha256"][branch]
        assert sha(RUN / branch / "K.npy") == f["K_sha256"][branch]
    assert sha(RUN / "protocol.json") == f["protocol_sha256"]
    assert sha(CHECKPOINT / "config.yaml") == f["checkpoint_config_sha256"]
    return f


def infer():
    import torch
    from PIL import Image
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array

    freeze = verify_freeze()
    assert torch.cuda.is_available()
    # Hash real weights once for both branches, instead of trusting only an old receipt.
    assert sha(CHECKPOINT / "model.pt") == freeze["checkpoint_model_pt_sha256"]
    started = time.perf_counter()
    print("Loading one shared BF16 checkpoint for A and B", flush=True)
    processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = MolmoMotion.from_pretrained(str(CHECKPOINT))
    finally:
        torch.set_default_dtype(original_dtype)
    model._internal = model._internal.cuda().eval()
    frames = [Image.open(RUN / "shared" / f"rgb_{i:04d}.png").convert("RGB") for i in HISTORY]
    uv = np.load(RUN / "shared/points_2d_at_t0.npy")
    for branch in ("A", "B"):
        verify_freeze()
        dest = RUN / branch
        if (dest / "prediction.npy").exists():
            status = read(dest / "model_run.json")
            assert status["status"] == "COMPLETE" and status["input_freeze_sha256"] == sha(RUN / "input_freeze.json")
            assert status["prediction_sha256"] == sha(dest / "prediction.npy")
            continue
        history = np.load(dest / "history.npy")
        torch.manual_seed(freeze["seed"])
        torch.cuda.manual_seed_all(freeze["seed"])
        torch.cuda.reset_peak_memory_stats()
        status = {"branch": branch, "status": "RUNNING", "input_freeze_sha256": sha(RUN / "input_freeze.json"),
                  "shared_parameters": {key: freeze[key] for key in ("model_id", "model_revision", "checkpoint_model_pt_sha256", "action_text", "future_horizon", "seed", "dtype", "point_ids", "history_frames")},
                  "history_sha256": sha(dest / "history.npy"), "torch_version": torch.__version__,
                  "gpu": torch.cuda.get_device_name(0), "shared_input_sha256": freeze["sha256"]}
        write(dest / "model_run.json", status)
        batch = processor(history_frames=frames, points_2d_at_t0=torch.from_numpy(uv),
                          points_3d_history=torch.from_numpy(history), action=freeze["action_text"], future_horizon=30)
        batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        print(f"Predicting branch {branch}; elapsed {time.perf_counter() - started:.1f}s", flush=True)
        prediction_start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict_trajectory(**batch)
        torch.cuda.synchronize()
        future = output.future_3d.detach().cpu().numpy()
        assert future.shape == (8, 30, 3) and np.isfinite(future).all() and not np.all(future == 0)
        raw = output.future_text
        tracks = parse_tracks_text(raw)
        assert tracks is not None
        delta, visible = tracks_to_array(tracks, num_points=8, num_frames=30, start_timestamp=3.0)
        assert np.asarray(visible).all()
        reconstruction_error = float(np.max(np.abs(np.asarray(delta) + history[-1, 0] - future)))
        assert reconstruction_error < 1e-4
        (dest / "raw_model_output.txt").write_text(raw, encoding="utf-8")
        np.save(dest / "prediction.npy", future)
        status.update(status="COMPLETE", prediction_shape=list(future.shape),
                      prediction_seconds=round(time.perf_counter() - prediction_start, 2),
                      elapsed_seconds=round(time.perf_counter() - started, 2),
                      prediction_sha256=sha(dest / "prediction.npy"),
                      raw_output_sha256=sha(dest / "raw_model_output.txt"),
                      parsed_visible_count=int(np.asarray(visible).sum()),
                      text_tensor_max_error_m=reconstruction_error,
                      all_positive_Z=bool((future[..., 2] > 0).all()),
                      peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                      peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
        write(dest / "model_run.json", status)
        print(f"Complete {branch}: {status['prediction_seconds']}s, parsed 240/240", flush=True)
        del batch, output
    a, b = [read(RUN / branch / "model_run.json") for branch in ("A", "B")]
    assert a["shared_parameters"] == b["shared_parameters"]
    assert a["shared_input_sha256"] == b["shared_input_sha256"]
    print("A/B inference complete; all non-geometry inputs identical", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "inputs", "infer"))
    {"prepare": prepare, "inputs": build_inputs, "infer": infer}[p.parse_args().mode]()
