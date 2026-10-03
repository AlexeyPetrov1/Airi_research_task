"""Causal 15 Hz Dobb·E -> ViPE -> AllTracker -> MolmoMotion experiment.

Run preparation in the existing CPU-capable environment, ViPE/AllTracker in
the separate environment, and MolmoMotion in its existing environment.
All prediction inputs are restricted to the saved prefix ending at t0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dobbe_vipe_v1"
VIPE = ROOT / "data_generation/third_party/vipe"
ALLTRACKER = ROOT / "data_generation/third_party/alltracker"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
SCENES = {
    "A": {"raw": "data/dobbe_oxe/target_raw", "t0": 98,
          "recording": "Pick_and_Place/Home15/Env1/2023-04-25--02-05-30",
          "action": "Pick up the red cup."},
    "B": {"raw": "data/dobbe_oxe/second_raw", "t0": 123,
          "recording": "Pick_and_Place/Home7/Env1/2023-04-27--10-47-40",
          "action": "Pick up the roll of tape."},
    "C": {"raw": "runs/plex_dobbe_preflight/dobbe_rgbd_sample", "t0": 60,
          "recording": "Drawer_Closing/Home10/Env2/2022-12-22--00-09-08",
          "action": "Close the drawer."},
}
# Heuristic acceptance thresholds, frozen before seeing reconstruction results.
# They permit estimated-geometry inference; they do not certify metric GT.
GATE = {"static_median_px_max": 4.0, "static_p90_px_max": 10.0,
        "static_pairs_min": 3, "static_observations_min": 30,
        "rigid_relative_median_max": 0.15, "history_visible_points_min": 8,
        "depth_ratio_median_min": 0.5, "depth_ratio_median_max": 2.0,
        "focal_relative_range_max": 0.15}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                              allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise RuntimeError(f"Frozen protocol differs: {path}")
    else:
        write(path, value)


def scene_dir(scene, branch="no_vda"):
    return RUN / scene if branch == "no_vda" else RUN / scene / branch


def prepare_branch(scene, branch):
    """Named controls preserve every baseline artifact and all raw query IDs."""
    if branch == "no_vda":
        return RUN / scene
    if branch == "sift_audited":
        raise ValueError("Use prepare_dobbe_vipe_sift_control.py; this is a separate post-hoc validation control")
    import shutil
    base, dest = RUN / scene, scene_dir(scene, branch)
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["query_points.npz", "tracks.npz", "alltracker_execution.json",
                 "measured_depth_prefix.npy", "mask_t0.png", "mask_annotation.json"]:
        if not (dest / name).exists():
            shutil.copy2(base / name, dest / name)
    if not (dest / "frames").exists():
        shutil.copytree(base / "frames", dest / "frames")
    p = read(base / "protocol.json")
    p.update(geometry_branch=branch, model_rgb_size_wh=[256, 256],
        control_reason="default: official VDA alignment; rectified: reverse known 256x192-to-256x256 export stretch, without new K, queries or future",
        source_baseline_protocol_sha256=sha(base / "protocol.json"))
    video = dest / "causal_15hz.mp4"
    if branch == "rectified":
        p["input_size_wh"] = [256, 192]
        p["vipe_to_model_pixel_map"] = "u_model=u_vipe; v_model=(v_vipe+0.5)*256/192-0.5"
        if not video.exists():
            frames = load_video(base / "causal_15hz.mp4")
            frames = np.stack([cv2.resize(img, (256, 192), interpolation=cv2.INTER_AREA) for img in frames])
            encode(video, frames)
            assert np.array_equal(load_video(video), frames)
    elif not video.exists():
        shutil.copy2(base / "causal_15hz.mp4", video)
    freeze(dest / "protocol.json", p)
    freeze(dest / "frame_map.json", {"vipe_to_raw": read(base / "frame_map.json")["vipe_to_raw"],
        "baseline_frame_map_sha256": sha(base / "frame_map.json"),
        "causal_video_sha256": sha(video), "branch": branch,
        "vipe_image_transformation": "pixel-center vertical resize 256->192" if branch == "rectified" else "identity",
        "molmo_history_pngs": "original 256x256, copied byte-identically from baseline"})
    return dest


def load_video(path, count=None):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while count is None or len(frames) < count:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    if count is not None and len(frames) != count:
        raise RuntimeError(f"Expected {count} frames from {path}, got {len(frames)}")
    return np.stack(frames)


def encode(path, frames, fps=15):
    import imageio_ffmpeg
    h, w = frames.shape[1:3]
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
            "-n", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264rgb",
            "-crf", "0", "-preset", "fast", str(path)]
    subprocess.run(args, input=frames.tobytes(), check=True)


def prepare(scene):
    config = SCENES[scene]
    folder = ROOT / config["raw"]
    dest = RUN / scene
    dest.mkdir(parents=True, exist_ok=True)
    raw_video = folder / "compressed_video_h264.mp4"
    cap = cv2.VideoCapture(str(raw_video))
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    t0 = config["t0"]
    if not np.isclose(fps, 30) or t0 + 60 >= n:
        raise RuntimeError(f"Unexpected source timing: {fps} Hz, {n} frames")
    ids = list(range(t0 % 2, t0 + 1, 2))
    protocol = {"version": 1, "scene": scene, **config, "raw_fps": fps,
                "input_fps": 15, "input_size_wh": [256, 256], "raw_frame_count": n,
                "observed_raw_ids": ids, "history_raw_ids": [t0-4, t0-2, t0],
                "history_vipe_indices": list(range(len(ids)-3, len(ids))),
                "future_raw_ids_evaluation_only": list(range(t0+2, t0+61, 2)),
                "geometry": "ViPE-estimated OpenCV world-frame geometry; not metric GT",
                "future_access_rule": "No RGB/depth/pose after t0 enters prediction, mask, tracking, filtering or geometry gate.",
                "source_sha256": {name: sha(folder / name) for name in
                     ["compressed_video_h264.mp4", "compressed_np_depth_float32.bin", "labels.json"]},
                "gate_thresholds": GATE}
    freeze(dest / "protocol.json", protocol)
    raw_prefix = load_video(raw_video, t0+1)
    frames = raw_prefix[ids]
    assert frames.shape[1:3] == (256, 256)
    pngs = dest / "frames"
    pngs.mkdir(exist_ok=True)
    for i, img in enumerate(frames):
        path = pngs / f"{i:05d}.png"
        if not path.exists():
            cv2.imwrite(str(path), img)
        assert np.array_equal(cv2.imread(str(path)), img)
    video = dest / "causal_15hz.mp4"
    if not video.exists():
        encode(video, frames)
    encoded = load_video(video)
    if not np.array_equal(encoded, frames):
        raise RuntimeError("Causal RGB lossless encoding changed frame pixels/order")
    freeze(dest / "frame_map.json", {
        "vipe_to_raw": {str(i): raw for i, raw in enumerate(ids)},
        "causal_video_sha256": sha(video), "pixel_identical_to_decoded_source": True,
        "frames_png_sha256": {str(i): sha(pngs / f"{i:05d}.png") for i in range(len(ids))}})
    # LZFSE is a single compressed stream: decode it, immediately discard future.
    sys.path.insert(0, str(ROOT.parent / ".tools/lzfse"))
    import liblzfse
    measured = np.frombuffer(liblzfse.decompress(
        (folder / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)[:t0+1][ids].copy()
    assert measured.shape == (len(ids), 192, 256)
    np.save(dest / "measured_depth_prefix.npy", measured)
    # Contact sheet contains observed frames only; no future is inspected.
    sample = sorted(set(np.linspace(0, len(ids)-1, 12).astype(int).tolist()
                        + list(range(len(ids)-3, len(ids)))))
    tiles = []
    for i in sample:
        tile = frames[i].copy()
        cv2.putText(tile, f"{scene} raw {ids[i]}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, f"{scene} raw {ids[i]}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(dest / "observed_contact.jpg"),
                np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]))
    print(scene, "prepared", len(ids), "frames; history", protocol["history_raw_ids"], flush=True)


def masks(scene):
    """Explicit saved masks drawn from observed t0 RGB; no future information."""
    dest = RUN / scene
    protocol = read(dest / "protocol.json")
    img = cv2.imread(str(dest / "frames" / f"{len(protocol['observed_raw_ids'])-1:05d}.png"))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    if scene == "A":
        polygon = [[56, 43], [183, 44], [164, 164], [143, 245], [94, 245], [65, 157]]
        cv2.fillPoly(mask, [np.array(polygon)], 255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask[((hsv[..., 0] > 14) & (hsv[..., 0] < 165)) | (hsv[..., 1] < 75)] = 0
        exclusions = [[[0, 255], [37, 255], [74, 171], [63, 159]],
                      [[147, 255], [194, 255], [171, 162], [160, 166]]]
    elif scene == "B":
        polygon = [[0, 136], [26, 124], [63, 126], [99, 150], [109, 188],
                   [105, 224], [87, 255], [0, 255]]
        hole = [[0, 177], [22, 144], [48, 143], [69, 163], [78, 194],
                [73, 225], [43, 247], [7, 254], [0, 240]]
        cv2.fillPoly(mask, [np.array(polygon)], 255)
        cv2.fillPoly(mask, [np.array(hole)], 0)
        exclusions = [[[0, 217], [0, 248], [83, 202], [77, 183]],
                      [[61, 213], [65, 255], [86, 255], [81, 207]],
                      [[104, 255], [161, 255], [135, 169], [110, 169]]]
    else:
        polygon = [[0, 149], [255, 117], [255, 255], [0, 255]]
        cv2.fillPoly(mask, [np.array(polygon)], 255)
        exclusions = [[[0, 255], [18, 255], [74, 174], [59, 155]],
                      [[72, 255], [95, 255], [83, 201], [66, 178]],
                      [[117, 255], [141, 255], [194, 174], [177, 156]],
                      [[207, 255], [229, 255], [196, 186], [182, 187]]]
    for poly in exclusions:
        cv2.fillPoly(mask, [np.array(poly)], 0)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    cv2.imwrite(str(dest / "mask_t0.png"), mask)
    # Background queries must stay outside target object and manipulation stick.
    bg_mask = 255 - cv2.dilate(mask, np.ones((21, 21), np.uint8))
    for poly in exclusions:
        cv2.fillPoly(bg_mask, [np.array(poly)], 0)
    if scene == "C":
        # Papers inside the drawer move with it, so exclude all drawer contents.
        bg_mask[77:] = 0
    bg_mask[:8] = bg_mask[-8:] = 0
    bg_mask[:, :8] = bg_mask[:, -8:] = 0
    pixels = np.column_stack(np.nonzero(mask)[::-1]).astype(np.float32)
    cv2.setRNGSeed(0)
    _, _, centers = cv2.kmeans(pixels, 100, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, .1),
        3, cv2.KMEANS_PP_CENTERS)
    # Snap centers into the mask to handle concave regions/hole in the tape roll.
    centers = pixels[((pixels[None] - centers[:, None]) ** 2).sum(-1).argmin(1)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 150, .01, 5, mask=bg_mask)
    if corners is None or len(corners) < 20:
        raise RuntimeError("Insufficient static RGB features")
    bg = corners[:, 0].round()
    query = np.concatenate([centers, bg])
    idx = len(protocol["observed_raw_ids"])-1
    np.savez(dest / "query_points.npz", query_points=np.column_stack([
        np.full(len(query), idx), query]), dim=(h, w), object_count=100)
    cv2.imwrite(str(dest / "background_mask_t0.png"), bg_mask)
    overlay = img.copy()
    overlay[mask > 0] = (.65 * overlay[mask > 0] + .35 * np.array([50, 200, 50])).astype(np.uint8)
    for x, y in centers.astype(int):
        cv2.circle(overlay, (x, y), 1, (255, 0, 255), -1)
    for x, y in bg.astype(int):
        cv2.circle(overlay, (x, y), 1, (255, 200, 0), -1)
    cv2.imwrite(str(dest / "query_overlay.png"), overlay)
    freeze(dest / "mask_annotation.json", {"t0_raw": protocol["t0"],
        "method": "manual foreground polygon; A also red HSV; erode 1 pixel; KMeans 100, snap to valid pixels",
        "polygon": polygon, "gripper_exclusions": exclusions,
        "query_frame": idx, "object_query_count": 100,
        "static_query_count": len(bg), "future_used": False,
        "mask_sha256": sha(dest / "mask_t0.png"),
        "query_sha256": sha(dest / "query_points.npz")})
    print(scene, "saved mask and queries", len(query), flush=True)


def run_receipt(dest, name, argv, cwd, env=None):
    """Persist command, input identity, timing and failure before raising."""
    receipt_path = dest / f"{name}_execution.json"
    if receipt_path.exists() and read(receipt_path).get("exit_code") == 0:
        prior = read(receipt_path)
        if prior["argv"] != [str(x) for x in argv] or prior["protocol_sha256"] != sha(dest / "protocol.json") or prior["causal_video_sha256"] != sha(dest / "causal_15hz.mp4"):
            raise RuntimeError("Successful cached execution has different inputs/command")
        return
    started = time.monotonic()
    value = {"argv": [str(x) for x in argv], "cwd": str(cwd),
             "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "protocol_sha256": sha(dest / "protocol.json"),
             "causal_video_sha256": sha(dest / "causal_15hz.mp4"),
             "status": "RUNNING"}
    write(receipt_path, value)
    with (dest / f"{name}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([str(x) for x in argv], cwd=cwd, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    import resource
    value.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-started,
                 status="COMPLETE" if result.returncode == 0 else "FAILED",
                 peak_child_ram_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20)
    write(receipt_path, value)
    if result.returncode:
        raise RuntimeError(f"{name} failed; see {dest / (name+'.log')}")


def vipe(scene, branch="no_vda"):
    dest = prepare_branch(scene, branch)
    env = os.environ.copy()
    env.update(HF_HOME=str(ROOT.parent / ".cache/huggingface"),
               TORCH_HOME=str(ROOT.parent / ".cache/torch"),
               CUDA_HOME="/usr/local/cuda-12.8", PYTHONUNBUFFERED="1")
    # The official no_vda preset. It retains GeoCalib, UniDepth and SLAM.
    preset = "default" if branch == "default" else "no_vda"
    run_receipt(dest, "vipe_no_vda" if preset == "no_vda" else "vipe_default", [sys.executable, "-m", "vipe.cli.main", "infer",
        dest / "causal_15hz.mp4", "-p", preset, "-o", dest / "vipe"], VIPE, env)


def track(scene):
    """The author AllTracker network, queried at t0 on reversed causal video.

    The query is the last observed frame. Reversal makes it frame zero and
    avoids running the vendored forward branch on a one-frame suffix.
    No temporal interpolation or substitute tracker is used.
    """
    import torch
    sys.path.insert(0, str(ALLTRACKER))
    from nets.alltracker import Net
    dest = RUN / scene
    p = read(dest / "protocol.json")
    frames = load_video(dest / "causal_15hz.mp4", len(p["observed_raw_ids"]))
    q = np.load(dest / "query_points.npz")
    points = q["query_points"][:, 1:].astype(np.float32)
    background_end = len(points)
    if scene == "A":
        old = ROOT / "runs/dobbe_rgbd_study/approx_history/points_2d_at_t0_candidate.npy"
        points = np.concatenate([points, np.load(old).astype(np.float32)])
    torch.manual_seed(0)
    torch.hub.set_dir(str(ROOT.parent / ".cache/torch/hub"))
    url = "https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth"
    checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    model = Net(16)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().eval()
    rgbs = torch.from_numpy(frames[::-1, :, :, ::-1].copy()).permute(0, 3, 1, 2)[None].float().cuda()
    started = time.monotonic()
    with torch.inference_mode():
        flows, confidence, _, _ = model(rgbs, iters=4, sw=None, is_training=False)
    assert flows.shape[1] == len(frames) and flows.shape[2:] == (2, 256, 256)
    # Author model output includes the anchor; exact t0 coordinates are retained.
    grid = torch.from_numpy(points/255*2-1).to(flows.device)[None, None].expand(len(frames), -1, -1, -1)
    xy = torch.nn.functional.grid_sample(flows[0], grid, align_corners=True)[:, :, 0].permute(0, 2, 1)
    xy = xy.cpu().numpy() + points[None]
    conf = torch.nn.functional.grid_sample(confidence[0, :, :1], grid,
                                           align_corners=True)[:, 0, 0].cpu().numpy()
    xy, conf = xy[::-1].copy(), conf[::-1].copy()
    xy[-1] = points
    conf[-1] = 1
    np.savez(dest / "tracks.npz", tracks=xy.astype(np.float32),
             confidence=conf, visibility=conf > .5, object_count=int(q["object_count"]),
             background_end=background_end, raw_ids=p["observed_raw_ids"])
    write(dest / "alltracker_execution.json", {"success": True,
        "source": "vendored author nets.alltracker.Net(16)", "checkpoint_url": url,
        "checkpoint_sha256": sha(ROOT.parent / ".cache/torch/hub/checkpoints/alltracker.pth"),
        "direction": "reverse causal prefix, query raw t0; reverse outputs back",
        "future_used": False, "inference_iters": 4, "dtype": "float32",
        "input_shape": list(rgbs.shape), "tracks_shape": list(xy.shape),
        "elapsed_seconds": time.monotonic()-started,
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
        "query_sha256": sha(dest / "query_points.npz"), "tracks_sha256": sha(dest / "tracks.npz")})
    print(scene, "AllTracker complete", xy.shape, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "masks", "branch_prepare", "vipe", "track", "geometry", "variants", "processor", "infer"])
    parser.add_argument("--scene", choices=[*SCENES, "all"], default="all")
    parser.add_argument("--variant", choices=["smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "hybrid", "paired_vipe", "paired_hybrid"], default="pure_vipe")
    parser.add_argument("--branch", choices=["no_vda", "default", "rectified", "sift_audited"], default="no_vda")
    args = parser.parse_args()
    if args.branch == "sift_audited" and args.stage not in ["variants", "processor", "infer"]:
        parser.error("sift_audited supports variants/processor/infer only; preserve the original geometry gate")
    scenes = list(SCENES) if args.scene == "all" else [args.scene]
    if args.stage == "geometry":
        from dobbe_vipe_geometry import geometry
        for scene in scenes:
            geometry(scene, args.branch)
        return
    if args.stage in ["variants", "processor", "infer"]:
        from dobbe_vipe_inference import export_variants, inputs, infer
        for scene in scenes:
            if args.stage == "variants":
                export_variants(scene, args.branch)
            elif args.stage == "processor":
                inputs(scene, args.variant, args.branch)
            else:
                infer(scene, args.variant, args.branch)
        return
    for scene in scenes:
        try:
            if args.stage == "branch_prepare":
                prepare_branch(scene, args.branch)
            elif args.stage == "vipe":
                vipe(scene, args.branch)
            else:
                globals()[args.stage](scene)
        except Exception as exc:
            import traceback
            write(RUN / scene / f"{args.stage}_failure.json", {
                "stage": args.stage, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "future_used": False})
            raise


if __name__ == "__main__":
    main()
