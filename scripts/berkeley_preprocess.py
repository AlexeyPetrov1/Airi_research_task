"""Observed-only Berkeley external-camera preparation for MolmoMotion.

All commands read RGB only from observed/rgb.npy (RGB uint8, chronological).
Nothing in this module reads evaluation/. Model H3 is always the final three
real 5 Hz frames; a longer observed prefix supports author trust filtering.

Example (from repo):
 python scripts/berkeley_preprocess.py depth --scene-dir runs/.../cup
 python scripts/berkeley_preprocess.py ground --scene-dir runs/.../cup \
   --pointing-json runs/.../cup/observed/molmopoint_grounding.json \
   --sam-checkpoint ../models/sam2.1_hiera_large.pt
 python scripts/berkeley_preprocess.py track --scene-dir runs/.../cup
 python scripts/berkeley_preprocess.py filter --scene-dir runs/.../cup
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
ALLTRACKER = ROOT / "data_generation/third_party/alltracker"
FILTER_SOURCE = ROOT / "data_generation/third_party/vipe/track-filter-smooth.py"
KMEANS_SOURCE = ROOT / "data_generation/third_party/sam3/querypoints_from_video.py"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_observed(scene):
    """Validate temporal fence using only files in observed/ and metadata."""
    scene = Path(scene)
    rgb = np.load(scene / "observed/rgb.npy", allow_pickle=False)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or len(rgb) < 3 or rgb.dtype != np.uint8:
        raise ValueError(f"Expected chronological RGB uint8 [T>=3,H,W,3], got {rgb.shape} {rgb.dtype}")
    indices_path = scene / "observed/source_indices.npy"
    meta_path = scene / "metadata.json"
    meta = read_json(meta_path) if meta_path.exists() else {}
    if indices_path.exists():
        indices = np.load(indices_path, allow_pickle=False)
    else:
        indices = np.asarray(meta.get("observed_source_indices", np.arange(len(rgb))))
    if indices.shape != (len(rgb),) or np.any(np.diff(indices) != 1):
        raise ValueError("Observed input must consist of consecutive real Berkeley frames")
    t0 = int(meta.get("t0_source_frame_index", meta.get("t0", indices[-1])))
    if int(indices[-1]) != t0 or np.any(indices > t0):
        raise ValueError(f"Temporal fence violation: indices={indices.tolist()}, t0={t0}")
    for name in ("geometry", "groups", "viz"):
        (scene / name).mkdir(exist_ok=True)
    return rgb, indices, meta


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(rgb)[..., ::-1]):
        raise OSError(path)


def draw_points(rgb, xy, ids=None, radius=3):
    import matplotlib
    colors = (matplotlib.colormaps["turbo"](np.linspace(0, 1, len(xy)))[:, :3] * 255).astype(np.uint8)
    image = rgb.copy()
    for n, (uv, color) in enumerate(zip(xy, colors)):
        if not np.isfinite(uv).all():
            continue
        u, v = np.rint(uv).astype(int)
        cv2.circle(image, (u, v), radius + 1, (0, 0, 0), -1)
        cv2.circle(image, (u, v), radius, tuple(map(int, color)), -1)
        if ids is not None:
            text = str(int(ids[n]))
            cv2.putText(image, text, (u+4, v-3), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, text, (u+4, v-3), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def official_kmeans(mask, t0):
    """Execute exactly the vendored KMeans function without SAM3 imports."""
    tree = ast.parse(KMEANS_SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kmeans_sample")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(KMEANS_SOURCE), "exec"), namespace)
    return namespace["kmeans_sample"](mask, k=100, frame_idx=int(t0), seed=0)[:, 1:].astype(np.float32)


def grounding(scene, point, sam_checkpoint, negative_points=(), mask_path=None, object_phrase=None, pointing_json=None):
    """User-approved official SAM2.1 prompted by real MolmoPoint at t0."""
    import torch
    scene = Path(scene)
    rgb, indices, meta = load_observed(scene)
    target = object_phrase or meta.get("target_object", "blue cup" if scene.name == "cup" else "ranch bottle")
    if pointing_json is None or mask_path is not None or point is not None or len(negative_points):
        raise ValueError("This run requires actual MolmoPoint --pointing-json and official SAM2.1; no manual points/masks")
    pointing = read_json(pointing_json)
    point = pointing["point_xy"]
    if pointing.get("future_used") is not False or int(pointing["source_frame"]) != int(indices[-1]):
        raise ValueError("MolmoPoint grounding metadata does not certify the observed t0 frame")
    started = time.monotonic()
    info = {"object_phrase": target,
            "prompt": f"point to {target} gripped and picked up by the robot gripper",
            "source_frame": int(indices[-1]), "future_used": False,
            "object_phrase_source": "explicit user target (no recaption needed)",
            "pointing_method": "actual official MolmoPoint-Vid-4B inference on observed t0",
            "pointing_metadata": str(pointing_json),
            "pointing_metadata_sha256": sha256(pointing_json),
            "segmentation_version_reason": "SAM3 weights access returned403 GatedRepo; user explicitly selected official Meta SAM2.1 earlier release",
            "point_xy": list(map(float, point)) if point is not None else None,
            "negative_points_xy": np.asarray(negative_points).tolist(),
            "kmeans_source": str(KMEANS_SOURCE.relative_to(ROOT)),
            "kmeans_source_sha256": sha256(KMEANS_SOURCE)}
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
        info["segmentation"] = "explicit supplied observed-only mask"
        info["mask_source"] = str(mask_path)
    else:
        if sam_checkpoint is None:
            raise ValueError("ground needs official SAM2.1 --sam-checkpoint")
        sam_parent = WORKSPACE / "third_party/sam2"
        sys.path.insert(0, str(sam_parent))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", str(sam_checkpoint), device="cuda")
        predictor = SAM2ImagePredictor(model)
        pts = np.asarray([point, *negative_points], np.float32)
        labels = np.asarray([1] + [0] * len(negative_points), np.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(rgb[-1])
            masks, scores, logits = predictor.predict(point_coords=pts, point_labels=labels, multimask_output=True)
        chosen = int(np.argmax(scores))
        mask = masks[chosen].astype(bool)
        np.savez_compressed(scene / "observed/sam_candidates.npz", masks=masks, scores=scores, logits=logits)
        import subprocess
        sam_commit = subprocess.check_output(["git", "-C", str(sam_parent), "rev-parse", "HEAD"], text=True).strip()
        info.update(segmentation="official Meta SAM2.1 Hiera-Large", sam_repository="https://github.com/facebookresearch/sam2",
                    sam_code_commit=sam_commit, sam_config="configs/sam2.1/sam2.1_hiera_l.yaml", checkpoint=str(sam_checkpoint),
                    checkpoint_sha256=sha256(sam_checkpoint), mask_scores=scores.tolist(), chosen_mask=chosen)
        del predictor, model
        gc.collect()
        torch.cuda.empty_cache()
    if mask.shape != rgb.shape[1:3] or int(mask.sum()) < 100:
        raise RuntimeError(f"Invalid target mask shape/area: {mask.shape}, {mask.sum()}")
    queries = official_kmeans(mask, indices[-1])
    if queries.shape != (100, 2):
        raise RuntimeError(f"Expected 100 KMeans points, got {queries.shape}")
    # Author KMeans centers can land outside concave masks. Keep exact author
    # points and mark them ineligible instead of silently relocating them.
    xy = np.rint(queries).astype(int)
    in_mask = mask[xy[:, 1], xy[:, 0]]
    np.save(scene / "observed/query_points_100.npy", queries)
    np.save(scene / "observed/query_in_mask.npy", in_mask)
    cv2.imwrite(str(scene / "observed/mask.png"), mask.astype(np.uint8) * 255)
    overlay = rgb[-1].copy()
    overlay[mask] = (.65 * overlay[mask] + .35 * np.array([80, 230, 70])).astype(np.uint8)
    save_rgb(scene / "observed/mask_overlay.png", overlay)
    save_rgb(scene / "viz/mask_and_100_queries.png", draw_points(overlay, queries, radius=2))
    info.update(mask_area_px=int(mask.sum()), query_count=len(queries), centers_inside_mask=int(in_mask.sum()),
                runtime_seconds=time.monotonic()-started, input_rgb_sha256=sha256(scene / "observed/rgb.npy"))
    write_json(scene / "observed/grounding_metadata.json", info)
    print("ground", scene.name, info["mask_area_px"], "mask pixels", flush=True)


def estimate_depth(scene):
    """Run installed UniDepthV2 on each observed RGB; never use future K."""
    import torch
    scene = Path(scene)
    rgb, indices, meta = load_observed(scene)
    sys.path.insert(0, str(WORKSPACE / "third_party/UniDepth"))
    from unidepth.models import UniDepthV2
    checkpoint = WORKSPACE / "models/unidepth-v2-vits14"
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    model = UniDepthV2.from_pretrained(str(checkpoint)).cuda().eval()
    model.resolution_level = 0
    depth_frames, matrices, confidence_frames = [], [], []
    for frame, index in zip(rgb, indices):
        tensor = torch.from_numpy(frame.copy()).permute(2, 0, 1).cuda()
        with torch.inference_mode():
            output = model.infer(tensor)
        depth_frames.append(output["depth"][0, 0].float().cpu().numpy())
        matrices.append(output["intrinsics"][0].float().cpu().numpy())
        confidence_frames.append(output["confidence"][0, 0].float().cpu().numpy())
        print("unidepth", scene.name, int(index), np.median(depth_frames[-1]), flush=True)
        del output, tensor
        torch.cuda.empty_cache()
    unidepth = np.asarray(depth_frames, np.float32)
    matrices = np.asarray(matrices, np.float32)
    median_k = np.median(matrices, axis=0).astype(np.float32)
    native_path = scene / "observed/native_depth.npy"
    if native_path.exists():
        provenance = read_json(scene / "observed/native_depth_metadata.json")
        if (not provenance.get("measured_metric_depth") or provenance.get("units") != "meters"
                or provenance.get("source_indices") != indices.tolist() or provenance.get("future_used") is not False):
            raise ValueError("Native depth lacks matching observed-only measured metric provenance")
        depth = np.load(native_path, allow_pickle=False).astype(np.float32)
        if depth.shape != unidepth.shape:
            raise ValueError(f"Native aligned depth shape mismatch {depth.shape} != RGB {unidepth.shape}")
        depth_source = "RealSense measured aligned metric depth"
        geometry_source = "measured-depth_estimated-intrinsics"
    else:
        depth = unidepth
        depth_source = "UniDepthV2 estimated metric depth"
        geometry_source = "estimated-depth"
    if not np.isfinite(matrices).all() or np.any(matrices[:, [0, 1], [0, 1]] <= 0):
        raise RuntimeError("Invalid inferred camera intrinsics")
    np.save(scene / "geometry/K_per_frame.npy", matrices)
    np.save(scene / "geometry/K_median.npy", median_k)
    np.save(scene / "geometry/depth_observed.npy", depth)
    np.save(scene / "geometry/unidepth_depth_observed.npy", unidepth)
    np.save(scene / "geometry/unidepth_confidence_observed.npy", np.asarray(confidence_frames, np.float32))
    components = np.stack([matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 0, 2], matrices[:, 1, 2]], axis=1)
    component_medians = np.median(components, axis=0)
    relative_range = np.ptp(components, axis=0) / np.maximum(np.abs(component_medians), 1e-6)
    stability = {"components": ["fx", "fy", "cx", "cy"], "source_indices": indices.tolist(),
                 "values_px": components.tolist(), "median_px": component_medians.tolist(),
                 "range_relative_to_median": relative_range.tolist(),
                 "focal_relative_range_max": float(np.max(relative_range[:2])),
                 "focal_stable_within_15_percent": bool(np.max(relative_range[:2]) < .15),
                 "future_used": False, "K_px": median_k.tolist()}
    write_json(scene / "geometry/intrinsics_stability.json", stability)
    info = {"depth_source": depth_source, "geometry_source": geometry_source,
            "K_source": "UniDepthV2 robust elementwise median of observed-frame intrinsics",
            "checkpoint": str(checkpoint), "source_indices": indices.tolist(), "future_used": False,
            "checkpoint_sha256": sha256(checkpoint / "model.safetensors"),
            "checkpoint_revision": (checkpoint / ".cache/huggingface/download/model.safetensors.metadata").read_text().splitlines()[0],
            "depth_patch_window_px": 5, "runtime_seconds": time.monotonic()-started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/2**30,
            "native_depth_used": native_path.exists(), "no_metric_scale_fit": True}
    write_json(scene / "geometry/depth_source.json", info)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for i, (name, ax) in enumerate(zip(stability["components"], axes.flat)):
        ax.plot(indices, components[:, i], "o-")
        ax.axhline(component_medians[i], color="black", linestyle="--", label="observed median")
        ax.set(xlabel="source frame", ylabel=f"{name} (px)")
        ax.grid(alpha=.2)
    fig.suptitle(f"{scene.name}: UniDepthV2 observed-only intrinsics")
    fig.tight_layout()
    fig.savefig(scene / "viz/intrinsics_stability.png", dpi=150)
    plt.close(fig)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("depth", scene.name, median_k.tolist(), flush=True)


def alltracker_tracks(rgb, points, reverse=False, max_side=512):
    """Author AllTracker with dense-flow sampling, returning source pixel units.

    This function can also be called separately by evaluation AFTER prediction.
    For observed input, reverse=True anchors the genuine last observed frame.
    """
    import torch
    sys.path.insert(0, str(ALLTRACKER))
    from nets.alltracker import Net
    checkpoint = WORKSPACE / ".cache/torch/hub/checkpoints/alltracker.pth"
    h, w = rgb.shape[1:3]
    scale = min(1.0, max_side / max(h, w))
    scaled_h = max(32, int(round(h*scale/8))*8)
    scaled_w = max(32, int(round(w*scale/8))*8)
    clip = rgb[::-1].copy() if reverse else rgb.copy()
    resized = np.stack([cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR) for frame in clip])
    scaled_points = np.asarray(points, np.float32) * np.array([scaled_w/w, scaled_h/h], np.float32)
    torch.manual_seed(0)
    model = Net(16)
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(weights["model"], strict=True)
    model = model.cuda().eval()
    del weights
    tensor = torch.from_numpy(resized).permute(0, 3, 1, 2)[None].float().cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.inference_mode():
        flows, confidence, _, _ = model(tensor, iters=4, sw=None, is_training=False)
        if flows.ndim != 5 or flows.shape[1] != len(clip):
            raise RuntimeError(f"AllTracker returned unexpected flow shape {flows.shape} for T={len(clip)}")
        norm = scaled_points / np.array([scaled_w-1, scaled_h-1], np.float32) * 2 - 1
        grid = torch.from_numpy(norm).cuda()[None, None].expand(len(clip), -1, -1, -1)
        sampled = torch.nn.functional.grid_sample(flows[0], grid, align_corners=True)[:, :, 0].permute(0, 2, 1)
        xy = sampled.cpu().numpy() + scaled_points[None]
        probabilities = torch.nn.functional.grid_sample(confidence[0], grid, align_corners=True)[:, :, 0].cpu().numpy()
        visprob, conf = probabilities[:, 0], probabilities[:, 1]
    xy *= np.array([w/scaled_w, h/scaled_h], np.float32)
    xy[0] = points
    conf[0] = 1
    visprob[0] = 1
    if reverse:
        xy, conf, visprob = xy[::-1].copy(), conf[::-1].copy(), visprob[::-1].copy()
    in_frame = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    visible = (visprob > .5) & in_frame & np.isfinite(xy).all(-1)
    info = {"source": "vendored author AllTracker Net(16)", "checkpoint_sha256": sha256(checkpoint),
            "iters": 4, "native_size_wh": [w, h], "network_size_wh": [scaled_w, scaled_h],
            "flow_sampling": "bilinear dense-flow sampling; source-pixel-unit output",
            "direction": "reverse observed-only prefix" if reverse else "forward evaluation continuation",
            "runtime_seconds": time.monotonic()-started,
            "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/2**30}
    del tensor, model, flows, confidence, sampled, grid
    gc.collect()
    torch.cuda.empty_cache()
    return xy.astype(np.float32), conf.astype(np.float32), visible, visprob.astype(np.float32), info


def track_observed(scene, max_side=512):
    scene = Path(scene)
    rgb, indices, meta = load_observed(scene)
    query = np.load(scene / "observed/query_points_100.npy", allow_pickle=False)
    tracks, confidence, visible, visprob, info = alltracker_tracks(rgb, query, reverse=True, max_side=max_side)
    np.savez_compressed(scene / "observed/observed_tracks_2d.npz", tracks=tracks, confidence=confidence,
                        visibility=visible, visibility_probability=visprob, source_indices=indices, dim=np.array(rgb.shape[1:3]))
    info.update(future_used=False, source_indices=indices.tolist(), query_count=len(query))
    write_json(scene / "observed/alltracker_execution.json", info)
    print("track", scene.name, tracks.shape, visible[-3:].all(0).sum(), "H3-visible", flush=True)


def camera_motion_audit(rgb, mask):
    """Observed-only sparse background optical flow with forward/backward check."""
    h, w = mask.shape
    allowed = (255 - cv2.dilate(mask.astype(np.uint8)*255, np.ones((41, 41), np.uint8)))
    allowed[:12] = allowed[-12:] = 0
    allowed[:, :12] = allowed[:, -12:] = 0
    # Evaluate distant background in all quadrants; moving robot features are
    # retained as outliers, while robust median measures the fixed scene.
    grays = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in rgb]
    pairs = []
    for t in range(1, len(rgb)):
        query = cv2.goodFeaturesToTrack(grays[t], maxCorners=400, qualityLevel=.02, minDistance=8, mask=allowed)
        if query is None:
            continue
        past, status, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t-1], query, None)
        back, status2, _ = cv2.calcOpticalFlowPyrLK(grays[t-1], grays[t], past, None)
        valid = status[:, 0].astype(bool) & status2[:, 0].astype(bool) & (np.linalg.norm(back[:, 0]-query[:, 0], axis=1) < 1)
        delta = past[valid, 0] - query[valid, 0]
        if len(delta):
            displacement = np.linalg.norm(delta, axis=1)
            pairs.append({"pair": [t-1, t], "count": len(delta),
                          "median_displacement_px": float(np.median(displacement)),
                          "p90_displacement_px": float(np.percentile(displacement, 90)),
                          "median_flow_xy_px": np.median(delta, axis=0).tolist(),
                          "fraction_below_1px": float(np.mean(displacement < 1))})
    if not pairs:
        raise RuntimeError("No background correspondences to verify fixed camera")
    max_median = max(item["median_displacement_px"] for item in pairs)
    fixed = max_median < 1.5 and min(item["count"] for item in pairs) >= 30
    return {"method": "observed-only background Shi-Tomasi + pyramidal LK; forward/backward <1px; object mask excluded",
            "pairs": pairs, "max_pair_median_displacement_px": max_median,
            "fixed_camera_supported": fixed, "median_threshold_px": 1.5, "minimum_pair_count": 30,
            "camera_pose": "identity" if fixed else "not accepted", "future_used": False,
            "limitations": "Robot-arm features may remain among background candidates; robust median, not upper tail, determines gate."}


def robust_lift(depth, tracks, visibility, K, patch_size=5):
    """Median of finite positive values in a 5x5 patch, preserving subpixel rays."""
    T, N, _ = tracks.shape
    h, w = depth.shape[1:]
    xyz = np.full((T, N, 3), np.nan, np.float32)
    vis = np.zeros((T, N), bool)
    valid_fraction = np.zeros((T, N), np.float32)
    z_spread = np.full((T, N), np.nan, np.float32)
    half = patch_size//2
    for t in range(T):
        for n in np.flatnonzero(visibility[t]):
            u, v = tracks[t, n]
            x, y = np.rint([u, v]).astype(int)
            if not (0 <= x < w and 0 <= y < h):
                continue
            patch = depth[t, max(0, y-half):min(h, y+half+1), max(0, x-half):min(w, x+half+1)]
            # 65.535 m is the uint16 saturation sentinel after mm->m; depths
            # beyond 10 m are irrelevant to this table scene and rejected.
            vals = patch[np.isfinite(patch) & (patch > 0) & (patch < 10)]
            valid_fraction[t, n] = len(vals)/patch.size
            if len(vals) < max(3, (patch.size+1)//2):
                continue
            z = float(np.median(vals))
            z_spread[t, n] = float(np.percentile(vals, 90)-np.percentile(vals, 10))
            xyz[t, n] = [(u-K[0, 2])/K[0, 0]*z, (v-K[1, 2])/K[1, 1]*z, z]
            vis[t, n] = True
    return xyz, vis, valid_fraction, z_spread


def load_author_filter():
    spec = importlib.util.spec_from_file_location("berkeley_author_filter", FILTER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spread_select(points, eligible, count, quality):
    """Deterministic spatial coverage among valid points, no future scores."""
    candidates = np.flatnonzero(eligible)
    if not len(candidates):
        return np.empty(0, np.int64)
    first = int(candidates[np.argmax(quality[candidates])])
    chosen = [first]
    distance = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, min(count, len(candidates))):
        score = distance.copy()
        score[~eligible] = -np.inf
        score[chosen] = -np.inf
        nxt = int(np.argmax(score))
        chosen.append(nxt)
        distance = np.minimum(distance, np.linalg.norm(points-points[nxt], axis=1))
    return np.asarray(chosen, np.int64)


def filter_and_export(scene):
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scene = Path(scene)
    rgb, indices, meta = load_observed(scene)
    tracks_npz = np.load(scene / "observed/observed_tracks_2d.npz", allow_pickle=False)
    tracks, confidence, visibility = (tracks_npz[key] for key in ("tracks", "confidence", "visibility"))
    depth = np.load(scene / "geometry/depth_observed.npy", allow_pickle=False)
    K = np.load(scene / "geometry/K_median.npy", allow_pickle=False)
    mask = cv2.imread(str(scene / "observed/mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    motion = camera_motion_audit(rgb, mask)
    write_json(scene / "geometry/camera_motion_audit.json", motion)
    if not motion["fixed_camera_supported"]:
        raise RuntimeError("Background optical flow does not support identity camera pose; inspect audit")
    np.save(scene / "geometry/camera_poses.npy", np.repeat(np.eye(4, dtype=np.float32)[None], len(rgb), axis=0))
    xyz, valid, depth_valid, spread = robust_lift(depth, tracks, visibility, K, patch_size=5)
    author = load_author_filter()
    started = time.monotonic()
    trust, anchor_ids = author.compute_trust_weights(xyz, valid, K=16)
    drop = author.filter_tracks_by_trust(trust, valid, z_thresh=2.)
    keep = ~drop
    filtered = xyz.copy()
    if keep.any():
        filtered[:, keep] = author.consensus_gated_smooth(xyz[:, keep], np.zeros((len(rgb), 3), np.float32),
                                                         valid[:, keep], trust[:, keep], device="cpu")
    in_mask = np.load(scene / "observed/query_in_mask.npy", allow_pickle=False)
    eligible = keep & valid[-3:].all(0) & np.isfinite(filtered[-3:]).all((0, 2)) & in_mask
    # Preserve all official-kept H3-valid candidates; spatially cover their mask.
    count = min(24, int(eligible.sum())//8*8)
    if count < 8:
        raise RuntimeError(f"Scene preparation FAILED: only {eligible.sum()} valid points")
    selected = spread_select(tracks[-1], eligible, count, trust[-3:].mean(0)*confidence[-3:].min(0))
    np.save(scene / "geometry/points_3d_raw.npy", xyz)
    np.save(scene / "geometry/points_3d_filtered.npy", filtered)
    np.savez_compressed(scene / "geometry/filter_diagnostics.npz", trust=trust, anchor_ids=anchor_ids,
                        keep=keep, eligible=eligible, depth_valid_fraction=depth_valid, depth_patch_spread_m=spread)
    np.save(scene / "observed/selected_point_ids.npy", selected)
    np.save(scene / "observed/history_rgb.npy", rgb[-3:])
    np.save(scene / "observed/points_2d_history.npy", tracks[-3:, selected])
    np.save(scene / "observed/points_3d_history.npy", filtered[-3:, selected])
    for i, frame in enumerate(rgb[-3:]):
        save_rgb(scene / f"observed/frame_t{i-2:+d}.png", frame)
    for group_idx in range(count//8):
        folder = scene / f"groups/group_{group_idx:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        ids = selected[group_idx*8:(group_idx+1)*8]
        point2 = tracks[-1, ids].astype(np.float32)
        point3 = filtered[-3:, ids].astype(np.float32)
        np.save(folder / "point_ids.npy", ids)
        np.save(folder / "points_2d_at_t0.npy", point2)
        np.save(folder / "points_3d_history.npy", point3)
        torch.save(torch.from_numpy(point2), folder / "points_2d_at_t0.pt")
        torch.save(torch.from_numpy(point3), folder / "points_3d_history.pt")
        write_json(folder / "metadata.json", {"group": group_idx, "point_ids": ids.tolist(),
                   "history_source_indices": indices[-3:].tolist(), "history_times_relative_t0_seconds": [-.4, -.2, 0],
                   "coordinates": "OpenCV camera XYZ in meters; fixed camera identity poses", "future_used": False})
    info = {"author_source": str(FILTER_SOURCE.relative_to(ROOT)), "author_source_sha256": sha256(FILTER_SOURCE),
            "author_functions": ["compute_trust_weights", "filter_tracks_by_trust", "consensus_gated_smooth"],
            "anchor_count": len(anchor_ids), "anchor_ids": anchor_ids.tolist(), "depth_patch_window_px": 5,
            "valid_depth_interval_m": [0, 10], "depth_interval_endpoints": "exclusive",
            "lift_wrapper": "finite-positive depth patch median; subpixel ray backprojection with fixed median K",
            "spatial_auto_split": "not needed: one grounded object", "trust_z_threshold": 2.,
            "smooth_parameters": {"alpha": 1000., "lambda_reg": .0001, "p": 2, "deltas": [1, 3, 5], "steps": 100, "lr": .05},
            "history_indices": indices[-3:].tolist(), "observed_indices": indices.tolist(), "future_used": False,
            "candidate_count": len(keep), "official_kept_count": int(keep.sum()), "eligible_h3_count": int(eligible.sum()),
            "selected_count": count, "group_count": count//8, "selected_point_ids": selected.tolist(),
            "selection": "spatial farthest point selection among official-kept H3-visible mask-contained points",
            "runtime_seconds": time.monotonic()-started,
            "max_smoothing_displacement_m": float(np.nanmax(np.linalg.norm(filtered-xyz, axis=-1))),
            "temporal_limitation": "Berkeley real H3 at 5 Hz (-0.4,-0.2,0s), versus MolmoMotion training 15 Hz; no frame interpolation/duplication"}
    write_json(scene / "geometry/filter_metadata.json", info)
    save_rgb(scene / "viz/selected_24_points.png", draw_points(rgb[-1], tracks[-1, selected], selected))
    tiles = []
    for index, frame, xy in zip(indices[-3:], rgb[-3:], tracks[-3:, selected]):
        tile = draw_points(frame, xy, selected)
        cv2.putText(tile, f"source {index} | {(index-indices[-1])/5:+.1f}s", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        tiles.append(tile)
    save_rgb(scene / "viz/history_contact_sheet.png", np.hstack(tiles))
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_depth = depth[-1][np.isfinite(depth[-1]) & (depth[-1] > 0) & (depth[-1] < 10)]
    lo, hi = np.percentile(valid_depth, [2, 98])
    handle = ax.imshow(depth[-1], cmap="magma", vmin=lo, vmax=hi)
    ax.scatter(tracks[-1, selected, 0], tracks[-1, selected, 1], s=18, facecolors="none", edgecolors="cyan")
    fig.colorbar(handle, ax=ax, label="depth (m)")
    ax.set_title(f"{scene.name}: t0 depth and selected points")
    fig.tight_layout()
    fig.savefig(scene / "viz/depth_t0.png", dpi=150)
    plt.close(fig)
    print("filter", scene.name, count, "selected points; groups", count//8, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["ground", "depth", "track", "filter"])
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--point", nargs=2, type=float)
    parser.add_argument("--negative-point", nargs=2, type=float, action="append", default=[])
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--object-phrase")
    parser.add_argument("--pointing-json", type=Path)
    parser.add_argument("--max-side", type=int, default=512)
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.stage == "ground":
        grounding(args.scene_dir, args.point, args.sam_checkpoint, args.negative_point, args.mask, args.object_phrase, args.pointing_json)
    elif args.stage == "depth":
        estimate_depth(args.scene_dir)
    elif args.stage == "track":
        track_observed(args.scene_dir, args.max_side)
    else:
        filter_and_export(args.scene_dir)


if __name__ == "__main__":
    main()
