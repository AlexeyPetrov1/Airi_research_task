"""Geometry I/O, diagnostics and selection; all arrays use causal ViPE indices."""
from __future__ import annotations

import importlib.util
import io
import json
import zipfile

import cv2
import numpy as np

from dobbe_vipe_v1 import RUN, VIPE, GATE, read, write, sha, scene_dir


def restore_intrinsics(intrinsics, native_wh):
    intrinsics = np.asarray(intrinsics, dtype=np.float64).copy()
    sx, sy = 256/native_wh[0], 256/native_wh[1]
    intrinsics[:, 0] *= sx
    intrinsics[:, 1] *= sy
    intrinsics[:, 2] = (intrinsics[:, 2]+.5)*sx-.5
    intrinsics[:, 3] = (intrinsics[:, 3]+.5)*sy-.5
    return intrinsics


def load_vipe(folder, count, native_wh=(256, 256)):
    name = "causal_15hz"
    expected = np.arange(count)
    camera_file = folder / "intrinsics" / f"{name}_camera.txt"
    if camera_file.exists():
        types = camera_file.read_text().splitlines()
        if types != [f"{i}: PINHOLE" for i in expected]:
            raise ValueError("Unsupported or incomplete ViPE camera model")
    arrays = []
    for kind, shape in [("pose", (count, 4, 4)), ("intrinsics", (count, 4))]:
        with np.load(folder / kind / f"{name}.npz", allow_pickle=False) as z:
            inds, data = z["inds"], z["data"]
            if not np.array_equal(inds, expected) or data.shape != shape:
                raise ValueError(f"Incomplete/misaligned ViPE {kind} IDs: {inds}")
            arrays.append(data.astype(np.float64))
    poses, intrinsics = arrays
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Nonfinite camera geometry")
    if not (intrinsics[:, :2] > 0).all():
        raise ValueError("Nonpositive focal length")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid homogeneous c2w row")
    r = poses[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(r), 1, atol=1e-3):
        raise ValueError("c2w is not a proper rigid transformation")
    import OpenEXR
    import Imath
    depths = []
    with zipfile.ZipFile(folder / "depth" / f"{name}.zip") as z:
        names = sorted(z.namelist())
        if names != [f"{i:05d}.exr" for i in expected]:
            raise ValueError("Incomplete/misaligned ViPE depth IDs")
        for name in names:
            exr = OpenEXR.InputFile(io.BytesIO(z.read(name)))
            dw = exr.header()["dataWindow"]
            width, height = dw.max.x-dw.min.x+1, dw.max.y-dw.min.y+1
            d = np.frombuffer(exr.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT)),
                              dtype=np.float32).reshape(height, width).copy()
            exr.close()
            if d.shape != tuple(native_wh[::-1]):
                raise ValueError(f"Depth resolution {d.shape} requires explicit mapping")
            if native_wh != (256, 256):
                d = cv2.resize(d, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
            depths.append(d)
    if native_wh != (256, 256):
        intrinsics = restore_intrinsics(intrinsics, native_wh)
    return poses, intrinsics, np.stack(depths)


def sample_depth(depth, points):
    """Median 3x3 optical-Z; never clamp out-of-frame track coordinates."""
    h, w = depth.shape
    z = np.full(len(points), np.nan)
    spread = np.full(len(points), np.inf)
    for i, (u, v) in enumerate(points):
        if not np.isfinite([u, v]).all() or not (1 <= u < w-1 and 1 <= v < h-1):
            continue
        x, y = np.round([u, v]).astype(int)
        patch = depth[y-1:y+2, x-1:x+2]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if len(valid) >= 7:
            z[i] = np.median(valid)
            spread[i] = np.ptp(valid)
    return z, spread


def lift(points, depths, intrinsics, c2w, visible):
    world = np.full((*points.shape[:2], 3), np.nan)
    valid = visible.copy()
    spreads = np.full(points.shape[:2], np.inf)
    for t in range(len(points)):
        z, spreads[t] = sample_depth(depths[t], points[t])
        fx, fy, cx, cy = intrinsics[t]
        cam = np.column_stack([(points[t, :, 0]-cx)*z/fx,
                               (points[t, :, 1]-cy)*z/fy, z])
        valid[t] &= np.isfinite(cam).all(-1)
        world[t, valid[t]] = cam[valid[t]] @ c2w[t, :3, :3].T + c2w[t, :3, 3]
    return world, valid, spreads


def project(world, c2w, intrinsics):
    cam = (world-c2w[:3, 3]) @ c2w[:3, :3]
    fx, fy, cx, cy = intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([cam[:, 0]/cam[:, 2]*fx+cx, cam[:, 1]/cam[:, 2]*fy+cy])
    uv[cam[:, 2] <= 0] = np.nan
    return uv


def stats(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return {"count": len(v), "median": float(np.median(v)) if len(v) else None,
            "p90": float(np.percentile(v, 90)) if len(v) else None}


def independent_sift_geometry(dest, depths, intrinsics, poses):
    """Validate the fixed RGB-only matches; never choose matches by depth error."""
    errors, pairs = [], []
    for path in sorted(dest.glob("static_sift_*.npz")):
        t = int(path.stem.rsplit("_", 1)[1])
        matches = np.load(path)
        world, valid, _ = lift(matches["t0_uv"][None], depths[-1:], intrinsics[-1:],
                              poses[-1:], np.ones((1, len(matches["t0_uv"])), bool))
        prediction = project(world[0], poses[t], intrinsics[t])
        keep = valid[0] & np.isfinite(prediction).all(-1)
        residual = np.linalg.norm(prediction[keep]-matches["earlier_uv"][keep], axis=-1)
        pairs.append({"vipe_index": t, **stats(residual)})
        errors.extend(residual)
    return {**stats(errors), "pairs": pairs}


def rigid_error(world):
    i, j = np.triu_indices(world.shape[1], 1)
    distances = np.linalg.norm(world[:, i]-world[:, j], axis=-1)
    deviation = np.abs(distances-distances[-1])
    baseline = np.median(distances[-1])
    return {"absolute_estimated_units": stats(deviation),
            "relative_to_t0_median_pair_distance": stats(deviation/baseline),
            "t0_median_pair_distance": float(baseline)}


def select_spread(points, candidates, quality, count=8):
    # Restrict to above-median quality, then deterministic farthest sampling.
    eligible = candidates[quality[candidates] >= np.median(quality[candidates])]
    if len(eligible) < count:
        eligible = candidates
    picked = [int(eligible[np.argmax(quality[eligible])])]
    while len(picked) < count:
        distance = np.linalg.norm(points[eligible, None]-points[None, picked], axis=-1).min(1)
        distance[np.isin(eligible, picked)] = -1
        picked.append(int(eligible[np.argmax(distance)]))
    return np.array(picked)


def geometry(scene, branch="no_vda"):
    dest = scene_dir(scene, branch)
    p = read(dest / "protocol.json")
    tcount = len(p["observed_raw_ids"])
    poses, k, depths = load_vipe(dest / "vipe", tcount, tuple(p["input_size_wh"]))
    tracks = np.load(dest / "tracks.npz", allow_pickle=False)
    uv, vis = tracks["tracks"], tracks["visibility"].copy()
    nobj = int(tracks["object_count"])
    if uv.shape != (tcount, len(vis[0]), 2) or not np.array_equal(tracks["raw_ids"], p["observed_raw_ids"]):
        raise ValueError("Tracking/ViPE frame identity mismatch")
    world, valid, spreads = lift(uv, depths, k, poses, vis)
    object_world, object_valid = world[:, :nobj], valid[:, :nobj]
    # Use the authors' exact sixteen-anchor consensus and MAD filtering functions.
    spec = importlib.util.spec_from_file_location("author_filter", VIPE / "track-filter-smooth.py")
    filt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(filt)
    trust, anchors = filt.compute_trust_weights(object_world, object_valid, K=16)
    drop = filt.filter_tracks_by_trust(trust, object_valid, z_thresh=1.5)
    history = p["history_vipe_indices"]
    smooth = filt.consensus_gated_smooth(object_world, poses[:, :3, 3],
                  object_valid, trust, steps=100, device="cpu")
    # Absolute gates always inspect unsmoothed geometry, avoiding circular validation.
    good = object_valid[history].all(0) & ~drop
    z_cam = np.stack([np.linalg.norm(object_world[t]-poses[t, :3, 3], axis=-1) for t in history])
    good &= (spreads[np.ix_(history, np.arange(nobj))] < .10*z_cam).all(0)
    candidates = np.flatnonzero(good)
    quality = trust[history].mean(0) * tracks["confidence"][history, :nobj].mean(0)
    selected = select_spread(smooth[-1], candidates, quality) if len(candidates) >= 8 else np.array([], int)
    # Anchor at t0, reproject into earlier frames using c2w/K, compare independently tracked RGB.
    bg = np.arange(nobj, int(tracks.get("background_end", len(uv[0]))))
    pair_reports, all_errors = [], []
    for t in np.linspace(0, tcount-4, 6).astype(int):
        pred = project(world[-1, bg], poses[t], k[t])
        allowed = valid[-1, bg] & vis[t, bg] & np.isfinite(pred).all(-1)
        # Keep bad off-screen predictions: discarding them would bias the gate.
        allowed &= np.isfinite(uv[t, bg]).all(-1) & ((uv[t, bg] >= 0) & (uv[t, bg] <= 255)).all(-1)
        errors = np.linalg.norm(pred[allowed]-uv[t, bg][allowed], axis=-1)
        pair_reports.append({"from_raw": p["t0"], "to_raw": p["observed_raw_ids"][t], **stats(errors)})
        all_errors.extend(errors)
    static = {**stats(all_errors), "pairs": pair_reports,
              "meaning": "t0 static depth lifted once, transformed into earlier cameras vs causal RGB tracks"}
    measured = np.load(dest / "measured_depth_prefix.npy")
    ratio_groups = {"object": [], "background": []}
    alignment = []
    for t in history:
        d = cv2.resize(measured[t], (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
        zv, _ = sample_depth(depths[t], uv[t])
        zd, _ = sample_depth(d, uv[t])
        for group, ids in [("object", np.arange(nobj)), ("background", bg)]:
            allowed = valid[t, ids] & np.isfinite(zd[ids]) & (zd[ids] > 0)
            ratio_groups[group].extend(zv[ids][allowed]/zd[ids][allowed])
        rgb = cv2.imread(str(dest / "frames" / f"{t:05d}.png"))
        edges_rgb = cv2.Canny(rgb, 50, 150)
        norm = np.nan_to_num(np.clip(d, 0, 3)/3*255).astype(np.uint8)
        edges_depth = cv2.Canny(norm, 15, 35) > 0
        near = cv2.distanceTransform(255-edges_rgb, cv2.DIST_L2, 3)
        alignment.append(float((near[edges_depth] <= 3).mean()) if edges_depth.any() else 0.0)
        rgb[edges_depth] = [0, 255, 0]
        cv2.imwrite(str(dest / f"measured_depth_edges_{t:05d}.png"), rgb)
    depth_ratios = {key: stats(vals) for key, vals in ratio_groups.items()}
    rigidity = rigid_error(object_world[history][:, selected]) if len(selected) else None
    relative_focal_range = np.ptp(k[:, :2], axis=0)/np.median(k[:, :2], axis=0)
    checks = {
        "complete_history_eight": len(selected) == 8,
        "sixteen_real_anchors": len(anchors) == 16,
        "static_evidence": static["count"] >= GATE["static_observations_min"] and sum(x["count"] >= 5 for x in pair_reports) >= GATE["static_pairs_min"],
        "static_median": static["median"] is not None and static["median"] <= GATE["static_median_px_max"],
        "static_p90": static["p90"] is not None and static["p90"] <= GATE["static_p90_px_max"],
        "rigid_history": rigidity is not None and rigidity["relative_to_t0_median_pair_distance"]["median"] <= GATE["rigid_relative_median_max"],
        "stable_focal": bool((relative_focal_range <= GATE["focal_relative_range_max"]).all()),
        "depth_scale_supported": all(v["median"] is not None and GATE["depth_ratio_median_min"] <= v["median"] <= GATE["depth_ratio_median_max"] for v in depth_ratios.values()),
    }
    decision = {"status": "PASS_ESTIMATED_GEOMETRY" if all(checks.values()) else "REJECT_ESTIMATED_GEOMETRY",
        "molmo_ready": all(checks.values()), "metric_ground_truth": False, "geometry_branch": branch,
        "checks": checks, "thresholds_frozen_before_vipe": GATE,
        "static_cross_frame_px": static, "rigid_history_unsmoothed": rigidity,
        "intrinsics_min": k.min(0).tolist(), "intrinsics_max": k.max(0).tolist(),
        "intrinsics_stability_interpretation": "ViPE estimates shared intrinsics and broadcasts them; temporal constancy is structural, not independent calibration evidence.",
        "focal_relative_range": relative_focal_range.tolist(),
        "depth_vipe_to_measured_ratio": depth_ratios,
        "measured_depth_edge_fraction_within_3px": alignment,
        "hybrid_alignment_status": "SPATIAL_ALIGNMENT_HYPOTHESIS_REQUIRES_VISUAL_REVIEW",
        "object_candidates": nobj, "after_filter": len(candidates), "anchor_ids": anchors.tolist(),
        "selected_ids": selected.tolist(), "selected_smoothing": "author ray-only gated, CPU 100 steps; diagnostics unsmoothed",
        "future_used": False, "protocol_sha256": sha(dest / "protocol.json"),
        "tracks_sha256": sha(dest / "tracks.npz")}
    np.savez(dest / "geometry_all.npz", points_world=world, valid=valid,
        object_smoothed=smooth, trust=trust, poses=poses, intrinsics=k, depths=depths)
    if len(selected):
        inputs = dest / "pure_vipe"
        inputs.mkdir(exist_ok=True)
        np.save(inputs / "points_3d_world.npy", smooth[history][:, selected].astype(np.float32))
        np.save(inputs / "points_2d_t0.npy", uv[-1, selected].astype(np.float32))
        np.save(inputs / "c2w_t0.npy", poses[-1].astype(np.float32))
    write(dest / "geometry_gate.json", decision)
    render_geometry(dest, p, uv, world, selected, poses, k, depths)
    render_causal_tracks(dest, p, uv, vis, selected)
    print(scene, decision["status"], json.dumps(checks), flush=True)


def render_geometry(dest, protocol, uv, world, selected, poses, k, depths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10))
    for row, t in enumerate(protocol["history_vipe_indices"]):
        rgb = cv2.cvtColor(cv2.imread(str(dest / "frames" / f"{t:05d}.png")), cv2.COLOR_BGR2RGB)
        ax = fig.add_subplot(3, 4, row*4+1)
        ax.imshow(rgb); ax.set_title(f"RGB raw {protocol['observed_raw_ids'][t]}"); ax.axis("off")
        ax = fig.add_subplot(3, 4, row*4+2)
        ax.imshow(depths[t], vmin=np.nanpercentile(depths, 2), vmax=np.nanpercentile(depths, 98), cmap="turbo")
        ax.set_title("ViPE estimated depth"); ax.axis("off")
        ax = fig.add_subplot(3, 4, row*4+3)
        ax.imshow(rgb)
        if len(selected):
            ax.scatter(*uv[t, selected].T, c=np.arange(8), cmap="tab10", vmin=0, vmax=9, s=25)
        ax.set_title("selected object tracks"); ax.axis("off")
        ax = fig.add_subplot(3, 4, row*4+4, projection="3d")
        if len(selected):
            for i, idx in enumerate(selected):
                ax.plot(*world[protocol["history_vipe_indices"], idx].T, color=plt.cm.tab10(i))
        ax.set_title("unsmoothed world 3D"); ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(dest / "geometry_diagnostics.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, name in enumerate(["fx", "fy", "cx", "cy"]):
        axes[0].plot(protocol["observed_raw_ids"], k[:, i], label=name)
    axes[0].legend(); axes[0].set_title("ViPE intrinsics (pixels)")
    axes[1].plot(poses[:, 0, 3], poses[:, 2, 3], "o-")
    axes[1].set_title("estimated camera trajectory X/Z")
    fig.tight_layout(); fig.savefig(dest / "camera_diagnostics.png", dpi=140); plt.close(fig)


def render_causal_tracks(dest, protocol, uv, visible, selected):
    import imageio_ffmpeg
    frames = []
    for t, raw in enumerate(protocol["observed_raw_ids"]):
        img = cv2.imread(str(dest / "frames" / f"{t:05d}.png"))
        points = selected if len(selected) else np.arange(min(100, len(uv[0])))
        for j, idx in enumerate(points):
            if not visible[t, idx] or not np.isfinite(uv[t, idx]).all():
                continue
            hue = int(j / max(1, len(points)) * 179)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            cv2.circle(img, tuple(uv[t, idx].round().astype(int)),
                       2 if len(selected) else 1, tuple(int(x) for x in color), -1)
        label = f"{protocol['scene']} raw {raw}; causal 15 Hz"
        cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3)
        cv2.putText(img, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        frames.append(img)
    # Shareable yuv420 MP4 is only a display artifact; model inputs stay lossless.
    args = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "256x256", "-r", "15",
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(dest / "causal_tracks.mp4")]
    import subprocess
    subprocess.run(args, input=np.stack(frames).tobytes(), check=True)
