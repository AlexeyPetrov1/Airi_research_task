"""Freeze an exploratory [3,8,3] red-cup history using transferred RGB-only K.

Only the protocol's three history frames, their measured HoNY depth, and their
published Dobb-E poses enter this artifact. The K comes from F2-NeRF's export
of control-scene RGB-only COLMAP. No future frame or model output is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inspect_hony_scenes import ROOT, liblzfse
from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, unproject


OUT = ROOT / "runs/dobbe_rgbd_study/approx_history"
STUDY = ROOT / "runs/dobbe_rgbd_study"
RAW = ROOT / "data/dobbe_oxe/target_raw"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cup_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 75, 25), (13, 255, 255))
    red |= cv2.inRange(hsv, (168, 75, 25), (179, 255, 255))
    region = np.zeros(red.shape, dtype=np.uint8)
    region[45:215, 60:178] = 255
    red &= region
    return cv2.erode(red, np.ones((5, 5), dtype=np.uint8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((STUDY / "protocol.json").read_text(encoding="utf-8"))
    history = protocol["main"]["history"]
    assert history == [96, 97, 98]
    receipt_path = STUDY / "f2nerf_second/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    k = np.asarray(receipt["K_256x256"], dtype=np.float64)
    assert k.shape == (3, 3)

    frames = [cv2.imread(str(OUT / f"rgb_{i:04d}.png")) for i in history]
    assert all(frame is not None and frame.shape[:2] == (256, 256)
               for frame in frames)
    gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    masks = [cup_mask(frame) for frame in frames]
    found = cv2.goodFeaturesToTrack(gray[0], maxCorners=180,
                                     qualityLevel=.002, minDistance=5,
                                     mask=masks[0], blockSize=5)
    if found is None:
        raise RuntimeError("No RGB cup features in H0")
    uv0 = found.astype(np.float32)
    uv1, ok1, _ = cv2.calcOpticalFlowPyrLK(
        gray[0], gray[1], uv0, None, winSize=(15, 15), maxLevel=2)
    uv2, ok2, _ = cv2.calcOpticalFlowPyrLK(
        gray[1], gray[2], uv1, None, winSize=(15, 15), maxLevel=2)
    back0, back_ok, _ = cv2.calcOpticalFlowPyrLK(
        gray[2], gray[0], uv2, None, winSize=(15, 15), maxLevel=2)
    depth = np.frombuffer(liblzfse.decompress(
        (RAW / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)[:history[-1]+1]
    candidates = []
    for j in range(len(uv0)):
        if not (ok1[j, 0] and ok2[j, 0] and back_ok[j, 0]):
            continue
        path = np.array([uv0[j, 0], uv1[j, 0], uv2[j, 0]], dtype=float)
        if np.linalg.norm(path[0] - back0[j, 0]) > .8:
            continue
        if not all(2 <= round(u) <= 253 and 2 <= round(v) <= 253 and
                   mask[round(v), round(u)] > 0
                   for (u, v), mask in zip(path, masks)):
            continue
        samples = [depth_at(depth[i], *uv) for i, uv in zip(history, path)]
        if any(not np.isfinite(z) or spread > .025
               for z, spread in samples):
            continue
        if max(z for z, _ in samples) - min(z for z, _ in samples) > .03:
            continue
        candidates.append({"rgb_uv": path.tolist(),
                           "depth_m": [z for z, _ in samples],
                           "depth_spread_m": [spread for _, spread in samples],
                           "round_trip_px": float(np.linalg.norm(
                               path[0] - back0[j, 0]))})
    if len(candidates) < 8:
        raise RuntimeError(f"Only {len(candidates)} valid cup features")

    # Deterministic RGB-space coverage after quality filtering; K is unused.
    chosen = [0]
    xy = np.asarray([c["rgb_uv"][0] for c in candidates])
    while len(chosen) < 8:
        distance = np.min(np.linalg.norm(
            xy[:, None, :] - xy[np.asarray(chosen)][None, :, :], axis=2),
            axis=1)
        distance[chosen] = -1
        chosen.append(int(np.argmax(distance)))
    selected = [candidates[j] for j in chosen]
    for j, point in enumerate(selected):
        point["id"] = f"cup_rgb_feature_{j:02d}"

    labels = json.loads((RAW / "labels.json").read_text(encoding="utf-8"))
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in history]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in history]
    xyz_world = np.empty((3, 8, 3), dtype=np.float64)
    for t in range(3):
        uv = np.asarray([p["rgb_uv"][t] for p in selected])
        z = np.asarray([p["depth_m"][t] for p in selected])
        optical = unproject(uv, z, k, np.zeros(4), "z")
        label = optical @ CAMERA_TO_LABEL.T
        xyz_world[t] = label @ poses[t, :3, :3].T + poses[t, :3, 3]
    if xyz_world.shape != (3, 8, 3) or not np.isfinite(xyz_world).all():
        raise RuntimeError("Invalid approximate history")
    # MolmoMotion expects every history point in the optical frame at t0=H2.
    xyz_t0_label = (xyz_world - poses[-1, :3, 3]) @ poses[-1, :3, :3]
    xyz_t0_optical = xyz_t0_label @ CAMERA_TO_LABEL
    if not (xyz_t0_optical[..., 2] > 0).all():
        raise RuntimeError("History has nonpositive depth in H2 optical frame")
    np.save(OUT / "points_3d_history_world_candidate.npy",
            xyz_world.astype(np.float32))
    np.save(OUT / "points_3d_history_candidate.npy",
            xyz_t0_optical.astype(np.float32))
    uv_t0 = np.asarray([p["rgb_uv"][-1] for p in selected], dtype=np.float32)
    np.save(OUT / "points_2d_at_t0_candidate.npy", uv_t0)
    center = xyz_t0_optical.mean(axis=0)
    residual = np.linalg.norm(xyz_t0_optical - center[None], axis=2)
    overlay = []
    for t, (frame, i) in enumerate(zip(frames, history)):
        canvas = frame.copy()
        for j, point in enumerate(selected):
            u, v = np.rint(point["rgb_uv"][t]).astype(int)
            cv2.circle(canvas, (u, v), 4, (0, 255, 255), -1)
            cv2.putText(canvas, str(j), (u+5, v-4),
                        cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
        overlay.append(canvas)
        cv2.imwrite(str(OUT / f"cup_points_{i:04d}.png"), canvas)
    cv2.imwrite(str(OUT / "cup_points_history.png"), np.hstack(overlay))
    manifest = {
        "status": "EXPLORATORY_APPROXIMATE",
        "history_frames": history,
        "future_used": False,
        "K_source": receipt["f2nerf_camera_meta"],
        "K_transfer_assumption": "same effective camera across two HoNY recordings; unverified",
        "K_256x256": k.tolist(),
        "rgb_to_depth": "u_d=u_r; v_d=(v_r+0.5)*0.75-0.5",
        "depth_semantics": "camera-Z hypothesis",
        "pose_semantics": "Dobb-E labels c2w, effective optical-to-label C",
        "optical_to_label": CAMERA_TO_LABEL.tolist(),
        "point_selection": "RGB GFTT H0 + bidirectional LK H0-H2; cup HSV mask; 3x3 depth spread <= 0.025 m each frame; history depth range <= 0.03 m; farthest point sampling in H0 RGB",
        "candidate_count_before_eight": len(candidates),
        "point_ids": [p["id"] for p in selected],
        "points": selected,
        "poses_c2w": poses.tolist(),
        "history_3d_shape": list(xyz_t0_optical.shape),
        "history_3d_frame": "H2=frame98 optical OpenCV: +X right, +Y down, +Z forward",
        "history_3d_sha256": digest(OUT / "points_3d_history_candidate.npy"),
        "history_3d_world_sha256": digest(
            OUT / "points_3d_history_world_candidate.npy"),
        "points_2d_at_t0_sha256": digest(OUT / "points_2d_at_t0_candidate.npy"),
        "static_residual_median_m": float(np.median(residual)),
        "static_residual_p90_m": float(np.percentile(residual, 90)),
        "static_residual_max_m": float(residual.max()),
        "rgb_sha256": {str(i): digest(OUT / f"rgb_{i:04d}.png")
                       for i in history},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps({"candidates": len(candidates),
                      "history_3d_shape": list(xyz_t0_optical.shape),
                      "t0_camera_z_min_m": float(xyz_t0_optical[..., 2].min()),
                      "static_residual_median_m": manifest["static_residual_median_m"],
                      "static_residual_p90_m": manifest["static_residual_p90_m"]},
                     indent=2))


if __name__ == "__main__":
    main()
