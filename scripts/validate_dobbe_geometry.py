"""Validate candidate RGB-only K against HoNY measured depth and labels.

Point correspondences are chosen from RGB by SIFT + fundamental-matrix RANSAC.
The same correspondences and measured depth samples are used for every K. Only
causal calibration frames are read. Record3D uses OpenGL camera axes. The old
Dobb-E exporter rotates portrait RGB clockwise before saving it; this image
rotation must be inverted when mapping an optical ray through historical P_old.
For landscape RGB the exporter instead uses P_new with no image rotation.
Both paths yield the same effective optical-to-label basis below. The only
pose/depth choices scored are c2w vs w2c and camera Z vs ray length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inspect_hony_scenes import ROOT, liblzfse


OUT = ROOT / "runs/dobbe_rgbd_study"
P_OLD = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64)
P_NEW = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
RGB_CW_TO_RAW_OPENGL = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]],
                                 dtype=np.float64)
CV_TO_OPENGL = np.diag([1.0, -1.0, -1.0])
assert np.array_equal(P_OLD @ RGB_CW_TO_RAW_OPENGL, P_NEW)
CAMERA_TO_LABEL = P_OLD @ RGB_CW_TO_RAW_OPENGL @ CV_TO_OPENGL
PAIRS = {
    "main": [(0, 20), (20, 40), (40, 60), (60, 80), (80, 95)],
    "second": [(0, 20), (20, 40), (40, 60), (60, 80),
               (80, 100), (100, 120)],
}


def load_scene(scene: str) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    folder = ROOT / "data/dobbe_oxe" / ("target_raw" if scene == "main" else "second_raw")
    indices = sorted(set(sum(([a, b] for a, b in PAIRS[scene]), [])))
    prefix_len = max(indices) + 1
    cap = cv2.VideoCapture(str(folder / "compressed_video_h264.mp4"))
    frames = {}
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode RGB frame {i}")
        frames[i] = frame
    cap.release()
    depth = np.frombuffer(liblzfse.decompress(
        (folder / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)[:prefix_len]
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    poses = np.repeat(np.eye(4)[None, ...], prefix_len, axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(prefix_len)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(prefix_len)]
    return frames, depth, poses


def depth_at(depth: np.ndarray, u: float, v: float) -> tuple[float, float]:
    x = int(round(u))
    y = int(round((v + .5) * .75 - .5))
    if x < 1 or x > 254 or y < 1 or y > 190:
        return float("nan"), float("inf")
    patch = depth[y-1:y+2, x-1:x+2]
    vals = patch[np.isfinite(patch) & (patch > .08) & (patch < 3.0)]
    if len(vals) < 7:
        return float("nan"), float("inf")
    return float(np.median(vals)), float(np.max(vals) - np.min(vals))


def choose_correspondences(scene: str, frames: dict[int, np.ndarray],
                           depth: np.ndarray) -> list[dict]:
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    by_frame = {}
    for i, frame in frames.items():
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        if scene == "main":
            mask[35:236, 36:222] = 255  # Cup, table, wall; remove moving rods at sides.
        else:
            mask[8:110, 10:246] = 255  # Static sofa, away from moving tape.
        by_frame[i] = sift.detectAndCompute(gray, mask)
    points = []
    for a, b in PAIRS[scene]:
        kp_a, desc_a = by_frame[a]
        kp_b, desc_b = by_frame[b]
        candidates = [m for m, n in matcher.knnMatch(desc_a, desc_b, k=2)
                      if m.distance < .78 * n.distance]
        if len(candidates) < 8:
            points.append({"pair": [a, b], "selected": [],
                           "reason": "fewer than 8 RGB matches"})
            continue
        xy_a = np.asarray([kp_a[m.queryIdx].pt for m in candidates])
        xy_b = np.asarray([kp_b[m.trainIdx].pt for m in candidates])
        _, inliers = cv2.findFundamentalMat(
            xy_a, xy_b, cv2.FM_RANSAC, 1.5, 0.99)
        if inliers is None:
            points.append({"pair": [a, b], "selected": [],
                           "reason": "RGB RANSAC failed"})
            continue
        accepted = []
        for idx in np.flatnonzero(inliers.ravel() > 0):
            m = candidates[int(idx)]
            uv_a, uv_b = xy_a[idx], xy_b[idx]
            z_a, spread_a = depth_at(depth[a], *uv_a)
            z_b, spread_b = depth_at(depth[b], *uv_b)
            if (not np.isfinite(z_a) or not np.isfinite(z_b)
                    or spread_a > .08 or spread_b > .08):
                continue
            accepted.append({"id": f"{a}-{b}-{len(accepted)}",
                             "rgb_a": uv_a.tolist(), "rgb_b": uv_b.tolist(),
                             "depth_a_m": z_a, "depth_b_m": z_b,
                             "depth_spread_a_m": spread_a,
                             "depth_spread_b_m": spread_b,
                             "descriptor_distance": float(m.distance)})
        accepted.sort(key=lambda p: p["descriptor_distance"])
        points.append({"pair": [a, b], "rgb_matches": len(candidates),
                       "rgb_ransac_inliers": int(inliers.sum()),
                       "selected": accepted[:35]})
    return points


def k_candidates(scene: str) -> dict[str, dict]:
    result = {"naive": {"params": [200, 200, 128, 128], "model": "PINHOLE"}}
    if scene == "main":
        result["previous_fit_diagnostic"] = {
            "params": [174, 199, 154, 155], "model": "PINHOLE"}
        names = [f"colmap_{m}_{subset}" for m in ("pinhole", "opencv")
                 for subset in ("all", "even", "odd")]
    else:
        names = ["colmap_pinhole_all", "colmap_pinhole_all_fixedpp_init62_107",
                 "colmap_pinhole_odd_fixedpp_init65_111",
                 "colmap_simple_pinhole_all_fixedpp_rectified",
                 "colmap_simple_pinhole_even_fixedpp_rectified",
                 "colmap_simple_pinhole_odd_fixedpp_rectified"]
    for name in names:
        path = OUT / scene / name / "summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        models = summary["reconstructions"]
        if not models or not models[0]["cameras"]:
            continue
        cam = models[0]["cameras"][0]
        params = (cam["published_256x256_K"]
                  if summary["model"] == "SIMPLE_PINHOLE" else cam["params"])
        result[name] = {"params": params,
                        "model": ("PINHOLE" if summary["model"] == "SIMPLE_PINHOLE"
                                  else summary["model"]),
                        "source_model": summary["model"],
                        "registered": models[0]["registered"],
                        "points3D": models[0]["points3D"]}
    return result


def get_intrinsics(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(candidate["params"], dtype=np.float64)
    k = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    dist = np.zeros(4) if len(p) == 4 else p[4:8]
    return k, dist


def unproject(uv: np.ndarray, z: np.ndarray, k: np.ndarray,
              dist: np.ndarray, depth_kind: str) -> np.ndarray:
    rays = cv2.undistortPoints(uv.reshape(-1, 1, 2), k, dist).reshape(-1, 2)
    xyz = np.column_stack((rays, np.ones(len(rays))))
    if depth_kind == "ray":
        xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz * z[:, None]


def evaluate(points: list[dict], poses: np.ndarray, candidate: dict,
             pose_kind: str, depth_kind: str) -> dict:
    k, dist = get_intrinsics(candidate)
    errors_3d, errors_px = [], []
    pair_results = []
    for item in points:
        a, b = item["pair"]
        selected = item["selected"]
        if not selected:
            continue
        uv_a = np.asarray([p["rgb_a"] for p in selected], dtype=np.float64)
        uv_b = np.asarray([p["rgb_b"] for p in selected], dtype=np.float64)
        z_a = np.asarray([p["depth_a_m"] for p in selected])
        z_b = np.asarray([p["depth_b_m"] for p in selected])
        optical_a = unproject(uv_a, z_a, k, dist, depth_kind)
        optical_b = unproject(uv_b, z_b, k, dist, depth_kind)
        label_a = optical_a @ CAMERA_TO_LABEL.T
        label_b = optical_b @ CAMERA_TO_LABEL.T
        t_a = poses[a] if pose_kind == "c2w" else np.linalg.inv(poses[a])
        t_b = poses[b] if pose_kind == "c2w" else np.linalg.inv(poses[b])
        common_a = label_a @ t_a[:3, :3].T + t_a[:3, 3]
        common_b = label_b @ t_b[:3, :3].T + t_b[:3, 3]
        metric = np.linalg.norm(common_a - common_b, axis=1)
        errors_3d.extend(metric.tolist())
        label_a_to_b = (common_a - t_b[:3, 3]) @ t_b[:3, :3]
        optical_a_to_b = label_a_to_b @ CAMERA_TO_LABEL
        projected, _ = cv2.projectPoints(optical_a_to_b,
                                         np.zeros(3), np.zeros(3), k, dist)
        projected = projected.reshape(-1, 2)
        pixel = np.linalg.norm(projected - uv_b, axis=1)
        positive = optical_a_to_b[:, 2] > .02
        errors_px.extend(pixel[positive].tolist())
        pair_results.append({"pair": [a, b], "n": len(selected),
                             "median_3d_m": float(np.median(metric)),
                             "median_px": (float(np.median(pixel[positive]))
                                           if np.any(positive) else None),
                             "positive_z": int(positive.sum())})
    return {"n_3d": len(errors_3d), "n_px": len(errors_px),
            "median_3d_m": float(np.median(errors_3d)) if errors_3d else None,
            "p90_3d_m": float(np.percentile(errors_3d, 90)) if errors_3d else None,
            "median_reprojection_px": (float(np.median(errors_px))
                                       if errors_px else None),
            "p90_reprojection_px": (float(np.percentile(errors_px, 90))
                                    if errors_px else None),
            "pairs": pair_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=PAIRS, required=True)
    args = parser.parse_args()
    frames, depth, poses = load_scene(args.scene)
    points = choose_correspondences(args.scene, frames, depth)
    (OUT / f"{args.scene}_static_correspondences.json").write_text(
        json.dumps(points, indent=2) + "\n", encoding="utf-8")
    candidates = k_candidates(args.scene)
    result = {"scene": args.scene, "causal_pairs": PAIRS[args.scene],
              "P_old": P_OLD.tolist(),
              "P_new": P_NEW.tolist(),
              "portrait_rgb_cw_to_raw_opengl": RGB_CW_TO_RAW_OPENGL.tolist(),
              "opencv_to_opengl": CV_TO_OPENGL.tolist(),
              "optical_to_label": CAMERA_TO_LABEL.tolist(),
              "rgb_depth_mapping": "u_d=u_r; v_d=(v_r+0.5)*192/256-0.5",
              "selection": "SIFT ratio + RGB-only fundamental RANSAC; scene ROI; 3x3 depth median; depth spread <=0.08 m",
              "point_counts": {f"{p['pair'][0]}-{p['pair'][1]}": len(p["selected"])
                               for p in points}, "candidates": {}}
    for name, candidate in candidates.items():
        fx, fy, cx, cy = candidate["params"][:4]
        candidate["K_depth"] = [fx, fy * 192 / 256,
                                cx, (cy + .5) * 192 / 256 - .5]
        result["candidates"][name] = {"camera": candidate, "variants": {}}
        for pose_kind in ("c2w", "w2c"):
            for depth_kind in ("z", "ray"):
                variant = f"{pose_kind}_{depth_kind}"
                result["candidates"][name]["variants"][variant] = evaluate(
                    points, poses, candidate, pose_kind, depth_kind)
    path = OUT / f"{args.scene}_static_geometry.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene": args.scene, "point_counts": result["point_counts"],
                      "naive": result["candidates"]["naive"]["variants"]}, indent=2))


if __name__ == "__main__":
    main()
