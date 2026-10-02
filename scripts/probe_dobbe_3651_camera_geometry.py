"""Probe RGB/depth/pose consistency for the matched HoNY red-cup capture.

This is a diagnostic. It does not silently accept a guessed camera matrix.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dobbe_oxe" / "target_raw"
OUT = ROOT / "runs" / "plex_dobbe_preflight"
PAIRS = [(0, 20), (20, 40), (40, 60), (60, 80),
         (180, 200), (200, 220)]


def load_frames(indices: set[int]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(DATA / "compressed_video_h264.mp4"))
    result = {}
    for index in sorted(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {index}")
        result[index] = frame
    capture.release()
    return result


def make_matches(frames: dict[int, np.ndarray]) -> list[dict]:
    sift = cv2.SIFT_create(nfeatures=1800)
    features = {}
    for index, image in frames.items():
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, descriptors = sift.detectAndCompute(gray, None)
        features[index] = (kps, descriptors)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    output = []
    for i, j in PAIRS:
        ki, di = features[i]
        kj, dj = features[j]
        knn = matcher.knnMatch(di, dj, k=2)
        good = [m for m, n in knn if m.distance < .72 * n.distance]
        uv_i = np.asarray([ki[m.queryIdx].pt for m in good], dtype=np.float64)
        uv_j = np.asarray([kj[m.trainIdx].pt for m in good], dtype=np.float64)
        # Exclude red object/cup; their motion may differ from camera motion.
        hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        red = ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170)) & (
            hsv[:, :, 1] > 75) & (hsv[:, :, 2] > 45)
        ix = np.rint(uv_i[:, 0]).astype(int).clip(0, 255)
        iy = np.rint(uv_i[:, 1]).astype(int).clip(0, 255)
        keep = ~red[iy, ix]
        uv_i, uv_j = uv_i[keep], uv_j[keep]
        output.append({"pair": [i, j], "uv_i": uv_i, "uv_j": uv_j,
                       "sift_matches": len(uv_i)})
    return output


def rotations() -> list[np.ndarray]:
    result = []
    for perm in itertools.permutations(range(3)):
        for sign in itertools.product((-1, 1), repeat=3):
            c = np.zeros((3, 3))
            for row, (axis, direction) in enumerate(zip(perm, sign)):
                c[row, axis] = direction
            if round(np.linalg.det(c)) == 1:
                result.append(c)
    return result


def reprojection(pair: dict, depth: np.ndarray, pose_r: np.ndarray,
                 pose_t: np.ndarray, c: np.ndarray,
                 fx: float = 200, fy: float = 200,
                 cx: float = 128, cy: float = 128) -> np.ndarray:
    i, j = pair["pair"]
    uv_i, uv_j = pair["uv_i"], pair["uv_j"]
    px = np.rint(uv_i[:, 0]).astype(int).clip(0, 255)
    py = np.rint((uv_i[:, 1] + .5) * .75 - .5).astype(int).clip(0, 191)
    z = depth[i, py, px]
    valid = np.isfinite(z) & (z > .08) & (z < 2)
    u, v, z, target = uv_i[valid, 0], uv_i[valid, 1], z[valid], uv_j[valid]
    p_i_opt = np.column_stack(((u - cx) / fx * z, (v - cy) / fy * z, z))
    p_i_label = p_i_opt @ c
    p_world = p_i_label @ pose_r[i].T + pose_t[i]
    p_j_label = (p_world - pose_t[j]) @ pose_r[j]
    p_j_opt = p_j_label @ c.T
    keep = p_j_opt[:, 2] > .02
    projected = np.column_stack((fx * p_j_opt[keep, 0] / p_j_opt[keep, 2] + cx,
                                 fy * p_j_opt[keep, 1] / p_j_opt[keep, 2] + cy))
    return np.linalg.norm(projected - target[keep], axis=1)


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    depth = np.frombuffer(liblzfse.decompress(
        (DATA / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    labels = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    pose_t = np.asarray([labels[str(i)]["xyz"] for i in range(len(labels))])
    pose_q = np.asarray([labels[str(i)]["quats"] for i in range(len(labels))])
    pose_r = Rotation.from_quat(pose_q).as_matrix()
    frames = load_frames(set(itertools.chain.from_iterable(PAIRS)))
    matches = make_matches(frames)
    candidates = []
    for c in rotations():
        errors = [reprojection(pair, depth, pose_r, pose_t, c)
                  for pair in matches]
        all_err = np.concatenate(errors)
        candidates.append({"C_label_to_optical": c.astype(int).tolist(),
                           "median_px": float(np.median(all_err)),
                           "p75_px": float(np.percentile(all_err, 75)),
                           "inliers_5px_fraction": float(np.mean(all_err < 5)),
                           "per_pair_median_px": [float(np.median(e)) if len(e) else None
                                                  for e in errors]})
    candidates.sort(key=lambda x: x["median_px"])
    result = {"pair_matches": [{"pair": x["pair"], "count": x["sift_matches"]}
                               for x in matches],
              "probe_K_rgb": [200, 200, 128, 128],
              "probe_rgb_to_depth": "u_depth=u_rgb; v_depth=(v_rgb+.5)*.75-.5",
              "pose_convention": "stored relative transforms, matrix from xyzw quaternion",
              "axis_candidates": candidates}
    (OUT / "dobbe_3651_camera_pose_probe.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_matches": result["pair_matches"],
                      "top_axes": candidates[:4]}, indent=2))


if __name__ == "__main__":
    main()
