"""Test the published 256x256 RGB ↔ 256x192 depth mapping on past frames."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_rgbd_study"
sys.path.insert(0, str(ROOT.parent / ".tools/lzfse"))
import liblzfse  # noqa: E402

SCENES = {"main": ("target_raw", [0, 20, 40, 60, 80]),
          "second": ("second_raw", [0, 30, 60, 90, 120])}


def read_frames(video: Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    result = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not decode frame {i}")
        result.append(frame)
    cap.release()
    return result


def score(rgb_edge: np.ndarray, depth_edge: np.ndarray, shift: tuple[int, int]) -> dict:
    dx, dy = shift
    aligned = np.zeros_like(depth_edge)
    x0, x1 = max(0, dx), min(256, 256 + dx)
    y0, y1 = max(0, dy), min(256, 256 + dy)
    aligned[y0:y1, x0:x1] = depth_edge[y0-dy:y1-dy, x0-dx:x1-dx]
    dist = cv2.distanceTransform(255 - rgb_edge, cv2.DIST_L2, 3)
    distances = dist[aligned > 0]
    if not len(distances):
        return {"n_depth_edge": 0, "median_distance_px": None,
                "p90_distance_px": None, "fraction_within_3px": None}
    return {"n_depth_edge": len(distances),
            "median_distance_px": float(np.median(distances)),
            "p90_distance_px": float(np.percentile(distances, 90)),
            "fraction_within_3px": float(np.mean(distances <= 3))}


def inspect_scene(name: str, source: Path, indices: list[int]) -> list[dict]:
    depth = np.frombuffer(liblzfse.decompress(
        (source / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    rgb = read_frames(source / "compressed_video_h264.mp4", indices)
    output = []
    tiles = []
    for i, frame in zip(indices, rgb):
        # OpenCV's half-pixel resize implements the stated v coordinate map.
        full_depth = cv2.resize(depth[i], (256, 256), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_edge = cv2.Canny(gray, 55, 130)
        # Depth discontinuities should align with visible silhouettes; ignore
        # invalid/very distant depth and mild surface slope.
        clean = cv2.medianBlur(full_depth, 5)
        gx = cv2.Sobel(clean, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(clean, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        valid = (clean > 0.08) & (clean < 3.0)
        edge = ((magnitude > 0.12) & valid).astype(np.uint8) * 255
        scores = {f"{dx:+d},{dy:+d}": score(rgb_edge, edge, (dx, dy))
                  for dx, dy in [(0, 0), (-8, 0), (8, 0), (0, -8), (0, 8)]}
        overlay = frame.copy()
        overlay[edge > 0] = (0.35 * overlay[edge > 0] +
                             0.65 * np.array([0, 255, 255])).astype(np.uint8)
        cv2.putText(overlay, f"{name} {i}", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4)
        cv2.putText(overlay, f"{name} {i}", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
        tiles.append(overlay)
        output.append({"frame": i, "scores": scores})
    cv2.imwrite(str(OUT / f"{name}_depth_edge_overlay.jpg"),
                np.hstack(tiles))
    return output


def main() -> None:
    data = ROOT / "data/dobbe_oxe"
    result = {name: inspect_scene(name, data / folder, indices)
              for name, (folder, indices) in SCENES.items()}
    (OUT / "rgb_depth_alignment.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
