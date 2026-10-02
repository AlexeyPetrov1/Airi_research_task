"""Inspect both selectively extracted HoNY captures before fixing a protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / ".tools/lzfse"))
import liblzfse  # noqa: E402


def inspect(name: str, folder: Path, output: Path) -> dict:
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    depth = np.frombuffer(liblzfse.decompress(
        (folder / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32).reshape(-1, 192, 256)
    capture = cv2.VideoCapture(str(folder / "compressed_video_h264.mp4"))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = list(range(0, total, max(1, total // 15)))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    tile_rows = []
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot read {name} frame {idx}")
        small = cv2.resize(img, (192, 192))
        cv2.putText(small, str(idx), (7, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(small, str(idx), (7, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (255, 255, 255), 2, cv2.LINE_AA)
        tile_rows.append(small)
    capture.release()
    while len(tile_rows) % 4:
        tile_rows.append(np.zeros_like(tile_rows[0]))
    sheet = np.vstack([np.hstack(tile_rows[i:i+4])
                       for i in range(0, len(tile_rows), 4)])
    cv2.imwrite(str(output / f"{name}_contact.jpg"), sheet)
    xyz = np.array([labels[str(i)]["xyz"] for i in range(len(labels))])
    quats = np.array([labels[str(i)]["quats"] for i in range(len(labels))])
    r = Rotation.from_quat(quats)
    start_distance = np.linalg.norm(xyz - xyz[0], axis=1)
    angles = (r[0].inv() * r).magnitude()
    summary = {
        "frames_rgb": total, "frames_depth": len(depth),
        "frames_labels": len(labels),
        "depth_shape": list(depth.shape),
        "depth_range_m": [float(np.min(depth)), float(np.median(depth)),
                          float(np.max(depth))],
        "translation_from_start_m": {str(i): float(start_distance[i])
                                     for i in indices},
        "rotation_from_start_deg": {str(i): float(np.rad2deg(angles[i]))
                                    for i in indices},
        "contact": f"{name}_contact.jpg",
    }
    return summary


def main() -> None:
    output = ROOT / "runs/dobbe_rgbd_study"
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "data/dobbe_oxe"
    result = {name: inspect(name, base / folder, output)
              for name, folder in [("main", "target_raw"),
                                   ("second", "second_raw")]}
    (output / "scene_inspection.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
