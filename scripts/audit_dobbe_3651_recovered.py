"""Audit the recovered HoNY source record against LeRobot episode 3651.

Requires the three target_raw members extracted by extract_dobbe_3651_rgbd.py,
the two LeRobot episode files, and the isolated pyliblzfse installation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "dobbe_oxe"
RAW = DATA / "target_raw"
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "dobbe_3651_recovered_audit.json"
FRAMES = (0, 40, 100, 200, 241)


def video_info(path: Path) -> tuple[dict, list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    info = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps_container": float(capture.get(cv2.CAP_PROP_FPS)),
        "hw": [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))],
    }
    frames = []
    for index in FRAMES:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode {path} frame {index}")
        frames.append(frame)
    capture.release()
    return info, frames


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    paths = {
        "raw_rgb": RAW / "compressed_video_h264.mp4",
        "raw_depth": RAW / "compressed_np_depth_float32.bin",
        "raw_labels": RAW / "labels.json",
        "lerobot_rgb": DATA / "lerobot_episode_003651.mp4",
        "lerobot_states": DATA / "lerobot_episode_003651.parquet",
    }
    files = {key: {"path": str(path.relative_to(ROOT)),
                   "bytes": path.stat().st_size,
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for key, path in paths.items()}
    raw_info, raw_frames = video_info(paths["raw_rgb"])
    lerobot_info, lerobot_frames = video_info(paths["lerobot_rgb"])
    frame_diffs = []
    for index, raw, converted in zip(FRAMES, raw_frames, lerobot_frames):
        if raw.shape != converted.shape:
            raise RuntimeError(f"RGB shape differs at frame {index}")
        delta = raw.astype(np.float32) - converted.astype(np.float32)
        frame_diffs.append({"frame": index,
                            "mean_abs_8bit": float(np.mean(np.abs(delta))),
                            "rmse_8bit": float(np.sqrt(np.mean(delta**2)))})

    labels = json.loads(paths["raw_labels"].read_text(encoding="utf-8"))
    states = pq.read_table(paths["lerobot_states"],
                           columns=["observation.state"])
    gripper = np.asarray([state[-1] for state in states.column(0).to_pylist()],
                         dtype=np.float32)
    label_gripper = np.asarray([labels[str(i)]["gripper"]
                                for i in range(len(gripper))], dtype=np.float32)
    depth_values = np.frombuffer(liblzfse.decompress(
        paths["raw_depth"].read_bytes()), dtype=np.float32)
    if depth_values.size % (192 * 256):
        raise RuntimeError("Unexpected depth payload size")
    depth = depth_values.reshape(-1, 192, 256)
    positive = depth_values[np.isfinite(depth_values) & (depth_values > 0)]
    audit = {
        "source_record": "Pick_and_Place/Home15/Env1/2023-04-25--02-05-30",
        "files": files,
        "raw_rgb": raw_info,
        "lerobot_rgb": lerobot_info,
        "labels_frames": len(labels),
        "labels_frame0_keys": sorted(labels["0"].keys()),
        "lerobot_state_frames": len(gripper),
        "gripper_equal_float32_first_242": bool(np.array_equal(gripper, label_gripper)),
        "raw_vs_lerobot_rgb": frame_diffs,
        "depth": {
            "shape": list(depth.shape), "dtype": str(depth.dtype),
            "positive_min_native_units": float(positive.min()),
            "positive_median_native_units": float(np.median(positive)),
            "positive_max_native_units": float(positive.max()),
            "zero_count": int(np.count_nonzero(depth == 0)),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(depth))),
        },
        "camera_K_in_target_labels": any(
            key.lower() in ("k", "intrinsics", "camera_intrinsics")
            for frame in labels.values() for key in frame),
        "frame_243_note": "Raw has 243 frames; LeRobot conversion has first 242.",
        "share_robot_pixel_check": "NOT_DONE: target ShareRobot PNG is packed in large planning tar.gz",
    }
    if not audit["gripper_equal_float32_first_242"]:
        raise RuntimeError("LeRobot gripper sequence differs from source")
    if not (raw_info["frames"] == len(labels) == depth.shape[0] == 243):
        raise RuntimeError("Raw RGB/depth/label frame counts differ")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
