"""Audit one HoNY RGB-D delivery record; do not infer missing camera calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "plex_dobbe_preflight"
SAMPLE = RUN / "dobbe_rgbd_sample"


def main() -> None:
    sys.path.insert(0, str(ROOT.parent / ".tools" / "lzfse"))
    import liblzfse

    labels = json.loads((SAMPLE / "labels.json").read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(SAMPLE / "compressed_video_h264.mp4"))
    rgb_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    rgb_fps = float(capture.get(cv2.CAP_PROP_FPS))
    rgb_shape = [int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                 int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))]
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError("Could not decode first video frame")
    floats = np.frombuffer(liblzfse.decompress(
        (SAMPLE / "compressed_np_depth_float32.bin").read_bytes()),
        dtype=np.float32)
    if floats.size % (192 * 256):
        raise RuntimeError("Unexpected depth payload shape")
    depth = floats.reshape(-1, 192, 256)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if not len(valid):
        raise RuntimeError("No positive finite depth samples")
    z0 = depth[0]
    z_norm = np.clip((z0 - .1) / 1.4, 0, 1)
    z_color = cv2.applyColorMap((255 * z_norm).astype(np.uint8), cv2.COLORMAP_TURBO)
    z_color = cv2.resize(z_color, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(RUN / "dobbe_rgbd_sample_frame0.png"),
                np.concatenate([frame, z_color], axis=1))
    result = {
        "source_record": "Drawer_Closing/Home10/Env2/2022-12-22--00-09-08",
        "scope": "dataset-level example, not ShareRobot dobbe#episode_3651",
        "rgb_frames": rgb_count, "rgb_fps_container": rgb_fps,
        "rgb_hw": rgb_shape, "depth_frames": int(depth.shape[0]),
        "depth_hw": list(depth.shape[1:]), "depth_dtype": str(depth.dtype),
        "labels_frames": len(labels), "label_keys_frame0": sorted(labels["0"]),
        "depth_positive_min_native_units": float(valid.min()),
        "depth_positive_median_native_units": float(np.median(valid)),
        "depth_positive_p99_native_units": float(np.percentile(valid, 99)),
        "depth_zero_count": int(np.sum(depth == 0)),
        "depth_nonfinite_count": int(np.sum(~np.isfinite(depth))),
        "camera_intrinsics_in_labels": any("K" in value or "intrinsics" in value
                                           for value in labels.values()),
        "intrinsic_status": "ABSENT_FROM_PUBLISHED_RECORD",
    }
    (RUN / "dobbe_rgbd_sample_audit.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
