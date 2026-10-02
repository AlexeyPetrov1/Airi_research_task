"""Save only the frozen episode_3651 history RGB frames for point inspection."""

from __future__ import annotations

import json

import cv2
import numpy as np

from inspect_hony_scenes import ROOT, liblzfse


def main() -> None:
    protocol = json.loads((ROOT / "runs/dobbe_rgbd_study/protocol.json")
                          .read_text(encoding="utf-8"))
    history = protocol["main"]["history"]
    assert history == [96, 97, 98]
    video = ROOT / "data/dobbe_oxe/target_raw/compressed_video_h264.mp4"
    depth_file = ROOT / "data/dobbe_oxe/target_raw/compressed_np_depth_float32.bin"
    output = ROOT / "runs/dobbe_rgbd_study/approx_history"
    output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    depth = np.frombuffer(liblzfse.decompress(depth_file.read_bytes()),
                          dtype=np.float32).reshape(-1, 192, 256)[:max(history)+1]
    for index in history:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok or frame.shape[:2] != (256, 256):
            raise RuntimeError(f"Missing history RGB {index}")
        cv2.imwrite(str(output / f"rgb_{index:04d}.png"), frame)
        scaled = np.clip((depth[index] - .15) / 1.35, 0, 1)
        colored = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
        colored = cv2.resize(colored, (256, 256), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(output / f"depth_{index:04d}.png"), colored)
        cv2.imwrite(str(output / f"rgb_depth_{index:04d}.png"),
                    np.hstack((frame, colored)))
    capture.release()
    print("Saved", history, "to", output)


if __name__ == "__main__":
    main()
