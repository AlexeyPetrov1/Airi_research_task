"""Inspect candidate 2 s windows in a second raw FMB episode before inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from inspect_fmb_2d_window import red_component


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy"
OUT = ROOT / "runs/fmb_second_example_1_M_L_3_vertical_n_3"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True).item()
    frames = data["obs/side_1"]
    depth = data["obs/side_1_depth"]
    tcp = data["obs/tcp_pose"]
    steps = (100, 110, 118, 120, 122, 124, 125, 126, 127, 132, 137, 142, 146)
    rows = []
    canvas = np.zeros((4*256, 4*256, 3), dtype="uint8")
    for i, step in enumerate(steps):
        if step >= len(frames):
            continue
        red, box = red_component(frames[step])
        values = depth[step][red]
        positive = values[values > 0]
        future = min(step+20, len(frames)-1)
        rows.append({"step": step, "primitive": str(data["primitive"][step]),
                     "red_xywh_area": box, "positive_depth_fraction_in_red": float(len(positive)/len(values)),
                     "positive_depth_median_raw": float(np.median(positive)) if len(positive) else None,
                     "future_20_available": step+20 < len(frames),
                     "tcp_motion_20_steps_mm": float(np.linalg.norm(tcp[future, :3]-tcp[step, :3])*1000)})
        image = frames[step].copy()
        cv2.putText(image, f"step {step}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    .6, (255, 255, 255), 2)
        canvas[i//4*256:(i//4+1)*256, i%4*256:(i%4+1)*256] = image
    cv2.imwrite(str(OUT / "window_candidates.png"), canvas)
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result = {"source": str(SOURCE), "source_sha256": digest,
              "RGB_shape": list(frames.shape), "depth_shape": list(depth.shape),
              "object_info": data["object_info"], "candidate_rows": rows,
              "huggingface_dataset_revision": "f99fd55c072eea5573523c96aa527aed3c665690"}
    (OUT / "window_probe.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
