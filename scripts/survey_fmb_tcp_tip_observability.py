"""Survey usable RGB peg-end observations during FMB horizontal episodes.

This does not assume that a mask extremum is a persistent CAD point. It records
where a prospective TCP-linked point is visible so a calibration interval can
be selected before fitting K.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from probe_fmb_tcp_tip_k import red_end


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/fmb/single_object_manipulation_dataset"
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
EPISODES = (0, 4, 8)


def ranges(labels: np.ndarray) -> list[dict]:
    result = []
    start = 0
    for i in range(1, len(labels)+1):
        if i == len(labels) or labels[i] != labels[start]:
            result.append({"primitive": str(labels[start]),
                           "first": start, "last": i-1})
            start = i
    return result


def main() -> None:
    episodes = []
    canvas = np.full((len(EPISODES)*192, 10*192, 3), 255, np.uint8)
    for n in EPISODES:
        path = DATA / f"1_M_L_3_horizontal_n_{n}.npy"
        source = np.load(path, allow_pickle=True).item()
        frames, labels = source["obs/side_1"], source["primitive"]
        tcp = source["obs/tcp_pose"]
        observations = []
        for step, frame in enumerate(frames):
            try:
                uv, quality = red_end(frame)
                x, y, w, h = quality["bounding_box_xywh"]
                vertical_shape = h >= 1.6*w
                rows = {"step": step, "primitive": str(labels[step]),
                        "uv": uv.tolist(), "vertical_shape": bool(vertical_shape),
                        **quality}
                observations.append(rows)
            except ValueError:
                pass
        interval_summary = []
        for item in ranges(labels):
            ids = np.arange(item["first"], item["last"]+1)
            chosen = [row for row in observations
                      if item["first"] <= row["step"] <= item["last"]]
            shape = [row for row in chosen if row["vertical_shape"]]
            positions = tcp[ids, :3]
            orient = Rotation.from_quat(tcp[ids, 3:])
            angle_from_first = (orient[0].inv()*orient).magnitude()*180/np.pi
            interval_summary.append({**item, "frames": len(ids),
                                     "red_end_count": len(chosen),
                                     "vertical_shape_count": len(shape),
                                     "vertical_shape_first_last": (
                                         [shape[0]["step"], shape[-1]["step"]]
                                         if shape else None),
                                     "tcp_position_axis_range_m":
                                         np.ptp(positions, axis=0).tolist(),
                                     "tcp_max_rotation_from_start_deg":
                                         float(np.max(angle_from_first))})
        insert = next(row for row in interval_summary
                      if row["primitive"] == "insert")
        for col, step in enumerate(np.linspace(insert["first"], insert["last"],
                                               10).round().astype(int)):
            tile = frames[step].copy()
            matching = [row for row in observations if row["step"] == step]
            if matching:
                u, v = matching[0]["uv"]
                cv2.circle(tile, (round(u), round(v)), 4, (0, 255, 0), -1)
            cv2.putText(tile, f"h{n} t{step}", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            row_index = EPISODES.index(n)
            canvas[row_index*192:(row_index+1)*192,
                   col*192:(col+1)*192] = cv2.resize(
                       tile, (192, 192), interpolation=cv2.INTER_AREA)
        episodes.append({"episode": n, "source": str(path),
                         "object_info": source["object_info"],
                         "intervals": interval_summary,
                         "observations": observations})
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "tcp_tip_insert_contact.png"), canvas)
    (OUT / "tcp_tip_observability_horizontal.json").write_text(
        json.dumps({"episodes": episodes}, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"intervals": {str(row["episode"]): row["intervals"]
                                    for row in episodes}}, indent=2))


if __name__ == "__main__":
    main()
