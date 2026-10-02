"""Make compact, unmodified-source previews for second-scene selection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data/fmb/single_object_manipulation_dataset"
OUT = ROOT / "runs/fmb_second_scene"
FIELDS = ["shape", "size", "length", "color", "angle", "distractor"]


def sheet(data: dict, name: str, camera: str) -> None:
    images = data[f"obs/{camera}"]
    n = len(images)
    indices = np.unique(np.linspace(0, n - 1, 16, dtype=int))
    width, height = 256, 285
    canvas = Image.new("RGB", (width * 4, height * 4), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    for slot, index in enumerate(indices):
        rgb = images[index][..., ::-1]
        x, y = slot % 4 * width, slot // 4 * height
        canvas.paste(Image.fromarray(rgb), (x, y))
        draw.text((x + 4, y + 258), f"{index}: {data['primitive'][index]}", fill="#111111")
    path = OUT / "candidate_previews" / f"{name}_{camera}.png"
    canvas.save(path)


def main() -> None:
    (OUT / "candidate_previews").mkdir(parents=True, exist_ok=True)
    rows = []
    for file in sorted(DATA.glob("*.npy")):
        data = np.load(file, allow_pickle=True).item()
        if not all(f"obs/{camera}" in data and f"obs/{camera}_depth" in data
                   for camera in ("side_1", "side_2", "wrist_1", "wrist_2")):
            continue
        name = file.stem
        n = len(data["primitive"])
        ref = {"shape": 1, "size": "M", "length": "L", "color": 3,
               "angle": "vertical", "distractor": "n"}
        info = data["object_info"]
        ranges = []
        start = 0
        for i in range(1, n + 1):
            if i == n or data["primitive"][i] != data["primitive"][start]:
                ranges.append({"primitive": str(data["primitive"][start]),
                               "start": start, "end": i - 1})
                start = i
        for camera in ("side_1", "side_2"):
            sheet(data, name, camera)
        rows.append({
            "source_file": file.name,
            "trajectory_id": name.rsplit("_", 1)[-1],
            "N": n,
            **info,
            "primitive_sequence": ">".join(dict.fromkeys(map(str, data["primitive"]))),
            "primitive_ranges": ranges,
            "side_1_quality": "PENDING_VISUAL_REVIEW",
            "side_2_quality": "PENDING_VISUAL_REVIEW",
            "depth_quality": "PENDING_OBJECT_ROI_REVIEW",
            "config_difference_count": sum(info.get(k) != ref[k] for k in FIELDS),
            "tcp_path_m": float(np.linalg.norm(np.diff(data["obs/tcp_pose"][:, :3], axis=0), axis=1).sum()),
            "accepted": False,
            "rejection_reason": "PENDING_SELECTION",
        })
        print(file.name, n, info, flush=True)
    with (OUT / "candidate_survey.json").open("w", encoding="utf-8") as output:
        json.dump(rows, output, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
