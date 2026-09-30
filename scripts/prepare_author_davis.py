"""Prepare the exact bmx-trees GT and original frames for the bundled example.

The two PointMotionBench track files are downloaded at the pinned revision in
the report. The DAVIS 2017 zip is only needed for original RGB future frames.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/data/davis_bmx_trees"
SOURCE = ROOT / "data/pointmotionbench/davis"
RUN = ROOT / "runs/author_davis_bmx_trees_f30"


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    meta = json.loads((SAMPLE / "meta.json").read_text())
    assert meta["video"] == "bmx-trees" and meta["obj"] == "bike_rider"
    assert meta["t0_absolute"] == 2 and meta["history_frame_indices"] == [0, 1, 2]
    indices = np.asarray(meta["point_indices"], dtype=np.int64)
    assert indices.shape == (8,) and np.unique(indices).size == 8
    with np.load(SOURCE / "bmx-trees_2d.npz", allow_pickle=True) as file:
        xy = file["tracks"].item()["bike_rider"]
        visible = file["visibility"].item()["bike_rider"]
        assert file["dim"].tolist() == [480, 854]
    with np.load(SOURCE / "bmx-trees_3d.npz", allow_pickle=True) as file:
        xyz = file["points_3d"].item()["bike_rider"]
    assert xy.shape[0] == xyz.shape[1] == visible.shape[0] == 80
    assert np.array_equal(xy[2, indices], torch.load(SAMPLE / "points_2d_at_t0.pt", weights_only=True).numpy())
    assert np.array_equal(xyz[indices, :3].transpose(1, 0, 2),
                          torch.load(SAMPLE / "points_3d_history.pt", weights_only=True).numpy())
    assert visible[:3, indices].all()

    future_indices = np.arange(3, 33, dtype=np.int64)
    gt_3d = xyz[indices][:, future_indices, :]
    gt_2d = xy[future_indices][:, indices, :].transpose(1, 0, 2)
    mask = visible[future_indices][:, indices].T & np.isfinite(gt_3d).all(axis=-1)
    assert gt_3d.shape == (8, 30, 3) and gt_2d.shape == (8, 30, 2)
    assert np.isfinite(gt_2d).all()
    np.savez_compressed(RUN / "ground_truth.npz", future_3d=gt_3d,
                        future_2d=gt_2d, valid=mask, point_indices=indices,
                        future_frame_indices=future_indices,
                        future_seconds_from_t0=np.arange(1, 31) / 24,
                        history_3d=xyz[indices, :3].transpose(1, 0, 2),
                        history_2d=xy[:3, indices],
                        intrinsics_K=torch.load(SAMPLE / "intrinsics_K.pt", weights_only=True).numpy())

    frames_dir = RUN / "frames"
    frames_dir.mkdir(exist_ok=True)
    archive = SOURCE / "DAVIS-2017-trainval-480p.zip"
    with zipfile.ZipFile(archive) as file:
        names = {Path(name).name: name for name in file.namelist()
                 if "/JPEGImages/480p/bmx-trees/" in name and name.lower().endswith(".jpg")}
        assert len(names) >= 33, len(names)
        for t in range(33):
            name = f"{t:05d}.jpg"
            target = frames_dir / name
            target.write_bytes(file.read(names[name]))
            with Image.open(target) as image:
                assert image.size == (854, 480)

    # The official archive must contain the same opening images as the sample.
    # Authors may re-encode JPEGs, so record rather than require byte identity.
    image_diffs = []
    for t in range(3):
        with Image.open(frames_dir / f"{t:05d}.jpg") as source, Image.open(SAMPLE / f"frame_t{t-2:+d}.jpg") as bundled:
            image_diffs.append(float(np.abs(np.asarray(source.convert("RGB"), dtype=np.int16)
                                            - np.asarray(bundled.convert("RGB"), dtype=np.int16)).mean()))
    assert max(image_diffs) < 5, f"Opening frames do not match bundled example: {image_diffs}"
    info = {
        "dataset": "allenai/PointMotionBench",
        "dataset_revision": "564ffa2e3cdb0ba443db8590b40e9691f487436c",
        "source_video": "DAVIS-2017-trainval-480p.zip/JPEGImages/480p/bmx-trees",
        "source_fps": 24,
        "history_frame_indices": [0, 1, 2],
        "t0_frame_index": 2,
        "future_frame_indices": future_indices.tolist(),
        "point_indices": indices.tolist(),
        "valid_gt_count": int(mask.sum()),
        "gt_count_at_fde_frame": int(mask[:, -1].sum()),
        "opening_frame_mean_absolute_pixel_differences": image_diffs,
    }
    (RUN / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
