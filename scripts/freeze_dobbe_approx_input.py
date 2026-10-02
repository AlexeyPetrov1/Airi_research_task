"""Seal the exploratory MolmoMotion input before any future RGB-D is used."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dobbe_rgbd_study/approx_history"
STUDY = ROOT / "runs/dobbe_rgbd_study"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
MODEL_PT_SHA256_PREVIOUSLY_VERIFIED = (
    "506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    target = RUN / "input_freeze.json"
    if target.exists():
        raise FileExistsError(f"Already frozen: {target}")
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["history_frames"] == [96, 97, 98]
    assert manifest["future_used"] is False
    xyz = np.load(RUN / "points_3d_history_candidate.npy")
    uv = np.load(RUN / "points_2d_at_t0_candidate.npy")
    assert xyz.shape == (3, 8, 3) and uv.shape == (8, 2)
    assert np.isfinite(xyz).all() and np.isfinite(uv).all()
    assert (xyz[..., 2] > 0).all()
    files = ["rgb_0096.png", "rgb_0097.png", "rgb_0098.png",
             "points_3d_history_candidate.npy",
             "points_3d_history_world_candidate.npy",
             "points_2d_at_t0_candidate.npy", "manifest.json"]
    checkpoint_hash = sha256(CHECKPOINT / "model.pt")
    if checkpoint_hash != MODEL_PT_SHA256_PREVIOUSLY_VERIFIED:
        raise RuntimeError("Model checkpoint changed from verified copy")
    receipt = STUDY / "f2nerf_second/receipt.json"
    freeze = {
        "status": "FROZEN_EXPLORATORY_APPROXIMATE",
        "task": "Pick_and_Place",
        "history_frames": [96, 97, 98],
        "future_evaluation_only": [99, 128],
        "action_text_exact_passed_to_model": "Pick up the red cup.",
        "future_horizon": 30,
        "seed": 0,
        "dtype": "bfloat16",
        "model_id": "allenai/MolmoMotion-4B-H3-F30",
        "model_revision": "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629",
        "checkpoint_model_pt_sha256": checkpoint_hash,
        "checkpoint_config_sha256": sha256(CHECKPOINT / "config.yaml"),
        "f2nerf_receipt_sha256": sha256(receipt),
        "history_source_nominal_hz": 30,
        "model_prediction_assumed_hz": 15,
        "timing_warning": "Processor has no real timestamp input; history sampling and model training cadence differ.",
        "sha256": {name: sha256(RUN / name) for name in files},
    }
    target.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"freeze": str(target), "files": len(files),
                      "checkpoint_sha256": checkpoint_hash}, indent=2))


if __name__ == "__main__":
    main()
