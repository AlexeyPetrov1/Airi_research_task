"""Run the supplied F2-NeRF COLMAP converter on the RGB-only control scene.

This exports camera metadata for the second HoNY scene. Its COLMAP world poses
are scene-specific and must never be applied to episode_3651; only the RGB-only
intrinsics are a possible transfer hypothesis, subject to independent checks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
STUDY = ROOT / "runs/dobbe_rgbd_study"
COLMAP = STUDY / "second/colmap_pinhole_all_fixedpp_init62_107"
STAGE = STUDY / "f2nerf_second"
CONVERTER = WORKSPACE / "third_party/f2-nerf/scripts/colmap2poses.py"


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    for name, source in (
        ("sparse", COLMAP / "sparse"),
        ("images", STUDY / "second/rgb_prefix"),
    ):
        link = STAGE / name
        if not source.is_dir():
            raise FileNotFoundError(source)
        if link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"Unexpected existing link: {link}")
        elif link.exists():
            raise RuntimeError(f"Refusing to replace existing directory: {link}")
        else:
            link.symlink_to(source, target_is_directory=True)

    command = [sys.executable, str(CONVERTER), "--data_dir", str(STAGE),
               "--out_mode", "cams_meta"]
    result = subprocess.run(command, text=True, capture_output=True)
    (STAGE / "converter.stdout.log").write_text(result.stdout, encoding="utf-8")
    (STAGE / "converter.stderr.log").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"F2-NeRF converter failed ({result.returncode}); "
                           f"see {STAGE / 'converter.stderr.log'}")

    array_path = STAGE / "cams_meta.npy"
    data = np.load(array_path)
    if data.ndim != 2 or data.shape != (121, 27):
        raise RuntimeError(f"Unexpected F2-NeRF shape: {data.shape}")
    k_all = data[:, 12:21].reshape(-1, 3, 3)
    if not np.allclose(k_all, k_all[0], atol=1e-9):
        raise RuntimeError("Expected one shared RGB camera")
    k = k_all[0]
    summary = json.loads((COLMAP / "summary.json").read_text(encoding="utf-8"))
    camera = summary["reconstructions"][0]["cameras"][0]
    expected = np.array([[camera["fx"], 0, camera["cx"]],
                         [0, camera["fy"], camera["cy"]],
                         [0, 0, 1]], dtype=float)
    if not np.allclose(k, expected, atol=1e-9):
        raise RuntimeError("F2-NeRF K differs from source COLMAP model")
    receipt = {"source_scene": summary["scene"],
               "source_run": str(COLMAP.relative_to(ROOT)),
               "f2nerf_source": str(CONVERTER.relative_to(WORKSPACE)),
               "f2nerf_camera_meta": str(array_path.relative_to(ROOT)),
               "f2nerf_camera_meta_sha256": hashlib.sha256(
                   array_path.read_bytes()).hexdigest(),
               "frames": int(data.shape[0]), "K_256x256": k.tolist(),
               "match_colmap_camera": True,
               "pose_use": "Second-scene NeRF poses are not episode_3651 poses"}
    (STAGE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                        encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
