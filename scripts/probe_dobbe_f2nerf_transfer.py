"""Test RGB-only F2-NeRF control-scene intrinsics on main measured RGB-D.

The transfer is an explicit hypothesis: the two recordings may have used
different phones/camera settings. No future frames or depth fit the candidate K.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

import validate_dobbe_geometry as geometry


def main() -> None:
    folder = geometry.ROOT / "data/dobbe_oxe/target_raw"
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    count = 96
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(count)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(count)]
    points = json.loads((geometry.OUT / "main_static_correspondences.json")
                        .read_text(encoding="utf-8"))
    f2nerf = json.loads((geometry.OUT / "f2nerf_second/receipt.json")
                        .read_text(encoding="utf-8"))
    matrix = np.asarray(f2nerf["K_256x256"], dtype=float)
    control = json.loads((geometry.OUT / "second_static_geometry.json")
                         .read_text(encoding="utf-8"))
    rectified = control["candidates"][
        "colmap_simple_pinhole_all_fixedpp_rectified"]["camera"]["params"]
    cases = {
        "naive_baseline": [200, 200, 128, 128],
        "f2nerf_control_pinhole": [matrix[0, 0], matrix[1, 1],
                                   matrix[0, 2], matrix[1, 2]],
        "control_aspect_pinhole": rectified,
    }
    output = {"source": f2nerf["f2nerf_camera_meta"],
              "hypothesis": "Same effective camera K across two HoNY recordings; unverified",
              "main_calibration_only": True, "candidates": {}}
    for name, params in cases.items():
        candidate = {"model": "PINHOLE", "params": list(map(float, params))}
        output["candidates"][name] = {
            "K_256x256": candidate["params"],
            "c2w_z": geometry.evaluate(points, poses, candidate, "c2w", "z"),
            "c2w_ray": geometry.evaluate(points, poses, candidate, "c2w", "ray")}
    path = geometry.OUT / "f2nerf_intrinsics_transfer.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for name, candidate in output["candidates"].items():
        metric = candidate["c2w_z"]
        print(name, candidate["K_256x256"],
              metric["median_3d_m"], metric["median_reprojection_px"])


if __name__ == "__main__":
    main()
