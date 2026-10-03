"""Immutable dobbe bundle with preserved dataset-specific arithmetic."""
import numpy as np

from ..bundles import load_bundle
from ..geometry import project_future


def load(config, root, evaluation=True):
    return load_bundle(config, root, evaluation=evaluation)

def finish(sample, raw):
    initial = sample.evaluation["initial_camera"]
    xyz = {"MolmoMotion": raw, "Static": np.repeat(initial[:, None], 30, axis=1)}
    uv = {name: project_future(values, sample.camera_intrinsics, sample.camera_poses)
          for name, values in xyz.items()}
    # Original convention marks camera-plane/behind-camera forecasts invalid.
    for name, values in xyz.items():
        for t in range(30):
            pose = sample.camera_poses[t+1]
            cam = (values[:, t]-pose[:3, 3]) @ pose[:3, :3]
            uv[name][cam[:, 2] <= 0, t] = np.nan
    return xyz, uv
