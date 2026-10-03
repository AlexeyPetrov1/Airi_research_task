"""Immutable davis bundle with preserved dataset-specific arithmetic."""
import numpy as np

from ..bundles import load_bundle
from ..geometry import project


def load(config, root, evaluation=True):
    return load_bundle(config, root, evaluation=evaluation)

def finish(sample, raw):
    history = sample.points_3d_history
    steps = np.arange(1, 31, dtype=np.float32)
    methods = {"MolmoMotion": raw,
               "Static": np.broadcast_to(history[-1, :, None], raw.shape).copy(),
               "Constant velocity": history[-1, :, None] + steps[None, :, None]*(history[-1]-history[-2])[:, None]}
    # Display only in the anchored first camera; never score this as video 2D GT.
    uv = {name: project(xyz, sample.camera_intrinsics) for name, xyz in methods.items()}
    return methods, uv
