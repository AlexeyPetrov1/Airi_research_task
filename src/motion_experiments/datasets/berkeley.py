"""Immutable berkeley bundle with preserved dataset-specific arithmetic."""
import numpy as np

from ..bundles import load_bundle
from ..geometry import linear_velocity, project


def load(config, root, evaluation=True):
    return load_bundle(config, root, evaluation=evaluation)

def finish(sample, raw):
    p0 = sample.points_3d_history[-1].astype(float)
    raw = raw.astype(float)
    alignment = np.arange(2, 30, 3)
    third = p0[:, None]+(raw-p0[:, None])/3
    if sample.episode_id == "cup":
        methods = {"original_physical": raw[:, alignment], "displacement_one_third": third[:, alignment],
                   "step_equals_frame": raw[:, :10]}
    else:
        t = np.arange(1, 31)/15
        weight = np.clip((t-1)/(2-1), 0, 1)
        weight = weight*weight*(3-2*weight)
        late = p0[:, None]+(1/3+(.36-1/3)*weight)[None, :, None]*(raw-p0[:, None])
        late[..., 1] -= .005*weight[None]
        phase = t/2
        constant = p0[:, None]+.36*(raw-p0[:, None])
        constant[..., 1] -= .005*(3*phase**2-2*phase**3)[None]
        methods = {"reference_0333": third[:, alignment], "scale_036_lift005": constant[:, alignment],
                   "late_036_lift005": late[:, alignment], "raw_MolmoMotion": raw[:, alignment]}
    # Preserve each executed script's arithmetic: cup centers float32 XYZ;
    # the bottle expansion script explicitly promotes its candidate geometry.
    if sample.episode_id == "cup":
        baseline_history = sample.points_3d_history
        observed_times = np.asarray(sample.metadata["history_timestamps"], dtype=np.float64)
        observed_times -= observed_times[-1]
    else:
        baseline_history = sample.points_3d_history.astype(float)
        observed_times = sample.history_timestamps.astype(float)
    velocity = linear_velocity(baseline_history, observed_times)
    methods.update({"Static": np.repeat(p0[:, None], 10, axis=1),
                    "Constant velocity": p0[:, None]+velocity[:, None]*sample.future_times[None, :, None]})
    if sample.episode_id == "bottle":
        methods["Object translation"] = p0[:, None]+np.median(velocity, axis=0)[None, None]*sample.future_times[None, :, None]
    return methods, {name: project(xyz, sample.camera_intrinsics) for name, xyz in methods.items()}
