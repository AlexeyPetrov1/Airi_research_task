"""Immutable fmb bundle with preserved dataset-specific arithmetic."""
import numpy as np

from ..bundles import load_bundle
from ..geometry import align_future, project_future
from ..robot_motion import predict


def load(config, root, evaluation=True):
    sample = load_bundle(config, root, evaluation=evaluation)
    if evaluation:
        winner = sample.observed_aux['motion_policy']
        sample.evaluation['motion'] = predict(
            sample.observed_aux['policy_reference'], 49, winner['window'],
            winner['damping_tau_s'], np.arange(1, 31)/15, winner['family']).astype(np.float32)
    return sample

def finish(sample, raw):
    t = sample.future_times
    h = sample.points_3d_history
    initial = h[-1]
    original = align_future(raw, initial, t)
    robot_initial = sample.evaluation["robot_initial"]
    selected = align_future(sample.evaluation["motion"], robot_initial, t)
    correction = np.median(original-initial[:, None], axis=0)
    correction *= np.minimum(1, sample.evaluation["cap"]/np.maximum(np.linalg.norm(correction, axis=-1), 1e-12))[:, None]
    methods = {"Selected_policy_plus_bounded_Molmo": selected+correction[None],
               "v3_Molmo_H3": original, "Observed_selected_motion_policy": selected,
               "v3_Static": np.repeat(initial[:, None], 20, axis=1),
               "v3_CV": initial[:, None]+((h[-1]-h[0])/.2)[:, None]*t[None, :, None]}
    return methods, {name: project_future(xyz, sample.camera_intrinsics, sample.camera_poses)
                     for name, xyz in methods.items()}
