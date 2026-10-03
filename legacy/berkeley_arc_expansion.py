import numpy as np
from .berkeley_temporal_diagnostics import scale_displacement


def expand(prediction, p0, alpha, lift_m, times_s):
    """Own-point displacement scale plus one smooth translation for all points.

    Camera X is right, Y is down, Z is forward. Lift is negative camera Y,
    not world/gravity height. Smoothstep is zero at t0 and reaches lift_m at 2s.
    The lift cannot change any pair distance at the same time; it changes no Z.
    """
    times_s = np.asarray(times_s, dtype=float)
    if times_s.shape != (prediction.shape[1],) or not np.isfinite(times_s).all():
        raise ValueError("One finite timestamp per prediction step is required")
    if np.any((times_s < 0) | (times_s > 2)) or not np.isfinite(lift_m) or lift_m < 0:
        raise ValueError("Lift requires 0..2 second timestamps and a nonnegative finite height")
    result = scale_displacement(prediction, p0, alpha)
    phase = times_s / 2.
    result[..., 1] -= float(lift_m) * (3 * phase**2 - 2 * phase**3)[None]
    if np.any(result[..., 2] <= 0):
        raise ValueError("Candidate lies behind the camera")
    return result

def extent_diagnostics(uv, baseline, uv0, xyz, baseline_xyz):
    distance = np.linalg.norm(uv - uv0[:, None], axis=-1)
    base_distance = np.linalg.norm(baseline - uv0[:, None], axis=-1)
    step = np.diff(np.concatenate([uv0[:, None], uv], axis=1), axis=1)
    base_step = np.diff(np.concatenate([uv0[:, None], baseline], axis=1), axis=1)
    return dict(
        median_farther_px_by_time=np.median(distance-base_distance, axis=0).tolist(),
        median_higher_px_by_time=np.median(baseline[..., 1]-uv[..., 1], axis=0).tolist(),
        farther_pair_fraction=float((distance > base_distance+1e-6).mean()),
        higher_pair_fraction=float((uv[..., 1] < baseline[..., 1]-1e-6).mean()),
        median_path_length_ratio_vs_one_third=float(np.median(
            np.linalg.norm(step, axis=-1).sum(1) / np.linalg.norm(base_step, axis=-1).sum(1))),
        median_endpoint_farther_px=float(np.median(distance[:, -1]-base_distance[:, -1])),
        median_endpoint_higher_px=float(np.median(baseline[:, -1, 1]-uv[:, -1, 1])),
        median_camera_up_difference_mm=float(np.median(baseline_xyz[..., 1]-xyz[..., 1])*1000),
    )
