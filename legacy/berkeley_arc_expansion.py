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
