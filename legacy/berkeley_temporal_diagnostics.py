import numpy as np


def scale_displacement(prediction, p0, alpha):
    """Scale around each point's own t0, preserving its spatial identity."""
    if prediction.ndim != 3 or p0.shape != (len(prediction), 3):
        raise ValueError("Expected point-major [P,T,3] and per-point [P,3] t0")
    if not np.isfinite(alpha) or not np.isfinite(prediction).all() or not np.isfinite(p0).all():
        raise ValueError("Nonfinite cadence input")
    return p0[:, None] + float(alpha) * (prediction-p0[:, None])
