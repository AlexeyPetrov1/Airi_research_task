import numpy as np


def temporal_expand(raw, p0, alpha_end, lift_m, onset_s, times_s):
    times_s = np.asarray(times_s, dtype=float)
    parameters = np.asarray([alpha_end, lift_m, onset_s], dtype=float)
    if (raw.ndim != 3 or p0.shape != (len(raw),3) or times_s.shape != (raw.shape[1],)
        or not np.isfinite(raw).all() or not np.isfinite(p0).all() or not np.isfinite(times_s).all()
        or not np.isfinite(parameters).all() or np.any((times_s<0)|(times_s>2))
        or not 0<=onset_s<2 or alpha_end<1/3 or lift_m<0):
        raise ValueError("Invalid shared phase schedule")
    q = np.clip((times_s-onset_s)/(2-onset_s), 0., 1.)
    weight = q*q*(3-2*q)
    alpha = 1/3+(alpha_end-1/3)*weight
    result = p0[:,None]+alpha[None,:,None]*(raw-p0[:,None])
    result[...,1] -= lift_m*weight[None]
    if np.any(result[...,2]<=0):
        raise ValueError("Candidate crosses the camera plane")
    return result
