import numpy as np


def project(world, c2w, intrinsics):
    cam = (world-c2w[:3, 3]) @ c2w[:3, :3]
    fx, fy, cx, cy = intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.column_stack([cam[:, 0]/cam[:, 2]*fx+cx, cam[:, 1]/cam[:, 2]*fy+cy])
    uv[cam[:, 2] <= 0] = np.nan
    return uv

def stats(values):
    v = np.asarray(values)
    v = v[np.isfinite(v)]
    return {"count": len(v), "median": float(np.median(v)) if len(v) else None,
            "p90": float(np.percentile(v, 90)) if len(v) else None}
