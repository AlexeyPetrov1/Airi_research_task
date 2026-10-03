import numpy as np
import re
from decimal import Decimal


def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict:
    if pred.shape != gt.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("prediction and GT must have matching (P,F,3) shape")
    if valid.shape != pred.shape[:2] or valid.dtype != np.bool_:
        raise ValueError("valid mask must be bool (P,F)")
    if not np.isfinite(pred).all():
        raise ValueError("prediction has NaN/Inf; cannot silently mask model omissions")
    if not np.isfinite(gt[valid]).all():
        raise ValueError("GT is nonfinite where mask is true")
    errors = np.linalg.norm(pred.astype(np.float64) - gt.astype(np.float64), axis=-1)
    counts = valid.sum(axis=0)
    per_time = [float(errors[valid[:, t], t].mean()) if counts[t] else None for t in range(valid.shape[1])]
    return {
        "ADE_m": float(errors[valid].mean()) if valid.any() else None,
        "FDE_m": float(errors[valid[:, -1], -1].mean()) if valid[:, -1].any() else None,
        "FDE_frame": pred.shape[1],
        "valid_pairs": int(valid.sum()),
        "total_pairs": int(valid.size),
        "FDE_valid_points": int(valid[:, -1].sum()),
        "error_by_horizon_m": per_time,
        "valid_points_by_horizon": counts.tolist(),
    }

def parse_raw_forecast(text: str, points: int = 8, horizon: int = 30) -> np.ndarray:
    """Strictly decode the saved quantized text without the model's permissive parser."""
    match = re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*', text)
    if match is None:
        raise ValueError("Expected exactly one complete <tracks> block")
    frames = match.group(1).split(";")
    if len(frames) != horizon:
        raise ValueError(f"Expected {horizon} timestamps, got {len(frames)}")
    delta = np.empty((points, horizon, 3), dtype=np.float32)
    for step, frame in enumerate(frames):
        tokens = frame.split()
        if len(tokens) != 1 + 4 * points:
            raise ValueError(f"Timestamp {step}: expected {points} complete point records")
        if Decimal(tokens[0]) != Decimal(step + 3):
            raise ValueError(f"Unexpected or repeated timestamp at step {step}: {tokens[0]}")
        seen = set()
        for offset in range(1, len(tokens), 4):
            record = tokens[offset:offset + 4]
            if any(re.fullmatch(r"[+-]?\d+", token) is None for token in record):
                raise ValueError(f"Non-integer point record: {record}")
            point_id, x, y, z = map(int, record)
            if point_id not in range(1, points + 1) or point_id in seen:
                raise ValueError(f"Missing, duplicate or unexpected point ID at step {step + 3}")
            seen.add(point_id)
            delta[point_id - 1, step] = (x / 1000.0, y / 1000.0, z / 1000.0)
        if seen != set(range(1, points + 1)):
            raise ValueError(f"Missing point IDs at step {step + 3}")
    if not np.isfinite(delta).all():
        raise ValueError("Dequantized answer contains NaN/Inf")
    return delta

def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    z = xyz[..., 2]
    px = np.full((*xyz.shape[:-1], 2), np.nan, dtype=np.float64)
    good = np.isfinite(xyz).all(axis=-1) & (z > 0)
    px[..., 0][good] = k[0, 0] * xyz[..., 0][good] / z[good] + k[0, 2]
    px[..., 1][good] = k[1, 1] * xyz[..., 1][good] / z[good] + k[1, 2]
    return px
