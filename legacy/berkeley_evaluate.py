import numpy as np


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([k[0, 0] * xyz[..., 0] / xyz[..., 2] + k[0, 2],
                         k[1, 1] * xyz[..., 1] / xyz[..., 2] + k[1, 2]], axis=-1)

def velocity(history: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    centered = timestamps - timestamps.mean()
    return np.sum(centered[:, None, None] * (history - history.mean(axis=0)), axis=0) / np.sum(centered**2)

def metric_values(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray, suffix: str) -> tuple[dict, np.ndarray]:
    error = np.linalg.norm(prediction - truth, axis=-1)
    if not np.isfinite(error[mask]).all():
        raise ValueError("A method is nonfinite on the common GT visibility mask; do not drop its bad predictions")
    by_time = [float(error[:, t][mask[:, t]].mean()) if mask[:, t].any() else None for t in range(10)]
    return {f"ADE_{suffix}": float(error[mask].mean()) if mask.any() else None,
            f"FDE_{suffix}": float(error[:, -1][mask[:, -1]].mean()) if mask[:, -1].any() else None,
            "error_by_horizon": by_time, "valid_pairs": int(mask.sum()),
            "valid_point_coverage": float(mask.mean()), "valid_points_by_horizon": mask.sum(axis=0).tolist(),
            "FDE_valid_points": int(mask[:, -1].sum())}, error
