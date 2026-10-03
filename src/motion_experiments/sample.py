from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ExperimentSample:
    dataset: str
    episode_id: str
    history_frames: np.ndarray
    history_timestamps: np.ndarray
    points_2d: np.ndarray
    points_3d_history: np.ndarray
    action: str
    point_ids: np.ndarray
    future_times: np.ndarray
    camera_intrinsics: np.ndarray | None = None
    camera_poses: np.ndarray | None = None
    c2w_at_t0: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)
    source_files: list[Path] = field(default_factory=list)
    model_input_files: list[Path] = field(default_factory=list)
    processor_fingerprints: list[dict] = field(default_factory=list)
    observed_aux: dict = field(default_factory=dict)
    saved_prediction: np.ndarray | None = None
    evaluation: dict = field(default_factory=dict)
    status: str = "COMPLETE"
    failure_reason: str | None = None


def validate_sample(sample):
    h, n = sample.points_3d_history.shape[:2]
    if h not in (1, 3) or n not in (8, 24):
        raise ValueError("Use the original H1/H3 and P8/P24 protocol")
    if sample.points_3d_history.shape != (h, n, 3) or sample.points_2d.shape != (n, 2):
        raise ValueError("Point/time dimensions do not agree")
    if sample.history_frames.shape[0] != h or sample.history_frames.shape[-1] != 3:
        raise ValueError("Expected the original RGB history")
    if sample.history_frames.dtype != np.uint8:
        raise ValueError("RGB must be uint8")
    for array in (sample.points_3d_history, sample.points_2d, sample.history_timestamps):
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite observed inputs")
    if len(np.unique(sample.point_ids)) != n or len(sample.history_timestamps) != h:
        raise ValueError("Point IDs or timestamps do not agree")
    if not sample.action or not np.all(np.diff(sample.history_timestamps) > 0):
        raise ValueError("Missing action or incorrect historical time order")
    if sample.metadata.get("future_used_for_model_input", False):
        raise ValueError("Future data cannot enter the forecast")
    return {"status": sample.status, "failure_stage": "geometry" if sample.failure_reason else None,
            "failure_reason": sample.failure_reason,
            "metric_3d_ground_truth": sample.metadata.get("metric_3d_ground_truth", False)}
