"""
Evaluator for 3D trajectory prediction (EgoDex + multi-dataset).

Parses <tracks> text output from the model, converts quantized delta
coordinates back to raw 3D space, and computes MSE / L2 metrics.

Metrics are computed in raw 3D coordinate space (camera frame) so that
results are directly comparable across different model types.

Aggregation hierarchy:
  per-config → per-(video, object) (weighted by n_visible) →
  per-dataset → overall (equal weight per (video, object) item)
"""

import re
from collections import defaultdict

import numpy as np
import torch
import torchmetrics


def parse_tracks_text(text):
    """Parse one or more <tracks coords="...">label</tracks> blocks into a dict.

    The answer may contain multiple <tracks> blocks (endpoint / history / full
    trajectory). All blocks are merged by timestamp; later blocks override
    earlier ones on duplicate timestamps so the full-trajectory block (always
    emitted last) wins against the single-frame endpoint prefix.

    Returns:
        dict mapping timestamp (float) -> list of (obj_id, x, y, z) tuples
        where x, y, z are quantized ints (0.001 units).
        Returns None if parsing fails.
    """
    frames = {}
    for m in re.finditer(r'<tracks\s+coords="([^"]*)"', text):
        coords_str = m.group(1)
        if not coords_str.strip():
            continue
        for frame_str in coords_str.split(";"):
            frame_str = frame_str.strip()
            if not frame_str:
                continue
            tokens = frame_str.split()
            if len(tokens) < 2:
                continue
            try:
                timestamp = float(tokens[0])
            except ValueError:
                continue

            points = []
            i = 1
            while i + 3 < len(tokens):
                try:
                    obj_id = int(tokens[i])
                    x = int(tokens[i + 1])
                    y = int(tokens[i + 2])
                    z = int(tokens[i + 3])
                    points.append((obj_id, x, y, z))
                    i += 4
                except (ValueError, IndexError):
                    i += 1
            frames[timestamp] = points

    return frames if frames else None


def tracks_to_array(parsed_tracks, num_points, num_frames, start_timestamp=1.0):
    """Convert parsed tracks dict to (P, F, 3) numpy array + visibility mask.

    Args:
        parsed_tracks: dict from parse_tracks_text()
        num_points: P
        num_frames: F (number of future frames)
        start_timestamp: first future timestamp (1.0 for H=1, 3.0 for H=3)

    Returns:
        delta: (P, F, 3) float32 — de-quantized deltas (÷1000)
        visibility: (P, F) bool
    """
    delta = np.zeros((num_points, num_frames, 3), dtype=np.float32)
    visibility = np.zeros((num_points, num_frames), dtype=bool)

    for fi in range(num_frames):
        timestamp = start_timestamp + fi
        if timestamp not in parsed_tracks:
            continue
        for obj_id, x, y, z in parsed_tracks[timestamp]:
            pi = obj_id - 1  # 1-indexed → 0-indexed
            if 0 <= pi < num_points:
                delta[pi, fi, 0] = x / 1000.0
                delta[pi, fi, 1] = y / 1000.0
                delta[pi, fi, 2] = z / 1000.0
                visibility[pi, fi] = True

    return delta, visibility


def compute_3d_metrics_raw(pred_raw, gt_raw, gt_vis):
    """Compute MSE and L2 in raw 3D coordinate space on visible points.

    Args:
        pred_raw: (P, F, 3) predicted raw 3D coordinates
        gt_raw: (P, F, 3) ground truth raw 3D coordinates
        gt_vis: (P, F) bool visibility mask

    Returns:
        dict with mse, l2, n_visible
    """
    valid = gt_vis
    n_visible = int(valid.sum())
    if n_visible == 0:
        return {"mse": float("nan"), "l2": float("nan"), "n_visible": 0}

    diff = (pred_raw - gt_raw).astype(np.float64)
    se = diff ** 2

    valid_3d = np.repeat(valid[:, :, np.newaxis], 3, axis=2)
    mse = se[valid_3d].mean()

    l2_per_point = np.sqrt(se.sum(axis=2))
    l2 = l2_per_point[valid].mean()

    return {
        "mse": float(mse),
        "l2": float(l2),
        "n_visible": n_visible,
    }


class EgoDex3DEvaluator:
    """Evaluator for 3D trajectory prediction (supports multi-dataset).

    Called by InfDatasetEvaluator with generated text predictions.
    Parses <tracks> output, reverses to raw 3D space using the stored anchor,
    and computes MSE / L2 against GT raw coordinates.

    Aggregation:
      per-config → per-(video, object) weighted by n_visible →
      per-dataset → overall
    """

    def __call__(self, metadatas, predictions, tokenizer, step=None):
        pred_texts = predictions["predictions_text"]

        n_parsed = 0
        n_failed = 0

        # Collect per-(video, object) results
        # key = (dataset, stem, obj_label) → list of {mse, l2, n_visible}
        per_item = defaultdict(list)
        item_dataset = {}  # key → dataset_name

        for ex_ix, pred_text in enumerate(pred_texts):
            metadata = metadatas[ex_ix]

            gt_anchor = np.array(metadata["gt_anchor"], dtype=np.float32)
            gt_future_raw = np.array(metadata["gt_future_raw"], dtype=np.float32)
            gt_future_vis = np.array(metadata["gt_future_vis"], dtype=bool)

            num_points = gt_future_raw.shape[0]
            num_frames = gt_future_raw.shape[1]

            # Determine history size from metadata or text to find start_timestamp
            # The input text contains history timestamps 0.0..H-1.0
            # Future timestamps start at H.0
            hist_frames = metadata.get("hist_frames")
            if hist_frames is not None:
                H = len(hist_frames)
            else:
                # Fallback: parse from input text (count timestamps in input tracks)
                H = 1  # legacy default
            start_timestamp = float(H)

            # Parse prediction
            parsed = parse_tracks_text(pred_text)
            if parsed is None:
                n_failed += 1
                continue

            pred_delta, pred_vis = tracks_to_array(
                parsed, num_points, num_frames, start_timestamp)

            # Convert predicted delta back to raw 3D space
            pred_raw = pred_delta + gt_anchor

            metrics = compute_3d_metrics_raw(pred_raw, gt_future_raw, gt_future_vis)
            if np.isnan(metrics["mse"]):
                n_failed += 1
                continue

            n_parsed += 1

            stem = metadata.get("video", str(ex_ix))
            obj_label = metadata.get("obj_label", 0)
            dataset_name = metadata.get("dataset_name", metadata.get("task_name", "unknown"))
            item_key = (dataset_name, stem, obj_label)

            per_item[item_key].append(metrics)
            item_dataset[item_key] = dataset_name

        # ── Aggregation ─────────────────────────────────────────────────
        # 1. Per-(video, object): weighted average across configs by n_visible
        item_mse = {}
        item_l2 = {}
        for item_key, config_metrics in per_item.items():
            weights = np.array([m["n_visible"] for m in config_metrics], dtype=np.float64)
            total_w = weights.sum()
            if total_w == 0:
                continue
            item_mse[item_key] = sum(m["mse"] * w for m, w in zip(config_metrics, weights)) / total_w
            item_l2[item_key] = sum(m["l2"] * w for m, w in zip(config_metrics, weights)) / total_w

        # 2. Per-dataset: average across items (equal weight per item)
        per_ds_mse = defaultdict(list)
        per_ds_l2 = defaultdict(list)
        for item_key in item_mse:
            ds = item_dataset[item_key]
            per_ds_mse[ds].append(item_mse[item_key])
            per_ds_l2[ds].append(item_l2[item_key])

        # 3. Overall: equal weight per (video, object) item
        all_item_mse = list(item_mse.values())
        all_item_l2 = list(item_l2.values())

        # ── Build output metrics ────────────────────────────────────────
        out = {}
        if all_item_mse:
            out["mse"] = torchmetrics.MeanMetric()
            out["mse"].update(torch.tensor(all_item_mse))
            out["l2"] = torchmetrics.MeanMetric()
            out["l2"].update(torch.tensor(all_item_l2))

            # Per-dataset metrics
            for ds_name in sorted(per_ds_mse.keys()):
                ds_mse_vals = per_ds_mse[ds_name]
                ds_l2_vals = per_ds_l2[ds_name]
                out[f"{ds_name}_mse"] = torchmetrics.MeanMetric()
                out[f"{ds_name}_mse"].update(torch.tensor(ds_mse_vals))
                out[f"{ds_name}_l2"] = torchmetrics.MeanMetric()
                out[f"{ds_name}_l2"].update(torch.tensor(ds_l2_vals))

        out["n_parsed"] = float(n_parsed)
        out["n_failed"] = float(n_failed)
        out["parse_rate"] = float(n_parsed / max(n_parsed + n_failed, 1))
        out["n_items"] = float(len(item_mse))
        out["n_datasets"] = float(len(per_ds_mse))

        return out
