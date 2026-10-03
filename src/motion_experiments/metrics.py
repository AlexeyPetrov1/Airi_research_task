"""Shared ADE/FDE on the unchanged legacy masks, plus descriptive diagnostics."""
import numpy as np


def displacement_metrics(prediction, truth, mask):
    if prediction.shape != truth.shape or mask.shape != prediction.shape[:2] or mask.dtype != bool:
        raise ValueError("Incompatible prediction/reference/mask")
    error = np.linalg.norm(prediction.astype(float)-truth.astype(float), axis=-1)
    if not np.isfinite(error[mask]).all():
        raise ValueError("A forecast is invalid on the reference mask; cannot hide it")
    by_time = [float(error[:, t][mask[:, t]].mean()) if mask[:, t].any() else None for t in range(mask.shape[1])]
    cumulative = [float(error[:, :t+1][mask[:, :t+1]].mean()) if mask[:, :t+1].any() else None for t in range(mask.shape[1])]
    return dict(ADE=float(error[mask].mean()) if mask.any() else None,
                FDE=float(error[:, -1][mask[:, -1]].mean()) if mask[:, -1].any() else None,
                error_by_time=by_time, cumulative_ADE=cumulative, valid_samples=int(mask.sum()),
                valid_final_points=int(mask[:, -1].sum()), total_samples=int(mask.size),
                coverage=float(mask.mean())), error


def trajectory_description(prediction, reference, start, mask, times):
    p = np.concatenate([start[:, None], prediction], axis=1)
    g = np.concatenate([start[:, None], reference], axis=1)
    segments = mask & np.c_[np.ones(len(mask), bool), mask[:, :-1]]
    vp, vg = np.diff(p, axis=1), np.diff(g, axis=1)
    npred, ngt = np.linalg.norm(vp, axis=-1), np.linalg.norm(vg, axis=-1)
    moving = segments & (ngt > 1) & (npred > 1)
    cosine = np.sum(vp*vg, axis=-1)/np.maximum(npred*ngt, 1e-12)
    endpoint = mask[:, -1] & (np.linalg.norm(reference[:, -1]-start, axis=-1) > 1)
    complete = segments.all(axis=1)
    valid_length = complete & (ngt.sum(axis=1) > 1)
    dt = np.diff(np.r_[0, times])
    return dict(
        direction_error_median_degrees=float(np.median(np.degrees(np.arccos(np.clip(cosine[moving], -1, 1))))) if moving.any() else None,
        endpoint_displacement_ratio_median=float(np.median(np.linalg.norm(prediction[:, -1]-start, axis=-1)[endpoint]/np.linalg.norm(reference[:, -1]-start, axis=-1)[endpoint])) if endpoint.any() else None,
        path_length_ratio_median=float(np.median(npred.sum(axis=1)[valid_length]/ngt.sum(axis=1)[valid_length])) if valid_length.any() else None,
        speed_MAE_px_per_s=float((np.abs(npred-ngt)/dt[None])[segments].mean()) if segments.any() else None,
        complete_point_paths=int(complete.sum()), moving_direction_samples=int(moving.sum()),
        predicted_endpoint_displacement_xy_median=np.median(prediction[:, -1]-start, axis=0).tolist())


def compute_metrics(sample, xyz, uv):
    result, errors = {}, {}
    height, width = sample.history_frames.shape[1:3]
    for name in xyz:
        row = {}
        if not sample.evaluation.get("diagnostic_projection"):
            row["2D_px"], error = displacement_metrics(uv[name], sample.evaluation["uv"], sample.evaluation["mask2"])
            errors[name+"_2d"] = error
            row["trajectory_2D"] = trajectory_description(uv[name], sample.evaluation["uv"], sample.points_2d,
                                                          sample.evaluation["mask2"], sample.future_times)
        else:
            row["2D_px"] = None
        if sample.evaluation.get("xyz") is not None:
            row["3D_m"], error = displacement_metrics(xyz[name], sample.evaluation["xyz"], sample.evaluation["mask3"])
            errors[name+"_3d"] = error
        else:
            row["3D_m"] = None
        finite = np.isfinite(uv[name]).all(axis=-1)
        outside = finite & ((uv[name] < 0).any(-1) | (uv[name] >= [width, height]).any(-1))
        row["projection"] = dict(outside_image_fraction=float(outside.mean()), invalid_projection_fraction=float((~finite).mean()),
                                  offimage_excluded_from_metrics=False,
                                  scope=sample.metadata.get("projection_scope", "per-frame camera projection"))
        pairs = np.triu_indices(len(sample.point_ids), 1)
        distances = np.linalg.norm(xyz[name][pairs[0]]-xyz[name][pairs[1]], axis=-1)
        initial = sample.points_3d_history[-1]
        if sample.c2w_at_t0 is not None:
            pose = sample.c2w_at_t0
            initial = (initial-pose[:3, 3]) @ pose[:3, :3]
        distance0 = np.linalg.norm(initial[pairs[0]]-initial[pairs[1]], axis=-1)
        row["pair_distance_drift_median_mm"] = float(np.median(np.abs(distances-distance0[:, None]))*1000)
        result[name] = row
    return dict(methods=result, primary=sample.metadata["primary"],
                definition="ADE: mean distance over fixed valid point/time pairs; FDE: mean distance at the fixed final horizon. No last-visible substitution.",
                reference=sample.metadata["reference"], metric_3d_ground_truth=False,
                conditional_3d=sample.metadata.get("conditional_3d", False),
                future_fit=False, metric_3d_status="AVAILABLE_ESTIMATED_REFERENCE" if sample.evaluation.get("xyz") is not None else "UNAVAILABLE_NO_3D_REFERENCE"), errors
