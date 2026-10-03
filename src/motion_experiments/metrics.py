"""Shared ADE/FDE on the unchanged legacy masks, plus descriptive diagnostics."""
import numpy as np


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


def displacement_metrics(prediction, truth, mask, thresholds=()):
    if prediction.shape != truth.shape or mask.shape != prediction.shape[:2] or mask.dtype != bool:
        raise ValueError("Incompatible prediction/reference/mask")
    error = np.linalg.norm(prediction.astype(float)-truth.astype(float), axis=-1)
    if not np.isfinite(error[mask]).all():
        raise ValueError("A forecast is invalid on the reference mask; cannot hide it")
    by_time = [float(error[:, t][mask[:, t]].mean()) if mask[:, t].any() else None for t in range(mask.shape[1])]
    cumulative = [float(error[:, :t+1][mask[:, :t+1]].mean()) if mask[:, :t+1].any() else None for t in range(mask.shape[1])]
    by_point = [float(error[p][mask[p]].mean()) if mask[p].any() else None for p in range(len(mask))]
    return dict(ADE=float(error[mask].mean()) if mask.any() else None,
                FDE=float(error[:, -1][mask[:, -1]].mean()) if mask[:, -1].any() else None,
                error_by_time=by_time, error_by_point=by_point, cumulative_ADE=cumulative,
                median=float(np.median(error[mask])) if mask.any() else None,
                P90=float(np.percentile(error[mask],90)) if mask.any() else None,
                PWT={str(t):float((error[mask]<t).mean()) if mask.any() else None for t in thresholds},
                valid_points_by_time=mask.sum(axis=0).tolist(), valid_samples=int(mask.sum()),
                valid_final_points=int(mask[:, -1].sum()), total_samples=int(mask.size),
                coverage=float(mask.mean())), error


def trajectory_description(prediction, reference, start, mask, times):
    p = np.concatenate([start[:, None], prediction], axis=1)
    g = np.concatenate([start[:, None], reference], axis=1)
    segments = mask & np.c_[np.ones(len(mask), bool), mask[:, :-1]]
    vp, vg = np.diff(p, axis=1), np.diff(g, axis=1)
    npred, ngt = np.linalg.norm(vp, axis=-1), np.linalg.norm(vg, axis=-1)
    actual_moving = segments & (ngt > 1)
    moving = actual_moving & (npred > 1)
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
        moving_step_direction_coverage=float(moving.sum()/actual_moving.sum()) if actual_moving.any() else None,
        predicted_endpoint_displacement_xy_median=np.median(prediction[:, -1]-start, axis=0).tolist())


def compute_metrics(sample, xyz, uv):
    result, errors = {}, {}
    height, width = sample.history_frames.shape[1:3]
    for name in xyz:
        row = {}
        if not sample.evaluation.get("diagnostic_projection"):
            row["2D_px"], error = displacement_metrics(uv[name], sample.evaluation["uv"], sample.evaluation["mask2"], (5,10,20,40))
            errors[name+"_2d"] = error
            row["2D_px"]["ADE_image_diagonal_fraction"] = row["2D_px"]["ADE"]/np.hypot(width,height)
            mask=sample.evaluation["mask2"]
            row["groups_P8"]=[dict(group=g,ADE_px=float(error[g*8:(g+1)*8][mask[g*8:(g+1)*8]].mean())
                                  if mask[g*8:(g+1)*8].any() else None) for g in range(len(mask)//8)]
            row["trajectory_2D"] = trajectory_description(uv[name], sample.evaluation["uv"], sample.points_2d,
                                                          sample.evaluation["mask2"], sample.future_times)
        else:
            row["2D_px"] = None
        if sample.evaluation.get("xyz") is not None:
            row["3D_m"], error = displacement_metrics(xyz[name], sample.evaluation["xyz"], sample.evaluation["mask3"], (.01,.02,.05,.1,.2))
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
        initial = sample.evaluation.get("method_initial",{}).get(name,sample.points_3d_history[-1]).astype(float)
        if sample.c2w_at_t0 is not None:
            pose = sample.c2w_at_t0
            initial = (initial-pose[:3, 3]) @ pose[:3, :3]
        distance0 = np.linalg.norm(initial[pairs[0]]-initial[pairs[1]], axis=-1)
        drift = np.abs(distances-distance0[:,None])*1000
        within = pairs[0]//8 == pairs[1]//8
        row["pair_distance_drift_median_mm"] = float(np.median(drift))
        row["pair_distance_drift_3D_mm"] = dict(median=float(np.median(drift)),P90=float(np.percentile(drift,90)),
               within_P8_median=float(np.median(drift[within])) if within.any() else None,
               across_P8_median=float(np.median(drift[~within])) if (~within).any() else None)
        row["median_endpoint_amplitude_m"] = float(np.median(np.linalg.norm(xyz[name][:,-1]-initial,axis=-1)))
        result[name] = row
    comparison = sample.metadata.get("comparison_method")
    if comparison:
        for name,row in result.items():
            row["expansion_vs_one_third"] = extent_diagnostics(uv[name],uv[comparison],sample.points_2d,
                                                              xyz[name],xyz[comparison])
    return dict(methods=result, primary=sample.metadata["primary"],
                definition="ADE: mean distance over fixed valid point/time pairs; FDE: mean distance at the fixed final horizon. No last-visible substitution.",
                reference=sample.metadata["reference"], metric_3d_ground_truth=False,
                conditional_3d=sample.metadata.get("conditional_3d", False),
                future_fit=False, metric_3d_status="AVAILABLE_ESTIMATED_REFERENCE" if sample.evaluation.get("xyz") is not None else "UNAVAILABLE_NO_3D_REFERENCE"), errors
