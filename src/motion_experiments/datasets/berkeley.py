"""Two prepared Berkeley native episodes, retaining physical 5 Hz timing."""
import numpy as np

from ..geometry import linear_velocity, project
from ..io import Source
from ..sample import ExperimentSample


def load(config, root, evaluation=True):
    s = Source(root)
    scene = config["scene"]
    base = f"runs/berkeley_ur5_molmomotion/{scene}"
    model = ("runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle" if scene == "bottle" else base)
    meta = s.json(model+"/metadata.json")
    frames = s.array(model+"/observed/history_rgb.npy")
    xyz = s.array(model+"/observed/points_3d_history.npy")
    uv = s.array(model+"/observed/points_2d_history.npy")[-1]
    ids = s.array(model+"/observed/selected_point_ids.npy")
    timestamps = s.array(model+"/observed/history_timestamps.npy")
    k = s.array(model+"/geometry/K_median.npy").astype(float)
    model_files = s.files.copy()
    saved = s.array(model+"/predictions/future_3d.npy")
    processors = [s.path(model+f"/predictions/group_{g:02d}/processor_inputs.pt") for g in range(3)]
    if not evaluation:
        return ExperimentSample("berkeley", scene, frames, timestamps, uv, xyz, meta["instruction"], ids,
                                np.arange(1, 11)/5, k, metadata=meta, source_files=s.files,
                                model_input_files=model_files, legacy_processor_files=processors, saved_prediction=saved)
    refs = s.array(base+"/evaluation/evaluation_results.npz")
    ref3 = s.array(base+"/evaluation/ground_truth_3d_est.npz")
    baseline_k = s.array(base+"/geometry/K_median.npy").astype(float)
    # Same legacy conditional reference: sensor Z/tracker pixels reexpressed
    # with the forecast's K. Do not mix the two candidate camera geometries.
    gt3 = ref3["xyz"] @ (np.linalg.inv(k) @ baseline_k).T if scene == "bottle" else ref3["xyz"]
    rgb = s.array(base+"/evaluation/future_rgb.npy")
    phase = "runs/berkeley_ur5_arc_expansion_v1/phase_schedule"
    old = phase+"/bottle/timing_group_02.mp4" if scene == "bottle" else "runs/berkeley_ur5_improvement_v1/cup/fixed_comparison.mp4"
    metadata = dict(meta, title="Berkeley UR5 · "+scene,
                    comparison_method="reference_0333" if scene == "bottle" else None,
                    primary="late_036_lift005" if scene == "bottle" else "original_physical",
                    shown_indices=list(range(16, 24)) if scene == "bottle" else list(range(8)),
                    geometry="fixed camera; "+meta["K_source"], conditional_3d=True,
                    reference="AllTracker 2D + sensor depth; candidate camera K, conditional 3D",
                    history_indices=meta["history_source_indices"], future_indices=meta["future_source_indices"],
                    old_video=old, metric_3d_ground_truth=False,
                    old_panel_crop=[1280,0,640,480] if scene == "bottle" else [0,0,640,480],
                    legacy_metric_file=phase+"/results.json" if scene == "bottle" else "runs/berkeley_ur5_improvement_v1/fixed_summary.json",
                    limitation=("Параметры bottle и вариант выбраны в прежнем исследовании после просмотра результатов; это exploratory postprocessing, не независимый тест." if scene == "bottle" else
                                "Сохранён исходный физический прогноз. Частоты входа 5 Hz и обучения 15 Hz различаются; K оценён, 3D reference условный."))
    expected = (s.array(phase+"/bottle/projected_2d.npz") if scene == "bottle" else
                s.array("runs/berkeley_ur5_improvement_v1/cup/fixed_arrays.npz"))
    return ExperimentSample("berkeley", scene, frames, timestamps, uv, xyz, meta["instruction"], ids,
                            np.arange(1, 11)/5, k, metadata=metadata, source_files=s.files,
                            model_input_files=model_files, legacy_processor_files=processors,
                            saved_prediction=saved,
                            evaluation=dict(rgb=rgb, xyz=gt3, uv=refs["ground_truth_2d"],
                                            mask3=ref3["valid"], mask2=refs["common_visibility_mask"],
                                            expected=expected, model_dir=model))


def finish(sample, raw):
    p0 = sample.points_3d_history[-1].astype(float)
    raw = raw.astype(float)
    alignment = np.arange(2, 30, 3)
    third = p0[:, None]+(raw-p0[:, None])/3
    if sample.episode_id == "cup":
        methods = {"original_physical": raw[:, alignment], "displacement_one_third": third[:, alignment],
                   "step_equals_frame": raw[:, :10]}
    else:
        t = np.arange(1, 31)/15
        weight = np.clip((t-1)/(2-1), 0, 1)
        weight = weight*weight*(3-2*weight)
        late = p0[:, None]+(1/3+(.36-1/3)*weight)[None, :, None]*(raw-p0[:, None])
        late[..., 1] -= .005*weight[None]
        phase = t/2
        constant = p0[:, None]+.36*(raw-p0[:, None])
        constant[..., 1] -= .005*(3*phase**2-2*phase**3)[None]
        methods = {"reference_0333": third[:, alignment], "scale_036_lift005": constant[:, alignment],
                   "late_036_lift005": late[:, alignment], "raw_MolmoMotion": raw[:, alignment]}
    # Preserve each executed script's arithmetic: cup centers float32 XYZ;
    # the bottle expansion script explicitly promotes its candidate geometry.
    if sample.episode_id == "cup":
        baseline_history = sample.points_3d_history
        observed_times = np.asarray(sample.metadata["history_timestamps"], dtype=np.float64)
        observed_times -= observed_times[-1]
    else:
        baseline_history = sample.points_3d_history.astype(float)
        observed_times = sample.history_timestamps.astype(float)
    velocity = linear_velocity(baseline_history, observed_times)
    methods.update({"Static": np.repeat(p0[:, None], 10, axis=1),
                    "Constant velocity": p0[:, None]+velocity[:, None]*sample.future_times[None, :, None]})
    if sample.episode_id == "bottle":
        methods["Object translation"] = p0[:, None]+np.median(velocity, axis=0)[None, None]*sample.future_times[None, :, None]
    return methods, {name: project(xyz, sample.camera_intrinsics) for name, xyz in methods.items()}
