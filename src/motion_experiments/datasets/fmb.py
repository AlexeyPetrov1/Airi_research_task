"""Prepared two-camera FMB episode and frozen observed motion policy."""
import numpy as np

from ..geometry import align_future, project_future
from ..io import Source
from ..robot_motion import predict
from ..sample import ExperimentSample


def load(config, root, evaluation=True):
    s = Source(root)
    camera = config["camera"]
    base = f"runs/fmb_wrist_v3/{camera}"
    improved = "runs/fmb_wrist_v4_improvement"
    dest = improved+"/"+camera
    branch = base+"/branches/C_sensor_tcp_official"
    meta = s.json(base+"/metadata.json")
    frames = s.array(base+"/observed/history_rgb.npy")
    xyz = s.array(branch+"/observed/points_3d_history.npy")
    uv = s.array(base+"/observed/points_2d_history.npy")[-1]
    observed = s.array(dest+"/observed_reference.npz")
    winner = s.json(improved+"/motion_policy_amendment_v2.json")["winner"]
    # Frozen observed-only choice. No new selection, future robot poses or fit.
    motion = predict(observed, 49, winner["window"], winner["damping_tau_s"],
                     np.arange(1, 31)/15, winner["family"]).astype(np.float32)
    np.testing.assert_allclose(motion, s.array(dest+"/selected_motion_policy_v2_15hz.npy"), atol=0, rtol=0)
    diagnosis = s.json(improved+"/observed_diagnosis.json")[camera]
    model_files = s.files.copy()
    saved = s.array(branch+"/predictions/future_3d.npy")
    processors = [s.path(branch+f"/predictions/group_{g:02d}/processor_inputs.pt") for g in range(3)]
    if not evaluation:
        return ExperimentSample("fmb", camera, frames, np.arange(-2, 1)/10, uv, xyz, meta["instruction"],
                                observed["ids"], np.arange(1, 21)/10, observed["K"], metadata=meta,
                                source_files=s.files, model_input_files=model_files,
                                legacy_processor_files=processors, saved_prediction=saved)
    ref = s.array(base+"/evaluation/future_reference.npz")
    rgb = s.array(base+"/evaluation/future_rgb.npy")[1:]
    expected = s.array(dest+"/predictions_and_projections.npz")
    metadata = dict(meta, title="FMB · "+camera, primary="Selected_policy_plus_bounded_Molmo",
                    shown_indices=list(range(0, 24, 3)),
                    geometry="camera_t0, moving camera with frozen observed hand-eye transform",
                    reference="AllTracker + sensor depth scale hypothesis + robot camera poses; estimated 3D",
                    metric_3d_ground_truth=False, conditional_3d=True,
                    history_indices=meta["history_source_indices"], future_indices=meta["future_source_indices"],
                    old_video=dest+"/viz/forecast_comparison.mp4", legacy_metric_file=dest+"/metrics.json",
                    old_panel_crop=[0,512,512,512],
                    motion_policy=winner,
                    limitation="Основной вариант использует кинематический прогноз по прошлым позам робота и ограниченную поправку MolmoMotion. Это не чистый выход нейросети; масштаб depth и hand-eye оценены.")
    return ExperimentSample("fmb", camera, frames, np.arange(-2, 1)/10, uv, xyz, meta["instruction"],
                            observed["ids"], ref["times"], ref["K"], ref["camera_c2w"],
                            metadata=metadata, source_files=s.files, model_input_files=model_files,
                            legacy_processor_files=processors, saved_prediction=saved,
                            evaluation=dict(rgb=rgb, xyz=ref["GT_3D_est"], uv=ref["GT_2D_est"],
                                            mask3=ref["common_mask3d"], mask2=ref["common_mask2d"],
                                            motion=motion, robot_initial=observed["camera_points"][-1],
                                            cap=diagnosis["camera_relative_target_drift_last10_p90_mm"]/1000,
                                            expected=expected))


def finish(sample, raw):
    t = sample.future_times
    h = sample.points_3d_history
    initial = h[-1]
    original = align_future(raw, initial, t)
    robot_initial = sample.evaluation["robot_initial"]
    selected = align_future(sample.evaluation["motion"], robot_initial, t)
    correction = np.median(original-initial[:, None], axis=0)
    correction *= np.minimum(1, sample.evaluation["cap"]/np.maximum(np.linalg.norm(correction, axis=-1), 1e-12))[:, None]
    methods = {"Selected_policy_plus_bounded_Molmo": selected+correction[None],
               "v3_Molmo_H3": original, "Observed_selected_motion_policy": selected,
               "v3_Static": np.repeat(initial[:, None], 20, axis=1),
               "v3_CV": initial[:, None]+((h[-1]-h[0])/.2)[:, None]*t[None, :, None]}
    return methods, {name: project_future(xyz, sample.camera_intrinsics, sample.camera_poses)
                     for name, xyz in methods.items()}
