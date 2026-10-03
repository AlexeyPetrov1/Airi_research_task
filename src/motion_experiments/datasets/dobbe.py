"""Dobb-E conditional diagnostic and the actual rejected geometry branch."""
import numpy as np

from ..geometry import project_future
from ..io import Source, video_frames
from ..sample import ExperimentSample


def load(config, root, evaluation=True):
    s = Source(root)
    base = "runs/dobbe_vipe_v1/C"
    variant = config.get("variant", "pure_vipe")
    folder = base+"/"+variant
    p = s.json(base+"/protocol.json")
    gate = s.json(folder+"/geometry_gate.json")
    frames = np.stack([s.rgb(base+f"/frames/{i:05d}.png") for i in p["history_vipe_indices"]])
    world = s.array(folder+"/points_3d_world.npy")
    pose = s.array(folder+"/c2w_t0.npy")
    uv = s.array(folder+"/points_2d_t0.npy")
    ids = np.array(gate["selected_ids"])
    model_files = s.files.copy()
    failed = [name for name, value in gate["checks"].items() if not value]
    meta = dict(title="Dobb-E · drawer", primary="MolmoMotion", geometry=p["geometry"],
                reference="independent AllTracker; projection conditional on published pose convention and causal ViPE K",
                history_indices=p["history_raw_ids"], future_indices=p["future_raw_ids_evaluation_only"],
                future_used_for_model_input=False, metric_3d_ground_truth=False,
                source_fps=30, model_fps=15, shown_indices=list(range(8)), variant=variant,
                geometry_gate=gate, old_video=folder+"/evaluation/future_overlay.mp4",
                old_panel_crop=[0,56,512,512],
                legacy_metric_file=folder+"/evaluation/metrics.json",
                limitation="2D ошибка условна: позы камер и K не подтверждены физической калибровкой. Эталонного 3D нет. Hybrid остановлен по static_median.")
    sample = ExperimentSample("dobbe", "C/"+variant, frames, np.array(p["history_raw_ids"])/30,
                              uv, world, p["action"], ids, np.arange(1, 31)/15,
                              c2w_at_t0=pose, metadata=meta, source_files=s.files,
                              model_input_files=model_files,
                              status="SKIPPED_GEOMETRY_GATE" if failed else "COMPLETE",
                              failure_reason=", ".join(failed) if failed else None)
    if failed:
        return sample
    saved = s.array(folder+"/prediction_15hz.npy")
    sample.saved_prediction = saved
    if not evaluation:
        sample.source_files = s.files
        return sample
    projection = s.array(folder+"/evaluation/conditional_projection.npz")
    observed = s.array(folder+"/evaluation/observed_future_tracks.npz")
    geometry = s.array(base+"/geometry_all.npz")
    fx, fy, cx, cy = geometry["intrinsics"][-1]
    sample.camera_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    sample.camera_poses = projection["relative_poses"]
    rgb = video_frames(s.path(folder+"/evaluation/future_15hz.mp4"))
    sample.saved_prediction = saved
    sample.evaluation = dict(rgb=rgb[1:], uv=projection["observed_uv"].transpose(1, 0, 2),
                             mask2=projection["valid"].T, visibility2=observed["visibility"][1:].T,
                             xyz=None, mask3=None, expected=projection,
                             initial_camera=s.array(folder+"/points_3d_camera_t0_diagnostic.npy")[-1])
    sample.source_files = s.files
    return sample


def finish(sample, raw):
    initial = sample.evaluation["initial_camera"]
    xyz = {"MolmoMotion": raw, "Static": np.repeat(initial[:, None], 30, axis=1)}
    uv = {name: project_future(values, sample.camera_intrinsics, sample.camera_poses)
          for name, values in xyz.items()}
    # Original convention marks camera-plane/behind-camera forecasts invalid.
    for name, values in xyz.items():
        for t in range(30):
            pose = sample.camera_poses[t+1]
            cam = (values[:, t]-pose[:3, 3]) @ pose[:3, :3]
            uv[name][cam[:, 2] <= 0, t] = np.nan
    return xyz, uv
