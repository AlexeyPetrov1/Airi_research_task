"""Bundled author example, preserving the documented first-camera frame issue."""
import numpy as np

from ..geometry import project
from ..io import Source, video_frames
from ..sample import ExperimentSample


def load(config, root, evaluation=True):
    s = Source(root)
    base = "runs/author_davis_bmx_trees_f30"
    sample = "examples/data/davis_bmx_trees"
    meta = s.json(sample+"/meta.json")
    frames = np.stack([s.rgb(sample+f"/frame_t{i:+d}.jpg") for i in (-2, -1, 0)])
    xyz = s.tensor(sample+"/points_3d_history.pt")
    uv = s.tensor(sample+"/points_2d_at_t0.pt")
    k = s.tensor(sample+"/intrinsics_K.pt")
    action = s.path(sample+"/caption.txt").read_text(encoding="utf8").strip()
    model_files = s.files.copy()
    saved = s.array(base+"/prediction.npz")["future_3d"]
    if not evaluation:
        return ExperimentSample("davis", "bmx-trees", frames, np.arange(-2, 1)/24, uv, xyz,
                                action, np.array(meta["point_indices"]), np.arange(1, 31)/24, k,
                                metadata=dict(meta, future_used_for_model_input=False),
                                source_files=s.files, model_input_files=model_files,
                                legacy_processor_files=[s.path(base+"/processor_inputs.pt")], saved_prediction=saved)
    ref = s.array(base+"/ground_truth.npz")
    video = video_frames(s.path(base+"/real_continuation_gt.mp4"))
    # This movie has real frames with legacy tracker dots. Raw evaluation RGB is
    # supplied independently by the repository bootstrap when available.
    raw = root / "evaluation_rgb/davis_bmx_trees.npy"
    if raw.exists():
        video = s.array("evaluation_rgb/davis_bmx_trees.npy")
    if len(video) != 30:
        raise ValueError("Expected the 30 genuine future frames")
    evidence = s.json(base+"/projection_check.json")
    metadata = dict(meta, title="DAVIS · BMX rider", primary="MolmoMotion", source_fps=24,
                    model_fps=15, geometry="author common frame anchored to the first camera",
                    projection_scope="first-camera diagnostic; unavailable per-frame extrinsics",
                    reference="author reconstructed 3D; independent author 2D tracks",
                    metric_3d_ground_truth=False, conditional_3d=True, future_used_for_model_input=False,
                    history_indices=meta["history_frame_indices"], future_indices=ref["future_frame_indices"].tolist(),
                    shown_indices=list(range(8)), old_video=base+"/real_continuation_gt.mp4",
                    legacy_metric_file=base+"/metrics.json", projection_evidence=evidence,
                    limitation="Авторские XYZ привязаны к первой камере. Нет поз для проекции прогноза на движущееся видео; 2D ADE/FDE не вычисляются.")
    return ExperimentSample("davis", "bmx-trees", frames, np.arange(-2, 1)/24, uv, xyz,
                            action, ref["point_indices"], ref["future_seconds_from_t0"], k,
                            metadata=metadata, source_files=s.files, model_input_files=model_files,
                            legacy_processor_files=[s.path(base+"/processor_inputs.pt")],
                            saved_prediction=saved,
                            evaluation=dict(rgb=video, xyz=ref["future_3d"], uv=ref["future_2d"],
                                            mask3=ref["valid"], mask2=ref["valid"],
                                            diagnostic_projection=True))


def finish(sample, raw):
    history = sample.points_3d_history
    steps = np.arange(1, 31, dtype=np.float32)
    methods = {"MolmoMotion": raw,
               "Static": np.broadcast_to(history[-1, :, None], raw.shape).copy(),
               "Constant velocity": history[-1, :, None] + steps[None, :, None]*(history[-1]-history[-2])[:, None]}
    # Display only in the anchored first camera; never score this as video 2D GT.
    uv = {name: project(xyz, sample.camera_intrinsics) for name, xyz in methods.items()}
    return methods, uv
