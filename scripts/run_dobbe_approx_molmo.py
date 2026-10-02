"""Run one exploratory MolmoMotion forecast from the sealed HoNY cup history."""

from __future__ import annotations

import hashlib
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/dobbe_rgbd_study/approx_history"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    freeze = json.loads((RUN / "input_freeze.json").read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_EXPLORATORY_APPROXIMATE"
    for name, expected in freeze["sha256"].items():
        if sha256(RUN / name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    if sha256(CHECKPOINT / "config.yaml") != freeze["checkpoint_config_sha256"]:
        raise RuntimeError("Checkpoint config changed")
    result_path = RUN / "prediction_15hz.npy"
    if result_path.exists():
        raise FileExistsError(result_path)
    xyz = np.load(RUN / "points_3d_history_candidate.npy")
    uv = np.load(RUN / "points_2d_at_t0_candidate.npy")
    assert xyz.shape == (3, 8, 3) and uv.shape == (8, 2)
    assert np.isfinite(xyz).all() and (xyz[..., 2] > 0).all()
    frames = [Image.open(RUN / f"rgb_{i:04d}.png").convert("RGB")
              for i in freeze["history_frames"]]
    started = time.perf_counter()
    status = {"status": "RUNNING", "success": False,
              "experiment": "exploratory transferred-K Dobb-E episode_3651",
              "history_frames": freeze["history_frames"],
              "geometry_file": "points_3d_history_candidate.npy",
              "geometry_sha256": freeze["sha256"][
                  "points_3d_history_candidate.npy"],
              "input_freeze_sha256": sha256(RUN / "input_freeze.json"),
              "model_id": freeze["model_id"],
              "model_revision": freeze["model_revision"],
              "checkpoint_model_pt_sha256": freeze[
                  "checkpoint_model_pt_sha256"],
              "action_text": freeze["action_text_exact_passed_to_model"],
              "future_horizon": freeze["future_horizon"],
              "seed": freeze["seed"], "dtype": freeze["dtype"],
              "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
                     else None}

    def update(stage: str) -> None:
        status["stage"] = stage
        status["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        status["peak_ram_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
        if torch.cuda.is_available():
            status["peak_cuda_allocated_gib"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 3)
        (RUN / "model_run.json").write_text(json.dumps(status, indent=2) + "\n",
                                             encoding="utf-8")
        print(stage, status["elapsed_seconds"], flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable")
        torch.manual_seed(freeze["seed"])
        update("load_processor")
        processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
        update("load_model_bf16")
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MolmoMotion.from_pretrained(str(CHECKPOINT))
        finally:
            torch.set_default_dtype(old_dtype)
        model._internal = model._internal.cuda().eval()
        update("prepare_inputs")
        batch = processor(history_frames=frames,
                          points_2d_at_t0=torch.from_numpy(uv),
                          points_3d_history=torch.from_numpy(xyz),
                          action=freeze["action_text_exact_passed_to_model"],
                          future_horizon=freeze["future_horizon"])
        status["processor_shapes"] = {
            key: list(value.shape) for key, value in batch.items()
            if torch.is_tensor(value)}
        batch = {key: value.cuda() if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        torch.cuda.synchronize()
        prediction_start = time.perf_counter()
        update("predict")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict_trajectory(**batch)
        torch.cuda.synchronize()
        status["prediction_seconds"] = round(
            time.perf_counter() - prediction_start, 2)
        (RUN / "raw_model_output.txt").write_text(output.future_text,
                                                   encoding="utf-8")
        future = output.future_3d.detach().cpu().numpy()
        np.save(result_path, future)
        status["prediction_shape"] = list(future.shape)
        status["prediction_finite_pairs"] = int(
            np.isfinite(future).all(axis=2).sum())
        status["all_zero_tensor"] = bool(np.all(future == 0))
        status["all_positive_Z"] = bool((future[..., 2] > 0).all())
        status["raw_output_sha256"] = sha256(RUN / "raw_model_output.txt")
        status["prediction_sha256"] = sha256(result_path)
        status["success"] = bool(future.shape == (8, 30, 3)
                                 and np.isfinite(future).all()
                                 and not status["all_zero_tensor"])
        status["status"] = "COMPLETE" if status["success"] else "INVALID_OUTPUT"
        update("complete")
    except Exception as exc:
        status["status"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc(limit=8)
        update("failed")
        raise


if __name__ == "__main__":
    main()
