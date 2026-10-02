"""Exploratory MolmoMotion H3/F30 run using the explicitly unverified CAD PnP input."""

from __future__ import annotations

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
RUN = ROOT / "runs/sharerobot_fmb_episode_5201/cad_history_candidate"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
ACTION = "Insert the red rectangular peg into the matching hole on the blue board."


def main() -> None:
    started = time.perf_counter()
    status = {"status": "EXPLORATORY_UNVALIDATED_GEOMETRY",
              "source_steps": [130,135,140], "future_horizon":30,
              "ground_truth_30_steps_available":False,"success":False}

    def update(stage: str) -> None:
        status["stage"] = stage
        status["elapsed_seconds"] = round(time.perf_counter()-started,2)
        status["peak_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,2)
        if torch.cuda.is_available():
            status["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated()/2**30,2)
        (RUN/"molmo_candidate_status.json").write_text(json.dumps(status,indent=2),encoding="utf-8")
        print(stage, status["elapsed_seconds"], flush=True)

    try:
        manifest = json.loads((RUN/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"].startswith("CANDIDATE_ONLY")
        points = np.load(RUN/"points_3d_history_candidate.npy")
        uv = np.load(RUN/"points_2d_at_t0_candidate.npy")
        assert points.shape == (3,8,3) and uv.shape == (8,2)
        assert np.isfinite(points).all() and (points[...,2]>0).all()
        frames = [Image.open(RUN/f"frame_{step}.png").convert("RGB")
                  for step in (130,135,140)]
        update("load_processor")
        processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
        update("load_model_bf16")
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MolmoMotion.from_pretrained(str(CHECKPOINT))
        finally:
            torch.set_default_dtype(previous_dtype)
        update("move_to_gpu")
        model._internal = model._internal.cuda().eval()
        update("prepare_inputs")
        batch = processor(history_frames=frames,
                          points_2d_at_t0=torch.from_numpy(uv),
                          points_3d_history=torch.from_numpy(points),
                          action=ACTION,future_horizon=30)
        status["processor_shapes"]={k:list(v.shape) for k,v in batch.items() if torch.is_tensor(v)}
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        update("predict")
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            output=model.predict_trajectory(**batch)
        (RUN/"molmo_candidate_output_raw.txt").write_text(output.future_text,encoding="utf-8")
        future=output.future_3d.detach().cpu().numpy()
        np.save(RUN/"molmo_candidate_future.npy",future)
        status["prediction_shape"]=list(future.shape)
        status["prediction_all_finite"]=bool(np.isfinite(future).all())
        status["raw_output_chars"]=len(output.future_text)
        status["success"]=bool(future.shape==(8,30,3) and np.isfinite(future).all())
        update("complete")
    except Exception as exc:
        status["error"]=f"{type(exc).__name__}: {exc}"
        status["traceback"]=traceback.format_exc(limit=8)
        update("failed")
        raise


if __name__ == "__main__":
    main()
