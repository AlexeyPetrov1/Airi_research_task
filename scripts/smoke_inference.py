"""Load the released checkpoint and optionally forecast one future frame.

This is a setup smoke test on the bundled historical clip, not an evaluation.
Build the CPU model in bf16 to keep peak RAM practical on this WSL VM. The
released weights are cast to the same bf16 used by the author quickstart.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="data/checkpoints/MolmoMotion-4B-H3-F30")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--output-prefix", default="runs/smoke_real_f1")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    assert (checkpoint / "config.yaml").is_file()
    assert (checkpoint / "model.pt").is_file()
    prefix = Path(args.output_prefix).resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)

    status: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "future_horizon": 0 if args.load_only else 1,
        "stage": "start",
        "success": False,
        "seconds": None,
    }
    start = time.perf_counter()

    def update(stage: str) -> None:
        status["stage"] = stage
        status["seconds"] = round(time.perf_counter() - start, 2)
        status["peak_process_ram_gib"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2
        )
        if torch.cuda.is_available():
            status["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            status["peak_cuda_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        print(json.dumps(status, indent=2), flush=True)
        prefix.with_suffix(".json").write_text(json.dumps(status, indent=2) + "\n")

    try:
        update("processor")
        processor = MolmoMotionProcessor.from_pretrained(str(checkpoint))
        update("construct_and_load_bf16_cpu")
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MolmoMotion.from_pretrained(str(checkpoint))
        finally:
            torch.set_default_dtype(previous_dtype)
        update("move_to_gpu")
        model._internal = model._internal.cuda().eval()
        torch.cuda.synchronize()
        update("gpu_loaded")

        if not args.load_only:
            sample = Path(__file__).resolve().parents[1] / "examples/data/davis_bmx_trees"
            frames = [
                Image.open(sample / f"frame_t{i:+d}.jpg").convert("RGB")
                for i in (-2, -1, 0)
            ]
            points_2d = torch.load(sample / "points_2d_at_t0.pt", weights_only=True)
            points_3d = torch.load(sample / "points_3d_history.pt", weights_only=True)
            action = (sample / "caption.txt").read_text().strip()
            inputs = processor(
                history_frames=frames,
                points_2d_at_t0=points_2d,
                points_3d_history=points_3d,
                action=action,
                future_horizon=1,
            )
            inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
            update("predict_trajectory")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = model.predict_trajectory(**inputs)
            future = output.future_3d.cpu().numpy()
            assert future.shape == (8, 1, 3), future.shape
            assert np.isfinite(future).all()
            np.savez_compressed(prefix.with_suffix(".npz"), future_3d=future)
            status["prediction_shape"] = list(future.shape)
        status["success"] = True
        update("complete")
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc(limit=8)
        update("failed")
        raise


if __name__ == "__main__":
    main()
