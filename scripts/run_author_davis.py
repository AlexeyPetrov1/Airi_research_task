"""Run the released H3/F30 model once on the bundled DAVIS bmx-trees episode."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor
from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples/data/davis_bmx_trees"
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
RUN = ROOT / "runs/author_davis_bmx_trees_f30"
HORIZON = 30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    inputs_dir = RUN / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    for path in SAMPLE.iterdir():
        if path.is_file():
            shutil.copy2(path, inputs_dir / path.name)

    start = time.perf_counter()
    status = {
        "episode": "davis_bmx_trees/bike_rider/t0=2/point_indices=[9,12,17,18,27,29,32,34]",
        "horizon": HORIZON,
        "success": False,
        "stage": "start",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint_config_sha256": sha256(CHECKPOINT / "config.yaml"),
        "checkpoint_model_bytes": (CHECKPOINT / "model.pt").stat().st_size,
        "checkpoint_hf_revision": "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629",
        "torch_version": torch.__version__,
    }

    def update(stage: str) -> None:
        status["stage"] = stage
        status["elapsed_seconds"] = round(time.perf_counter() - start, 2)
        status["peak_process_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
        if torch.cuda.is_available():
            status["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            status["peak_cuda_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        (RUN / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2), flush=True)

    try:
        update("processor")
        processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
        update("construct_and_load_bf16_cpu")
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MolmoMotion.from_pretrained(str(CHECKPOINT))
        finally:
            torch.set_default_dtype(previous_dtype)
        update("move_to_gpu")
        model._internal = model._internal.cuda().eval()
        torch.cuda.synchronize()
        update("prepare_inputs")
        frames = [Image.open(SAMPLE / f"frame_t{i:+d}.jpg").convert("RGB") for i in (-2, -1, 0)]
        points_2d = torch.load(SAMPLE / "points_2d_at_t0.pt", weights_only=True)
        points_3d = torch.load(SAMPLE / "points_3d_history.pt", weights_only=True)
        inputs = processor(
            history_frames=frames,
            points_2d_at_t0=points_2d,
            points_3d_history=points_3d,
            action=(SAMPLE / "caption.txt").read_text().strip(),
            future_horizon=HORIZON,
        )
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in inputs.items()}, RUN / "processor_inputs.pt")
        status["processor_input_shapes"] = {k: list(v.shape) for k, v in inputs.items() if torch.is_tensor(v)}
        inputs = {k: v.cuda() if torch.is_tensor(v) else v for k, v in inputs.items()}
        update("predict_trajectory")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict_trajectory(**inputs)
        torch.cuda.synchronize()
        (RUN / "model_output_raw.txt").write_text(output.future_text)
        status["raw_output_chars"] = len(output.future_text)
        parsed = parse_tracks_text(output.future_text)
        if parsed is None:
            raise ValueError("Model text does not contain a complete parseable <tracks> block")
        delta, parsed_vis = tracks_to_array(parsed, num_points=8, num_frames=HORIZON, start_timestamp=3.0)
        status["parsed_visibility_count"] = int(np.asarray(parsed_vis).sum())
        status["parsed_visibility_shape"] = list(np.asarray(parsed_vis).shape)
        if np.asarray(parsed_vis).shape != (8, HORIZON) or not np.asarray(parsed_vis).all():
            raise ValueError("Model text lacks at least one of the 8 x 30 point/timestamp entries")
        future = output.future_3d.cpu().numpy()
        if future.shape != (8, HORIZON, 3) or not np.isfinite(future).all():
            raise ValueError(f"Invalid future: shape={future.shape}, finite={np.isfinite(future).all()}")
        if not np.allclose(future, delta + inputs["anchor_3d"].cpu().numpy(), atol=1e-4):
            raise ValueError("Parsed raw text differs from output.future_3d")
        np.savez_compressed(RUN / "prediction.npz", future_3d=future)
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
