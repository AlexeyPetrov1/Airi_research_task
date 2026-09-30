"""Run the released H3/F30 checkpoint once per distinct WorldTrack input."""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from molmo_motion import MolmoMotion
from decode import strict_decode


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CHECKPOINT = REPO / "data/checkpoints/MolmoMotion-4B-H3-F30"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(stage, status):
    status["stage"] = stage
    status["process_peak_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    if torch.cuda.is_available():
        status["cuda_peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        status["cuda_peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
    (ROOT / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"stage": stage, "ram_gib": status["process_peak_ram_gib"], "elapsed_s": status.get("elapsed_seconds")}), flush=True)


def main():
    first = time.perf_counter()
    preflight = json.loads((ROOT / "preflight.json").read_text())
    inputs_cpu = {
        label: torch.load(ROOT / f"processor_{label}.pt", map_location="cpu", weights_only=False)
        for label in ("A", "B")
    }
    if torch.equal(inputs_cpu["A"]["input_ids"], inputs_cpu["B"]["input_ids"]):
        labels = ("A",)
    else:
        labels = ("A", "B")
    if not torch.equal(inputs_cpu["A"]["images"], inputs_cpu["B"]["images"]):
        raise ValueError("Visual inputs differ")
    with np.load(ROOT / "episode.npz") as ep:
        w2c_t0 = ep["w2c_t0"].astype(np.float64)

    torch.manual_seed(0)
    status = {
        "success": False,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "episode": preflight["source_clip"],
        "history_frames": preflight["history_frames"],
        "future_frames": preflight["future_frames"],
        "source_point_ids": preflight["source_point_ids"],
        "horizon": 30,
        "labels_run": list(labels),
        "checkpoint_config_sha256": sha(CHECKPOINT / "config.yaml"),
        "checkpoint_model_bytes": (CHECKPOINT / "model.pt").stat().st_size,
        "checkpoint_model_sha256": sha(CHECKPOINT / "model.pt"),
        "checkpoint_hf_revision": "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "processor_inputs_sha256": {label: sha(ROOT / f"processor_{label}.pt") for label in labels},
        "per_run": {},
    }
    try:
        snapshot("load_checkpoint_cpu", status)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = MolmoMotion.from_pretrained(str(CHECKPOINT))
        finally:
            torch.set_default_dtype(old_dtype)
        snapshot("move_checkpoint_to_gpu", status)
        model._internal = model._internal.cuda().eval()
        torch.cuda.synchronize()
        snapshot("checkpoint_ready", status)
        for label in labels:
            torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(0)
            run_start = time.perf_counter()
            cpu = inputs_cpu[label]
            gpu = {k: v.cuda() if torch.is_tensor(v) else v for k, v in cpu.items()}
            snapshot(f"generate_{label}", status)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.predict_trajectory(**gpu)
            torch.cuda.synchronize()
            raw = result.future_text
            (ROOT / f"model_output_{label}_raw.txt").write_text(raw)
            deltas = strict_decode(raw)
            absolute = result.future_3d.detach().cpu().numpy()
            if absolute.shape != (8, 30, 3) or not np.isfinite(absolute).all():
                raise ValueError(f"Invalid {label} absolute output")
            expected = deltas + cpu["anchor_3d"].numpy()
            if not np.allclose(absolute, expected, atol=1e-6, rtol=0):
                raise ValueError(f"{label} raw text and result.future_3d disagree")
            if label == "A":
                future_world = absolute.astype(np.float64)
            else:
                future_world = (absolute.astype(np.float64) - w2c_t0[:3, 3]) @ w2c_t0[:3, :3]
            np.savez_compressed(
                ROOT / f"prediction_{label}.npz",
                quantized_deltas=deltas,
                native_xyz=absolute,
                future_world=future_world,
                source_point_ids=np.asarray(preflight["source_point_ids"], dtype=np.int32),
                source_future_frames=np.asarray(preflight["future_frames"], dtype=np.int32),
            )
            status["per_run"][label] = {
                "seconds": round(time.perf_counter() - run_start, 2),
                "raw_chars": len(raw),
                "strict_positions": 240,
                "peak_cuda_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "peak_cuda_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
                "process_peak_ram_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
                "raw_sha256": sha(ROOT / f"model_output_{label}_raw.txt"),
                "prediction_sha256": sha(ROOT / f"prediction_{label}.npz"),
            }
            status["elapsed_seconds"] = round(time.perf_counter() - first, 2)
            snapshot(f"complete_{label}", status)
            del gpu, result
        status["success"] = True
        status["elapsed_seconds"] = round(time.perf_counter() - first, 2)
        snapshot("complete", status)
    except Exception as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["traceback"] = traceback.format_exc(limit=8)
        status["elapsed_seconds"] = round(time.perf_counter() - first, 2)
        snapshot("failed", status)
        raise


if __name__ == "__main__":
    main()
