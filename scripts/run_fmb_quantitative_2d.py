"""Run the frozen FMB t0=126 H3/F30 input with MolmoMotion in WSL GPU env."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotion, MolmoMotionProcessor


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
MODEL_REVISION = "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629"
MODEL_PT_SHA256_PREVIOUSLY_VERIFIED = "506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze() -> None:
    frozen = json.loads((RUN / "input_freeze.json").read_text(encoding="utf8"))
    for name, expected in frozen["sha256"].items():
        actual = sha256(RUN / name)
        if actual != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    preflight = json.loads((RUN / "preflight.json").read_text(encoding="utf8"))
    assert preflight["decision"] == "PASS_WITH_CALIBRATION_UNCERTAINTY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("nominal", "nominal_repeat", "simple", "focal", "pnp", "cad_silhouette"),
                        default="nominal")
    args = parser.parse_args()
    verify_freeze()
    if args.variant == "nominal":
        dest = RUN
        history_path = RUN / "history_sensor_3d.npy"
    else:
        dest = RUN / "variants" / args.variant
        dest.mkdir(parents=True, exist_ok=True)
        history_path = (RUN / {"nominal_repeat": "history_sensor_3d.npy",
                               "simple": "history_sensor_3d_K_simple.npy",
                               "focal": "history_sensor_3d_K_focal_0925.npy",
                               "pnp": "history_pnp_3d.npy",
                               "cad_silhouette": "history_cad_silhouette_tcp_3d.npy"}[args.variant])
    if (dest / "prediction_15hz.npy").exists():
        raise FileExistsError(f"Prediction already exists: {dest}")
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    points = np.load(history_path)
    uv = np.load(RUN / "points_2d_at_t0.npy")
    assert points.shape == (3, 8, 3) and uv.shape == (8, 2)
    assert np.isfinite(points).all() and (points[..., 2] > 0).all()
    frames = [Image.open(RUN / f"frame_{step}.png").convert("RGB")
              for step in config["history_steps"]]
    started = time.perf_counter()
    status = {"variant": args.variant, "status": "RUNNING", "success": False,
              "source_steps": config["history_steps"], "t0": config["t0"],
              "action_text": config["action_text_exact_passed_to_model"],
              "geometry_file": history_path.name,
              "geometry_sha256": sha256(history_path),
              "model_id": "allenai/MolmoMotion-4B-H3-F30",
              "model_revision": MODEL_REVISION,
              "model_pt_sha256_previously_verified": MODEL_PT_SHA256_PREVIOUSLY_VERIFIED,
              "checkpoint_config_sha256": sha256(CHECKPOINT / "config.yaml"),
              "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "seed": 0, "dtype": "bfloat16",
              "history_source_nominal_hz": 10,
              "model_prediction_assumed_hz": 15,
              "temporal_domain_shift": "Three real history observations are 0.1 s apart; model typical training/prediction timing is about 15 Hz. Processor has no explicit timestamp input."}

    def update(stage: str) -> None:
        status["stage"] = stage
        status["elapsed_seconds"] = round(time.perf_counter()-started, 2)
        status["peak_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20, 3)
        if torch.cuda.is_available():
            status["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated()/2**30, 3)
            status["peak_cuda_reserved_gib"] = round(torch.cuda.max_memory_reserved()/2**30, 3)
        (dest / "model_run.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf8")
        print(args.variant, stage, status["elapsed_seconds"], flush=True)

    try:
        assert torch.cuda.is_available()
        torch.manual_seed(0)
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
                          points_3d_history=torch.from_numpy(points),
                          action=config["action_text_exact_passed_to_model"],
                          future_horizon=30)
        status["processor_shapes"] = {key: list(value.shape) for key, value in batch.items()
                                      if torch.is_tensor(value)}
        batch = {key: value.cuda() if torch.is_tensor(value) else value
                 for key, value in batch.items()}
        torch.cuda.synchronize()
        start_predict = time.perf_counter()
        update("predict")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model.predict_trajectory(**batch)
        torch.cuda.synchronize()
        status["prediction_seconds"] = round(time.perf_counter()-start_predict, 2)
        raw = output.future_text
        future = output.future_3d.detach().cpu().numpy()
        (dest / "raw_model_output.txt").write_text(raw, encoding="utf8")
        np.save(dest / "prediction_15hz.npy", future)
        status["prediction_shape"] = list(future.shape)
        status["prediction_finite_pairs"] = int(np.isfinite(future).all(axis=2).sum())
        status["prediction_completeness"] = status["prediction_finite_pairs"] / 240
        status["all_positive_Z"] = bool((future[..., 2] > 0).all())
        status["all_zero_tensor"] = bool(np.all(future == 0))
        status["raw_output_chars"] = len(raw)
        status["raw_output_sha256"] = sha256(dest / "raw_model_output.txt")
        status["parser_status"] = "FULL_8x30x3" if (
            future.shape == (8, 30, 3) and np.isfinite(future).all()) else "INCOMPLETE"
        status["success"] = bool(status["parser_status"] == "FULL_8x30x3" and
                                 not status["all_zero_tensor"])
        status["status"] = "COMPLETE" if status["success"] else "INVALID_OUTPUT"
        update("complete")
    except Exception as error:
        status["status"] = "FAILED"
        status["error"] = f"{type(error).__name__}: {error}"
        status["traceback"] = traceback.format_exc(limit=8)
        update("failed")
        raise


if __name__ == "__main__":
    main()
