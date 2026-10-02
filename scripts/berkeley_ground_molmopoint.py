"""Genuine official MolmoPoint pointing using only each Berkeley scene's t0.

The installed Vid checkpoint implements native image pointing, so the default
passes exactly one original observed RGB image and uses extract_image_points.
This removes video sampling ambiguity and never duplicates MolmoMotion H3.
Author loader, constrained generation and official patch-coordinate decoder
are reused. No manual coordinates or segmentation-derived point selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
CHECKPOINT = WORKSPACE / "models/MolmoPoint-Vid-4B"
REVISION = "b331ed6c6352e6db967325493c6dae515541a919"
AUTHOR = ROOT / "data_generation/third_party/sam3/molmo2_pointing.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


def checkpoint_info(checkpoint: Path, require_weights: bool) -> dict:
    config = checkpoint / "config.json"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (checkpoint / name).exists()]
    if require_weights and missing:
        raise FileNotFoundError(f"Pinned model download incomplete; missing {missing}")
    revision_file = checkpoint / ".cache/huggingface/download/config.json.metadata"
    revision = revision_file.read_text().splitlines()[0] if revision_file.exists() else REVISION
    if revision != REVISION:
        raise ValueError(f"Expected pinned MolmoPoint revision {REVISION}, found {revision}")
    code_names = ("config.json", "processor_config.json", "modeling_molmo_point.py",
                  "processing_molmo2.py", "image_processing_molmo2.py", "chat_template.jinja")
    return {"model_id": "allenai/MolmoPoint-Vid-4B", "checkpoint": str(checkpoint),
            "checkpoint_revision": revision, "checkpoint_code_sha256": {name: sha256(checkpoint/name) for name in code_names},
            "checkpoint_shards": {name: (checkpoint/name).stat().st_size if (checkpoint/name).exists() else None for name in shards},
            "author_loader": str(AUTHOR.relative_to(ROOT)), "author_loader_sha256": sha256(AUTHOR)}


def numpy_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, list):
        return [numpy_metadata(item) for item in value]
    if isinstance(value, tuple):
        return tuple(numpy_metadata(item) for item in value)
    if isinstance(value, dict):
        return {key: numpy_metadata(item) for key, item in value.items()}
    return value


def prepare_scene(scene: Path, processor, mode: str) -> tuple[dict, dict, dict]:
    import torch
    from berkeley_preprocess import load_observed
    if (scene / "predictions/model_run.json").exists():
        raise RuntimeError("Cannot change observed grounding after a saved prediction receipt")
    if (scene / "observed/molmopoint_grounding.json").exists():
        raise FileExistsError("Refusing to overwrite actual MolmoPoint grounding; inspect existing receipt")
    rgb, indices, metadata = load_observed(scene)
    target = metadata["target_object"]
    if target not in ("blue cup", "ranch bottle"):
        raise ValueError(f"Unexpected task target: {target}")
    prompt = f"point to {target} gripped and picked up by the robot gripper"
    image_path = scene / "observed/molmopoint_t0.png"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(rgb[-1], cv2.COLOR_RGB2BGR)):
        raise IOError(image_path)
    if mode == "image":
        content = {"type": "image", "image": str(image_path.resolve())}
    else:
        # This optional author-video API receives one real t0 frame, not H3.
        import imageio.v2 as imageio
        video_path = scene / "observed/molmopoint_t0_single_frame.mp4"
        with imageio.get_writer(video_path, fps=5, codec="libx264", quality=10, macro_block_size=2) as writer:
            writer.append_data(rgb[-1])
        try:
            import decord
            decord.bridge.set_bridge("native")
        except ImportError:
            pass
        content = {"type": "video", "video": str(video_path.resolve())}
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, content]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True, padding=True,
                                           return_pointing_metadata=True)
    pointing_metadata = numpy_metadata(inputs.pop("metadata"))
    required = {"token_pooling", "subpatch_mapping", "image_sizes"} if mode == "image" else {
        "token_pooling", "subpatch_mapping", "timestamps", "video_size"}
    if not required.issubset(pointing_metadata):
        raise ValueError(f"Missing native {mode} pointing metadata: {required-set(pointing_metadata)}")
    if mode == "image":
        if len(pointing_metadata["image_sizes"]) != 1:
            raise ValueError("Only the original t0 image may be sent to MolmoPoint")
    elif not np.allclose(pointing_metadata["timestamps"], 0):
        raise ValueError("Single-frame t0 video unexpectedly contains other timestamps")
    # Saving the model's actual processed token IDs provides auditability without
    # keeping another large copy of image patch tensors.
    np.save(scene / "observed/molmopoint_input_ids.npy", inputs["input_ids"].cpu().numpy())
    info = {"scene": scene.name, "source_frame": int(indices[-1]),
            "t0_source_frame_index": int(indices[-1]), "source_camera": "observation.images.image",
            "object_phrase": target, "prompt": prompt, "future_used": False,
            "input_modality": mode, "input_real_frame_count": 1,
            "input_source_indices": [int(indices[-1])], "frame_0_maps_to_source_frame": int(indices[-1]),
            "model_input_rgb_sha256": sha256(image_path), "observed_rgb_sha256": sha256(scene / "observed/rgb.npy"),
            "input_shapes": {key: list(value.shape) for key, value in inputs.items() if torch.is_tensor(value)},
            "pointing_metadata_shapes": {key: list(value.shape) if isinstance(value, np.ndarray) else str(type(value).__name__)
                                         for key, value in pointing_metadata.items()},
            "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(scene / "observed/molmopoint_preparation.json", info)
    return inputs, pointing_metadata, info


def select_point(points, mode: str, width: int, height: int) -> tuple[np.ndarray, list, int]:
    values = np.asarray(points, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Official MolmoPoint generated no valid target points")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError(f"Unexpected native point decoder shape: {values.shape}")
    # extract_image_points column1=image index. extract_video_points column1=
    # timestamp (not sampled-frame index). For exactly one t0 both must be zero.
    eligible = (np.isfinite(values).all(axis=1) & np.isclose(values[:, 1], 0)
                & (values[:, 2] >= 0) & (values[:, 2] < width)
                & (values[:, 3] >= 0) & (values[:, 3] < height))
    if not eligible.any():
        raise ValueError(f"No generated point lies on observed t0 inside the image: {values.tolist()}")
    chosen = int(np.flatnonzero(eligible)[0])
    return values[chosen, 2:].astype(np.float32), values.tolist(), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-mode", choices=("image", "single-frame-video"), default="image")
    parser.add_argument("--prepare-only", action="store_true", help="CPU processor validation; performs no generation")
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(WORKSPACE / ".cache/huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers
    if not transformers.__version__.startswith("4.57."):
        raise RuntimeError(f"Author MolmoPoint requires Transformers 4.57.x; got {transformers.__version__}")
    provenance = checkpoint_info(args.checkpoint, require_weights=not args.prepare_only)
    started = time.monotonic()
    if args.prepare_only:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(args.checkpoint), trust_remote_code=True, local_files_only=True)
        model = None
    else:
        sys.path.insert(0, str(AUTHOR.parent))
        from molmo2_pointing import load_molmo2
        model, processor = load_molmo2(model_id=str(args.checkpoint), device=args.device, dtype=torch.bfloat16)
    load_seconds = time.monotonic()-started
    for scene in args.scene_dir:
        inputs, pointing_metadata, info = prepare_scene(scene, processor, args.input_mode)
        if args.prepare_only:
            print(json.dumps({"scene": scene.name, "prepared": True, "modality": args.input_mode,
                              "input_shapes": info["input_shapes"]}), flush=True)
            continue
        device = next(model.parameters()).device
        inputs = {key: value.to(device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            generated_ids = model.generate(**inputs,
                logits_processor=model.build_logit_processor_from_inputs(inputs), max_new_tokens=512, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.monotonic()-started
        generated_tokens = generated_ids[:, inputs["input_ids"].size(1):]
        text = processor.post_process_image_text_to_text(generated_tokens, skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False)[0]
        (scene / "observed/molmopoint_raw_output.txt").write_text(text, encoding="utf8")
        np.save(scene / "observed/molmopoint_generated_token_ids.npy", generated_tokens.cpu().numpy())
        if args.input_mode == "image":
            points = model.extract_image_points(text, pointing_metadata["token_pooling"],
                                                 pointing_metadata["subpatch_mapping"], pointing_metadata["image_sizes"])
            width, height = pointing_metadata["image_sizes"][0]
            decoder = "model.extract_image_points"
        else:
            points = model.extract_video_points(text, pointing_metadata["token_pooling"],
                pointing_metadata["subpatch_mapping"], pointing_metadata["timestamps"], pointing_metadata["video_size"])
            width, height = pointing_metadata["video_size"]
            decoder = "model.extract_video_points"
        point, candidates, chosen = select_point(points, args.input_mode, width, height)
        result = {**info, **provenance, "success": True, "point_xy": point.tolist(),
                  "candidate_points": candidates, "chosen_candidate_index": chosen,
                  "selection_rule": "first valid native decoded point at t0 in deterministic generated order",
                  "pointing_method": "actual official MolmoPoint-Vid-4B constrained greedy generation",
                  "coordinate_decoder": decoder, "raw_output": text, "runtime_seconds": elapsed,
                  "model_load_seconds": load_seconds,
                  "peak_gpu_memory_gib": torch.cuda.max_memory_allocated()/2**30 if device.type == "cuda" else 0,
                  "transformers_version": transformers.__version__, "torch_version": torch.__version__,
                  "dtype": "bfloat16", "seed": 0, "max_new_tokens": 512,
                  "logits_processor": "model.build_logit_processor_from_inputs(inputs)",
                  "no_manual_point_or_segmentation_selection": True,
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
        write_json(scene / "observed/molmopoint_grounding.json", result)
        image = cv2.imread(str(scene / "observed/molmopoint_t0.png"))
        for index, record in enumerate(candidates):
            x, y = map(int, np.rint(record[2:]))
            color = (50, 255, 70) if index == chosen else (0, 190, 255)
            cv2.circle(image, (x, y), 7, color, 2, cv2.LINE_AA)
            cv2.putText(image, f"MolmoPoint {index}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        cv2.imwrite(str(scene / "observed/molmopoint_overlay.png"), image)
        print(json.dumps({"scene": scene.name, "point_xy": point.tolist(), "source_frame": info["source_frame"],
                          "runtime_seconds": elapsed, "raw_output": text}), flush=True)
        del inputs, generated_ids, generated_tokens
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
