"""Run one pretrained monocular metric model on selected FMB side_1 frames.

The RGB stream is stored as BGR bytes in the source npy. Predictions are
saved in camera coordinates, without any scale fitting to sensor depth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT.parent / "third_party"
DATA = ROOT / "runs/sharerobot_fmb_episode_5201"
STEPS = (0, 37, 130, 135, 140, 145)


def load_model(name: str):
    if name in ("moge2", "moge3"):
        sys.path.insert(0, str(EXTERNAL / "MoGe"))
        if name == "moge2":
            from moge.model.v2 import MoGeModel
            path = ROOT.parent / "models/moge-2-vitl/model.pt"
        else:
            from moge.model.v3 import MoGeModel
            path = ROOT.parent / "models/moge-3-vitl/model.pt"
        return MoGeModel.from_pretrained(path).cuda().eval()
    if name == "unidepthv2":
        sys.path.insert(0, str(EXTERNAL / "UniDepth"))
        from unidepth.models import UniDepthV2

        path = ROOT.parent / "models/unidepth-v2-vits14"
        model = UniDepthV2.from_pretrained(str(path)).cuda().eval()
        model.resolution_level = 0
        return model
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["moge2", "moge3", "unidepthv2"], required=True)
    parser.add_argument("--aspect-restore", action="store_true",
                        help="Undo official 640x480 to 256x256 RGB aspect stretch")
    parser.add_argument("--fov-candidate", action="store_true",
                        help="MoGe only: use candidate 256px horizontal FOV")
    args = parser.parse_args()
    if args.fov_candidate and args.model not in ("moge2", "moge3"):
        parser.error("--fov-candidate applies only to MoGe")
    model = load_model(args.model)
    source = np.load(DATA / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    arrays = {"steps": np.asarray(STEPS, dtype=np.int32)}
    summary = {"model": args.model, "steps": list(STEPS),
               "preprocessing": "source BGR to RGB; no sensor calibration or depth scale fit",
               "metric_geometry_status": "MODEL_PREDICTION_UNVALIDATED"}
    if args.model == "moge3":
        summary["refine_steps"] = 3
        summary["checkpoint_revision"] = "184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be"
        summary["checkpoint_sha256"] = "9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925"
    if args.aspect_restore:
        summary["preprocessing"] += "; inverse aspect stretch 256x256 to 256x192"
    if args.fov_candidate:
        summary["preprocessing"] += "; MoGe fov_x from candidate fx=152.0836px"
    for step in STEPS:
        rgb = cv2.cvtColor(source["obs/side_1"][step], cv2.COLOR_BGR2RGB)
        if args.aspect_restore:
            rgb = cv2.resize(rgb, (256, 192), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).cuda()
        with torch.inference_mode():
            if args.model in ("moge2", "moge3"):
                fov = float(np.degrees(2 * np.arctan(256 / (2 * 152.0836)))) if args.fov_candidate else None
                infer_kwargs = {"num_tokens": 1200, "use_fp16": True,
                                "apply_mask": False, "fov_x": fov}
                if args.model == "moge3":
                    infer_kwargs["refine_steps"] = 3
                output = model.infer(tensor.float() / 255, **infer_kwargs)
                depth = output["depth"].float().cpu().numpy()
                points = output["points"].float().cpu().numpy()
                intrinsics = output["intrinsics"].float().cpu().numpy()
                intrinsics[0, :] *= rgb.shape[1]
                intrinsics[1, :] *= rgb.shape[0]  # MoGe reports normalized K.
                confidence = None
            else:
                output = model.infer(tensor)
                depth = output["depth"][0, 0].float().cpu().numpy()
                points = output["points"][0].permute(1, 2, 0).float().cpu().numpy()
                intrinsics = output["intrinsics"][0].float().cpu().numpy()
                confidence = output["confidence"][0, 0].float().cpu().numpy()
        if args.aspect_restore:
            depth = cv2.resize(depth, (256, 256), interpolation=cv2.INTER_LINEAR)
            points = cv2.resize(points, (256, 256), interpolation=cv2.INTER_LINEAR)
            if confidence is not None:
                confidence = cv2.resize(confidence, (256, 256), interpolation=cv2.INTER_LINEAR)
            intrinsics[1, :] *= 256 / 192
        if depth.shape != (256, 256) or points.shape != (256, 256, 3):
            raise ValueError((step, depth.shape, points.shape))
        arrays[f"depth_{step}"] = depth.astype("float32")
        arrays[f"points_{step}"] = points.astype("float32")
        arrays[f"K_{step}"] = intrinsics.astype("float32")
        if confidence is not None:
            arrays[f"confidence_{step}"] = confidence.astype("float32")
        row = {"step": step, "K_px": intrinsics.tolist(),
               "depth_median_m": float(np.median(depth[np.isfinite(depth)])),
               "depth_min_m": float(np.nanmin(depth)),
               "depth_max_m": float(np.nanmax(depth)),
               "point_depth_max_abs_m": float(np.nanmax(np.abs(points[..., 2] - depth)))}
        summary.setdefault("frames", []).append(row)
        print(args.model, step, row["depth_median_m"], row["K_px"], flush=True)
        del output, tensor
        torch.cuda.empty_cache()
    suffix = ("_aspect" if args.aspect_restore else "") + ("_fov" if args.fov_candidate else "")
    np.savez_compressed(DATA / f"{args.model}{suffix}_metric_predictions.npz", **arrays)
    (DATA / f"{args.model}{suffix}_metric_predictions.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
