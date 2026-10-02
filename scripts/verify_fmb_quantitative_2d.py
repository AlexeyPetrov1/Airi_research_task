"""Independently parse raw MolmoMotion tracks for the frozen FMB 2D experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))


def verify_one(variant: str) -> dict:
    dest = RUN if variant == "nominal" else RUN / "variants" / variant
    history_path = RUN / {"nominal": "history_sensor_3d.npy",
                          "nominal_repeat": "history_sensor_3d.npy",
                          "simple": "history_sensor_3d_K_simple.npy",
                          "focal": "history_sensor_3d_K_focal_0925.npy",
                          "pnp": "history_pnp_3d.npy",
                          "cad_silhouette": "history_cad_silhouette_tcp_3d.npy"}[variant]
    raw = (dest / "raw_model_output.txt").read_text(encoding="utf8")
    tracks = parse_tracks_text(raw)
    if tracks is None:
        raise ValueError("No complete <tracks> block")
    delta, visibility = tracks_to_array(tracks, num_points=8, num_frames=30,
                                        start_timestamp=3.0)
    history = np.load(history_path)
    prediction = np.load(dest / "prediction_15hz.npy")
    reconstructed = np.asarray(delta)+history[-1, 0]
    max_error = float(np.max(np.abs(reconstructed-prediction)))
    status = {"variant": variant,
              "raw_track_block_complete": True,
              "parsed_visibility_shape": list(np.asarray(visibility).shape),
              "parsed_visible_count": int(np.asarray(visibility).sum()),
              "expected_positions": 240,
              "prediction_shape": list(prediction.shape),
              "all_prediction_finite": bool(np.isfinite(prediction).all()),
              "all_prediction_positive_Z": bool((prediction[..., 2] > 0).all()),
              "all_zero_tensor": bool(np.all(prediction == 0)),
              "max_abs_text_vs_tensor_difference_m": max_error,
              "status": "PASS" if max_error < 1e-4 and np.asarray(visibility).all()
                        and prediction.shape == (8, 30, 3) and np.isfinite(prediction).all()
                        and not np.all(prediction == 0) else "FAIL"}
    (dest / "parser_validation.json").write_text(json.dumps(status, indent=2) + "\n",
                                                 encoding="utf8")
    print(json.dumps(status, indent=2))
    if status["status"] != "PASS":
        raise ValueError("Raw text does not match saved 8x30x3 forecast")
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("nominal", "nominal_repeat", "simple", "focal", "pnp", "cad_silhouette", "all"),
                        default="nominal")
    args = parser.parse_args()
    variants = ("nominal", "nominal_repeat", "simple", "focal", "pnp", "cad_silhouette") if args.variant == "all" else (args.variant,)
    for variant in variants:
        if variant == "nominal" or (RUN / "variants" / variant / "prediction_15hz.npy").exists():
            verify_one(variant)


if __name__ == "__main__":
    main()
