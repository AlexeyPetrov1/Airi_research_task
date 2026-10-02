"""Detect temporally constant decoded 3D forecasts without changing the metric."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get(
    "FMB_QUANT_RUN",
    ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))


def main() -> None:
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    gt = np.load(RUN / "gt_2d.npy")
    p0 = np.load(RUN / "points_2d_at_t0.npy")
    variants = {}
    for variant in ("nominal", "simple", "focal", "pnp", "cad_silhouette", "nominal_repeat"):
        folder = RUN if variant == "nominal" else RUN / "variants" / variant
        path = folder / "prediction_15hz.npy"
        if not path.exists():
            continue
        forecast = np.load(path)
        if forecast.shape != (8, 30, 3):
            raise ValueError(f"Unexpected forecast shape for {variant}: {forecast.shape}")
        unique = [int(np.unique(forecast[i], axis=0).shape[0]) for i in range(8)]
        token_payloads = []
        raw = (folder / "raw_model_output.txt").read_text(encoding="utf8")
        if '<tracks coords="' in raw:
            body = raw.split('<tracks coords="', 1)[1].split('"', 1)[0]
            token_payloads = [part.strip().split(" ", 1)[1] for part in body.split(";")]
        variants[variant] = {
            "unique_3D_positions_per_point_over_30_steps": unique,
            "all_8_points_temporally_constant": all(value == 1 for value in unique),
            "max_coordinate_range_over_time_m": float(np.ptp(forecast, axis=1).max()),
            "raw_time_rows": len(token_payloads),
            "raw_unique_coordinate_payloads": len(set(token_payloads)),
        }
    result = {"source_file": config["source_file"], "t0": config["t0"],
              "GT_mean_final_displacement_px": float(np.linalg.norm(gt[:, -1]-p0, axis=1).mean()),
              "method": "Exact unique decoded positions and raw coordinate payloads across 30 future steps",
              "variants": variants}
    (RUN / "temporal_forecast_diagnostic.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
