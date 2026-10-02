"""Inspect how sub-mm K changes alter quantized MolmoMotion history tokens."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("FMB_QUANT_RUN", ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))


def quantized(history: np.ndarray) -> np.ndarray:
    anchor = history[-1, 0]
    delta = history-anchor
    return np.asarray([int(round(float(value)*1000)) for value in delta.ravel()],
                      dtype=int).reshape(history.shape)


def main() -> None:
    base = np.load(RUN / "history_sensor_3d.npy")
    base_q = quantized(base)
    output = {"processor_format": "history XYZ minus last-frame point-0 anchor; each component round(meters*1000) to integer",
              "source_code": "src/molmo_motion/processor.py::_format_history_tracks",
              "history_coordinates": int(base_q.size), "variants": {}}
    for variant, history_name in (("simple", "history_sensor_3d_K_simple.npy"),
                                  ("focal", "history_sensor_3d_K_focal_0925.npy")):
        history = np.load(RUN / history_name)
        q = quantized(history)
        changed = np.argwhere(q != base_q)
        records = [{"history_index": int(h), "point_id": int(p),
                    "axis": "XYZ"[int(axis)], "nominal_integer": int(base_q[h, p, axis]),
                    "variant_integer": int(q[h, p, axis])}
                   for h, p, axis in changed]
        row = {"median_input_XYZ_shift_mm": float(np.median(
                   np.linalg.norm(history-base, axis=2))*1000),
               "anchor_shift_mm": float(np.linalg.norm(history[-1, 0]-base[-1, 0])*1000),
               "quantized_history_components_changed": len(records),
               "changed_components": records}
        pred_path = RUN / "variants" / variant / "prediction_15hz.npy"
        if pred_path.exists():
            a, b = np.load(RUN / "prediction_15hz.npy"), np.load(pred_path)
            distance = np.linalg.norm(a-b, axis=2)
            row.update({"mean_3D_forecast_disagreement_mm": float(distance.mean()*1000),
                        "final_mean_3D_forecast_disagreement_mm": float(distance[:, -1].mean()*1000)})
        output["variants"][variant] = row
    (RUN / "prompt_quantization_sensitivity.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
