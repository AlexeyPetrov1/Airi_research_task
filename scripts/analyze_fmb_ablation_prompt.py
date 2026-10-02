"""Count changed millimetre-quantized history components in FMB ablations."""

from __future__ import annotations

import csv
import json

import numpy as np

from analyze_fmb_prompt_sensitivity import quantized
from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, write_json


def main() -> None:
    rows = []
    for episode, run in RUNS.items():
        base = np.load(run / "history_sensor_3d.npy")
        base_q = quantized(base)
        base_anchor = base[-1, 0]
        base_pred = np.load(run / "prediction_15hz.npy")
        for variant in VARIANTS:
            folder = run / SUITE / variant
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
            history = np.load(folder / "history_3d.npy")
            q = quantized(history)
            row = {
                "episode": episode, "variant": variant,
                "changed_quantized_history_components_out_of_72": int(np.sum(q != base_q)),
                "max_abs_integer_delta_mm": int(np.max(np.abs(q - base_q))),
                "anchor_shift_mm": float(np.linalg.norm(history[-1, 0] - base_anchor) * 1000),
                "text_changed": manifest["action"] != json.loads((run / "experiment_config.json").read_text(encoding="utf8"))["action_text_exact_passed_to_model"],
            }
            prediction_path = folder / "prediction_15hz.npy"
            if prediction_path.exists():
                prediction = np.load(prediction_path)
                row["unique_future_configurations"] = int(np.unique(prediction.transpose(1, 0, 2).reshape(30, -1), axis=0).shape[0])
                row["mean_3D_forecast_change_from_nominal_mm"] = float(np.linalg.norm(prediction - base_pred, axis=2).mean() * 1000)
                row["final_mean_3D_forecast_change_from_nominal_mm"] = float(np.linalg.norm(prediction[:, -1] - base_pred[:, -1], axis=1).mean() * 1000)
            rows.append(row)
    dest = ROOT / "runs/fmb_ablation_prompt_analysis_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "comparison.csv").open("w", newline="", encoding="utf8") as f:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(dest / "comparison.json", {
        "rows": rows,
        "source_format": "src/molmo_motion/processor.py::_format_history_tracks, history minus final-frame point-0 anchor, components rounded to integer millimetres",
        "note": "Changed tokenized history components are an input difference, not a causal decomposition of forecast error. Text-only variants leave numeric history unchanged.",
    })
    print(len(rows), "prompt records")


if __name__ == "__main__":
    main()
