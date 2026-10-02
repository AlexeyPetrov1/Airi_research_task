"""Assemble paired, descriptive FMB ablation effects from saved evaluations."""

from __future__ import annotations

import csv
import json

import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, write_json


def main() -> None:
    rows = []
    for episode, run in RUNS.items():
        history_uv = np.load(run / "history_points_2d.npy")
        history_motion = float(np.median(np.linalg.norm(history_uv[-1] - history_uv[0], axis=1)))
        base = json.loads((run / "metrics.json").read_text(encoding="utf8"))["metrics"]
        nominal = base["MolmoMotion"]
        stationary = base["stationary"]
        velocity = base["constant_velocity"]
        for variant in VARIANTS:
            report = json.loads((run / SUITE / variant / "evaluation.json").read_text(encoding="utf8"))
            metric = report["metrics"]
            alt = report["alternative_dense_flow_metrics"]
            rows.append({
                "episode": episode, "variant": variant,
                "ADE_2D_px": metric["ADE_2D_px"], "FDE_2D_px": metric["FDE_2D_px"],
                "delta_ADE_vs_nominal_px": metric["ADE_2D_px"] - nominal["ADE_2D_px"],
                "delta_FDE_vs_nominal_px": metric["FDE_2D_px"] - nominal["FDE_2D_px"],
                "ADE_vs_stationary_px": metric["ADE_2D_px"] - stationary["ADE_2D_px"],
                "ADE_vs_velocity_px": metric["ADE_2D_px"] - velocity["ADE_2D_px"],
                "FDE_vs_stationary_px": metric["FDE_2D_px"] - stationary["FDE_2D_px"],
                "FDE_vs_velocity_px": metric["FDE_2D_px"] - velocity["FDE_2D_px"],
                "alternative_GT_ADE_2D_px": alt["ADE_2D_px"],
                "alternative_GT_FDE_2D_px": alt["FDE_2D_px"],
                "mean_prediction_change_vs_nominal_px": report["mean_forecast_2d_change_vs_nominal_px"],
                "counterfactual_ADE_not_goal_success": variant == "text_counterfactual",
                "median_H3_endpoint_motion_px": history_motion,
                "coverage": metric["valid_pairs"],
            })
    dest = ROOT / "runs/fmb_ablation_conclusions_v1"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "paired_effects.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(dest / "paired_effects.json", {
        "rows": rows,
        "status": "DESCRIPTIVE_TWO_EPISODE_COMPARISON",
        "statistical_unit": "FMB demonstration; point-frame pairs are not independent trials",
        "counterfactual_note": "Error against the original future measures response change, not success at the changed action",
        "selection_note": "Do not select a geometry, K, scale, point order or wording using the evaluation future and then report its error as an unbiased estimate",
    })
    print(len(rows), "paired ablation rows")


if __name__ == "__main__":
    main()
