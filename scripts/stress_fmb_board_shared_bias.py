"""Stress board K against shared RGB feature-localization bias across episodes."""

from __future__ import annotations

import json

import numpy as np

from calibrate_fmb_board_k import HOLDOUT_FEATURES, TRAIN, cad_features, extract
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from calibrate_fmb_two_boards_k import fit_multi
from probe_fmb_tcp_tip_k import OUT


def main() -> None:
    points, names, _ = cad_features(1)
    features = np.array([i for i in range(len(names)) if i not in HOLDOUT_FEATURES])
    observed = []
    for number in EPISODES:
        source = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        images = source["obs/side_1"]
        observed.append(np.array([extract(images[s], 1)[0] for s in TRAIN]))
        del source
    baseline = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    seed = np.array(baseline["best_fit"]["x"])
    rng = np.random.default_rng(5201)
    scenarios = []
    for anchor_sigma in (1., 2.):
        params = []
        for _ in range(100):
            # Same feature displacement in every episode and frame represents
            # a persistent semantic/segmentation bias invisible to frame repeatability.
            common = rng.normal(0, .5, (len(names), 2))
            common[:6] = rng.normal(0, anchor_sigma, (6, 2))
            changed = tuple(a+common[None] for a in observed)
            row = fit_multi(points, changed, features, seed, max_nfev=140)
            if row["success"]:
                k = np.array(row["K"])
                params.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        values = np.array(params)
        scenarios.append({"shared_anchor_sigma_px": anchor_sigma,
                          "shared_hole_center_sigma_px": .5,
                          "attempts": 100, "successes": len(values),
                          "parameter_order": ["fx", "fy", "cx", "cy"],
                          "percentiles_5_50_95":
                              np.percentile(values, [5, 50, 95], axis=0).tolist(),
                          "K_samples": params})
    result = {"status": "SYSTEMATIC_ANNOTATION_BIAS_STRESS_NOT_A_CAMERA_POSTERIOR",
              "method": "Apply one shared landmark offset to all five board placements and all repeated frames, then refit K using RGB only.",
              "scenarios": scenarios,
              "limitations": ["Gaussian offsets are hypothetical systematic-error scenarios.",
                              "Hole-center semantics and CAD fabrication tolerances are not measured.",
                              "No sensor depth or future GT used for fitting or selecting K."]}
    (OUT / "shared_annotation_bias_stress.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf8")
    print(json.dumps([{k: v for k, v in row.items() if k != "K_samples"}
                      for row in scenarios], indent=2))


if __name__ == "__main__":
    main()
