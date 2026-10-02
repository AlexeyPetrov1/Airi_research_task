"""Resample board hole features to assess RGB-only K stability."""

from __future__ import annotations

import csv
import json

import numpy as np

from calibrate_fmb_board_k import HOLDOUT_FEATURES, TRAIN, cad_features, extract
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from calibrate_fmb_two_boards_k import fit_multi
from probe_fmb_tcp_tip_k import OUT


def main() -> None:
    points, names, _ = cad_features(1)
    anchors = np.arange(6)
    holes = np.array([i for i in range(6, len(names)) if i not in HOLDOUT_FEATURES])
    observations = []
    for number in EPISODES:
        source = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        images = source["obs/side_1"]
        observations.append(np.array([extract(images[s], 1)[0] for s in TRAIN]))
        del source
    previous = json.loads((OUT / "multi_board_bundle_calibration.json").read_text(encoding="utf8"))
    seed = np.array(previous["best_fit"]["x"])
    rng = np.random.default_rng(5201)
    rows = []
    for trial in range(100):
        sampled = np.r_[anchors, rng.choice(holes, size=len(holes), replace=True)]
        fit = fit_multi(points, tuple(observations), sampled, seed, max_nfev=120)
        k = np.array(fit["K"])
        rows.append({"trial": trial, "success": fit["success"],
                     "fx_px": k[0, 0], "fy_px": k[1, 1],
                     "cx_px": k[0, 2], "cy_px": k[1, 2],
                     "cost": fit["cost"],
                     "resampled_hole_ids": ",".join(map(str, sampled[6:]))})
    with (OUT / "bootstrap_K.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    array = np.array([[r["fx_px"], r["fy_px"], r["cx_px"], r["cy_px"]]
                      for r in rows if r["success"]])
    summary = {"method": "Resample the 10 training hole-center features with replacement; retain six outer/side anchors and all five board placements.",
               "attempts": 100,
               "successes": len(array),
               "parameter_order": ["fx", "fy", "cx", "cy"],
               "percentiles_5_50_95": np.percentile(array, [5, 50, 95], axis=0).tolist(),
               "note": "Feature bootstrap does not model common systematic errors in virtual corners or hole-center semantics."}
    (OUT / "bootstrap_K_summary.json").write_text(json.dumps(summary, indent=2)+"\n",
                                                   encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
