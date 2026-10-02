"""RGB/TCP calibration probe with one common tip offset across three episodes.

The independent-offset fit yielded similar ~150 mm TCP->tip distances. This
probe tests whether sharing the offset removes enough ambiguity to identify K.
No sensor depth or future GT is used until after RGB fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_fmb_cad_pnp import K as K_NOMINAL
from probe_fmb_tcp_tip_k import fit, score


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "molmo-motion/runs/fmb_effective_k_256_calibration"
EPISODES = (0, 4, 8)


def main() -> None:
    previous = json.loads((OUT / "multi_episode_tcp_tip_K_probe.json").read_text())
    records = previous["records"]
    train = [r for r in records if r["split"] == "train"]
    hold = [r for r in records if r["split"] == "holdout"]
    poses = np.asarray([r["tcp_pose"] for r in train])
    uv = np.asarray([r["uv_rgb_px"] for r in train])
    hold_pose = np.asarray([r["tcp_pose"] for r in hold])
    hold_uv = np.asarray([r["uv_rgb_px"] for r in hold])
    old_x = np.asarray(previous["best_fit"]["x"])
    mean_offset = np.mean(old_x[10:].reshape(3, 3), axis=0)
    seed = np.r_[old_x[:10], mean_offset]
    starts = []
    for k_name, k in (("TCP_independent", np.asarray(previous["best_fit"]["K"])),
                      ("nominal", K_NOMINAL),
                      ("board_peg", np.asarray(json.loads(
                          (OUT / "board_peg_table_contact_K_probe.json").read_text())
                          ["best_fit"]["K"]))):
        trial = seed.copy()
        trial[:4] = [np.log(k[0, 0]), np.log(k[1, 1]), k[0, 2], k[1, 2]]
        fitted = fit(poses, uv, trial)
        fitted["seed_K_name"] = k_name
        starts.append(fitted)
    starts.sort(key=lambda row: row["cost"])
    best = starts[0]
    bx = np.asarray(best["x"])
    nominal_seed = seed.copy()
    nominal_seed[:4] = [np.log(K_NOMINAL[0, 0]),
                        np.log(K_NOMINAL[1, 1]),
                        K_NOMINAL[0, 2], K_NOMINAL[1, 2]]
    nominal = fit(poses, uv, nominal_seed, free_k=False)
    subsets = []
    for exclude in EPISODES:
        selected = [r for r in train if r["episode"] != exclude]
        p = np.asarray([r["tcp_pose"] for r in selected])
        v = np.asarray([r["uv_rgb_px"] for r in selected])
        trial = fit(p, v, bx)
        subsets.append({"excluded_episode": exclude, "K": trial["K"],
                        "train_rmse_px": trial["rmse_px"],
                        "success": trial["success"]})
    rng = np.random.default_rng(5201)
    monte_carlo = []
    for _ in range(50):
        trial = fit(poses, uv+rng.normal(0, 2., uv.shape), bx)
        if trial["success"]:
            k = np.asarray(trial["K"])
            monte_carlo.append([k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
    result = {"status": "DIAGNOSTIC_RGB_TCP_SHARED_OFFSET_K",
              "source_episodes": EPISODES,
              "assumption": "All three grasps have one identical TCP-to-free-tip offset.",
              "mean_independent_offset_seed_m": mean_offset.tolist(),
              "train_count": len(train), "holdout_count": len(hold),
              "seed_fits": starts, "best_fit": best,
              "best_holdout": score(bx, hold_pose, hold_uv),
              "fixed_nominal_K_fit": nominal,
              "fixed_nominal_K_holdout": score(np.asarray(nominal["x"]),
                                               hold_pose, hold_uv),
              "leave_one_episode_out": subsets,
              "monte_carlo_tip_sigma_px": 2.,
              "monte_carlo_K_fx_fy_cx_cy": monte_carlo,
              "monte_carlo_K_5_50_95_percentiles": (
                  np.percentile(monte_carlo, [5, 50, 95], axis=0).tolist()
                  if monte_carlo else None),
              "limitations": [
                  "RGB mask extremum is not a verified persistent material point.",
                  "Grasp offsets can differ among episodes; sharing them is a testable assumption.",
                  "Temporal RGB/TCP synchronization is assumed.",
                  "No sensor depth or future tracking GT enters the fit."]}
    (OUT / "shared_tcp_offset_K_probe.json").write_text(
        json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"K": best["K"], "train_rmse_px": best["rmse_px"],
                      "holdout_rmse_px": result["best_holdout"]["rmse_px"],
                      "nominal_holdout_rmse_px": result["fixed_nominal_K_holdout"]["rmse_px"],
                      "TCP_tip_offset_m": bx[10:13].tolist(),
                      "subset_K": subsets,
                      "MC_percentiles": result["monte_carlo_K_5_50_95_percentiles"]},
                     indent=2))


if __name__ == "__main__":
    main()
