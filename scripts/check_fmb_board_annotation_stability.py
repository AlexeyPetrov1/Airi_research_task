"""Stress the automatic board landmark extraction against mask thresholds."""

from __future__ import annotations

import json

import numpy as np

from calibrate_fmb_board_k import cad_features, extract
from calibrate_fmb_multi_boards_k import DATA, EPISODES
from probe_fmb_tcp_tip_k import OUT


FRAMES = (0, 16, 32)


def main() -> None:
    _, names, _ = cad_features(1)
    rows, failures = [], []
    for number in EPISODES:
        source = np.load(DATA/f"1_M_L_3_vertical_n_{number}.npy", allow_pickle=True).item()
        images = source["obs/side_1"]
        for frame in FRAMES:
            baseline = extract(images[frame], 1)[0]
            for top_v in (80, 100, 120):
                for dark_v in (90, 100, 110):
                    try:
                        alternate = extract(images[frame], 1, top_v=top_v,
                                            dark_v=dark_v)[0]
                        offset = np.linalg.norm(alternate-baseline, axis=1)
                        rows.append({"episode": number, "frame": frame,
                                     "top_V_threshold": top_v,
                                     "hole_max_V_threshold": dark_v,
                                     "offset_px_by_feature": dict(zip(names, offset.tolist()))})
                    except (ValueError, IndexError) as exc:
                        failures.append({"episode": number, "frame": frame,
                                         "top_V_threshold": top_v,
                                         "hole_max_V_threshold": dark_v,
                                         "reason": str(exc)})
        del source
    array = np.array([[item["offset_px_by_feature"][name] for name in names]
                      for item in rows])
    summary = {}
    for label, subset in (("top_outer_corners", array[:, :4]),
                          ("front_bottom_tangencies", array[:, 4:6]),
                          ("hole_centers", array[:, 6:]),
                          ("all", array)):
        values = subset.ravel()
        summary[label] = {"mean_px": float(np.mean(values)),
                          "median_px": float(np.median(values)),
                          "p90_px": float(np.percentile(values, 90)),
                          "max_px": float(np.max(values))}
    result = {"status": "AUTOMATIC_SEGMENTATION_STABILITY_NOT_MANUAL_REPEATABILITY",
              "source_episodes": EPISODES,
              "frames_each_episode": FRAMES,
              "threshold_grid": {"top_V": [80, 100, 120],
                                 "hole_max_V": [90, 100, 110]},
              "attempts": len(EPISODES)*len(FRAMES)*9,
              "successes": len(rows), "failures": failures,
              "summary": summary,
              "per_variant": rows,
              "limitations": ["Threshold variation tests algorithmic stability, not independent manual marking.",
                              "It does not cover systematic CAD feature semantic error."]}
    (OUT / "board_annotation_stability.json").write_text(json.dumps(result, indent=2)+"\n",
                                                      encoding="utf8")
    print(json.dumps({"successes": len(rows), "failures": len(failures),
                      "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
