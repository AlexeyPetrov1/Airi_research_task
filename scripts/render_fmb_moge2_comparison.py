"""Render actual future RGB with saved sensor and MoGe-2 forecasts."""

from __future__ import annotations

import json

import cv2
import numpy as np

from render_fmb_study_media import BLUE, ORANGE, GREEN, header, point, video
from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json


COLORS = {"sensor": ORANGE, "moge2_raw": GREEN, "moge2_scaled": (205, 75, 175)}
LABELS = {"sensor": "FMB sensor Z", "moge2_raw": "MoGe-2 RGB depth",
          "moge2_scaled": "MoGe-2 / H3 scale"}


def main() -> None:
    for episode, run in RUNS.items():
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source = np.load(ROOT.parent / "data/fmb/single_object_manipulation_dataset" /
                         config["source_file"], allow_pickle=True).item()
        gt = np.load(run / "gt_2d.npy")
        mask = np.load(run / "validity_mask.npy")
        predictions = {"sensor": np.load(run / "prediction_2d.npy")}
        for variant in ("moge2_raw", "moge2_scaled"):
            predictions[variant] = np.load(run / SUITE / variant / "prediction_2d.npy")
        frames = []
        for index, step in enumerate(config["future_steps"]):
            bgr = source["obs/side_1"][step]
            observed = bgr.copy()
            for point_id in range(8):
                if mask[point_id, index]:
                    point(observed, gt[point_id, index], point_id, BLUE, "circle")
            header(observed, "Observed RGB continuation",
                   f"step {step}; t = {(index + 1) / 10:.1f} s; blue = RGB proxy", bottom=True)
            panels = [observed]
            for name, trajectory in predictions.items():
                panel = bgr.copy()
                outside = 0
                for point_id in range(8):
                    if mask[point_id, index]:
                        point(panel, gt[point_id, index], point_id, BLUE, "circle")
                    outside += not point(panel, trajectory[point_id, index], point_id,
                                         COLORS[name], "cross")
                error = float(np.linalg.norm(trajectory[:, index] - gt[:, index], axis=1)
                              [mask[:, index]].mean())
                header(panel, LABELS[name], f"mean {error:.1f} px; outside {outside}/8",
                       bottom=True)
                panels.append(panel)
            frames.append(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
        dest = run / "moge2_history_v1"
        video(dest / "forecast_comparison.mp4", frames)
        selected = [4, 9, 14, 19]
        cv2.imwrite(str(dest / "forecast_comparison_overview.png"),
                    np.vstack((np.hstack([frames[index] for index in selected[:2]]),
                               np.hstack([frames[index] for index in selected[2:]]))))
        write_json(dest / "forecast_media_manifest.json", {
            "episode": episode, "source_file": config["source_file"],
            "future_steps": config["future_steps"], "GT": "frozen RGB-derived 2D proxy",
            "forecasts": list(predictions), "all_predictions_loaded_from_saved_files": True,
            "video_frame_count": len(frames) * 3, "panel_size_px": [256, 256],
            "out_of_image_predictions_counted_not_clipped_in_metrics": True,
        })
        print(episode, len(frames), "future frames")


if __name__ == "__main__":
    main()
