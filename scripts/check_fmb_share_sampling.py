"""Match locally available ShareRobot PNGs to all four FMB RGB streams.

This only tests observed ShareRobot frames. It never fills missing frames from
an assumed sampling rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_fmb_episode_5201 import phash, ssim


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_fmb_episode_5201"
SHARE = ROOT / "runs" / "sharerobot_transfer_berkeley_ur5_episode_26"
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")


def main() -> None:
    source = np.load(OUT / "1_M_L_3_vertical_n_2.npy", allow_pickle=True).item()
    available = {}
    for path in SHARE.glob("reserve_fmb_episode_5201_frame_*.png"):
        frame_index = int(path.stem.rsplit("_", 1)[1])
        available[frame_index] = path
    if not available:
        raise FileNotFoundError("No local ShareRobot PNGs for episode_5201")

    expected = {
        "linspace_145": np.linspace(0, 144, 30, dtype=int).tolist(),
        "linspace_148": np.linspace(0, 147, 30, dtype=int).tolist(),
    }
    matches = []
    for frame_index, path in sorted(available.items()):
        share = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        all_scores = []
        by_camera = {}
        for camera in CAMERAS:
            stored = source[f"obs/{camera}"]
            assert stored.shape == (148, 256, 256, 3)
            options = {}
            # Source .npy is described as BGR, but ShareRobot may have copied
            # those bytes into an RGB-labelled image without converting them.
            for mode, frames in (("as_stored", stored), ("bgr_to_rgb", stored[:, :, :, ::-1])):
                mae = np.abs(frames.astype(np.int16) - share).mean(axis=(1, 2, 3))
                ranking = np.argsort(mae)
                options[mode] = {
                    "best_step": int(ranking[0]),
                    "best_mae": float(mae[ranking[0]]),
                    "second_step": int(ranking[1]),
                    "second_mae": float(mae[ranking[1]]),
                    "at_linspace_145_step": float(mae[expected["linspace_145"][frame_index]]),
                    "at_linspace_148_step": float(mae[expected["linspace_148"][frame_index]]),
                }
                all_scores.extend((float(score), camera, mode, int(step)) for step, score in enumerate(mae))
            by_camera[camera] = options
        all_scores.sort()
        best_mae, best_camera, best_mode, best_step = all_scores[0]
        best_rgb = source[f"obs/{best_camera}"][best_step]
        if best_mode == "bgr_to_rgb":
            best_rgb = best_rgb[:, :, ::-1]
        entry = {
            "share_frame_index": frame_index,
            "share_file": str(path.relative_to(ROOT)),
            "best_camera": best_camera,
            "best_channel_mode": best_mode,
            "best_source_step": best_step,
            "best_pixel_mae": best_mae,
            "global_top_5": [{"camera": camera, "channel_mode": mode,
                              "step": step, "pixel_mae": score}
                             for score, camera, mode, step in all_scores[:5]],
            "phash_distance": int(np.count_nonzero(phash(best_rgb) != phash(share.astype(np.uint8)))),
            "ssim": ssim(best_rgb, share.astype(np.uint8)),
            "by_camera": by_camera,
            "hypotheses": {
                name: {"predicted_step": steps[frame_index],
                       "residual_steps": best_step - steps[frame_index]}
                for name, steps in expected.items()
            },
        }
        matches.append(entry)

    result = {
        "source_file": "1_M_L_3_vertical_n_2.npy",
        "source_step_count": 148,
        "camera_keys": list(CAMERAS),
        "available_share_png_count": len(matches),
        "share_pngs_expected_by_description": 30,
        "comparisons_per_share_png": 148 * len(CAMERAS) * 2,
        "total_comparisons": len(matches) * 148 * len(CAMERAS) * 2,
        "method": "Mean absolute pixel error for as-stored and BGR-to-RGB source channels; SSIM and pHash on best pair",
        "matches": matches,
        "matched_steps_monotonic": all(a["best_source_step"] < b["best_source_step"]
                                       for a, b in zip(matches, matches[1:])),
        "linspace_sequences": expected,
        "conclusion_limit": "Only the locally available ShareRobot PNGs are tested; these cannot establish the entire 30-frame sampling sequence or a true 145-frame source length.",
    }
    (OUT / "four_camera_sampling_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    print(json.dumps({"available_share_png_count": len(matches),
                      "total_comparisons": result["total_comparisons"],
                      "matches": [{"share_frame_index": m["share_frame_index"],
                                   "best_camera": m["best_camera"],
                                   "best_channel_mode": m["best_channel_mode"],
                                   "best_source_step": m["best_source_step"],
                                   "best_pixel_mae": m["best_pixel_mae"],
                                   "other_camera_best_mae": {camera: min(z["best_mae"] for z in d.values())
                                                             for camera, d in m["by_camera"].items()
                                                             if camera != m["best_camera"]},
                                   "hypotheses": m["hypotheses"]}
                                  for m in matches]}, indent=2))


if __name__ == "__main__":
    main()
