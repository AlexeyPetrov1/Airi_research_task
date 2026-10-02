"""Audit duplicate and depth-edge sensitivity of saved causal RGB matches.

This does not estimate K or tune the geometry gate. It applies one fixed
stricter quality filter to both scenes and scores the same K candidates on
identical filtered observations, plus a leave-one-frame-pair-out check.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

import validate_dobbe_geometry as geometry


def poses_from_labels(scene: str) -> np.ndarray:
    folder = geometry.ROOT / "data/dobbe_oxe" / (
        "target_raw" if scene == "main" else "second_raw")
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    count = max(max(pair) for pair in geometry.PAIRS[scene]) + 1
    poses = np.repeat(np.eye(4)[None], count, axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(count)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(count)]
    return poses


def filtered_group(group: dict) -> tuple[dict, dict]:
    saved = group["selected"]
    kept = []
    near_duplicate_count = 0
    for point in saved:
        a = np.asarray(point["rgb_a"])
        b = np.asarray(point["rgb_b"])
        near_duplicate = any(
            np.linalg.norm(a - np.asarray(other["rgb_a"])) < 3.0
            or np.linalg.norm(b - np.asarray(other["rgb_b"])) < 3.0
            for other in kept)
        if near_duplicate:
            near_duplicate_count += 1
            continue
        kept.append(point)
    clean = [point for point in kept
             if point["depth_spread_a_m"] <= .02
             and point["depth_spread_b_m"] <= .02]
    return ({**group, "selected": clean},
            {"pair": group["pair"], "original": len(saved),
             "spatially_unique": len(kept),
             "depth_spread_le_2cm": len(clean),
             "near_duplicates_removed": near_duplicate_count})


def main() -> None:
    output = {"selection": {
        "min_distance_any_view_px": 3.0,
        "max_3x3_depth_spread_each_view_m": .02,
        "rule": "Fixed across both scenes and all candidate K; diagnostic only"}}
    for scene in ("main", "second"):
        raw = json.loads((geometry.OUT / f"{scene}_static_correspondences.json")
                         .read_text(encoding="utf-8"))
        pairs = [filtered_group(group) for group in raw]
        filtered = [group for group, _ in pairs]
        counts = [count for _, count in pairs]
        poses = poses_from_labels(scene)
        candidates = geometry.k_candidates(scene)
        selected_k = (["naive", "previous_fit_diagnostic", "colmap_pinhole_all",
                       "colmap_opencv_all"] if scene == "main" else
                      ["naive", "colmap_pinhole_all_fixedpp_init62_107",
                       "colmap_simple_pinhole_all_fixedpp_rectified"])
        results = {}
        for name in selected_k:
            candidate = candidates[name]
            full = geometry.evaluate(filtered, poses, candidate, "c2w", "z")
            drop_each = {}
            for i, group in enumerate(filtered):
                if not group["selected"]:
                    continue
                subset = filtered[:i] + filtered[i + 1:]
                metric = geometry.evaluate(subset, poses, candidate, "c2w", "z")
                drop_each[f"{group['pair'][0]}-{group['pair'][1]}"] = {
                    key: metric[key] for key in ("n_3d", "median_3d_m",
                                               "median_reprojection_px")}
            results[name] = {"full": full, "leave_one_pair_out": drop_each}
        output[scene] = {"pairs": counts,
                         "original_observations": sum(c["original"] for c in counts),
                         "filtered_observations": sum(c["depth_spread_le_2cm"]
                                                      for c in counts),
                         "candidates": results}
    path = geometry.OUT / "correspondence_robustness.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for scene in ("main", "second"):
        info = output[scene]
        print(scene, f"{info['original_observations']} -> "
              f"{info['filtered_observations']} observations")
        for name, result in info["candidates"].items():
            metric = result["full"]
            print(" ", name, metric["median_3d_m"],
                  metric["median_reprojection_px"])


if __name__ == "__main__":
    main()
