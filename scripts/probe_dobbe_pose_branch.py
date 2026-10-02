"""Measure the effect of the required portrait RGB rotation on static pairs.

The exporter selects P_old and rotates RGB clockwise for portrait metadata;
for landscape metadata it selects P_new and leaves RGB alone. Both imply
P_new @ CV_TO_OPENGL for rays in the published frames. The old calculation
P_old @ CV_TO_OPENGL forgot the RGB rotation. This script quantifies its effect.
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation

import validate_dobbe_geometry as geometry


def main() -> None:
    output = {}
    for scene in ("main", "second"):
        folder = geometry.ROOT / "data/dobbe_oxe" / (
            "target_raw" if scene == "main" else "second_raw")
        labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
        count = max(max(pair) for pair in geometry.PAIRS[scene]) + 1
        poses = np.repeat(np.eye(4)[None], count, axis=0)
        poses[:, :3, :3] = Rotation.from_quat(
            [labels[str(i)]["quats"] for i in range(count)]).as_matrix()
        poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(count)]
        points = json.loads((geometry.OUT / f"{scene}_static_correspondences.json")
                            .read_text(encoding="utf-8"))
        candidates = geometry.k_candidates(scene)
        selected = ["naive"]
        if scene == "main":
            selected.append("previous_fit_diagnostic")
        result = {}
        for name, p in (("missing_rgb_rotation", geometry.P_OLD),
                        ("exporter_consistent", geometry.P_NEW)):
            geometry.CAMERA_TO_LABEL = p @ geometry.CV_TO_OPENGL
            result[name] = {}
            for k_name in selected:
                result[name][k_name] = geometry.evaluate(
                    points, poses, candidates[k_name], "c2w", "z")
        output[scene] = result
    path = geometry.OUT / "image_rotation_diagnostic.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for scene, variants in output.items():
        for branch, results in variants.items():
            metric = results["naive"]
            print(scene, branch, metric["median_3d_m"],
                  metric["median_reprojection_px"])


if __name__ == "__main__":
    main()
