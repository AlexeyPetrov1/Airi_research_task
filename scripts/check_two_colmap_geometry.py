"""Checks real calibration provenance and geometry without touching inference."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dobbe_two_colmap import ROOT, RAW, RUN, A_RUN, read, write, sha, verify_freeze
from validate_dobbe_geometry import CAMERA_TO_LABEL, evaluate


def main():
    freeze = verify_freeze()
    labels = read(RAW / "labels.json")
    poses = np.repeat(np.eye(4)[None], 96, axis=0)
    poses[:, :3, :3] = Rotation.from_quat([labels[str(i)]["quats"] for i in range(96)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(96)]
    source = ROOT / "runs/dobbe_colmap_clean_rerun_v1/causal_validation_correspondences.json"
    points = read(source)
    assert sum(len(item["selected"]) for item in points) == 22
    results = {}
    for b in ("A", "B"):
        k = np.load(RUN / b / "K.npy")
        candidate = {"params": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]], "model": "PINHOLE"}
        results[b] = {kind: evaluate(points, poses, candidate, "c2w", kind) for kind in ("z", "ray")}
        # Numerical independent path: history XYZ must use exactly shared UV/Z and labels.
        uv = np.load(RUN / "shared/history_uv.npy")
        z = np.load(RUN / "shared/history_depth_m.npy")
        hp = np.load(RUN / "shared/poses_c2w.npy")
        check = np.empty((3, 8, 3))
        for ti in range(3):
            for pi in range(8):
                optical = np.linalg.inv(k) @ np.r_[uv[ti, pi], 1.] * z[ti, pi]
                world = hp[ti, :3, :3] @ CAMERA_TO_LABEL @ optical + hp[ti, :3, 3]
                check[ti, pi] = CAMERA_TO_LABEL.T @ hp[-1, :3, :3].T @ (world-hp[-1, :3, 3])
        np.testing.assert_allclose(check, np.load(RUN / b / "history.npy"), atol=5e-8, rtol=1e-6)
    # The clean prefix used by B must have exactly A's decoded RGB pixels/hashes.
    for i, expected in read(RUN / "protocol.json")["rgb_sha256"].items():
        assert sha(RUN / f"official/images/{int(i):04d}.png") == expected
        assert sha(A_RUN.parent / f"images/main/{int(i):04d}.png") == expected
    write(RUN / "geometry_audit.json", {"status": "PASS_K_ONLY_ARTIFACT_AUDIT", "calibration_validation": "unvalidated candidates; not a calibration certification",
          "correspondences_sha256": sha(source), "correspondence_count": 22, "external_diagnostics": results,
          "independent_unprojection_transform_check": "PASS", "same_calibration_RGB_prefix": "PASS", "only_K_differs": "PASS",
          "limits": "22 automatic feature correspondences, depth-Z/ray ambiguity and uncertain labels; do not treat a smaller diagnostic error as certification"})
    print("Geometry audit PASS; external depth/labels diagnostics saved", flush=True)
    for b in ("A", "B"):
        print(b, results[b]["z"], flush=True)


if __name__ == "__main__":
    main()
