"""Audit the exploratory F2-NeRF transfer, frozen MolmoMotion run, and sparse GT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from validate_dobbe_geometry import CAMERA_TO_LABEL


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/dobbe_rgbd_study"
RUN = STUDY / "approx_history"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    protocol = read(STUDY / "protocol.json")
    manifest = read(RUN / "manifest.json")
    freeze = read(RUN / "input_freeze.json")
    model = read(RUN / "model_run.json")
    evaluation = read(RUN / "future_evaluation.json")
    projection = read(RUN / "future_projection_diagnostic.json")
    receipt = read(STUDY / "f2nerf_second/receipt.json")
    assert freeze["history_frames"] == protocol["main"]["history"]
    assert freeze["future_evaluation_only"] == [
        min(protocol["main"]["future_evaluation_only"]),
        max(protocol["main"]["future_evaluation_only"])]
    assert manifest["future_used"] is False
    for name, expected in freeze["sha256"].items():
        assert sha256(RUN / name) == expected, name
    assert sha256(STUDY / "f2nerf_second/receipt.json") == freeze[
        "f2nerf_receipt_sha256"]
    assert sha256(STUDY / "f2nerf_second/cams_meta.npy") == receipt[
        "f2nerf_camera_meta_sha256"]
    assert receipt["frames"] == 121 and receipt["match_colmap_camera"]
    assert np.allclose(np.asarray(receipt["K_256x256"]),
                       np.asarray(manifest["K_256x256"]), atol=0, rtol=0)

    xyz = np.load(RUN / "points_3d_history_candidate.npy")
    world = np.load(RUN / "points_3d_history_world_candidate.npy")
    uv = np.load(RUN / "points_2d_at_t0_candidate.npy")
    assert xyz.shape == world.shape == (3, 8, 3) and uv.shape == (8, 2)
    assert np.isfinite(xyz).all() and np.isfinite(world).all()
    assert (xyz[..., 2] > 0).all()
    assert len(set(manifest["point_ids"])) == 8
    poses = np.asarray(manifest["poses_c2w"])
    expected = (world - poses[-1, :3, 3]) @ poses[-1, :3, :3]
    expected = expected @ CAMERA_TO_LABEL
    assert np.allclose(xyz, expected, atol=1e-6)
    k = np.asarray(manifest["K_256x256"])
    projected = np.column_stack((
        k[0, 0] * xyz[-1, :, 0] / xyz[-1, :, 2] + k[0, 2],
        k[1, 1] * xyz[-1, :, 1] / xyz[-1, :, 2] + k[1, 2]))
    t0_reprojection_max_px = float(np.max(np.linalg.norm(projected - uv, axis=1)))
    assert t0_reprojection_max_px < 1e-3

    assert model["status"] == "COMPLETE" and model["success"]
    assert model["input_freeze_sha256"] == sha256(RUN / "input_freeze.json")
    assert model["geometry_sha256"] == freeze["sha256"][
        "points_3d_history_candidate.npy"]
    assert model["checkpoint_model_pt_sha256"] == freeze[
        "checkpoint_model_pt_sha256"]
    prediction_path = RUN / "prediction_15hz.npy"
    assert model["prediction_sha256"] == sha256(prediction_path)
    prediction = np.load(prediction_path)
    assert prediction.shape == (8, 30, 3) and np.isfinite(prediction).all()
    gt = np.load(RUN / "future_3d_gt_candidate.npy")
    valid = np.load(RUN / "future_gt_valid.npy")
    assert gt.shape == (8, 30, 3) and valid.shape == (8, 30)
    assert np.array_equal(np.isfinite(gt).all(axis=2), valid)
    assert int(valid.sum()) == evaluation["valid_point_timestamps"]
    assert int(valid[:, -1].sum()) == evaluation["valid_final_points"]
    assert not evaluation["full_eight_point_30_frame_gt"]
    errors = np.linalg.norm(prediction - gt, axis=2)
    observed_ade = float(np.nanmean(errors))
    observed_fde = float(np.nanmean(errors[:, -1]))
    m = evaluation["metrics"]["MolmoMotion_index_aligned"]
    assert abs(m["observed_only_ADE_3D_m"] - observed_ade) < 1e-6
    assert abs(m["observed_only_FDE_3D_m"] - observed_fde) < 1e-6
    assert m["ADE_3D_m_full_8x30"] is None
    assert m["FDE_3D_m_full_8_points"] is None
    assert projection["possible"] == 240
    assert len(projection["by_method"]["MolmoMotion"]) == 30
    report = {"status": "PASS_EXPLORATORY_ARTIFACT_AUDIT",
              "formal_geometry_gate": read(STUDY / "decision.json")["status"],
              "f2nerf_frames": receipt["frames"],
              "history_shape": list(xyz.shape),
              "t0_reprojection_max_px": t0_reprojection_max_px,
              "model_prediction_shape": list(prediction.shape),
              "observed_future_gt": [int(valid.sum()), 240],
              "final_valid_points": int(valid[:, -1].sum()),
              "full_ADE_FDE_available": False}
    (RUN / "audit.json").write_text(json.dumps(report, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
