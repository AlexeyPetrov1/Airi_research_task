"""Verify saved inference inputs stay separate from evaluation and final media."""
import os
from pathlib import Path

import numpy as np
import pytest

from motion_experiments.datasets import ADAPTERS
from motion_experiments.io import ROOT, read_json, sha


@pytest.mark.parametrize("name", ["author_davis", "fmb_wrist_1", "fmb_wrist_2",
                                  "berkeley_bottle", "berkeley_cup", "dobbe"])
def test_saved_inputs_and_forecast_provenance(name):
    folder = Path(os.environ.get("MOLMO_RUN", ROOT / "outputs/final_corrected")) / name
    if not folder.exists():
        pytest.skip("Create the final run to check its actual artifacts")
    config = read_json(folder / "config.json")
    adapter = ADAPTERS[config["dataset"]]
    sample = adapter.load(config, ROOT / "data/legacy", evaluation=False)
    with np.load(folder / "inputs/canonical.npz") as saved:
        assert "camera_poses" not in saved and "evaluation_camera_poses" not in saved
        assert "xyz" not in saved and "uv" not in saved and "mask2" not in saved
        for key, field in [("history_rgb", "history_frames"), ("points_2d", "points_2d"),
                           ("points_3d_history", "points_3d_history"),
                           ("point_ids", "point_ids"), ("history_timestamps", "history_timestamps")]:
            np.testing.assert_array_equal(saved[key], getattr(sample, field))
    for relative, digest in read_json(folder / "inputs/freeze.json").items():
        assert sha(ROOT / "data/legacy" / relative) == digest
    forecast = folder / "predictions/future_3d.npy"
    np.testing.assert_allclose(np.load(forecast), sample.saved_prediction, atol=1e-7, rtol=0)
    assert sha(forecast) == read_json(folder / "future_access_receipt.json")["prediction_sha256"]
    receipt = read_json(folder / "predictions/model_run.json")
    if receipt.get("predictions_from"):
        origin = Path(receipt["predictions_from"])
        assert read_json(origin / "status.json")["fresh_inference"]
        assert sha(forecast) == sha(origin / "predictions/future_3d.npy")
        for group in sorted((origin / "predictions").glob("group_*")):
            actual = read_json(group / "model_run.json")
            assert actual["success"] and actual["generation_started"]
            assert sha(group / "raw_model_output.txt") == actual["raw_sha256"]
            assert sha(group / "future_3d.npy") == actual["prediction_sha256"]
    evaluated = adapter.load(config, ROOT / "data/legacy", evaluation=True)
    with np.load(folder / "evaluation/reference.npz") as reference:
        np.testing.assert_array_equal(reference["uv"], evaluated.evaluation["uv"])
        if evaluated.camera_poses is not None:
            np.testing.assert_array_equal(reference["evaluation_camera_poses"], evaluated.camera_poses)
