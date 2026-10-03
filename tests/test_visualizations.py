import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_experiments.io import ROOT, read_json
from motion_experiments.datasets import ADAPTERS


@pytest.mark.parametrize("name",["author_davis","fmb_wrist_1","fmb_wrist_2","berkeley_bottle","berkeley_cup","dobbe"])
def test_encoded_media_preserves_time_and_coordinates(name):
    folder=Path(os.environ.get("MOLMO_RUN",ROOT/"outputs/packaging_final_inference"))/name/"visualizations"
    if not folder.exists():
        pytest.skip("Run the shared pipeline to validate actual encoded media")
    receipt=read_json(folder/"render_receipt.json")
    coords=np.load(folder/"drawn_coordinates.npz")
    for filename,expected in receipt["videos"].items():
        cap=cv2.VideoCapture(str(folder/filename));count=0
        while cap.read()[0]:count+=1
        assert count == len(coords["time_s"]) == expected["frame_count"]
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == expected["size_wh"][0]
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == expected["size_wh"][1]
        cap.release()
    assert coords["predicted_uv"].shape[:2] == coords["reference_uv"].shape[:2]
    assert len(coords["point_ids"]) == receipt["evaluated_points"]
    assert not receipt["future_camera_poses_in_prediction_video"]
    config=read_json(folder.parent/"config.json")
    sample=ADAPTERS[config["dataset"]].load(config,ROOT/"fixtures",evaluation=True)
    np.testing.assert_array_equal(coords["reference_initial_uv"],sample.points_2d)
    assert (folder/"legacy_vs_unified.png").is_file()
    assert read_json(folder/"legacy_visualization_parity.json")["success"]
