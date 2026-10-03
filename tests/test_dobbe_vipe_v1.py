"""Scientific correctness checks for the causal ViPE adapter."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dobbe_vipe_geometry import lift, project, sample_depth, load_vipe, restore_intrinsics, independent_sift_geometry
from dobbe_vipe_v1 import SCENES
from molmo_motion.processor import MolmoMotionProcessor
from molmo_motion.public_config import MolmoMotionConfig


@pytest.mark.parametrize("scene,history,count,future_end", [
    ("A", [94, 96, 98], 50, 158), ("B", [119, 121, 123], 62, 183),
    ("C", [56, 58, 60], 31, 120)])
def test_causal_15hz_protocol(scene, history, count, future_end):
    t0 = SCENES[scene]["t0"]
    observed = list(range(t0 % 2, t0+1, 2))
    assert len(observed) == count and observed[-3:] == history
    future = list(range(t0+2, t0+61, 2))
    assert len(future) == 30 and future[-1] == future_end
    assert max(observed) < min(future)


def test_static_world_points_reproject_with_camera_translation():
    depths = np.full((2, 40, 40), 2.0)
    poses = np.repeat(np.eye(4)[None], 2, 0)
    poses[1, 0, 3] = .1
    intrinsics = np.array([[100, 100, 20, 20]] * 2)
    uv = np.array([[[20., 20.]], [[15., 20.]]])
    world, valid, _ = lift(uv, depths, intrinsics, poses, np.ones((2, 1), bool))
    np.testing.assert_allclose(world[0], world[1], atol=1e-12)
    np.testing.assert_allclose(project(world[0], poses[1], intrinsics[1]), uv[1])
    assert valid.all()


def test_depth_sampling_never_clamps_invalid_tracks():
    depth = np.ones((10, 10))
    values, _ = sample_depth(depth, np.array([[-2, 5], [10, 5], [np.nan, 2], [5, 5]]))
    assert np.isnan(values[:3]).all() and values[3] == 1


def test_rectified_camera_rays_match_original_pixel_centers():
    k = np.array([[210., 210., 128., 96.]])
    world = np.array([[.1, -.15, .8], [-.2, .12, 1.3]])
    small = project(world, np.eye(4), k[0])
    raw = project(world, np.eye(4), restore_intrinsics(k, (256, 192))[0])
    np.testing.assert_allclose(raw[:, 0], small[:, 0])
    np.testing.assert_allclose(raw[:, 1], (small[:, 1]+.5)*256/192-.5)


def test_vipe_missing_frame_is_rejected(tmp_path):
    (tmp_path / "pose").mkdir()
    np.savez(tmp_path / "pose/causal_15hz.npz", inds=[0, 2], data=np.repeat(np.eye(4)[None], 2, 0))
    with pytest.raises(ValueError, match="misaligned"):
        load_vipe(tmp_path, 3)


def test_independent_static_audit_detects_depth_pose_scale_mismatch(tmp_path):
    poses = np.repeat(np.eye(4)[None], 2, 0)
    poses[1, 0, 3] = .1
    k = np.array([[100., 100., 20., 20.]]*2)
    np.savez(tmp_path / "static_sift_00000.npz", t0_uv=[[15., 20.], [20., 22.]],
             earlier_uv=[[20., 20.], [25., 22.]])
    correct = independent_sift_geometry(tmp_path, np.full((2, 40, 40), 2.), k, poses)
    wrong = independent_sift_geometry(tmp_path, np.full((2, 40, 40), 1.), k, poses)
    assert correct["count"] == wrong["count"] == 2
    assert correct["median"] == pytest.approx(0.)
    assert wrong["median"] == pytest.approx(5.)


def test_public_processor_world_and_camera_inputs_are_equivalent():
    # Capture the exact serialized prompt through the public processor path.
    def preprocess(example):
        prompt = example["message_list"][0]["question"]
        return {"input_tokens": np.frombuffer(prompt.encode(), dtype=np.uint8).astype(np.int64)}
    processor = MolmoMotionProcessor(None, MolmoMotionConfig(
        num_points=8, history_size=3, future_size=30), preprocess, object())
    theta = .4
    pose = torch.tensor([[np.cos(theta), 0, np.sin(theta), .3],
                         [0, 1, 0, -.2], [-np.sin(theta), 0, np.cos(theta), .1],
                         [0, 0, 0, 1]], dtype=torch.float32)
    camera = torch.arange(72, dtype=torch.float32).reshape(3, 8, 3)/113 + torch.tensor([-.2, -.1, 1.5])
    world = camera @ pose[:3, :3].T + pose[:3, 3]
    inv = torch.linalg.inv(pose)
    manual = world @ inv[:3, :3].T + inv[:3, 3]
    args = dict(history_frames=[Image.new("RGB", (256, 256))]*3,
                points_2d_at_t0=torch.ones(8, 2)*100, action="Pick up the red cup.", future_horizon=30)
    a = processor(points_3d_history=world, c2w_at_t0=pose, **args)
    b = processor(points_3d_history=manual, **args)
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert torch.equal(a["anchor_3d"], b["anchor_3d"])
