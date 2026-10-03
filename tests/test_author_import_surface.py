"""Smoke tests — verify the public + internal import graph is intact.

These tests intentionally do NOT instantiate the 4B model. They guard
against import-time errors after refactors (deleted modules,
renamed-but-forgotten imports, broken HF adapter wiring).
"""

import pytest


def test_top_level_public_api():
    import molmo_motion
    assert hasattr(molmo_motion, "MolmoMotion")
    assert hasattr(molmo_motion, "MolmoMotionProcessor")
    assert hasattr(molmo_motion, "MolmoMotionConfig")
    assert hasattr(molmo_motion, "MolmoMotionOutput")
    assert isinstance(molmo_motion.__version__, str)


def test_data_layer():
    from molmo_motion.data.data_loader import DataLoaderConfig  # noqa: F401
    from molmo_motion.data.get_dataset import get_dataset_by_name  # noqa: F401
    from molmo_motion.data.trajectory_3d_dataset import (
        Trajectory3DDataset,
        DATASET_CONFIG,
    )
    # Release ships the 7 base datasets in molmo-motion-1m, plus the 3
    # PointMotionBench eval subsets.
    expected = {
        "egodex", "ytvis", "hepic", "xperience",
        "droid", "stereo4d", "molmospaces",
        "hot3d_bench", "worldtrack_bench", "davis_bench",
    }
    assert expected.issubset(set(DATASET_CONFIG.keys())), (
        f"Missing datasets: {expected - set(DATASET_CONFIG.keys())}"
    )


def test_eval_dispatcher_minimal():
    """Only the trajectory metric remains in the EvaluatorConfig surface."""
    from molmo_motion.eval.inf_evaluator import EvaluatorConfig
    fields = {f.name for f in EvaluatorConfig.__dataclass_fields__.values()}
    # The only inference metric we ship for trajectory:
    assert "egodex_3d_eval" in fields
    # QA fields should have been stripped:
    assert "vqa_eval" not in fields
    assert "math_vista_eval" not in fields
    assert "tomato" not in fields


def test_get_dataset_rejects_non_trajectory():
    from molmo_motion.data.get_dataset import get_dataset_by_name
    with pytest.raises(NotImplementedError):
        get_dataset_by_name("coco_2014_vqa_multi", "train")


def test_hf_adapter_imports():
    from molmo_motion.hf_model import (
        Molmo2Config,
        Molmo2ForConditionalGeneration,
        Molmo2Processor,
    )
    assert Molmo2Config is not None
    assert Molmo2ForConditionalGeneration.config_class is Molmo2Config
    assert Molmo2Processor is not None


def test_cli_entry_points_importable():
    """The console-script targets must be importable for `pip` to install them."""
    from molmo_motion.cli import train, eval as eval_cli
    assert callable(train.main)
    assert callable(eval_cli.main)
    from molmo_motion.hf_model import convert_molmo_motion_to_hf as conv
    assert callable(conv.main)


def test_format_history_tracks_uses_shared_3d_anchor():
    """The shared anchor is the first point at t0; other points retain offsets."""
    import numpy as np
    from molmo_motion.processor import _format_history_tracks
    H, P = 3, 4
    hist = np.random.RandomState(0).randn(H, P, 3).astype(np.float32) * 0.1
    anchor = hist[-1, 0]
    text = _format_history_tracks(hist, anchor)
    # The anchor point's final record is zero; the cloud must not collapse.
    last_row = text.split(";")[-1].rsplit('"', 1)[0]
    parts = last_row.split()
    # First token is the timestamp; remaining come in groups of 4: [obj_id, x, y, z]
    deltas = [int(x) for x in parts[1:]]
    assert deltas[:4] == [1, 0, 0, 0]
    assert any(deltas[i + j] != 0 for i in range(4, len(deltas), 4) for j in (1, 2, 3))


def test_public_config_history_size_validation():
    from molmo_motion import MolmoMotionConfig
    # Released variants only:
    MolmoMotionConfig(history_size=1)
    MolmoMotionConfig(history_size=3)
    with pytest.raises(ValueError):
        MolmoMotionConfig(history_size=5)
