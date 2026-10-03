import os
from pathlib import Path

import pytest

from motion_experiments.datasets import ADAPTERS
from motion_experiments.io import ROOT, read_json
from motion_experiments.molmomotion import build_inputs


@pytest.mark.parametrize("name",["author_davis","fmb_wrist_1","fmb_wrist_2","berkeley_bottle","berkeley_cup","dobbe"])
def test_exact_original_processor_packets(name):
    checkpoint=Path(os.environ.get("MOLMO_CHECKPOINT",ROOT/"data/checkpoints/MolmoMotion-4B-H3-F30"))
    if not (checkpoint/"config.yaml").exists():
        pytest.skip("Install the pinned checkpoint config to exercise processor parity")
    config=read_json(ROOT/"configs"/(name+".json"))
    sample=ADAPTERS[config["dataset"]].load(config,ROOT/"data/legacy",evaluation=False)
    batches,checks=build_inputs(sample,checkpoint)
    assert len(batches) == len(sample.point_ids)//8
    assert all(all(group.values()) for group in checks)


def test_vendored_model_is_imported_from_this_repository():
    import molmo_motion
    assert Path(molmo_motion.__file__).resolve().is_relative_to(ROOT)
