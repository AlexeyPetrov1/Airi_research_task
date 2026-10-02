"""Synthetic checks for the proper rigid fit used in the FMB postprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_fmb_rigid_correction import rigid_fit  # noqa: E402


def test_recovers_known_proper_motion() -> None:
    rng = np.random.default_rng(8)
    source = rng.normal(size=(8, 3))
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    target = source @ rotation + np.array([.1, .2, .3])
    fitted, estimated_rotation, _ = rigid_fit(source, target)
    assert np.max(np.abs(fitted - target)) < 1e-12
    assert abs(np.linalg.det(estimated_rotation) - 1) < 1e-12


def test_reflection_is_not_admitted() -> None:
    rng = np.random.default_rng(13)
    source = rng.normal(size=(8, 3))
    reflected = source * np.array([-1, 1, 1])
    fitted, rotation, _ = rigid_fit(source, reflected)
    assert np.linalg.det(rotation) > .999999
    assert np.linalg.norm(fitted - reflected) > 0.1


if __name__ == "__main__":
    test_recovers_known_proper_motion()
    test_reflection_is_not_admitted()
    print("FMB rigid correction synthetic checks: PASS")
