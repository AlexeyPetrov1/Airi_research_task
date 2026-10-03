"""Independent legacy artifacts/formulas, including the genuine negative case."""
from pathlib import Path

import numpy as np
import pytest

from motion_experiments.datasets import ADAPTERS
from motion_experiments.geometry import project, project_future, transform
from motion_experiments.io import ROOT, read_json, sha
from motion_experiments.metrics import compute_metrics, displacement_metrics
from motion_experiments.sample import validate_sample

CASES = ["author_davis", "fmb_wrist_1", "fmb_wrist_2", "berkeley_bottle", "berkeley_cup", "dobbe", "dobbe_blocked"]
SOURCE = ROOT/"data/legacy"
GOLDEN = ROOT/"tests/golden/source"


@pytest.mark.parametrize("name", CASES)
def test_canonical_input_and_temporal_parity(name):
    config = read_json(ROOT/"configs"/(name+".json"))
    adapter = ADAPTERS[config["dataset"]]
    actual = adapter.load(config, SOURCE, evaluation=False)
    expected = adapter.load(config, GOLDEN, evaluation=False)
    validate_sample(actual)
    for field in ("history_frames", "history_timestamps", "points_2d", "points_3d_history", "point_ids", "future_times"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert actual.action == expected.action
    assert actual.status == expected.status
    assert actual.failure_reason == expected.failure_reason
    # Preparing inference must not read any evaluation artifact.
    assert not any("evaluation" in p.parts or p.name.startswith("future_reference") for p in actual.source_files)


@pytest.mark.parametrize("name", CASES[:-1])
def test_forecast_and_projection_parity(name):
    config = read_json(ROOT/"configs"/(name+".json"))
    adapter = ADAPTERS[config["dataset"]]
    sample = adapter.load(config, SOURCE)
    xyz, uv = adapter.finish(sample, sample.saved_prediction)
    if name.startswith("fmb"):
        from legacy.fmb_wrist_v4_evaluate import align
        from legacy.fmb_wrist_math import from_anchor, project as original_project
        expected = sample.evaluation["expected"]
        for method in xyz:
            np.testing.assert_allclose(xyz[method], expected[method], atol=1e-12, rtol=0)
            np.testing.assert_allclose(uv[method], expected[method+"_uv"], atol=1e-9, rtol=0)
        np.testing.assert_array_equal(xyz["v3_Molmo_H3"], align(sample.saved_prediction, sample.points_3d_history[-1]))
    elif name == "berkeley_cup":
        for method in ("original_physical", "displacement_one_third", "step_equals_frame"):
            np.testing.assert_allclose(xyz[method], sample.evaluation["expected"][method], atol=1e-14, rtol=0)
    elif name == "berkeley_bottle":
        for method in ("reference_0333", "scale_036_lift005", "late_036_lift005"):
            expected_path = SOURCE/"runs/berkeley_ur5_arc_expansion_v1"/("bottle" if method != "late_036_lift005" else "phase_schedule/bottle")
            np.testing.assert_allclose(xyz[method], np.load(expected_path/"aligned_3d.npz")[method], atol=1e-14, rtol=0)
            np.testing.assert_allclose(uv[method], np.load(expected_path/"projected_2d.npz")[method], atol=1e-9, rtol=0)
    elif name == "dobbe":
        np.testing.assert_allclose(uv["MolmoMotion"], sample.evaluation["expected"]["predicted_uv"].transpose(1,0,2), atol=1e-8, rtol=0)
        np.testing.assert_allclose(uv["Static"], sample.evaluation["expected"]["stationary_uv"].transpose(1,0,2), atol=1e-8, rtol=0)
    else:
        from legacy.evaluate_author_davis import parse_raw_forecast
        raw=(SOURCE/"runs/author_davis_bmx_trees_f30/model_output_raw.txt").read_text()
        np.testing.assert_allclose(sample.saved_prediction, parse_raw_forecast(raw)+sample.points_3d_history[-1,0], atol=1e-7, rtol=0)


@pytest.mark.parametrize("name", CASES[:-1])
def test_metrics_and_baselines_reproduce_saved_legacy(name):
    config=read_json(ROOT/"configs"/(name+".json"))
    adapter=ADAPTERS[config["dataset"]]
    sample=adapter.load(config,SOURCE)
    xyz,uv=adapter.finish(sample,sample.saved_prediction)
    current,_=compute_metrics(sample,xyz,uv)
    previous=read_json(SOURCE/sample.metadata["legacy_metric_file"])
    if name.startswith("fmb"):
        previous=previous["methods"]
        for method,row in current["methods"].items():
            for old,new in [("3D_est_m","3D_m"),("2D_px","2D_px")]:
                for metric in ("ADE","FDE"):
                    assert row[new][metric] == pytest.approx(previous[method][old][metric], abs=1e-12)
    elif name == "author_davis":
        for method,row in current["methods"].items():
            for metric in ("ADE","FDE"):
                assert row["3D_m"][metric] == pytest.approx(previous[method][metric+"_m"], abs=1e-12)
            assert row["2D_px"] is None
        assert current["methods"]["MolmoMotion"]["3D_m"]["error_by_time"][14] is None
    elif name == "berkeley_cup":
        from legacy.berkeley_evaluate import metric_values
        for method in ("original_physical","displacement_one_third","step_equals_frame"):
            for old,new in [("3D_est_m","3D_m"),("2D_px","2D_px")]:
                for metric in ("ADE","FDE"):
                    assert current["methods"][method][new][metric] == pytest.approx(previous["cup"][method][old][metric+"_"+old], abs=1e-12)
        for method in ("Static","Constant velocity"):
            old,_=metric_values(uv[method],sample.evaluation["uv"],sample.evaluation["mask2"],"2D_px")
            assert current["methods"][method]["2D_px"]["ADE"] == old["ADE_2D_px"]
    elif name == "berkeley_bottle":
        for method in ("late_036_lift005","scale_036_lift005","reference_0333"):
            old = previous["bottle"][method] if method=="late_036_lift005" else None
            if old:
                for metric in ("ADE","FDE"):
                    assert current["methods"][method]["2D_px"][metric] == pytest.approx(old[metric+"_2D_px"], abs=1e-10)
                    assert current["methods"][method]["3D_m"][metric] == pytest.approx(old["conditional_3D_est"][metric+"_m"], abs=1e-12)
    else:
        row=current["methods"]["MolmoMotion"]
        assert row["2D_px"]["ADE"] == pytest.approx(previous["conditional_forecast_pixel_errors"]["mean_ADE_px"], abs=1e-10)
        assert row["2D_px"]["FDE"] == pytest.approx(previous["conditional_forecast_pixel_errors"]["FDE_visible_mean_px"], abs=1e-10)
        assert row["3D_m"] is None


def test_real_negative_branch_remains_rejected():
    config=read_json(ROOT/"configs/dobbe_blocked.json")
    sample=ADAPTERS["dobbe"].load(config,SOURCE)
    assert sample.status == "SKIPPED_GEOMETRY_GATE"
    assert sample.failure_reason == "static_median"
    assert sample.saved_prediction is None
    assert sample.evaluation == {}
    assert sample.metadata["geometry_gate"]["static_cross_frame_px"]["median"] > 4


def test_mask_and_final_horizon_rules():
    gt=np.zeros((2,3,2));pred=gt.copy();pred[...,0]=2
    mask=np.ones((2,3),bool)
    score,_=displacement_metrics(pred,gt,mask)
    assert score["ADE"] == score["FDE"] == 2
    mask[:,-1]=False
    score,_=displacement_metrics(pred,gt,mask)
    assert score["FDE"] is None
    pred[0,0,0]=np.inf
    with pytest.raises(ValueError):displacement_metrics(pred,gt,mask)


def test_original_artifacts_remain_unchanged():
    manifest=read_json(ROOT/"tests/golden/manifest.json")
    for relative,digest in manifest["source_sha256"].items():
        assert sha(GOLDEN/relative) == digest
        assert sha(SOURCE/relative) == digest
    rgb = read_json(ROOT/"docs/evaluation_rgb_source.json")
    for folder in (SOURCE, GOLDEN):
        assert sha(folder/rgb["array_path"]) == rgb["array_sha256"]


def test_model_source_was_copied_without_changes():
    manifest=read_json(ROOT/"docs/model_source_snapshot.json")
    for relative,digest in manifest["sha256"].items():
        assert sha(ROOT/relative) == digest


def test_neighboring_research_archive_is_unchanged_when_present():
    archive=ROOT.parent/"molmo-motion"
    if not archive.is_dir():
        pytest.skip("The standalone final repository does not require the research archive")
    manifest=read_json(ROOT/"tests/golden/manifest.json")
    for relative,digest in manifest["source_sha256"].items():
        assert sha(archive/relative) == digest
    for relative,digest in manifest["legacy_script_sha256"].items():
        assert sha(archive/relative) == digest
    for frame in read_json(ROOT/"docs/evaluation_rgb_source.json")["source_frames"]:
        assert sha(archive/frame["path"]) == frame["sha256"]
