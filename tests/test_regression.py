"""Immutable assertions captured from baseline before the storage migration."""
import numpy as np
import pytest

from motion_experiments.datasets import ADAPTERS
from motion_experiments.fingerprints import array_fingerprint
from motion_experiments.io import ROOT, read_json, sha
from motion_experiments.metrics import compute_metrics, displacement_metrics
from motion_experiments.sample import validate_sample

GOLDEN = read_json(ROOT/'tests/golden_manifest.json')
CASES = list(GOLDEN['cases'])
SOURCE = ROOT/'fixtures'


def load(name, evaluation=True):
    config = read_json(ROOT/'configs'/(name+'.json'))
    adapter = ADAPTERS[config['dataset']]
    return adapter, adapter.load(config, SOURCE, evaluation=evaluation)


@pytest.mark.parametrize('name', CASES)
def test_canonical_input_and_temporal_parity(name):
    _, actual = load(name, evaluation=False)
    expected = GOLDEN['cases'][name]
    validate_sample(actual)
    for field, frozen in expected['canonical'].items():
        assert array_fingerprint(getattr(actual,field)) == frozen, field
        assert list(getattr(actual,field).strides) == expected['canonical_strides'][field], field
    for field in ('camera_intrinsics','c2w_at_t0'):
        if field not in expected['canonical']: assert getattr(actual,field) is None
    assert array_fingerprint(actual.future_times) == expected['future_times']
    assert (actual.action,actual.status,actual.failure_reason) == (expected['action'],expected['status'],expected['failure_reason'])
    assert not any(p.name.startswith('evaluation') or 'media' in p.parts for p in actual.source_files)


@pytest.mark.parametrize('name', [n for n in CASES if n!='dobbe_blocked'])
def test_forecast_and_projection_parity(name):
    adapter, sample = load(name)
    expected = GOLDEN['cases'][name]
    assert array_fingerprint(sample.saved_prediction) == expected['prediction']
    xyz, uv = adapter.finish(sample, sample.saved_prediction)
    for space, methods in [('xyz',xyz),('uv',uv)]:
        assert set(methods) == set(expected[space])
        for method,value in methods.items():
            assert array_fingerprint(value) == expected[space][method], (space,method)


@pytest.mark.parametrize('name', [n for n in CASES if n!='dobbe_blocked'])
def test_metrics_and_baselines_reproduce_saved_legacy(name):
    adapter, sample = load(name)
    xyz, uv = adapter.finish(sample, sample.saved_prediction)
    metrics, errors = compute_metrics(sample,xyz,uv)
    expected = GOLDEN['cases'][name]
    # Every scalar, time/point statistic, trajectory diagnostic and baseline;
    # exact equality is stronger than the former subset of approximate checks.
    assert metrics == expected['metrics']
    assert set(errors) == set(expected['errors'])
    for key,value in errors.items(): assert array_fingerprint(value) == expected['errors'][key]
    for key,value in expected['evaluation'].items():
        assert array_fingerprint(sample.evaluation[key]) == value
    if expected['camera_poses'] is not None:
        assert array_fingerprint(sample.camera_poses) == expected['camera_poses']


def test_real_negative_branch_remains_rejected():
    _, sample = load('dobbe_blocked')
    assert sample.status == 'SKIPPED_GEOMETRY_GATE'
    assert sample.failure_reason == 'static_median'
    assert sample.saved_prediction is None and sample.evaluation == {}
    assert sample.processor_fingerprints == []
    assert sample.metadata['geometry_gate']['static_cross_frame_px']['median'] > 4


def test_mask_and_final_horizon_rules():
    gt=np.zeros((2,3,2));pred=gt.copy();pred[...,0]=2
    mask=np.ones((2,3),bool)
    score,_=displacement_metrics(pred,gt,mask)
    assert score['ADE'] == score['FDE'] == 2
    mask[:,-1]=False
    score,_=displacement_metrics(pred,gt,mask)
    assert score['FDE'] is None
    pred[0,0,0]=np.inf
    with pytest.raises(ValueError):displacement_metrics(pred,gt,mask)


def test_original_artifacts_remain_unchanged():
    for relative,digest in GOLDEN['fixture_sha256'].items():
        assert sha(ROOT/relative) == digest, relative
    assert not (ROOT/'tests/golden/source').exists()
    assert not (ROOT/'data/legacy').exists()
    assert not (ROOT/'legacy').exists()


def test_model_source_was_copied_without_changes():
    manifest=read_json(ROOT/'docs/model_source_snapshot.json')
    for relative,digest in manifest['sha256'].items(): assert sha(ROOT/relative) == digest


def test_neighboring_research_archive_is_unchanged_when_present():
    archive=ROOT.parent/'molmo-motion'
    if not archive.is_dir():
        pytest.skip('The standalone repository does not require the research archive')
    for relative,digest in {**GOLDEN['historical_source_sha256'],**GOLDEN['historical_script_sha256']}.items():
        assert sha(archive/relative) == digest
    for frame in GOLDEN['rgb_provenance']['source_frames']:
        assert sha(archive/frame['path']) == frame['sha256']
