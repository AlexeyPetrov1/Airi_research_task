"""Causal access, lossless storage and exact processor structure contracts."""
import copy

import numpy as np
import pytest

from motion_experiments.datasets import ADAPTERS
from motion_experiments.fingerprints import fingerprint
from motion_experiments.io import ROOT, Source, read_json
from motion_experiments.run import run_case


@pytest.mark.parametrize('config_path', sorted((ROOT/'configs').glob('*.json')), ids=lambda p:p.stem)
def test_prepare_cannot_open_future_artifacts(config_path, monkeypatch):
    original = Source.path
    def causal_only(self,name):
        if 'evaluation' in name or '/media/' in name:
            raise AssertionError('Future data was opened during preparation')
        return original(self,name)
    monkeypatch.setattr(Source,'path',causal_only)
    config=read_json(config_path)
    ADAPTERS[config['dataset']].load(config,ROOT/'fixtures',evaluation=False)


def test_blocked_run_never_builds_processor_or_calls_model(tmp_path, monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('Geometry gate must precede processor/inference')
    monkeypatch.setattr('motion_experiments.run.build_inputs',forbidden)
    config=read_json(ROOT/'configs/dobbe_blocked.json')
    run_case(config,ROOT/'fixtures',tmp_path/'blocked',tmp_path/'missing_checkpoint','inference')
    assert not (tmp_path/'blocked/predictions').exists()
    assert not (tmp_path/'blocked/evaluation').exists()
    assert not (tmp_path/'blocked/metrics.json').exists()


def test_outputs_cannot_modify_immutable_fixtures():
    config=read_json(ROOT/'configs/dobbe_blocked.json')
    with pytest.raises(ValueError,match='immutable fixtures'):
        run_case(config,ROOT/'fixtures',ROOT/'fixtures/forbidden',ROOT,'replay')
    assert not (ROOT/'fixtures/forbidden').exists()


def test_fingerprint_preserves_nested_types_and_every_byte():
    import torch
    value={'tensor':torch.arange(6,dtype=torch.bfloat16).reshape(2,3),
           'nested':[np.array([1,2],dtype=np.int16),('text',None,False,-0.0)]}
    frozen=fingerprint(value)
    assert fingerprint(copy.deepcopy(value)) == frozen
    changed=copy.deepcopy(value);changed['tensor'][0,0]=1
    assert fingerprint(changed) != frozen
    changed=copy.deepcopy(value);changed['tensor']=changed['tensor'].float()
    assert fingerprint(changed) != frozen
    changed=copy.deepcopy(value);changed['nested']=tuple(changed['nested'])
    assert fingerprint(changed) != frozen
    changed=copy.deepcopy(value);changed['nested'][1]=('text',None,False,0.0)
    assert fingerprint(changed) != frozen
