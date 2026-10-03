"""Mandatory complete verification: installed wheel, fresh forecasts, all seven cases.

Baseline outputs must be generated from commit 70ed2c4 before migration.
All output directories are new; cached inference receipts cannot satisfy this run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from motion_experiments.fingerprints import array_fingerprint
from motion_experiments.io import read_json,sha,write_json

CASES=['author_davis','fmb_wrist_1','fmb_wrist_2','berkeley_bottle','berkeley_cup','dobbe','dobbe_blocked']
BASELINE='70ed2c413eee5e3b9dc1324e2c87ff333ee88334'


def require(condition,message):
    if not condition: raise AssertionError(message)


def npz_equal(left,right):
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        require(set(a)==set(b),f'NPZ fields differ: {left}')
        for key in a:
            require(array_fingerprint(a[key])==array_fingerprint(b[key]),f'Array differs: {left} / {key}')


def verify_case(run,baseline,name,fresh,started):
    folder=run/name
    previous=baseline/name
    status=read_json(folder/'status.json')
    row=dict(config=name,status=status['status'],fresh_inference=False,prediction_max_difference_m=None,
             metrics_match=False,baselines_match=False,visualization_geometry_match=False,
             runtime=status['runtime_seconds'],peak_cuda_memory=None,result='FAIL')
    expected='SKIPPED_GEOMETRY_GATE' if name=='dobbe_blocked' else 'COMPLETE'
    require(status['status']==expected,f'{name}: unexpected status')
    npz_equal(folder/'inputs/canonical.npz',previous/'inputs/canonical.npz')
    if name=='dobbe_blocked':
        require(status['failure_reason']=='static_median',name+': wrong gate reason')
        require(status['expected_failure'] and not status['success'],name+': wrong negative status')
        for path in ('predictions','evaluation','metrics.json','model_input_parity.json'):
            require(not (folder/path).exists(),name+': gate opened '+path)
        row.update(metrics_match=True,baselines_match=True,visualization_geometry_match=True,
                   checks_applicability='Forecast/metrics/media intentionally absent after expected geometry gate',result='PASS')
        return row
    raw=np.load(folder/'predictions/future_3d.npy')
    ref=np.load(previous/'predictions/future_3d.npy')
    require(raw.shape==ref.shape and np.isfinite(raw).all(),name+': invalid forecast')
    difference=float(np.max(np.abs(raw.astype(float)-ref.astype(float))))
    np.testing.assert_allclose(raw,ref,atol=1e-7,rtol=0)
    require(read_json(folder/'metrics.json')==read_json(previous/'metrics.json'),name+': metrics differ')
    for artifact in ('predictions/aligned.npz','errors.npz','evaluation/reference.npz','visualizations/drawn_coordinates.npz'):
        npz_equal(folder/artifact,previous/artifact)
    receipt=read_json(folder/'visualizations/render_receipt.json')
    before=read_json(previous/'visualizations/render_receipt.json')
    require({k:v for k,v in receipt.items() if k!='videos'}=={k:v for k,v in before.items() if k!='videos'},name+': rendering association differs')
    require(set(receipt['videos'])==set(before['videos']),name+': missing video')
    for filename,expected_video in receipt['videos'].items():
        for key in ('frame_count','size_wh','fps'):
            require(expected_video[key]==before['videos'][filename][key],name+': video '+key+' differs')
        cap=cv2.VideoCapture(str(folder/'visualizations'/filename))
        count=0
        while cap.read()[0]: count+=1
        size=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        cap.release()
        require(count==expected_video['frame_count'] and size==expected_video['size_wh'],name+': invalid encoded video')
    require((folder/'index.html').is_file(),name+': HTML missing')
    require(read_json(folder/'model_input_parity.json')['success'],name+': processor parity missing')
    parity=read_json(folder/'prediction_parity.json')
    require(parity['fresh_inference']==fresh and status['fresh_inference']==fresh,name+': wrong inference mode')
    model_receipt=read_json(folder/'predictions/model_run.json')
    groups=model_receipt['groups']
    if fresh:
        require(model_receipt['mode']=='inference' and model_receipt['predictions_from'] is None,name+': reused forecasts')
        require(len(groups)==raw.shape[0]//8,name+': incomplete P8 calls')
        for i,group in enumerate(groups):
            local=folder/'predictions'/f'group_{i:02d}'
            require(group['success'] and group['generation_started'],name+': missing actual generation')
            require((local/'model_run.json').stat().st_mtime>=started-2,name+': cached receipt')
            require(sha(local/'future_3d.npy')==group['prediction_sha256'],name+': group forecast changed')
            require(sha(local/'raw_model_output.txt')==group['raw_sha256'],name+': raw text changed')
        row['peak_cuda_memory']=dict(allocated_gib=max(g['peak_cuda_allocated_gib'] for g in groups),
                                     reserved_gib=max(g['peak_cuda_reserved_gib'] for g in groups))
    row.update(fresh_inference=fresh,prediction_max_difference_m=difference,metrics_match=True,
               baselines_match=True,visualization_geometry_match=True,result='PASS')
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--baseline-run',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--run-prefix',default='verification_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f'))
    args=parser.parse_args()
    checkpoint=args.checkpoint.resolve();baseline=args.baseline_run.resolve();python=args.python.absolute()
    golden=read_json(REPO/'tests/golden_manifest.json')
    require(golden['baseline_commit']==BASELINE,'Incorrect frozen baseline')
    require(set(golden['cases'])==set(CASES),'All seven cases are mandatory')
    checkpoint_snapshot=read_json(REPO/'docs/checkpoint_snapshot.json')
    require(checkpoint_snapshot['revision']=='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629','Wrong checkpoint revision')
    for filename,expected in checkpoint_snapshot['files'].items():
        require(sha(checkpoint/filename)==expected['sha256'],'Checkpoint bytes differ: '+filename)
    started=time.time()
    runs={mode:REPO/'outputs'/(args.run_prefix+'_'+mode) for mode in ('preflight','replay','inference','rerender')}
    require(not any(p.exists() for p in runs.values()),'Final verification requires new output directories')
    logs=REPO/'outputs'/(args.run_prefix+'_logs');logs.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,MOLMO_CHECKPOINT=str(checkpoint))
    env.setdefault('HF_HOME',str(REPO.parent/'.cache/huggingface'))
    env.setdefault('TORCH_HOME',str(REPO.parent/'.cache/torch'))
    runtime_hashes={str(p.relative_to(REPO)):sha(p) for p in (REPO/'src/motion_experiments').rglob('*.py')}
    report=dict(success=False,baseline_commit=BASELINE,started_at_utc=datetime.now(timezone.utc).isoformat(),
                execution_cwd=tempfile.gettempdir(),python=str(python),checkpoint_revision='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629',
                inference_run=str(runs['inference'].relative_to(REPO)),checkpoint_sha256={k:v['sha256'] for k,v in checkpoint_snapshot['files'].items()},
                prediction_atol_m=1e-7,prediction_rtol=0,runtime_sha256=runtime_hashes,
                cases=[dict(config=n,result='FAIL',reason='Verification has not completed') for n in CASES])
    def invoke(stage,command,cwd=tempfile.gettempdir()):
        print(stage,flush=True)
        log=logs/(stage+'.log')
        with log.open('w',encoding='utf8') as stream:
            subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        return log.read_text(encoding='utf8')
    cli=[str(python),'-m','motion_experiments.run','--config']+['configs/'+n+'.json' for n in CASES]+['--checkpoint',str(checkpoint)]
    console_cli=[str(python.parent/'molmo-motion-experiment')]+cli[3:]
    try:
        identity=invoke('installed_package',[str(python),'-c',
            'import json,molmo_motion,motion_experiments;from motion_experiments.io import ROOT;'
            'print(json.dumps(dict(model=molmo_motion.__file__,package=motion_experiments.__file__,data=str(ROOT))))'])
        identity=json.loads(identity.strip().splitlines()[-1]);report['installed_package']=identity
        for key in ('model','package'):
            require('site-packages' in identity[key], 'Verification must use an installed wheel: '+key)
        installed=Path(identity['data'])
        for relative,digest in runtime_hashes.items():
            require(sha(Path(identity['package']).parent.parent/Path(relative).relative_to('src'))==digest,'Installed experiment code differs: '+relative)
        for relative,digest in golden['fixture_sha256'].items():
            require(sha(REPO/relative)==digest and sha(installed/relative)==digest,'Fixture differs from golden: '+relative)
        for relative,digest in read_json(REPO/'docs/model_source_snapshot.json')['sha256'].items():
            require(sha(REPO/relative)==digest,'Vendored model changed')
            require(sha(Path(identity['model']).parent.parent/Path(relative).relative_to('src'))==digest,'Installed model changed')
        # Produce actual final-layout media for the artifact tests before Level A.
        invoke('preflight',console_cli+['--mode','replay','--output',str(runs['preflight'])])
        env['MOLMO_RUN']=str(runs['preflight'])
        report['pytest_before']=invoke('pytest_before',[str(python),'-m','pytest','-q'],cwd=REPO)
        require(' skipped' not in report['pytest_before'],'Full verification may not skip checks')
        invoke('replay',cli+['--mode','replay','--output',str(runs['replay'])])
        report['replay_cases']=[verify_case(runs['replay'],baseline,n,False,started) for n in CASES]
        invoke('fresh_inference',cli+['--mode','inference','--output',str(runs['inference'])])
        report['cases']=[verify_case(runs['inference'],baseline,n,n!='dobbe_blocked',started) for n in CASES]
        invoke('predictions_from',console_cli+['--mode','replay','--predictions-from',str(runs['inference']),'--output',str(runs['rerender'])])
        report['predictions_from_cases']=[verify_case(runs['rerender'],baseline,n,False,started) for n in CASES]
        env['MOLMO_RUN']=str(runs['inference'])
        report['pytest_after']=invoke('pytest_after',[str(python),'-m','pytest','-q'],cwd=REPO)
        require(' skipped' not in report['pytest_after'],'Fresh output verification may not skip checks')
        for relative,digest in golden['fixture_sha256'].items():
            require(sha(REPO/relative)==digest and sha(installed/relative)==digest,'Fixture mutated during verification')
        for relative,digest in runtime_hashes.items():
            require(sha(REPO/relative)==digest,'Runtime changed during verification: '+relative)
        report['fresh_model_calls']=sum(len(read_json(runs['inference']/n/'predictions/model_run.json')['groups']) for n in CASES[:-1])
        require(report['fresh_model_calls']==14,'Expected all fourteen fresh P8 calls')
        report['success']=True
    except Exception as exc:
        report['failure']=str(exc)
        raise
    finally:
        report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
        report['runtime_seconds']=time.time()-started
        write_json(REPO/'docs/final_verification.json',report)
        rows=['# Final verification','',('PASS' if report['success'] else 'FAIL')+' — baseline `'+BASELINE+'`.','',
              'Installed package; all seven replay cases; six fresh neural forecasts; expected pre-inference geometry gate; predictions-from; complete tests.','',
              '| Config | Status | Fresh inference | Max difference, m | Metrics | Geometry | Result |',
              '|---|---|---|---:|---|---|---|']
        for case in report['cases']:
            rows.append('| '+' | '.join(str(case.get(k,'—')) for k in ('config','status','fresh_inference','prediction_max_difference_m','metrics_match','visualization_geometry_match','result'))+' |')
        rows += ['',f"Fresh model calls: {report.get('fresh_model_calls',0)}. Runtime: {report['runtime_seconds']:.1f} s.",
                 '', 'Tolerance: atol=1e-7 m, rtol=0. All metrics, aligned baselines, errors and visualization coordinates are compared exactly.',
                 '', 'Machine-readable details: [final_verification.json](final_verification.json). Logs: `'+str(logs.relative_to(REPO))+'`.']
        if report.get('failure'): rows += ['',report['failure']]
        (REPO/'docs/final_verification.md').write_text('\n'.join(rows)+'\n',encoding='utf8')
    print('PASS: complete fresh end-to-end verification',flush=True)


if __name__=='__main__': main()
