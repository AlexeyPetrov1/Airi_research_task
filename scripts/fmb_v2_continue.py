"""Continue this local experiment after H3, preserving the prediction/evaluation fence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from fmb_v2_scene import ROOT,RUN,SPECS
from berkeley_preprocess import write_json

def main():
    scenes=[RUN/name for name in SPECS]
    receipt=RUN/'continuation_status.json'
    log=RUN/'continuation.log'
    start=time.monotonic();last=-100
    while True:
        completed=[]
        for scene in scenes:
            path=scene/'predictions/model_run.json'
            completed.append(path.exists() and json.loads(path.read_text()).get('success') is True)
            for group in (scene/'predictions').glob('group_*/model_run.json'):
                data=json.loads(group.read_text())
                if data.get('error') and not data.get('success'):raise RuntimeError(f'H3 failed: {group}')
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        ready=all(completed) and used<1500
        state={'stage':'waiting_for_H3_and_free_GPU','H3_scenes_complete':completed,
               'gpu_memory_used_mib':used,'queue_seconds':time.monotonic()-start,'never_stops_other_jobs':True}
        write_json(receipt,state)
        if time.monotonic()-last>30 or ready:
            print(json.dumps(state),flush=True);last=time.monotonic()
        if ready:break
        time.sleep(5)
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    commands=[]
    primary=scenes[0]
    if not (primary/'predictions_h1/model_run.json').exists():
        commands.append(('H1',[sys.executable,str(ROOT/'scripts/fmb_v2_infer.py'),'--mode','h1','--scene-dir',str(primary)]))
    for scene in scenes:
        if not (scene/'evaluation/future_rgb.npy').exists():
            commands.append((f'export_{scene.name}',[sys.executable,str(ROOT/'scripts/fmb_v2_scene.py'),scene.name,'--future']))
        if not (scene/'evaluation/alltracker_future.npz').exists():
            commands.append((f'track_{scene.name}',[sys.executable,str(ROOT/'scripts/fmb_v2_evaluate.py'),'track','--scene-dir',str(scene)]))
    commands.append(('evaluate',[sys.executable,str(ROOT/'scripts/fmb_v2_evaluate.py'),'evaluate','--scene-dir',*map(str,scenes)]))
    with log.open('a',encoding='utf-8',buffering=1) as handle:
        for stage,command in commands:
            state={'stage':stage,'command':command,'started_epoch':time.time()};write_json(receipt,state)
            print(stage,'starting',flush=True)
            code=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT).returncode
            if code:
                write_json(receipt,{**state,'success':False,'returncode':code});raise SystemExit(code)
        write_json(receipt,{'stage':'awaiting_root_visual_review_and_report','success':True,'elapsed_seconds':time.monotonic()-start})
        print('Predictions, references, metrics and media complete; visual review pending',flush=True)

if __name__=='__main__':main()
