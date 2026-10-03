"""Finish the authorized matched runs serially on the local GPU.

Wait for the already-running H3, measure it, then generate/measure H1 and an
optional matched no-control run. Completed diffusion outputs are never replaced.
"""
import json
import argparse
from pathlib import Path
import subprocess
import time
from das_full_motion_diagnose import ROOT, OUT
from das_prepare_control import write_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation-script',type=Path,default=ROOT/'scripts/das_full_motion_generate.py')
    args=parser.parse_args()
    generation='/mnt/f/AIRI_task/.venv-das/bin/python'
    analysis='/mnt/f/AIRI_task/.venv/bin/python'
    deadline=time.monotonic()+1800
    while True:
        state=json.loads((OUT/'H3_group00_6s_prior025/resource_usage.json').read_text())
        if state.get('error'):raise RuntimeError(state['error'])
        if state.get('success'):break
        if time.monotonic()>deadline:raise TimeoutError('H3 has not completed; preserve outputs and inspect its resource log')
        time.sleep(5)
    def free_gpu():
        while True:
            free=subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
            if int(free.strip().splitlines()[0])>9000:return
            time.sleep(5)
    def measure(name):
        free_gpu()
        subprocess.run([analysis,str(ROOT/'scripts/das_full_motion_evaluate.py'),'--name',name],check=True)
    measure('H3_group00_6s_prior025')
    variants=[('H1_group00_2s_no_prior','group00_physical_2s',['--no-prior']),
        ('H4_no_trajectory_control','group00_stretched_6s',['--no-prior','--no-control'])]
    for name,preparation,flags in variants:
        out=OUT/name
        if not (out/'generated_seed42.mp4').exists():
            free_gpu()
            subprocess.run([generation,str(args.generation_script),
                '--name',name,'--preparation-subdir',preparation,*flags],check=True)
        else:
            assert json.loads((out/'resource_usage.json').read_text())['success']
        measure(name)
    write_json(OUT/'queue_completion.json',{'success':True,'sequential_gpu':True,
        'variants':['H3_group00_6s_prior025']+[name for name,_,_ in variants]})


if __name__=='__main__':main()
