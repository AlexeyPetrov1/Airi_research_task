"""Run the requested whole-body experiment after the matched queue finishes."""
import json
import subprocess
import time
from das_full_motion_diagnose import ROOT, OUT
from das_prepare_control import write_json


def main():
    while True:
        try:
            queue=json.loads((OUT/'queue_completion.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            time.sleep(5);continue
        if queue.get('success'):break
        time.sleep(5)
    name='H5_group00_6s_whole_robot_prior025'
    out=OUT/name
    if not (out/'generated_seed42.mp4').exists():
        subprocess.run(['/mnt/f/AIRI_task/.venv-das/bin/python',
            str(ROOT/'scripts/das_full_motion_generate.py'),'--name',name,
            '--preparation-subdir','group00_stretched_6s_whole_robot_v1','--strength','0.25'],check=True)
    else:
        assert json.loads((out/'resource_usage.json').read_text())['success']
    subprocess.run(['/mnt/f/AIRI_task/.venv/bin/python',
        str(ROOT/'scripts/das_full_motion_evaluate.py'),'--name',name],check=True)
    subprocess.run(['/mnt/f/AIRI_task/.venv/bin/python',
        str(ROOT/'scripts/das_full_motion_robot_evaluate.py'),'--name',name],check=True)
    write_json(OUT/'whole_robot_queue_completion.json',{'success':True,
        'variant':name,'started_after_matched_queue':True,'future_used_in_generation':False})


if __name__=='__main__':main()
