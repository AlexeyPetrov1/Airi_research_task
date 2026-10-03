"""Continue already-authorized full cases after the live H1 pilot finishes.

Waits for a specific live process identity, then runs GPU jobs sequentially.
No automatic repeat of failed generations; recovery uses actual saved outputs.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1'


def write(status):
    (OUT/'continuation.json').write_text(json.dumps(status,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait-pid',type=int,required=True)
    a=p.parse_args();start=time.monotonic()
    while True:
        cmd=Path(f'/proc/{a.wait_pid}/cmdline')
        live=cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else ''
        if 'berkeley_geometry_infer.py' not in live or 'CASE-H1' not in live:break
        status=dict(stage='waiting_live_H1_pilot',pid=a.wait_pid,elapsed_s=time.monotonic()-start)
        write(status);print(json.dumps(status),flush=True);time.sleep(15)
    h1=OUT/'CASE-H1/cup/predictions/model_run.json'
    if not h1.exists() or not json.loads(h1.read_text()).get('success'):
        raise RuntimeError('H1 terminated without successful output. Inspect actual process and receipts before recovery.')
    commands=[[sys.executable,str(ROOT/'scripts/berkeley_geometry_compare.py'),'--case','CASE-H1','--scene','cup']]
    for case in ('CASE-AUGE','CASE-NOK'):
        commands.append([sys.executable,str(ROOT/'scripts/berkeley_geometry_infer.py'),'--scenes',
            str(OUT/case/'cup'),str(OUT/case/'bottle'),'--groups','0','1','2'])
        for scene in ('cup','bottle'):
            commands.append([sys.executable,str(ROOT/'scripts/berkeley_geometry_compare.py'),'--case',case,'--scene',scene])
    for command in commands:
        status=dict(stage='running',command=command,elapsed_s=time.monotonic()-start)
        write(status);print(json.dumps(status),flush=True)
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:
            write(dict(status,stage='failed',returncode=result.returncode));raise SystemExit(result.returncode)
    write(dict(stage='complete',elapsed_s=time.monotonic()-start,
               remaining='Root must visually review full-case movies, verify evidence, complete ShareRobot mapping and publish final report'))


if __name__=='__main__':main()
