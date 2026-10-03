"""Run one authorized case pilot after an identified live workflow completes."""
from pathlib import Path
import argparse,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--wait-text',required=True)
    p.add_argument('--case',required=True);a=p.parse_args();start=time.monotonic()
    record=OUT/a.case/'queue.json'
    while True:
        cmd=Path(f'/proc/{a.wait_pid}/cmdline')
        live=cmd.read_bytes().replace(b'\0',b' ').decode() if cmd.exists() else ''
        if a.wait_text not in live:break
        status=dict(stage='waiting_verified_workflow',pid=a.wait_pid,command=live,elapsed_s=time.monotonic()-start)
        record.write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True);time.sleep(15)
    for command in ([sys.executable,str(ROOT/'scripts/berkeley_geometry_infer.py'),'--scenes',str(OUT/a.case/'cup'),'--groups','0'],
                    [sys.executable,str(ROOT/'scripts/berkeley_geometry_compare.py'),'--case',a.case,'--scene','cup']):
        record.write_text(json.dumps(dict(stage='running',command=command),indent=2)+'\n')
        result=subprocess.run(command,cwd=ROOT)
        if result.returncode:
            record.write_text(json.dumps(dict(stage='failed',returncode=result.returncode,command=command),indent=2)+'\n')
            raise SystemExit(result.returncode)
    record.write_text(json.dumps(dict(stage='pilot_complete',visual_review_pending=True),indent=2)+'\n')


if __name__=='__main__':main()
