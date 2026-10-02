"""Queue genuine Berkeley grounding behind existing GPU work; never stop other jobs."""
from pathlib import Path
import argparse,json,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[1]


def command_line(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    except FileNotFoundError:return ''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--wait-command',required=True)
    p.add_argument('--scene-dir',type=Path,nargs='+',required=True)
    args=p.parse_args()
    receipt=args.scene_dir[0].parent/'gpu_queue.json'
    start=time.monotonic()
    last_print=-100
    while True:
        other=command_line(args.wait_pid)
        occupied=args.wait_command in other
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        ready=not occupied and memory<1500
        status={'stage':'ready' if ready else 'waiting_for_existing_gpu_work','wait_pid':args.wait_pid,
                'existing_command':other,'gpu_memory_used_mib':memory,'queue_seconds':time.monotonic()-start,
                'never_stops_other_jobs':True}
        receipt.write_text(json.dumps(status,indent=2)+'\n')
        if time.monotonic()-last_print>30 or ready:
            print(json.dumps(status),flush=True)
            last_print=time.monotonic()
        if ready:break
        time.sleep(5)
    command=[sys.executable,str(ROOT/'scripts/berkeley_ground_molmopoint.py'),'--scene-dir',*map(str,args.scene_dir)]
    status['stage']='grounding';receipt.write_text(json.dumps(status,indent=2)+'\n')
    result=subprocess.run(command,cwd=ROOT)
    status.update(stage='complete' if result.returncode==0 else 'grounding_failed',returncode=result.returncode,
                  elapsed_seconds=time.monotonic()-start)
    receipt.write_text(json.dumps(status,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
