"""Wait for verified process identities before starting another GPU command."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--after-pid',type=int,action='append',default=[])
p.add_argument('--receipt',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
command=a.command[1:] if a.command[:1]==['--'] else a.command
def identity(pid):
    try:
        # Start ticks protect against PID reuse without importing Torch/psutil.
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
identities={pid:identity(pid) for pid in a.after_pid};start=time.monotonic()
while any(token is not None and identity(pid)==token for pid,token in identities.items()):
    state={'state':'waiting','pid':os.getpid(),'after_pids':identities,'seconds':time.monotonic()-start,'command':command}
    a.receipt.write_text(json.dumps(state,indent=2)+'\n');print('Waiting for existing GPU process identities',identities,flush=True)
    time.sleep(15)
a.receipt.write_text(json.dumps({'state':'started','wait_seconds':time.monotonic()-start,'command':command},indent=2)+'\n')
raise SystemExit(subprocess.call(command))
