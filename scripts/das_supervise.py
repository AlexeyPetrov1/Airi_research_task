"""Persist child exit codes, including OS OOM kills that bypass Python finally."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--receipt',type=Path,required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args()
command=a.command[1:] if a.command and a.command[0]=='--' else a.command
start=time.monotonic()
process=subprocess.Popen(command)
receipt={'command':command,'pid':process.pid,'running':True,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
code=process.wait()
receipt.update(running=False,returncode=code,elapsed_seconds=time.monotonic()-start)
a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
sys.exit(code if code>=0 else 128-code)
