"""Preserve this run's slow partial call before an explicit CPU-thread correction."""
import json,os,signal,time,shutil
from datetime import datetime,timezone
from pathlib import Path
from fmb_wrist_prepare import write,sha
from fmb_wrist_v4_diagnose import OUT

state=json.loads((OUT/'inference_status.json').read_text());pid=state['pid']
cmdpath=Path('/proc')/str(pid)/'cmdline'
cmd=cmdpath.read_bytes()
assert b'scripts/fmb_wrist_v4_infer.py' in cmd and b'/mnt/f/AIRI_task/.venv/bin/python' in cmd
group=OUT/state['camera']/'variants'/state['variant']/f'group_{state["group"]:02d}'
archive=OUT/'runtime_interruption_00';assert not archive.exists()
start=time.monotonic();os.kill(pid,signal.SIGTERM)
while cmdpath.exists():
    try:current=cmdpath.read_bytes()
    except OSError:break
    if current!=cmd:break
    assert time.monotonic()-start<20,'Process still live: do not move its outputs'
    time.sleep(.2)
archive.mkdir()
status=json.loads((group/'model_run.json').read_text())
if status.get('success'):
    partial=False
else:
    partial=True;shutil.move(str(group),str(archive/group.name))
write(archive/'receipt.json',{'recorded_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,'verified_command':cmd.replace(b'\0',b' ').decode(),
      'state_at_interruption':state,'partial_group_archived':partial,'generation_started':status['generation_started'],
      'completed_group_discarded':False,'external_process_terminated':False,'reason':'First native call took692s; explicit resumption with4 CPU threads, matching the earlier continuation configuration. Frozen scientific inputs/checkpoint/greedy decoding unchanged.',
      'original_inference_script_sha256':sha(OUT/'code_snapshots/fmb_wrist_v4_infer_initial.py')})
write(OUT/'inference_status.json',{'status':'INTERRUPTED_FOR_RUNTIME_CONFIGURATION','pid':pid,'preserved_completed_groups':True})
print('Only our verified process stopped; completed groups preserved; partial attempt archived',flush=True)
