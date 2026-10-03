"""Archive this run's unfinished call before a documented resource-only retry."""
import os,signal,time
from pathlib import Path
from datetime import datetime,timezone
from fmb_wrist_prepare import ROOT,RUN,write,sha

pid=16863;proc=Path('/proc')/str(pid)
cmd=(proc/'cmdline').read_bytes()
assert b'fmb_v2_infer.py' in cmd and b'fmb_wrist_v3/wrist_2/branches/A_vipe_full' in cmd,'Unexpected process; refuse termination'
out=RUN/'wrist_2/branches/A_vipe_full/predictions';group=out/'group_00'
assert (group/'model_run.json').exists() and not(group/'raw_model_output.txt').exists(),'Inspect completed outputs instead of aborting'
os.kill(pid,signal.SIGTERM)
for _ in range(60):
    if not proc.exists():break
    time.sleep(.5)
assert not proc.exists(),'Own process has not stopped; archive refused'
archive=out/'interrupted_resource_attempt_00';assert not archive.exists();archive.mkdir()
group.rename(archive/'group_00')
write(archive/'interruption_receipt.json',{'interrupted_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,
      'verified_cmdline':cmd.replace(b'\0',b' ').decode(),'generation_started':True,'complete_output_available':False,
      'reason':'Concurrent DAS GPU workload, total device occupancy >11.6GiB on12GB GPU; first native call unfinished after >13min. Stop only own process to serialize compute.',
      'external_process_terminated':False,'future_opened':False,'retry_same_frozen_inputs':True,
      'original_status_sha256':sha(archive/'group_00/model_run.json')})
print('Own unfinished attempt preserved; external DAS process left running',flush=True)
