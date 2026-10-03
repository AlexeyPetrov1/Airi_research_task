"""Serialize this experiment with other live GPU work, then gate future evaluation."""
import os,subprocess,time
from pathlib import Path
from fmb_wrist_prepare import ROOT,RUN,write

py=str(ROOT.parent/'.venv/bin/python')
def busy():
    found=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=path.read_bytes()
        except OSError:continue
        if any(name in cmd for name in [b'das_generate.py',b'berkeley_infer.py',b'fmb_v2_infer.py',b'vipe.cli.main']):
            found.append({'pid':int(path.parent.name),'command':cmd.replace(b'\0',b' ').decode(errors='replace')})
    return found

quiet_since=None;last=None
while True:
    active=busy()
    if active:
        quiet_since=None
        if active!=last:print('Waiting for existing GPU work',active,flush=True)
        write(RUN/'forecast_pipeline_status.json',{'status':'WAITING_FOR_GPU','external_active_processes':active,'future_used':False})
    else:
        if quiet_since is None:quiet_since=time.monotonic()
        if time.monotonic()-quiet_since>=30:break
    last=active;time.sleep(10)

scenes=[RUN/'wrist_2/branches'/name for name in ['A_vipe_full','B_sensor_vipe','C_sensor_tcp_official']]+[RUN/'wrist_1/branches/C_sensor_tcp_official']
tasks=[['fmb_v2_infer.py','--scene-dir',*[str(scene) for scene in scenes]],
       ['fmb_wrist_evaluate.py','export'],['fmb_wrist_evaluate.py','track'],['fmb_wrist_evaluate.py','evaluate'],
       ['fmb_wrist_blinded_media.py'],['fmb_wrist_reference_uncertainty.py'],['fmb_wrist_future_crossview.py'],['fmb_wrist_side_future.py']]
with (RUN/'forecast_pipeline.log').open('w') as log:
    for args in tasks:
        command=[py,'-u',str(ROOT/'scripts'/args[0]),*args[1:]];print('RUN',args[0],*args[1:2],flush=True)
        write(RUN/'forecast_pipeline_status.json',{'status':'RUNNING','command':command,'future_evaluation_started':args[0]!='fmb_v2_infer.py'})
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            write(RUN/'forecast_pipeline_status.json',{'status':'FAILED','command':command,'exit_code':result.returncode});raise RuntimeError('See forecast_pipeline.log')
write(RUN/'forecast_pipeline_status.json',{'status':'COMPLETE','builtin_future_review_pending':True})
print('Model/evaluation outputs ready; built-in blinded image review pending',flush=True)
