"""Run remaining observed-only steps and keep a failure receipt."""
import subprocess
from fmb_wrist_prepare import ROOT,RUN,write

tasks=[['fmb_wrist_geometry.py','diagnose'],['fmb_wrist_geometry.py','export'],['fmb_wrist_preprocess.py','track'],['fmb_wrist_visuals.py'],['fmb_wrist_crossview.py']]
with (RUN/'observed_finalization.log').open('w') as log:
    for args in tasks:
        command=[str(ROOT.parent/'.venv-vipe/bin/python'),str(ROOT/'scripts'/args[0]),*args[1:]]
        print('RUN',args,flush=True);result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(RUN/'geometry_status.json',{'status':'RUNNING' if result.returncode==0 else 'FAILED','command':command,'exit_code':result.returncode,'future_used':False})
        if result.returncode:raise RuntimeError('See observed_finalization.log')
write(RUN/'geometry_status.json',{'status':'COMPLETE','future_used':False,'visual_review_pending':True})
