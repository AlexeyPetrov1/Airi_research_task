"""Wait for verified observed preprocessing, then run causal geometry checks."""
import argparse,json,subprocess,time
from pathlib import Path
from fmb_wrist_prepare import ROOT,RUN,write

def main(pid,wait_text):
    while True:
        try:cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        except OSError:break
        if wait_text.encode() not in cmd:break
        time.sleep(15)
    assert json.loads((RUN/'preprocess_status.json').read_text())['status']=='COMPLETE'
    py=str(ROOT.parent/'.venv-vipe/bin/python');tasks=[['scripts/fmb_wrist_geometry.py','diagnose'],['scripts/fmb_wrist_geometry.py','export'],['scripts/fmb_wrist_crossview.py']]
    with (RUN/'geometry_pipeline.log').open('a') as log:
        for argv in tasks:
            print('RUN',argv,flush=True);result=subprocess.run([py,*argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            write(RUN/'geometry_status.json',{'status':'RUNNING' if result.returncode==0 else 'FAILED','command':argv,'exit_code':result.returncode})
            if result.returncode:raise RuntimeError('Geometry pipeline failed; see geometry_pipeline.log')
    write(RUN/'geometry_status.json',{'status':'COMPLETE','future_used':False,'visual_review_pending':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);p.add_argument('--wait-text',default='fmb_wrist_after_vipe.py');a=p.parse_args();main(a.wait_pid,a.wait_text)
