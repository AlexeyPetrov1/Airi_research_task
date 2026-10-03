"""Continue observed-only preprocessing after the specific live ViPE worker."""
import argparse,os,subprocess,time
from pathlib import Path
from fmb_wrist_prepare import ROOT,RUN,write

def main(pid):
    while True:
        try:cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        except OSError:break
        if b'fmb_wrist_vipe.py' not in cmd:break
        write(RUN/'preprocess_wait.json',{'status':'WAITING_FOR_VERIFIED_VIPE_PROCESS','pid':pid,'checked_at_unix':time.time()});time.sleep(30)
    env=os.environ.copy();env.update(HF_HOME=str(ROOT.parent/'.cache/huggingface'),TORCH_HOME=str(ROOT.parent/'.cache/torch'),PYTHONUNBUFFERED='1')
    py=str(ROOT.parent/'.venv/bin/python');scene_args=[str(RUN/c) for c in ['wrist_2','wrist_1']]
    missing=[scene for scene in scene_args if not(Path(scene)/'observed/molmopoint_grounding.json').exists()]
    commands=[['scripts/fmb_wrist_ground.py','--scene-dir',*missing]] if missing else []
    commands += [['scripts/fmb_wrist_preprocess.py',stage] for stage in ['sam','track','unidepth','h3-masks']]
    logpath=RUN/'observed_preprocess.log'
    with logpath.open('a') as log:
        for argv in commands:
            print('RUN',argv,flush=True);result=subprocess.run([py,*argv],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(RUN/'preprocess_status.json',{'command':argv,'exit_code':result.returncode,'status':'RUNNING' if result.returncode==0 else 'FAILED'})
            if result.returncode:raise RuntimeError('Observed preprocessing failed; inspect '+str(logpath))
    write(RUN/'preprocess_status.json',{'status':'COMPLETE','future_used':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int,required=True);a=p.parse_args();main(a.wait_pid)
