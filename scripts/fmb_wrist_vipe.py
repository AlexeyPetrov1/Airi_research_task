"""Run official ViPE after currently live independent GPU work finishes."""
from __future__ import annotations
import argparse, json, os, resource, subprocess, sys, time
from pathlib import Path
from fmb_wrist_prepare import ROOT,RUN,write,sha

def busy_processes():
    busy=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:text=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except OSError:continue
        if 'python' in text and any(s in text for s in ['berkeley_geometry_infer.py','berkeley_improvement_continue.py','berkeley_improvement_queue.py']):
            busy.append({'pid':int(p.name),'cmdline':text})
    return busy

def run(camera,preset='default',wait=True):
    dest=RUN/camera
    if wait:
        while busy:=busy_processes():
            write(RUN/'gpu_wait.json',{'status':'WAITING_FOR_LIVE_PROCESSES','processes':busy,'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
            print('Verified other live GPU work',[(b['pid'],b['cmdline'].split('/scripts/')[-1]) for b in busy],flush=True)
            time.sleep(30)
    receipt=dest/f'vipe_{preset}_receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='COMPLETE':return
    env=os.environ.copy();env.update(HF_HOME=str(ROOT.parent/'.cache/huggingface'),TORCH_HOME=str(ROOT.parent/'.cache/torch'),CUDA_HOME='/usr/local/cuda-12.8',PYTHONUNBUFFERED='1')
    video=dest/'observed/observed_10hz.mp4';assert video.exists()
    argv=[sys.executable,'-m','vipe.cli.main','infer',str(video),'-p',preset,'-o',str(dest/f'vipe_{preset}')]
    info={'status':'RUNNING','camera':camera,'argv':argv,'nominal_fps':10,'observed_video_sha256':sha(video),
          'future_used':False,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'protocol_sha256':sha(RUN/'PROTOCOL.md')}
    write(receipt,info);start=time.monotonic()
    with (dest/f'vipe_{preset}.log').open('w') as log:
        result=subprocess.run(argv,cwd=ROOT/'data_generation/third_party/vipe',env=env,stdout=log,stderr=subprocess.STDOUT)
    info.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,
                elapsed_seconds=time.monotonic()-start,peak_child_ram_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20)
    write(receipt,info);print(camera,preset,info,flush=True)
    if result.returncode:raise RuntimeError(f'ViPE failed: {dest}/{preset}.log')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);p.add_argument('--preset',default='default');p.add_argument('--no-wait',action='store_true')
    a=p.parse_args()
    for camera in a.cameras:run(camera,a.preset,not a.no_wait)
