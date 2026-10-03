"""Restore recorded 4:3 aspect before default ViPE; keep source 10Hz and raw model RGB."""
from __future__ import annotations
import argparse,json,os,resource,subprocess,time
from pathlib import Path
import cv2,numpy as np,imageio.v2 as imageio
from fmb_wrist_prepare import ROOT,RUN,write,sha

def run(camera):
    scene=RUN/camera;folder=scene/'observed/aspect_rectified';folder.mkdir(exist_ok=True)
    rgb=np.load(scene/'observed/rgb.npy');restored=np.array([cv2.resize(frame,(256,192),interpolation=cv2.INTER_AREA) for frame in rgb])
    video=folder/'observed_10hz.mp4'
    if not video.exists():
        imageio.mimwrite(video,restored,fps=10,codec='libx264rgb',pixelformat='rgb24',macro_block_size=None,ffmpeg_params=['-crf','0','-preset','fast'])
        assert np.array_equal(np.array(imageio.mimread(video)),restored)
    output=scene/'vipe_rectified_default';receipt=scene/'vipe_rectified_default_receipt.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('status')=='COMPLETE':return
    info={'status':'RUNNING','source_rgb_sha256':sha(scene/'observed/rgb.npy'),'video_sha256':sha(video),
          'source_real_frame_count':len(rgb),'source_fps':10,'restored_size_wh':[256,192],
          'transformation':'Undo official 640x480 -> 256x256 anisotropic RGB resize: 256x256 -> 256x192 pixel-center area resize',
          'original_MolmoMotion_RGB_changed':False,'temporal_resampling':False,'future_used':False,
          'pipeline':'Native default ViPE; source aspect restoration is explicit input preprocessing',
          'evidence':'Pinned official franka_fmb_env.py _get_im cv2.resize(rgb,(256,256)); rs_capture.py 640x480 color profile'}
    py=str(ROOT.parent/'.venv-vipe/bin/python');argv=[py,'-m','vipe.cli.main','infer',str(video),'-p','default','-o',str(output)]
    info['argv']=argv;write(receipt,info);started=time.monotonic()
    env=os.environ.copy();env.update(HF_HOME=str(ROOT.parent/'.cache/huggingface'),TORCH_HOME=str(ROOT.parent/'.cache/torch'),PYTHONUNBUFFERED='1')
    with (scene/'vipe_rectified_default.log').open('w') as log:result=subprocess.run(argv,cwd=ROOT/'data_generation/third_party/vipe',env=env,stdout=log,stderr=subprocess.STDOUT)
    info.update(status='COMPLETE' if result.returncode==0 else 'FAILED',exit_code=result.returncode,elapsed_seconds=time.monotonic()-started,peak_child_ram_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/2**20)
    write(receipt,info)
    if result.returncode:raise RuntimeError('Aspect-restored ViPE failed')
    write(scene/'geometry/vipe_source_selection.json',{'folder':'vipe_rectified_default','native_wh':[256,192],
         'selection_reason':'Physical aspect ratio restored from pinned capture-code evidence; canonical 256x256 ViPE retained as control',
         'future_used':False,'K_pixel_map':'fx_model=fx; fy_model=fy*256/192; cx_model=cx; cy_model=(cy+.5)*256/192-.5',
         'depth_pixel_map':'nearest-neighbor 256x192 -> 256x256, recorded approximate sensor registration unchanged'})
    print(camera,'rectified ViPE completed',info['elapsed_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--wait-pid',type=int);p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);a=p.parse_args()
    if a.wait_pid:
        while True:
            try:cmd=(Path('/proc')/str(a.wait_pid)/'cmdline').read_bytes()
            except OSError:break
            if b'fmb_wrist_after_vipe.py' not in cmd:break
            time.sleep(15)
        assert json.loads((RUN/'preprocess_status.json').read_text())['status']=='COMPLETE'
    for camera in a.cameras:run(camera)
