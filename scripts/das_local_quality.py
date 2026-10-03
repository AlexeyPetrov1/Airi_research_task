"""Local motion-region quality prevents a static background dominating scores."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from das_evaluate import frames
from das_quality import temporal_warp_error
from das_prepare_control import write_json,sha256

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--variant',type=Path,required=True);a=p.parse_args()
root=a.scene/'das_wanfun';out=a.variant
bg=np.load(out/'background_completion.npz');moving=bg['moving_mask'];old=bg['original_mask']
support=np.stack([cv2.dilate((m|old).astype(np.uint8),np.ones((15,15),np.uint8))>0 for m in moving[:17]])
query=np.rint(np.arange(1,11)/5*8).astype(int);support=support[query]
real=np.load(a.scene/'evaluation/future_rgb.npy')
result={'region':'Same union of original cup mask and rendered predicted cup support, dilated 7 px; matched nearest-8Hz frames at real 5Hz times',
        'mean_region_fraction':float(support.mean()),'timestamp_error_max_s':.05,
        'limitation':'Region includes occlusion/background. Low warp error can still reward stationary objects; no identity or perceptual-quality proof.',
        'real':temporal_warp_error(real,support),'runs':{}}
for name,path in [('original',root/'generated_molmomotion_seed42.mp4'),('variant',out/'generated_molmomotion_seed42.mp4')]:
    video=np.stack([cv2.resize(f,(640,480)) for f in frames(path)])[query]
    error=np.abs(video.astype(float)-real).mean(-1)
    result['runs'][name]={'sha256':sha256(path),'local_RGB_MAE_0_255':float(error[support].mean()),
        'local_temporal_warp':temporal_warp_error(video,support),
        'global_temporal_warp':temporal_warp_error(video)}
write_json(out/'local_quality.json',result);print(result)
