"""Measure residual initial cup motif; a localized proxy, not instance counting."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_evaluate import frames
from das_prepare_control import sheet,write_json,sha256

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--variant',type=Path,required=True);a=p.parse_args()
original=a.scene/'das_wanfun';out=a.variant
reference=np.array(Image.open(a.scene/'observed/frame_000063.png').convert('RGB'))
# Fixed visible cartoon motif, selected from observed t0 before new-video inspection.
# Limited displacement tolerance catches the same stationary cup after small model drift.
box=(128,249,180,292);x1,y1,x2,y2=box;pad=12
template=cv2.cvtColor(reference[y1:y2,x1:x2],cv2.COLOR_RGB2GRAY)
videos={'original':original/'generated_molmomotion_seed42.mp4','repaired':out/'generated_molmomotion_seed42.mp4'}
result={'template_xyxy':box,'search_padding_px':pad,'coordinate_system':'640x480 source pixels',
        'method':'Maximum grayscale normalized template correlation in fixed initial-cup neighbourhood; every decoded frame',
        'limitation':'A texture-presence proxy, not calibrated probability, segmentation, physical identity, or automatic cup count.',
        'videos':{}}
crops=[]
for name,path in videos.items():
    video=np.stack([cv2.resize(f,(640,480)) for f in frames(path)])
    scores=[]
    for f in video:
        search=cv2.cvtColor(f[y1-pad:y2+pad,x1-pad:x2+pad],cv2.COLOR_RGB2GRAY)
        scores.append(float(cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED).max()))
    result['videos'][name]={'sha256':sha256(path),'NCC_by_frame':scores,
        'mean_NCC_frames_4_to_16':float(np.mean(scores[4:17])),
        'mean_NCC_frames_17_to_48':float(np.mean(scores[17:]))}
    crops.append(video[:,190:330,100:205])
    for kind,tiles,size in [('site',video[:,190:330,100:205],(150,200)),
                            ('full',video,(192,144))]:
        w,h=size
        canvas=Image.new('RGB',(7*w,7*(h+22)),'white');draw=ImageDraw.Draw(canvas)
        for i,f in enumerate(tiles):
            x=i%7*w;y=i//7*(h+22)
            canvas.paste(Image.fromarray(f).resize(size),(x,y+22))
            draw.text((x+4,y+4),f'{name} {i}: {i/8:.3f}s',fill='black')
        canvas.save(out/f'all_49_{kind}_{name}.png')
sheet(out/'initial_site_audit.png',crops,['OLD VIDEO: INITIAL CUP SITE','NEW VIDEO: SAME FIXED SITE'],tile=(210,280))
real=np.concatenate([reference[None],np.load(a.scene/'evaluation/future_rgb.npy')])
real_scores=[float(cv2.matchTemplate(cv2.cvtColor(f[y1-pad:y2+pad,x1-pad:x2+pad],cv2.COLOR_RGB2GRAY),
                                  template,cv2.TM_CCOEFF_NORMED).max()) for f in real]
result['real_evaluation']={'times_s':(np.arange(11)/5).tolist(),'NCC_by_frame':real_scores,
                          'source_sha256':sha256(a.scene/'evaluation/future_rgb.npy'),
                          'mean_NCC_0p6_to_2p0_s':float(np.mean(real_scores[3:])),
                          'future_used_only_after_generation':True}
fig,ax=plt.subplots(figsize=(9,4))
for name,values in result['videos'].items():ax.plot(np.arange(49)/8,values['NCC_by_frame'],label=name)
ax.plot(np.arange(11)/5,real_scores,'o-',label='real continuation (5Hz)')
ax.set(xlabel='Physical time (s)',ylabel='Initial cartoon motif NCC',ylim=(-.1,1.05),title='Residual texture at initial cup site; proxy only')
ax.legend();ax.grid(alpha=.3);fig.tight_layout();fig.savefig(out/'initial_site_NCC.png',dpi=150)
write_json(out/'duplicate_texture_audit.json',result)
print({name:values['mean_NCC_frames_4_to_16'] for name,values in result['videos'].items()})
