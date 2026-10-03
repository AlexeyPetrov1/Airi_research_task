"""Create synchronized full-duration before/after views without altering results."""
import argparse
from pathlib import Path
import cv2,numpy as np
from das_evaluate import frames,label
from das_prepare_control import save_video,sheet,write_json,sha256

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--variant',type=Path,required=True);a=p.parse_args()
old_path=a.scene/'das_wanfun/generated_molmomotion_seed42.mp4'
new_path=a.variant/'generated_molmomotion_seed42.mp4'
native=lambda path:np.stack([cv2.resize(f,(640,480)) for f in frames(path)])
old,new=native(old_path),native(new_path);assert old.shape==new.shape==(49,480,640,3)
save_video(a.variant/'before_after_full_duration.mp4',[
    np.concatenate([label(x,f'ORIGINAL {i/8:.3f}s'),label(y,f'REPAIRED {i/8:.3f}s')],axis=1)
    for i,(x,y) in enumerate(zip(old,new))])
sheet(a.variant/'before_after_contact_sheet.png',[old,new],['ORIGINAL','REPAIRED'],indices=(0,2,4,8,16,48))
crop=lambda f:cv2.resize(f[180:340,96:224],(384,480))
save_video(a.variant/'initial_site_before_after.mp4',[
    np.concatenate([label(crop(x),f'ORIGINAL {i/8:.3f}s'),label(crop(y),f'REPAIRED {i/8:.3f}s')],axis=1)
    for i,(x,y) in enumerate(zip(old,new))])
write_json(a.variant/'before_after_receipt.json',{'original_sha256':sha256(old_path),
    'repaired_sha256':sha256(new_path),'frames':49,'fps':8,'source_coordinates':'640x480',
    'site_crop_xyxy':[96,180,224,340],'result_videos_modified':False})
