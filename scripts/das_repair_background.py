"""Causal disocclusion repair for dense DaS control, without future RGB/depth.

The initial object is removed from the static depth layer and its footprint
is filled with an estimated background. This is an explicit estimate of an
unobserved surface, not measured depth and not a correction to MolmoMotion.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from das_prepare_control import backproject, project, splat, save_video, sheet, sha256, write_json


def complete_background(depth, mask, dilation=5, min_depth=None):
    repair=cv2.dilate(mask.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilation+1,)*2))>0
    valid=np.isfinite(depth)&(depth>0)&(depth<10)&~repair
    # Fill invalid measurements for numerical neighbourhoods only. Values outside
    # the repair mask are never written to the actual background layer.
    indices=distance_transform_edt(~valid,return_distances=False,return_indices=True)
    inverse=1/depth[tuple(indices)]
    repaired=cv2.inpaint(inverse.astype(np.float32),repair.astype(np.uint8)*255,5,cv2.INPAINT_NS)
    ring=(cv2.dilate(repair.astype(np.uint8),np.ones((21,21),np.uint8))>0)&~repair&valid
    lo,hi=np.percentile(inverse[ring],[2,98])
    result=depth.copy()
    result[repair]=1/np.clip(repaired[repair],lo,hi)
    if min_depth is not None:
        result[repair]=np.maximum(result[repair],min_depth)
    assert np.isfinite(result[repair]).all() and (result[repair]>0).all()
    return result,repair


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    original=a.scene/'das_wanfun';out=a.output;out.mkdir(parents=True,exist_ok=True)
    if (out/'generated_molmomotion_seed42.mp4').exists():raise FileExistsError('Preserve generated experiment')
    freeze=json.loads((original/'control_input_freeze.json').read_text())
    assert all(sha256(a.scene/n)==h for n,h in freeze['sha256'].items())
    for name in ['control_input_freeze.json','rigid_motion.npz','rigid_motion_49.npz','dense_predicted_motion.npz',
                 'object_cloud_t0.npz','rigid_fit.json','temporal_mapping.json','das_wanfun_commit.txt',
                 'checkpoint_receipt.json','image_t0_720x480.png']:
        shutil.copyfile(original/name,out/name)
    rgb=np.array(Image.open(a.scene/'observed/frame_000063.png').convert('RGB'))
    mask=np.array(Image.open(a.scene/'observed/mask.png').convert('L'))>0
    depth=np.load(a.scene/'observed/native_depth.npy')[-1]
    k=np.load(a.scene/'geometry/K_median.npy')
    obj=np.load(original/'object_cloud_t0.npz')
    background_near_limit=float(obj['xyz'][:,2].max()+.02)
    completed,repair=complete_background(depth,mask,min_depth=background_near_limit)
    valid=np.isfinite(depth)&(depth>0)&(depth<10)
    p2,p98=np.percentile(1/depth[valid],[2,98])
    y,x=np.nonzero((valid&~repair)|repair);uv=np.c_[x,y]
    bg_depth=completed[y,x]
    colors=np.stack([x/639*255,y/479*255,np.clip((1/bg_depth-p2)/(p98-p2),0,1)*255],axis=1).astype(np.uint8)
    bg=np.zeros_like(rgb);bg_z=np.full(mask.shape,np.inf)
    splat(backproject(uv,bg_depth,k),colors,k,bg,bg_z,radius=0)
    dense=np.load(original/'dense_predicted_motion.npz')['xyz']
    native=[];moving=[]
    for xyz in dense:
        canvas,z=bg.copy(),bg_z.copy();splat(xyz,obj['colors'],k,canvas,z,radius=1)
        native.append(canvas)
        foreground=np.zeros_like(rgb);foreground_z=np.full(mask.shape,np.inf)
        splat(xyz,np.full_like(obj['colors'],255),k,foreground,foreground_z,radius=1)
        moving.append(foreground[...,0]>0)
    resized=[np.array(Image.fromarray(f).resize((720,480),Image.Resampling.BILINEAR)) for f in native]
    save_video(out/'control_molmomotion_640x480.mp4',native)
    save_video(out/'control_molmomotion_720x480.mp4',resized)
    Image.fromarray(bg).save(out/'completed_background_control.png')
    Image.fromarray(repair.astype(np.uint8)*255).save(out/'background_repair_mask.png')
    np.savez_compressed(out/'background_completion.npz',depth=completed,repair_mask=repair,
                        moving_mask=np.stack(moving),original_mask=mask)
    import imageio.v2 as imageio
    with imageio.get_reader(str(original/'control_molmomotion_640x480.mp4')) as reader:
        before=[frame for frame in reader]
    sheet(out/'control_contact_sheet.png',[before,native],['ORIGINAL CONTROL: EMPTY HOLE','REPAIRED: ESTIMATED BACKGROUND'])
    review={'generation_ready':False,'visual_review_required':True,'future_used':False,
            'control_sha256':sha256(out/'control_molmomotion_720x480.mp4'),
            'variant':'background_completion_v1','forecast_changed':False,'reference_RGB_changed':False,
            'estimated_depth_pixels':int(repair.sum()),'object_mask_pixels':int(mask.sum()),
            'background_depth_lower_bound_m':background_near_limit,
            'method':'Navier-Stokes interpolation of t0 inverse depth in dilated object footprint; neighbour depth clipping; unchanged measured background elsewhere',
            'limitation':'Depth behind observed cup is not measured. This is an explicit causal disocclusion estimate.',
            'source_control_sha256':sha256(original/'control_molmomotion_720x480.mp4'),
            'identical_dense_motion_sha256':sha256(out/'dense_predicted_motion.npz')}
    write_json(out/'control_validation.json',review)
    print(json.dumps(review,indent=2))


if __name__=='__main__':main()
