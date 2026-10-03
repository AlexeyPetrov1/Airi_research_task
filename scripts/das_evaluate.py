"""Measure generated -> control and generated -> real on physical times.

Real future is used exclusively here, after generation and frozen controls.
AllTracker correspondence is a measurement proxy, not physical ground truth.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from das_prepare_control import write_json, sha256, save_video, sheet, project
from das_quality import temporal_warp_error, cup_appearance


def frames(path):
    with imageio.get_reader(str(path)) as reader:
        return np.stack([frame for frame in reader])


def interpolate(tracks,times,query):
    return np.stack([np.stack([np.interp(query,times,tracks[:,i,d]) for d in range(2)],-1)
                     for i in range(tracks.shape[1])],axis=1)


def metrics(reference,actual,valid):
    error=np.linalg.norm(actual-reference,axis=-1)
    valid=valid&np.isfinite(error)
    count=valid.sum(1)
    per_time=[float(error[i,valid[i]].mean()) if count[i] else None for i in range(len(error))]
    common=valid.all(0)
    return {'ADE_px':float(error[valid].mean()) if valid.any() else None,
            'FDE_px':per_time[-1], 'visible_pairs':int(valid.sum()),'total_pairs':int(valid.size),
            'coverage':float(valid.mean()),'valid_points_by_time':count.tolist(),
            'error_by_time_px':per_time,'complete_track_count':int(common.sum()),
            'complete_track_ADE_px':float(error[:,common].mean()) if common.any() else None,
            'complete_track_FDE_px':float(error[-1,common].mean()) if common.any() else None}


def track_generated(scene,out,generated,tracker_repo,checkpoint,destination):
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,str(tracker_repo/'data_generation/third_party/alltracker'))
    from nets.alltracker import Net
    uv=np.load(scene/'observed/points_2d_history.npy')[-1].astype(np.float32)
    ids=np.load(scene/'observed/selected_point_ids.npy')
    # Undo identical anisotropic RGB/control resize before measuring in source pixels.
    native=np.stack([cv2.resize(f,(640,480),interpolation=cv2.INTER_LINEAR) for f in generated[:17]])
    t0=np.array(Image.open(scene/'observed/frame_000063.png').convert('RGB'))
    # A reference image anchor provides correspondence into generated frame 0;
    # discard its extra index before evaluating 0..2 s.
    clip=np.concatenate([t0[None],native])
    scaled=np.stack([cv2.resize(f,(512,384),interpolation=cv2.INTER_LINEAR) for f in clip])
    model=Net(16)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    model=model.cuda().eval()
    rgbs=torch.from_numpy(scaled.copy()).permute(0,3,1,2)[None].float().cuda()
    torch.cuda.reset_peak_memory_stats(); started=time.monotonic()
    with torch.inference_mode():
        flow,scores,_,_=model(rgbs,iters=4,sw=None,is_training=False)
        points=uv*np.array([.8,.8],np.float32)
        norm=points/np.array([511,383],np.float32)*2-1
        grid=torch.from_numpy(norm).cuda()[None,None].expand(len(clip),-1,-1,-1)
        sampled=F.grid_sample(flow[0],grid,align_corners=True)[:,:,0].permute(0,2,1)
        conf=F.grid_sample(scores[0],grid,align_corners=True)[:,:,0].permute(0,2,1).cpu().numpy()
        xy=(sampled.cpu().numpy()+points[None])/.8
    torch.cuda.synchronize()
    xy,conf=xy[1:],conf[1:]
    np.savez_compressed(destination,tracks=xy.astype(np.float32),visibility=conf[...,0]>.5,
                        visibility_probability=conf[...,0],confidence=conf[...,1],point_ids=ids,
                        time_from_t0_s=np.arange(17)/8,dim=np.array([480,640]))
    write_json(destination.with_suffix('.json'),{'tracker':'official AllTracker Net(16)','iterations':4,
        'checkpoint_sha256':sha256(checkpoint),'generated_sha256':sha256(out/'generated_molmomotion_seed42.mp4')
        if destination.name=='generated_tracking.npz' else sha256(out/'generated_no_control_seed42.mp4'),
        'reference_anchor':'real t0 followed by generated frame 0..16; discard extra real anchor index',
        'future_real_frames_used_for_tracking':False,'source_resolution':[640,480],
        'tracker_resolution':[512,384],'seconds':time.monotonic()-started,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'limitation':'Tracker cannot certify object identity if it deforms/disappears. Use visibility masks and visual audit.'})


def label(frame,text):
    image=Image.fromarray(frame.copy())
    draw=ImageDraw.Draw(image);draw.rectangle((0,0,image.width,26),fill='white');draw.text((8,7),text,fill='black')
    return np.array(image)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--tracker-repo',type=Path,required=True)
    p.add_argument('--tracker-checkpoint',type=Path,required=True)
    p.add_argument('--no-control',action='store_true')
    p.add_argument('--artifact-dir',type=Path)
    a=p.parse_args();scene=a.scene;out=a.artifact_dir or scene/'das_wanfun'
    generated_path=out/('generated_no_control_seed42.mp4' if a.no_control else 'generated_molmomotion_seed42.mp4')
    generated=frames(generated_path)
    assert generated.shape==(49,480,720,3)
    frozen=json.loads((out/'control_input_freeze.json').read_text())['sha256']
    assert all(sha256(scene/name)==h for name,h in frozen.items()),'Causal inputs changed'
    suffix='_no_control' if a.no_control else ''
    destination=out/f'generated_tracking{suffix}.npz'
    if not destination.exists():
        track_generated(scene,out,generated,a.tracker_repo,a.tracker_checkpoint,destination)
    tracking=np.load(destination)
    gen_xy=tracking['tracks'];gen_vis=tracking['visibility']
    times=np.arange(1,11)/5
    gen_eval=interpolate(gen_xy,np.arange(17)/8,times)
    # Require both endpoints for interpolated correspondence visibility.
    lo=np.floor(times*8).astype(int);hi=np.ceil(times*8).astype(int)
    visible=gen_vis[lo]&gen_vis[hi]
    # First real future access occurs below, with generation and input hashes checked.
    real=np.load(scene/'evaluation/future_rgb.npy')
    real_tracks=np.load(scene/'evaluation/evaluation_tracks_2d.npz')
    assert np.array_equal(real_tracks['point_ids'],tracking['point_ids'])
    real_xy=real_tracks['tracks'][1:];real_vis=real_tracks['visibility'][1:]
    predicted=np.load(scene/'predictions/future_3d.npy')
    k=np.load(scene/'geometry/K_median.npy')
    pred_xy=np.stack([project(predicted[:,i],k) for i in range(30)])
    pred_eval=pred_xy[np.arange(2,30,3)]
    sparse=np.load(out/'dense_predicted_motion.npz')['sparse_xyz']
    control_xy=np.stack([project(points,k) for points in sparse[:17]])
    control_eval=interpolate(control_xy,np.arange(17)/8,times)
    fitted=np.load(out/'rigid_motion.npz')
    p0=np.load(scene/'observed/points_3d_history.npy')[-1]
    rigid_15hz=np.einsum('tij,nj->tni',fitted['R'],p0)+fitted['t'][:,None]
    rigid_eval=np.stack([project(points,k) for points in rigid_15hz])[np.arange(2,30,3)]
    validity=lambda xy:np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<640)&(xy[...,1]>=0)&(xy[...,1]<480)
    control_vis=validity(control_eval)
    shared=visible&real_vis&control_vis&validity(gen_eval)
    report={'coordinate_units':'original 640x480 pixels','evaluation_times_s':times.tolist(),
        'sampling_generated':'linear interpolation of 8 Hz AllTracker positions to physical 5 Hz real times; endpoint visibility required',
        'molmomotion_to_real':metrics(real_xy,pred_eval,real_vis),
        'rigid_to_molmomotion':metrics(pred_eval,rigid_eval,np.ones_like(shared)),
        'resampled_control_to_rigid':metrics(rigid_eval,control_eval,np.ones_like(shared)),
        'generated_to_control':metrics(control_eval,gen_eval,visible&control_vis&validity(gen_eval)),
        'generated_to_real':metrics(real_xy,gen_eval,visible&real_vis&validity(gen_eval)),
        'shared_mask_generated_to_control':metrics(control_eval,gen_eval,shared),
        'shared_mask_generated_to_real':metrics(real_xy,gen_eval,shared),
        'shared_mask_molmomotion_to_real':metrics(real_xy,pred_eval,shared),
        'generated_anchor_error_px':float(np.linalg.norm(gen_xy[0]-np.load(scene/'observed/points_2d_history.npy')[-1],axis=-1).mean()),
        'generated_frame_sha256':sha256(generated_path),'real_tracking_sha256':sha256(scene/'evaluation/evaluation_tracks_2d.npz'),
        'image_quality_scope':'descriptive single-clip metrics; no calibrated perceptual quality or distribution-level FVD claim',
        'tracking_limitation':'AllTracker proxy; visibility filtering may leave too few valid target correspondences.',
        'RGB_quality_sampling':'nearest generated 8 Hz frame to real 5 Hz time; maximum timestamp offset 0.05 s, separate from interpolated motion metrics',
        'RGB_quality_sample_times_s':(np.rint(times*8)/8).tolist(),
        'no_control_ablation':a.no_control}
    native=np.stack([cv2.resize(f,(640,480),interpolation=cv2.INTER_LINEAR) for f in generated])
    sampled=np.stack([native[int(round(t*8))] for t in times])
    report['image_metrics']={'mean_generated_to_real_MAE_0_255':float(np.abs(sampled.astype(float)-real).mean()),
        'mean_generated_to_real_PSNR_db':float(np.mean([cv2.PSNR(g,r) for g,r in zip(sampled,real)])),
        'mean_generated_laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(g,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for g in native[:17]])),
        'mean_real_laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(r,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for r in real])),
        'mean_generated_adjacent_MAE_0_255_8Hz':float(np.abs(np.diff(native[:17].astype(float),axis=0)).mean()),
        'mean_real_adjacent_MAE_0_255_5Hz':float(np.abs(np.diff(real.astype(float),axis=0)).mean()),
        'mean_generated_adjacent_MAE_0_255_at_real_5Hz_times':float(np.abs(np.diff(sampled.astype(float),axis=0)).mean()),
        'interpretation':'MAE/PSNR penalize different valid motion; adjacent MAE includes motion and different FPS. Sharpness variance is not perceptual quality.'}
    report['temporal_consistency_generated_nearest_5Hz_times']=temporal_warp_error(sampled)
    report['temporal_consistency_real_at_5Hz']=temporal_warp_error(real)
    reference=np.array(Image.open(scene/'observed/frame_000063.png').convert('RGB'))
    mask=np.array(Image.open(scene/'observed/mask.png').convert('L'))>0
    report['object_appearance_proxy']=cup_appearance(native[:17],gen_xy,gen_vis,reference,mask)
    write_json(out/f'metrics{suffix}.json',report)
    np.savez_compressed(out/f'aligned_evaluation{suffix}.npz',generated=gen_eval,control=control_eval,
                        molmomotion=pred_eval,real=real_xy,generated_visible=visible,real_visible=real_vis,
                        shared_mask=shared,times=times,point_ids=tracking['point_ids'])
    controls=frames(out/'control_molmomotion_640x480.mp4')
    real_with_t0=np.concatenate([np.array(Image.open(scene/'observed/frame_000063.png').convert('RGB'))[None],real])
    real_indices=np.clip(np.rint(np.arange(17)/8*5).astype(int),0,10)
    real_8hz=real_with_t0[real_indices]
    comparison=[];real_gen=[];control_gen=[]
    for i in range(17):
        t=i/8
        row=[label(real_8hz[i],f'REAL {real_indices[i]/5:.3f}s (nearest to {t:.3f}s)'),label(controls[i],f'DAS CONTROL {t:.3f}s'),label(native[i],f'GENERATED {t:.3f}s')]
        comparison.append(np.concatenate(row,axis=1));real_gen.append(np.concatenate([row[0],row[2]],axis=1));control_gen.append(np.concatenate(row[1:],axis=1))
    save_video(out/f'triple_comparison{suffix}.mp4',comparison)
    save_video(out/f'real_vs_generated{suffix}.mp4',real_gen)
    save_video(out/f'control_vs_generated{suffix}.mp4',control_gen)
    sheet(out/f'generated_contact_sheet{suffix}.png',[real_8hz,controls,native],['REAL (nearest 5 Hz)','DAS CONTROL','GENERATED'])
    sheet(out/f'generated_full_duration{suffix}.png',[native],['GENERATED; control held after 2 s'],indices=(0,8,16,24,32,40,48))
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,xy,color in [('Control',control_eval,'magenta'),('Generated',gen_eval,'blue'),('Real',real_xy,'green'),('MolmoMotion',pred_eval,'orange')]:
        axes[0].plot(*xy.mean(1).T,'o-',label=name,color=color)
    axes[0].invert_yaxis();axes[0].set(xlabel='X (px)',ylabel='Y (px)',title='Mean of 24 queried positions (visibility unfiltered)');axes[0].legend()
    for key in ['generated_to_control','generated_to_real','molmomotion_to_real']:
        axes[1].plot(times,[np.nan if v is None else v for v in report[key]['error_by_time_px']],label=key)
    axes[1].set(xlabel='Physical time (s)',ylabel='Visible-point mean error (px)',title='Different masks; see JSON shared-mask metrics');axes[1].legend()
    fig.tight_layout();fig.savefig(out/f'motion_comparison{suffix}.png',dpi=160);plt.close(fig)
    print(json.dumps({key:report[key] for key in ['generated_to_control','generated_to_real']},indent=2))


if __name__=='__main__': main()
