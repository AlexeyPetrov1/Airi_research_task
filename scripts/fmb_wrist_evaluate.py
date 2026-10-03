"""Prediction-gated, moving-camera future reference, common-mask metrics, media."""
from __future__ import annotations
import argparse,json,subprocess
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from fmb_wrist_prepare import ROOT,RUN,write,sha,CAMERAS
from fmb_wrist_math import *
import berkeley_preprocess as b

TIMES=np.arange(1,21)/10

def prediction_gate():
    receipts=[]
    for dest in sorted((RUN/'wrist_2/branches').glob('*')):
        path=dest/'predictions/model_run.json'
        assert path.exists() and json.loads(path.read_text())['success'],f'Incomplete prediction: {dest}'
        receipts.append({'path':str(path),'sha256':sha(path)})
    assert len(receipts)==3
    assert (RUN/'builtin_visual_review_observed.json').exists(),'Built-in observed visual review not saved'
    return receipts

def export_future():
    receipts=prediction_gate();meta=json.loads((RUN/'wrist_2/metadata.json').read_text())
    d=np.load(meta['source_path'],allow_pickle=True).item();ids=np.arange(meta['t0'],meta['t0']+21)
    for camera in CAMERAS:
        dest=RUN/camera/'evaluation';dest.mkdir(exist_ok=True)
        if (dest/'future_rgb.npy').exists():continue
        np.save(dest/'future_rgb.npy',d['obs/'+camera][ids][...,::-1].copy())
        np.save(dest/'future_sensor_depth.npy',d['obs/'+camera+'_depth'][ids].astype(np.float32)*1e-4)
        np.save(dest/'future_tcp_pose_xyzw.npy',d['obs/tcp_pose'][ids]);np.save(dest/'source_indices.npy',ids)
        write(dest/'export_receipt.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'prediction_receipts':receipts,'source_indices':ids.tolist(),
                'future_used_for_evaluation_only':True,'nominal_source_fps':10,'source_sha256':sha(Path(meta['source_path']))})
    print('Prediction-gated future RGB, Z16 and robot poses exported',flush=True)

def track_future(camera):
    prediction_gate();scene=RUN/camera;dest=scene/'evaluation';path=dest/'future_alltracker.npz'
    if path.exists():return
    rgb=np.load(dest/'future_rgb.npy');selected=np.load(scene/'observed/selected_point_ids.npy')
    tracks=np.load(scene/'observed/observed_tracks_2d.npz')['tracks'];query=tracks[-1,selected]
    xy,confidence,visible,probability,info=b.alltracker_tracks(rgb,query,reverse=False,max_side=256)
    np.savez_compressed(path,tracks=xy,confidence=confidence,visibility=visible,visibility_probability=probability,
                        selected_ids=selected,source_indices=np.load(dest/'source_indices.npy'))
    write(dest/'alltracker_receipt.json',{'future_used_for_evaluation_only':True,**info})
    print(camera,'future tracking complete',flush=True)

def align_prediction(future,initial):
    source=np.r_[0,np.arange(1,31)/15];values=np.concatenate([initial[:,None],future],axis=1)
    return np.array([[np.interp(TIMES,source,values[n,:,axis]) for axis in range(3)] for n in range(24)]).transpose(0,2,1)

def scores(pred,gt,mask):
    dist=np.linalg.norm(pred-gt,axis=-1);valid=mask&np.isfinite(dist)
    return {'ADE':float(np.mean(dist[valid])) if valid.any() else None,'FDE':float(np.mean(dist[:,-1][valid[:,-1]])) if valid[:,-1].any() else None,
            'valid_samples':int(valid.sum()),'valid_final_points':int(valid[:,-1].sum()),
            'per_time_mean':[float(np.mean(dist[:,t][valid[:,t]])) if valid[:,t].any() else None for t in range(20)],
            'per_point_mean':[float(np.mean(dist[i][valid[i]])) if valid[i].any() else None for i in range(24)]}

def evaluate(camera):
    prediction_gate();scene=RUN/camera;dest=scene/'evaluation';viz=scene/'viz';viz.mkdir(exist_ok=True)
    selection=json.loads((scene/'geometry/branch_selection.json').read_text());names=list(selection['branches'])
    if camera=='wrist_1':names=[name for name in names if (scene/'branches'/name/'predictions/future_3d.npy').exists()]
    if not names:raise ValueError('No complete forecasts to evaluate')
    primary=selection['primary'];reference_branch='C_sensor_tcp_official'
    if primary in names:names=[primary]+[name for name in names if name!=primary]
    branch=scene/'geometry'/reference_branch;k=np.load(branch/'K.npy');k=k[-1] if k.ndim==3 else k
    # One common reference per camera, independently tracked after prediction, evaluated with TCP-derived poses.
    x=np.load(scene/'geometry/selected_hand_eye_X.npy');tcp=tcp_matrices(np.load(dest/'future_tcp_pose_xyzw.npy'));poses=tcp@x
    tracked=np.load(dest/'future_alltracker.npz');uv=tracked['tracks'];vis=tracked['visibility']
    depth=np.load(dest/'future_sensor_depth.npy');local=[];valid=[];depth_spreads=[];depth_coverages=[]
    for z,p,v in zip(depth,uv,vis):
        zz,spread,coverage=sample_z(z,p);local.append(backproject(p,zz,k));valid.append(v&np.isfinite(zz)&(spread<.03));depth_spreads.append(spread);depth_coverages.append(coverage)
    common=np.stack([transform(np.linalg.inv(poses[0])@T,p) for T,p in zip(poses,local)])
    gt3d=common[1:].transpose(1,0,2);gt2d=uv[1:].transpose(1,0,2);mask3=np.array(valid)[1:].T;mask2=vis[1:].T
    predictions={};projections={};initials={};data={}
    for name in names:
        model=scene/'branches'/name
        future=np.load(model/'predictions/future_3d.npy');hist=np.load(model/'observed/points_3d_history.npy')
        methods={name:align_prediction(future,hist[-1]),name+'/Static':np.repeat(hist[-1,:,None,:],20,axis=1),
                 name+'/CV':hist[-1,:,None,:]+((hist[2]-hist[0])/.2)[:,None,:]*TIMES[None,:,None]}
        for method,xyz in methods.items():
            predictions[method]=xyz;initials[method]=hist[-1]
            camera_xyz=np.stack([from_anchor(xyz[:,t],poses[t+1],poses[0]) for t in range(20)],axis=1)
            projections[method]=project(camera_xyz,k)
            data[method]={'geometry_gate':selection['branches'][name]['validated_geometry_gate'],
                 'outside_image_fraction':float(np.mean((projections[method]<0).any(-1)|(projections[method]>=256).any(-1))),
                 'behind_camera_fraction':float(np.mean(camera_xyz[...,2]<=0)),
                 'endpoint_mean_amplitude_m':float(np.mean(np.linalg.norm(xyz[:,-1]-hist[-1],axis=-1)))}
    # Masks are shared across every method; off-image projections remain in errors.
    for xyz in predictions.values():mask3 &= np.isfinite(xyz).all(-1)
    for pixels in projections.values():mask2 &= np.isfinite(pixels).all(-1)
    for method in predictions:
        data[method]['3D_est_m']=scores(predictions[method],gt3d,mask3);data[method]['2D_px']=scores(projections[method],gt2d,mask2)
        data[method]['displacement_3D_est_m']=scores(predictions[method]-initials[method][:,None],gt3d-common[0][:,None],mask3)
        target=(gt3d[:,-1]-common[0]);pred=predictions[method][:,-1]-initials[method]
        valid_dir=mask3[:,-1]&(np.linalg.norm(target,axis=-1)>.003)&(np.linalg.norm(pred,axis=-1)>1e-5)
        cosine=np.sum(target*pred,-1)/(np.linalg.norm(target,axis=-1)*np.linalg.norm(pred,axis=-1)+1e-12)
        data[method]['median_endpoint_direction_error_deg']=float(np.median(np.degrees(np.arccos(np.clip(cosine[valid_dir],-1,1))))) if valid_dir.any() else None
        valid_amp=mask3[:,-1]&(np.linalg.norm(target,axis=-1)>.003)
        ratio=np.linalg.norm(pred,axis=-1)/np.maximum(np.linalg.norm(target,axis=-1),1e-12)
        data[method]['median_endpoint_amplitude_ratio']=float(np.median(ratio[valid_amp])) if valid_amp.any() else None
        data[method]['initial_reference_offset_m']=float(np.nanmedian(np.linalg.norm(initials[method]-common[0],axis=-1)))
    np.savez_compressed(dest/'future_reference.npz',GT_3D_est=gt3d,GT_2D_est=gt2d,common_mask3d=mask3,common_mask2d=mask2,
                       sensor_patch_depth_spread=np.array(depth_spreads),sensor_patch_depth_coverage=np.array(depth_coverages),
                       camera_c2w=poses,K=k,source_indices=np.load(dest/'source_indices.npy'),times=TIMES)
    info={'camera':camera,'primary_geometry_frozen_before_future':primary,'common_reference_geometry':reference_branch,
          'methods':data,'reference':'Future AllTracker + sensor Z16 scale hypothesis + frozen observed hand-eye X and future TCP',
          'reference_not_physical_GT':True,'reference_K_status':'One frozen official K sensor/TCP control for every geometry; no branch-specific future reference',
          'source_time_basis':'nominal 10Hz; model future 15Hz interpolated XYZ by time',
          'common_valid_2d_samples':int(mask2.sum()),'common_valid_3d_samples':int(mask3.sum()),
          'outside_predictions_excluded':False,'future_camera_motion_used_for_evaluation_only':True,
          'reference_depth_rule':'5x5 median, at least13 positive finite samples, p90-p10<0.03m; common for all methods',
          'limitations':'Reference inherits observed hand-eye/K/scale uncertainty and can drift on low-texture target. All rejected geometries are explicitly diagnostic.'}
    write(dest/'metrics.json',info)
    rgb=np.load(dest/'future_rgb.npy');selected=tracked['selected_ids'];display=np.arange(0,24,3);frames=[];over=[]
    for t in range(20):
        reference=cv2.resize(rgb[t+1],(512,512));combined=reference.copy()
        for j in display:
            if np.isfinite(gt2d[j,t]).all():cv2.circle(reference,tuple(np.rint(gt2d[j,t]*2).astype(int)),4,(40,255,40),-1)
        cv2.putText(reference,f'Reference t={TIMES[t]:.1f}s',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
        panels=[reference]
        for n,name in enumerate(names):
            im=cv2.resize(rgb[t+1],(512,512))
            outside_count=0
            for j in display:
                p=projections[name][j,t]*2
                outside_count+=int(not((p>=0).all() and (p<512).all()))
                if np.isfinite(p).all() and (np.abs(p)<1e6).all():cv2.circle(im,tuple(np.rint(p).astype(int)),4,(255,70,240),-1)
                real=gt2d[j,t]*2
                if np.isfinite(real).all():cv2.circle(im,tuple(np.rint(real).astype(int)),3,(40,255,40),-1)
            cv2.putText(im,f'{name}; outside {outside_count}/8',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),2)
            cv2.putText(im,'Green: reference | Magenta: forecast',(10,495),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(im)
        frames.append(np.hstack(panels));over.append(panels[1])
    for filename,video in [('geometry_forecasts_side_by_side.mp4',frames),('primary_prediction_vs_real.mp4',over)]:
        imageio.mimwrite(viz/filename,video,fps=10,codec='libx264',macro_block_size=None,ffmpeg_params=['-crf','19'])
    for t in [0,9,19]:b.save_rgb(viz/f'forecast_review_{t+1:02d}.png',frames[t])
    # Readability sheet shows every frozen ID; judge reference drift separately from model quality.
    panels=[]
    for t in [0,9,19]:
        source_index=int(np.load(dest/'source_indices.npy')[t+1])
        im=b.draw_points(cv2.resize(rgb[t+1],(768,768)),uv[t+1]*3,selected,radius=3);cv2.putText(im,f'{camera}: future source {source_index}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(im)
    b.save_rgb(viz/'reference_all24_review.png',np.hstack(panels))
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for method,row in data.items():
        axes[0].plot(TIMES,np.array(row['3D_est_m']['per_time_mean'])*1000,label=method)
        axes[1].plot(TIMES,row['2D_px']['per_time_mean'],label=method)
    axes[0].set(ylabel='3D_est error (mm)',xlabel='future nominal time (s)');axes[1].set(ylabel='2D error (px)',xlabel='future nominal time (s)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(viz/'forecast_errors.png',dpi=160);plt.close(fig)
    # Full 30-step XYZ range, never cropped to the image or reference-visible points.
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name in names:
        future=np.load(scene/'branches'/name/'predictions/future_3d.npy')
        for axis in range(3):
            for j in range(24):axes[axis].plot(np.arange(1,31)/15,future[j,:,axis],alpha=.35,label=name if j==0 else None)
    for axis,ax in enumerate(axes):ax.set(xlabel='model future time (s)',ylabel='XYZ'[axis]+' in C_t0 (m)');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(viz/'full_30step_xyz.png',dpi=160);plt.close(fig)
    print(camera,'EVALUATED',{m:(v['3D_est_m']['ADE'],v['2D_px']['ADE']) for m,v in data.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['export','track','evaluate']);p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);a=p.parse_args()
    if a.stage=='export':export_future()
    else:
        for camera in a.cameras:{'track':track_future,'evaluate':evaluate}[a.stage](camera)
