"""Prediction-gated independent AllTracker reference, nominal-time metrics and media."""
from pathlib import Path
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from berkeley_preprocess import alltracker_tracks,write_json,read_json,sha256,save_rgb
from berkeley_evaluate import project,velocity,points_on,trails_on,label,write_video
from fmb_v2_preprocess import lift

TIMES=np.arange(1,21)/10

def interpolate_prediction(prediction, times=TIMES):
    source_times=np.arange(1,prediction.shape[1]+1)/15
    if np.min(times)<source_times[0] or np.max(times)>source_times[-1]:raise ValueError('Prediction extrapolation forbidden')
    return np.stack([np.stack([np.interp(times,source_times,point[:,c]) for c in range(3)],-1) for point in prediction])

def check_freeze(scene):
    freeze=read_json(scene/'predictions/input_freeze.json')
    changed=[name for name,digest in freeze['sha256'].items() if sha256(scene/name)!=digest]
    if changed:raise ValueError(f'Frozen input changed: {changed}')
    return len(freeze['sha256'])

def track(scene):
    assert read_json(scene/'predictions/model_run.json')['success']
    if read_json(scene/'metadata.json')['primary_example']:assert read_json(scene/'predictions_h1/model_run.json')['success']
    check_freeze(scene)
    destination=scene/'evaluation/alltracker_future.npz'
    if destination.exists():raise FileExistsError('Refusing to overwrite future reference')
    observed=np.load(scene/'observed/rgb.npy');future=np.load(scene/'evaluation/future_rgb.npy')
    xy=np.load(scene/'observed/points_2d_history.npy')[-1]
    clip=np.concatenate([observed[-1:],future]);assert clip.shape==(21,256,256,3)
    tracks,confidence,visible,probability,info=alltracker_tracks(clip,xy,reverse=False)
    np.savez_compressed(destination,tracks=tracks,confidence=confidence,visibility=visible,visibility_probability=probability,
                        selected_point_ids=np.load(scene/'observed/selected_point_ids.npy'))
    info.update(independent_of_history_tracking=True,same_t0_point_ids=True,prediction_gated=True,
                future_only_for_evaluation=True,reference='AllTracker estimated RGB correspondence, not physical GT')
    write_json(scene/'evaluation/alltracker_execution.json',info)
    print(scene.name,'future tracked',tracks.shape,flush=True)

def metric(pred,truth,mask,suffix):
    error=np.linalg.norm(pred-truth,axis=-1)
    if not np.isfinite(error[mask]).all():raise ValueError('Nonfinite prediction on shared reference mask')
    per_time=[float(error[:,t][mask[:,t]].mean()) if mask[:,t].any() else None for t in range(mask.shape[1])]
    per_point=[float(error[p][mask[p]].mean()) if mask[p].any() else None for p in range(mask.shape[0])]
    return {f'ADE_{suffix}':float(error[mask].mean()) if mask.any() else None,
            f'FDE_{suffix}':float(error[:,-1][mask[:,-1]].mean()) if mask[:,-1].any() else None,
            'error_by_time':per_time,'per_point_ADE':per_point,'valid_pairs':int(mask.sum()),
            'coverage':float(mask.mean()),'valid_by_time':mask.sum(0).tolist()},error

def uncertainty(scene,clip,xy,gt,mask,model_error):
    from probe_fmb_mask_ecc import follow
    positions,info=follow(clip[...,::-1].copy(),0,20,xy)
    ecc=np.stack([positions[t] for t in range(1,21)]).transpose(1,0,2)
    disagreement=np.linalg.norm(ecc-gt,axis=-1)
    values=disagreement[mask]
    np.savez_compressed(scene/'evaluation/ecc_uncertainty.npz',tracks=ecc,disagreement_px=disagreement)
    result={'point_ids':np.load(scene/'observed/selected_point_ids.npy').tolist(),
        'method':'Same v1 sequential color-mask affine ECC initialized with the v2 24 t0 coordinates',
        'reference_not_changed':True,'mean_ECC_AllTracker_disagreement_px':float(values.mean()),
        'median_ECC_AllTracker_disagreement_px':float(np.median(values)),
        'p90_ECC_AllTracker_disagreement_px':float(np.percentile(values,90)),
        'max_ECC_AllTracker_disagreement_px':float(values.max()),
        'ratio_mean_disagreement_to_MolmoMotion_ADE':float(values.mean()/model_error[mask].mean()),
        'ecc_fits':info,'limitations':'Both trackers can share errors on a low-texture face. Agreement is not proof of material-point identity.'}
    write_json(scene/'evaluation/reference_uncertainty.json',result)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(TIMES,[disagreement[:,t][mask[:,t]].mean() if mask[:,t].any() else np.nan for t in range(20)],label='ECC vs AllTracker')
    ax.plot(TIMES,[model_error[:,t][mask[:,t]].mean() if mask[:,t].any() else np.nan for t in range(20)],label='H3 vs AllTracker')
    ax.set(xlabel='nominal future time (s)',ylabel='mean error (px)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(scene/'viz/reference_uncertainty.png',dpi=150);plt.close(fig)
    # This contact sheet is read by the root agent for a human visual audit.
    rows=[]
    ids=np.load(scene/'observed/selected_point_ids.npy')
    for t in [0,9,19]:
        frame=clip[t+1].copy()
        points_on(frame,gt[:8,t],ids[:8],(40,245,100),mask[:8,t])
        points_on(frame,ecc[:8,t],ids[:8],(255,190,40),mask[:8,t])
        tile=cv2.resize(frame,(768,768),interpolation=cv2.INTER_NEAREST)
        label(tile,f't={TIMES[t]:.1f}s | green AllTracker | orange ECC')
        rows.append(tile)
    save_rgb(scene/'viz/manual_reference_audit.png',np.hstack(rows))

def visualize(scene,rgb,history_uv,ids,gt,mask,methods_uv,methods_xyz,errors,metrics):
    viz=scene/'viz';hero=8
    colors={'MolmoMotion_H3':(255,65,170),'static':(60,190,255),'constant_velocity':(255,185,30),'MolmoMotion_H1':(180,90,255)}
    side=[];overlays=[]
    for t,frame in enumerate(rgb):
        left=frame.copy();right=frame.copy();combined=frame.copy()
        trails_on(left,gt[:hero,:t+1],(40,245,100),mask[:hero,:t+1]);points_on(left,gt[:hero,t],ids[:hero],(40,245,100),mask[:hero,t])
        trails_on(right,methods_uv['MolmoMotion_H3'][:hero,:t+1],colors['MolmoMotion_H3']);points_on(right,methods_uv['MolmoMotion_H3'][:hero,t],ids[:hero],colors['MolmoMotion_H3'])
        trails_on(combined,gt[:hero,:t+1],(40,245,100),mask[:hero,:t+1]);trails_on(combined,methods_uv['MolmoMotion_H3'][:hero,:t+1],colors['MolmoMotion_H3'])
        points_on(combined,gt[:hero,t],ids[:hero],(40,245,100),mask[:hero,t])
        points_on(combined,methods_uv['MolmoMotion_H3'][:hero,t],ids[:hero],colors['MolmoMotion_H3'])
        left=cv2.resize(left,(512,512));right=cv2.resize(right,(512,512));combined=cv2.resize(combined,(512,512))
        label(left,f'AllTracker reference | {TIMES[t]:.1f}s');label(right,'MolmoMotion H3');label(combined,'Green reference | pink H3')
        side.append(np.hstack([left,right]));overlays.append(combined)
    receipts={'side_by_side':write_video(viz/'side_by_side.mp4',side,fps=10),
              'overlay':write_video(viz/'prediction_vs_real.mp4',overlays,fps=10)}
    save_rgb(viz/'prediction_vs_real_contact.png',np.hstack([side[t] for t in [0,9,19]]))
    save_rgb(viz/'real_future_alltracker.png',np.hstack([side[t][:,:512] for t in [0,9,19]]))
    fig,axes=plt.subplots(1,len(methods_uv),figsize=(5*len(methods_uv),5))
    for ax,(name,uv) in zip(axes,methods_uv.items()):
        model_folder={'MolmoMotion_H3':'predictions','MolmoMotion_H1':'predictions_h1'}.get(name)
        if model_folder and (scene/model_folder/'future_3d.npy').exists():
            uv=project(np.load(scene/model_folder/'future_3d.npy'),np.load(scene/'geometry/K_median.npy'))
        ax.imshow(np.load(scene/'observed/rgb.npy')[-1]);ax.plot(gt[:,:,0].T,gt[:,:,1].T,c='green',alpha=.25)
        ax.plot(uv[:,:,0].T,uv[:,:,1].T,c=np.array(colors[name])/255,alpha=.3)
        valid=np.isfinite(uv).all(-1)
        bounds=np.concatenate([uv[valid],gt[mask],[[0,0],[255,255]]])
        ax.set_xlim(bounds[:,0].min()-10,bounds[:,0].max()+10);ax.set_ylim(bounds[:,1].max()+10,bounds[:,1].min()-10)
        ax.set_title(name);ax.set_xlabel('u (px)');ax.set_ylabel('v (px)')
    fig.suptitle('All 24 points: green reference, full 30/32-step model trajectories');fig.tight_layout();fig.savefig(viz/'full_extent_trajectories.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in methods_uv:
        axes[0].plot(TIMES,metrics['methods'][name]['2D']['error_by_time'],label=name)
        axes[1].plot(TIMES,metrics['methods'][name]['3D_est']['error_by_time'],label=name)
    for ax,suffix in zip(axes,['px','m under scale hypothesis']):
        ax.set(xlabel='nominal future time (s)',ylabel=f'mean error ({suffix})');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(viz/'error_vs_time.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    for name in methods_uv:ax.plot(ids,metrics['methods'][name]['2D']['per_point_ADE'],'o',label=name)
    ax.set(xlabel='frozen candidate point ID',ylabel='2D ADE (px)');ax.legend();fig.tight_layout();fig.savefig(viz/'per_point_ADE.png',dpi=150);plt.close(fig)
    if 'MolmoMotion_H1' in methods_uv:
        fig,ax=plt.subplots(figsize=(8,4))
        for name in ['MolmoMotion_H3','MolmoMotion_H1']:ax.plot(TIMES,metrics['methods'][name]['2D']['error_by_time'],label=name)
        ax.set(xlabel='nominal future time (s)',ylabel='2D error (px)');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(viz/'h1_vs_h3.png',dpi=150);plt.close(fig)
    write_json(viz/'video_receipts.json',receipts)

def evaluate(scene):
    check_freeze(scene);meta=read_json(scene/'metadata.json');evaldir=scene/'evaluation'
    ids=np.load(scene/'observed/selected_point_ids.npy');data=np.load(evaldir/'alltracker_future.npz')
    assert np.array_equal(data['selected_point_ids'],ids)
    rgb=np.load(evaldir/'future_rgb.npy');hxyz=np.load(scene/'observed/points_3d_history.npy');huv=np.load(scene/'observed/points_2d_history.npy')
    k=np.load(scene/'geometry/K_median.npy');gt=data['tracks'][1:].transpose(1,0,2)
    mask=data['visibility'][1:].T&np.isfinite(gt).all(-1)&(gt[...,0]>=0)&(gt[...,0]<256)&(gt[...,1]>=0)&(gt[...,1]<256)
    htimes=np.load(scene/'observed/history_timestamps.npy');ftimes=np.load(evaldir/'timestamps.npy')
    assert np.allclose(ftimes-htimes[-1],TIMES) and np.allclose(htimes-htimes[-1],[-.2,-.1,0])
    methods={'MolmoMotion_H3':interpolate_prediction(np.load(scene/'predictions/future_3d.npy')),
        'static':np.repeat(hxyz[-1,:,None,:],20,axis=1),
        'constant_velocity':hxyz[-1,:,None,:]+velocity(hxyz,htimes)[:,None,:]*TIMES[None,:,None]}
    if meta['primary_example']:methods['MolmoMotion_H1']=interpolate_prediction(np.load(scene/'predictions_h1/future_3d.npy'))
    gtxyz,depthvalid,coverage,spread=lift(np.load(evaldir/'native_depth_future.npy'),data['tracks'][1:],data['visibility'][1:],k)
    gtxyz=gtxyz.transpose(1,0,2);mask3=mask&depthvalid.T&np.isfinite(gtxyz).all(-1)
    methods_uv={name:project(xyz,k) for name,xyz in methods.items()};metrics={'scene':scene.name,'methods':{},
        'time_basis':'nominal FMB 10Hz and MolmoMotion 15Hz, no hardware timestamps',
        'prediction_alignment':'Linear interpolation of predicted XYZ only; source frames/reference not interpolated',
        'future_times_s':TIMES.tolist(),'reference':'AllTracker estimated RGB correspondences of frozen 24 IDs',
        'unit_of_experiment':'episode; point-times are correlated, no significance testing',
        'depth_scale_status':meta['depth_scale_status'],'K_source':'observed-only UniDepthV2 median',
        'common_mask_2D_pairs':int(mask.sum()),'common_mask_3D_est_pairs':int(mask3.sum()),
        'frozen_inputs_verified':check_freeze(scene)}
    errors={}
    for name,xyz in methods.items():
        uv=methods_uv[name];m2,err=metric(uv,gt,mask,'2D_px');m3,_=metric(xyz,gtxyz,mask3,'3D_est_m');errors[name]=err
        outside=(uv[...,0]<0)|(uv[...,0]>=256)|(uv[...,1]<0)|(uv[...,1]>=256)
        d_pred=uv[:,-1]-huv[-1];d_gt=gt[:,-1]-huv[-1];eligible=mask[:,-1]
        norm_pred=np.linalg.norm(d_pred,axis=-1);norm_gt=np.linalg.norm(d_gt,axis=-1)
        directional=eligible&(norm_gt>1e-3)&(norm_pred>1e-3)
        angle=np.degrees(np.arccos(np.clip(np.sum(d_pred*d_gt,axis=-1)[directional]/(norm_pred[directional]*norm_gt[directional]),-1,1)))
        metrics['methods'][name]={'2D':m2,'3D_est':m3,'outside_image_fraction_all_predictions':float(outside.mean()),
            'outside_image_fraction_on_common_mask':float(outside[mask].mean()),'nonpositive_z_fraction':float((xyz[...,2]<=0).mean()),
            'endpoint_displacement_px_mean':float(norm_pred[eligible].mean()),
            'reference_endpoint_displacement_px_mean':float(norm_gt[eligible].mean()),
            'direction_error_degrees_mean':float(angle.mean()) if len(angle) else None,
            'direction_valid_points':int(directional.sum()),
            'amplitude_ratio_mean':float(np.mean(norm_pred[eligible]/np.maximum(norm_gt[eligible],1e-3)))}
    np.savez_compressed(evaldir/'references_and_methods.npz',GT_2D_est=gt,GT_3D_est=gtxyz,common_mask_2D=mask,
                       common_mask_3D_est=mask3,selected_point_ids=ids,**{name:xyz for name,xyz in methods.items()})
    write_json(evaldir/'metrics.json',metrics)
    visualize(scene,rgb,huv,ids,gt,mask,methods_uv,methods,errors,metrics)
    if meta['primary_example']:
        clip=np.concatenate([np.load(scene/'observed/rgb.npy')[-1:],rgb])
        uncertainty(scene,clip,huv[-1],gt,mask,errors['MolmoMotion_H3'])
    print(scene.name,json_summary(metrics),flush=True)

def json_summary(metrics):
    import json
    return json.dumps({name:{'ADE':row['2D']['ADE_2D_px'],'FDE':row['2D']['FDE_2D_px']} for name,row in metrics['methods'].items()})

def audit_reference(scene):
    check_freeze(scene)
    data=np.load(scene/'evaluation/references_and_methods.npz')
    k=np.load(scene/'geometry/K_median.npy')
    gt=data['GT_2D_est'];mask=data['common_mask_2D']
    error=np.linalg.norm(project(data['MolmoMotion_H3'],k)-gt,axis=-1)
    clip=np.concatenate([np.load(scene/'observed/rgb.npy')[-1:],np.load(scene/'evaluation/future_rgb.npy')])
    uncertainty(scene,clip,np.load(scene/'observed/points_2d_history.npy')[-1],gt,mask,error)
    print(scene.name,'additional reference audit complete',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['track','evaluate','audit-reference'])
    p.add_argument('--scene-dir',type=Path,nargs='+',required=True);args=p.parse_args()
    for scene in args.scene_dir:{'track':track,'evaluate':evaluate,'audit-reference':audit_reference}[args.stage](scene)
