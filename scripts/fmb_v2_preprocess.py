"""Berkeley methods applied to FMB, with FMB depth/registration quality checks."""
from pathlib import Path
import argparse
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import berkeley_preprocess as b

def depth(scene):
    b.estimate_depth(scene)
    info=b.read_json(scene/'geometry/depth_source.json')
    info.update(depth_source='Measured FMB Z16 with supported, unconfirmed 0.0001 m/raw scale',
                geometry_source='sensor-depth_under_scale_hypothesis_estimated-intrinsics',
                registration_status='observed audit required; not instrumentally certified',
                depth_scale_status='SUPPORTED_HYPOTHESIS_NOT_RECORD_CONFIRMED')
    b.write_json(scene/'geometry/depth_source.json',info)
    k=np.load(scene/'geometry/K_per_frame.npy')
    info=b.read_json(scene/'geometry/intrinsics_stability.json')
    values=np.array(info['values_px'])
    info.update(min_px=values.min(0).tolist(),max_px=values.max(0).tolist())
    b.write_json(scene/'geometry/intrinsics_stability.json',info)

def registration(scene):
    rgb,idx,_=b.load_observed(scene)
    depths=np.load(scene/'observed/native_depth.npy')
    mask=cv2.imread(str(scene/'observed/mask.png'),0)>0
    panels=[]; rows=[]
    for t in [0,3,7]:
        z=depths[t]; valid=np.isfinite(z)&(z>0)&(z<10)
        # Fill holes for display only. Metrics suppress hole-adjacent pixels.
        z_display=z.copy(); z_display[~valid]=np.median(z[valid])
        lo,hi=np.percentile(z[valid],[2,98])
        ze=(np.hypot(cv2.Sobel(z_display,cv2.CV_32F,1,0),cv2.Sobel(z_display,cv2.CV_32F,0,1))>.012)
        ze &= cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8))>0
        re=cv2.Canny(rgb[t],50,120)>0
        color=(matplotlib.colormaps['magma'](np.clip((z_display-lo)/(hi-lo),0,1))[...,:3]*255).astype(np.uint8)
        color[~valid]=0
        over_rgb=rgb[t].copy(); over_rgb[ze]=[0,255,50]
        over_depth=color.copy(); over_depth[re]=[0,255,255]
        tiles=[]
        for title,im in [('RGB',rgb[t]),('Sensor depth',color),('Depth edges on RGB',over_rgb),('RGB edges on depth',over_depth)]:
            tile=cv2.resize(im,(384,384),interpolation=cv2.INTER_NEAREST)
            cv2.putText(tile,f'{idx[t]}: {title}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            tiles.append(tile)
        panels.append(np.hstack(tiles))
        distance=cv2.distanceTransform((~ze).astype(np.uint8),cv2.DIST_L2,5)
        regions={'whole_scene':np.ones(mask.shape,bool),'board':np.zeros(mask.shape,bool),'object_t0':mask}
        regions['board'][95:210,65:195]=True
        for name,region in regions.items():
            if name=='object_t0' and t!=7: continue
            values=distance[re&region&valid]
            rows.append({'source_index':int(idx[t]),'region':name,'rgb_edge_count':len(values),
                         'median_nearest_depth_edge_px':float(np.median(values)) if len(values) else None,
                         'fraction_within_3px':float(np.mean(values<=3)) if len(values) else None})
    b.save_rgb(scene/'viz/rgb_depth_registration_audit.png',np.vstack(panels))
    b.write_json(scene/'geometry/registration_audit.json',{
        'future_used':False,'source_indices':idx[[0,3,7]].tolist(),'diagnostics':rows,
        'depth_edge_threshold_sobel_m':.012,'status':'APPROXIMATE_REGISTRATION_AUDIT_ONLY',
        'conclusion':'Inspect object, board and gripper edges; sensor holes and unequal contours remain. No correction fitted.',
        'limitations':'Nearest-edge distances are descriptive and include texture/internal board edges; do not certify pointwise registration.',
        'roundtrip_is_registration_proof':False})

def lift(depth,xy,visible,k):
    """Exactly the Berkeley 5x5 finite positive median, >=13/25 on a full patch."""
    return b.robust_lift(depth,xy,visible,k,patch_size=5)

def author_filter():
    """Guard an upstream diagnostic-print crash; keep all filtering mathematics."""
    import inspect
    module=b.load_author_filter()
    source=inspect.getsource(module.filter_tracks_by_trust)
    original='        print(f"  Dropped tracks: mean_w range [{dropped_mw.min():.3f}, {dropped_mw.max():.3f}]")'
    assert source.count(original)==1
    source=source.replace(original,'        if dropped_mw.size:\n    '+original)
    exec(compile(source,str(b.FILTER_SOURCE),'exec'),module.__dict__)
    return module

def observed_masks(scene):
    """Observed H3 membership audit using SAM, never the v1 color mask."""
    import sys
    import torch
    assert not (scene/'predictions/input_freeze.json').exists()
    rgb,idx,_=b.load_observed(scene)
    data=np.load(scene/'observed/observed_tracks_2d.npz');query=np.load(scene/'observed/query_points_100.npy')
    prompt=np.array(b.read_json(scene/'observed/molmopoint_grounding.json')['point_xy'])
    nearest=int(np.argmin(np.linalg.norm(query-prompt,axis=-1)))
    sys.path.insert(0,str(b.WORKSPACE/'third_party/sam2'))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(b.WORKSPACE/'models/sam2.1_hiera_large.pt'),device='cuda')
    predictor=SAM2ImagePredictor(model);masks=[];receipt=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for t in [5,6]:
            point=data['tracks'][t,nearest]
            predictor.set_image(rgb[t]);options,scores,logits=predictor.predict(point_coords=point[None],point_labels=np.array([1]),multimask_output=True)
            chosen=int(np.argmax(scores));masks.append(options[chosen])
            np.savez_compressed(scene/'observed'/f'sam_historical_{idx[t]}.npz',masks=options,scores=scores,logits=logits)
            receipt.append({'source_index':int(idx[t]),'prompt_xy':point.tolist(),'chosen':chosen,'scores':scores.tolist()})
    masks.append(cv2.imread(str(scene/'observed/mask.png'),0)>0)
    np.save(scene/'observed/historical_masks_h3.npy',np.array(masks,bool))
    b.write_json(scene/'observed/historical_masks_metadata.json',{'future_used':False,'source_indices':idx[-3:].tolist(),
        'prompt_origin':'Nearest t0 KMeans candidate to actual MolmoPoint point, followed by observed AllTracker',
        'prompt_candidate_id':nearest,'selected_by':'maximum SAM score; t0 original mask reused','frames':receipt})
    del model,predictor;torch.cuda.empty_cache()

def filter_scene(scene):
    import torch
    started=time.monotonic()
    assert not (scene/'predictions/input_freeze.json').exists()
    rgb,idx,meta=b.load_observed(scene)
    data=np.load(scene/'observed/observed_tracks_2d.npz')
    xy,visible,confidence=[data[name] for name in ['tracks','visibility','confidence']]
    z=np.load(scene/'geometry/depth_observed.npy'); k=np.load(scene/'geometry/K_median.npy')
    mask=cv2.imread(str(scene/'observed/mask.png'),0)>0
    camera=b.camera_motion_audit(rgb,mask)
    b.write_json(scene/'geometry/camera_motion_audit.json',camera)
    assert camera['fixed_camera_supported'], camera
    np.save(scene/'geometry/camera_poses.npy',np.repeat(np.eye(4,dtype=np.float32)[None],8,axis=0))
    xyz,valid,coverage,spread=lift(z,xy,visible,k)
    author=author_filter()
    trust,anchors=author.compute_trust_weights(xyz,valid,K=16)
    keep=~author.filter_tracks_by_trust(trust,valid,z_thresh=2.)
    smoothed=xyz.copy()
    smoothed[:,keep]=author.consensus_gated_smooth(xyz[:,keep],np.zeros((8,3),np.float32),valid[:,keep],trust[:,keep],device='cpu')
    margin=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    uv=np.rint(xy[-1]).astype(int)
    m=margin[np.clip(uv[:,1],0,255),np.clip(uv[:,0],0,255)]
    stable=(np.linalg.norm(np.diff(xy[-3:],axis=0),axis=-1)<20).all(0)
    hmasks=np.load(scene/'observed/historical_masks_h3.npy');inside=np.zeros((3,100),bool)
    for t in range(3):
        rounded=np.rint(xy[t+5]).astype(int)
        allowed=(rounded[:,0]>=0)&(rounded[:,0]<256)&(rounded[:,1]>=0)&(rounded[:,1]<256)
        inside[t,allowed]=hmasks[t,rounded[allowed,1],rounded[allowed,0]]
    eligible=(keep&valid[-3:].all(0)&np.isfinite(smoothed[-3:]).all((0,2))&
              np.load(scene/'observed/query_in_mask.npy')&(m>=2.5)&stable&(spread[-3:]<.03).all(0)&inside.all(0))
    b.write_json(scene/'geometry/selection_gates.json',{
        'future_used':False,'min_mask_boundary_distance_px':2.5,'max_h3_step_jump_px':20,
        'max_depth_patch_p90_p10_m':.03,'depth_min_valid_count_full_patch':13,
        'inside_SAM_mask_all_H3_frames':True,
        'author_keep':int(keep.sum()),'eligible':int(eligible.sum()),
        'selection_policy':'Deterministic farthest point selection; first point highest history trust*confidence; no future scores'})
    assert eligible.sum()>=24, f'Insufficient points: {eligible.sum()} (do not relax after prediction)'
    ids=b.spread_select(xy[-1],eligible,24,trust[-3:].mean(0)*confidence[-3:].min(0))
    np.save(scene/'geometry/points_3d_raw.npy',xyz)
    np.save(scene/'geometry/points_3d_filtered.npy',smoothed)
    np.savez_compressed(scene/'geometry/filter_diagnostics.npz',trust=trust,anchor_ids=anchors,keep=keep,
                        eligible=eligible,depth_valid_fraction=coverage,depth_patch_spread_m=spread,mask_margin_px=m)
    for name,value in [('selected_point_ids',ids),('points_2d_history',xy[-3:,ids]),('points_3d_history',smoothed[-3:,ids])]:
        np.save(scene/'observed'/f'{name}.npy',value)
    for group_index in range(3):
        folder=scene/'groups'/f'group_{group_index:02d}';folder.mkdir(exist_ok=True)
        group_ids=ids[group_index*8:(group_index+1)*8]
        for name,value in [('point_ids',group_ids),('points_2d_at_t0',xy[-1,group_ids]),('points_3d_history',smoothed[-3:,group_ids])]:
            np.save(folder/f'{name}.npy',value)
        b.write_json(folder/'metadata.json',{'group':group_index,'point_ids':group_ids.tolist(),
             'history_source_indices':idx[-3:].tolist(),'history_times_relative_t0_seconds':[-.2,-.1,0],
             'coordinates':'OpenCV fixed side_1 XYZ, meters under scale hypothesis','future_used':False})
    selected=smoothed[-3:,ids]; raw=xyz[-3:,ids]
    pairs=np.triu_indices(24,1)
    distances=np.linalg.norm(selected[:,:,None]-selected[:,None,:],axis=-1)[:,pairs[0],pairs[1]]
    reproj=selected[...,:2]/selected[...,2,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
    audit={'success':True,'future_used':False,'selected_points':24,'historical_xyz_shape':list(selected.shape),
        'finite':bool(np.isfinite(selected).all()),'positive_z':bool((selected[...,2]>0).all()),
        'history_smoothing_max_displacement_m':float(np.max(np.linalg.norm(selected-raw,axis=-1))),
        'raw_history_speed_max_m_s':float(np.max(np.linalg.norm(np.diff(raw,axis=0),axis=-1)/.1)),
        'history_speed_max_m_s':float(np.max(np.linalg.norm(np.diff(selected,axis=0),axis=-1)/.1)),
        'pairwise_distance_change_max_m':float(np.max(np.ptp(distances,axis=0))),
        'pairwise_distance_change_median_m':float(np.median(np.ptp(distances,axis=0))),
        'implementation_roundtrip_max_error_px':float(np.max(np.linalg.norm(reproj-xy[-3:,ids],axis=-1))),
        'depth_coverage_h3':float(valid[-3:,ids].mean()),'mask_area_px':int(mask.sum()),
        'official_kept_count':int(keep.sum()),'eligible_h3_count':int(eligible.sum()),
        'camera_fixed':True,'author_filter_source_sha256':b.sha256(b.FILTER_SOURCE),
        'author_filter_adapter':'Guard empty dropped_mw only in diagnostic print; no algorithm change or vendored-file edit',
        'author_filter_parameters':'Same compute_trust_weights, z_thresh=2, default consensus_gated_smooth as Berkeley',
        'temporal_limitation':'H3 uses nominal 10Hz [-0.2,-0.1,0] rather than training 15Hz',
        'runtime_seconds':time.monotonic()-started}
    assert audit['finite'] and audit['positive_z'] and audit['implementation_roundtrip_max_error_px']<.01
    from berkeley_input_audit import validate_group_payloads
    audit['group_payload_audit']=validate_group_payloads(scene)
    b.write_json(scene/'geometry/input_audit.json',audit)
    b.save_rgb(scene/'viz/selected_24_points.png',b.draw_points(rgb[-1],xy[-1,ids],ids,radius=2))
    tiles=[]
    for frame,index,points in zip(rgb,idx,xy[:,ids]):
        im=b.draw_points(frame,points,ids,radius=1)
        im=cv2.resize(im,(384,384),interpolation=cv2.INTER_NEAREST)
        cv2.putText(im,f'frame {index}',(7,22),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
        tiles.append(im)
    b.save_rgb(scene/'viz/historical_tracks.png',np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])]))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(a,c) in zip(axes,[(0,1),(0,2),(1,2)]):
        for t in range(3): ax.scatter(selected[t,:,a],selected[t,:,c],s=12,label=f'{idx[-3+t]}')
        ax.set(xlabel='XYZ'[a]+' (m)',ylabel='XYZ'[c]+' (m)');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('History sensor XYZ (scale hypothesis) after Berkeley ray smoothing');fig.tight_layout()
    fig.savefig(scene/'viz/history_xyz_clouds.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5))
    valid_z=z[-1][(z[-1]>0)&(z[-1]<10)];lo,hi=np.percentile(valid_z,[2,98])
    im=ax.imshow(z[-1],vmin=lo,vmax=hi,cmap='magma');ax.scatter(xy[-1,ids,0],xy[-1,ids,1],s=12,c='cyan')
    fig.colorbar(im,ax=ax,label='m under scale hypothesis');fig.tight_layout();fig.savefig(scene/'viz/sensor_depth.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3));ax.plot([p['pair'][1] for p in camera['pairs']], [p['median_displacement_px'] for p in camera['pairs']],'o-')
    ax.axhline(1.5,c='r',ls='--');ax.set(xlabel='observed local frame',ylabel='background median flow (px)');fig.tight_layout()
    fig.savefig(scene/'viz/fixed_camera_check.png',dpi=150);plt.close(fig)
    registration(scene)
    print(scene.name,'PREPARED',ids.tolist(),audit,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['depth','ground','track','observed-masks','filter'])
    p.add_argument('--scene-dir',type=Path,nargs='+',required=True);a=p.parse_args()
    for scene in a.scene_dir:
        if a.stage=='depth':depth(scene)
        elif a.stage=='ground':b.grounding(scene,None,b.WORKSPACE/'models/sam2.1_hiera_large.pt',pointing_json=scene/'observed/molmopoint_grounding.json')
        elif a.stage=='track':b.track_observed(scene)
        elif a.stage=='observed-masks':observed_masks(scene)
        else:filter_scene(scene)
