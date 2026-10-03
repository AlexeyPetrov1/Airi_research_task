"""Compare observed geometry, calibrate hand-eye, and export identical H3 IDs."""
from __future__ import annotations
import argparse,io,json,shutil,zipfile
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import ROOT,RUN,write,sha
from fmb_wrist_math import *
import berkeley_preprocess as b
from fmb_v2_preprocess import author_filter

def load_vipe(scene,preset='default',use_selection=True):
    folder=scene/f'vipe_{preset}';name='observed_10hz';count=len(np.load(scene/'observed/source_indices.npy'));arrays=[]
    selection=scene/'geometry/vipe_source_selection.json';native_wh=(256,256)
    if preset=='default' and use_selection and selection.exists():
        chosen=json.loads(selection.read_text());folder=scene/chosen['folder'];native_wh=tuple(chosen['native_wh'])
    for kind,shape in [('pose',(count,4,4)),('intrinsics',(count,4))]:
        with np.load(folder/kind/(name+'.npz')) as z:
            assert np.array_equal(z['inds'],np.arange(count)) and z['data'].shape==shape
            arrays.append(z['data'].astype(float))
    camera,values=arrays;K=np.repeat(np.eye(3)[None],count,axis=0)
    K[:,0,0]=values[:,0];K[:,1,1]=values[:,1];K[:,0,2]=values[:,2];K[:,1,2]=values[:,3]
    assert np.allclose(camera[:,3],[0,0,0,1]) and np.allclose(np.linalg.det(camera[:,:3,:3]),1,atol=1e-3)
    import OpenEXR,Imath
    depths=[]
    with zipfile.ZipFile(folder/'depth'/(name+'.zip')) as archive:
        assert sorted(archive.namelist())==[f'{i:05d}.exr' for i in range(count)]
        for name in sorted(archive.namelist()):
            exr=OpenEXR.InputFile(io.BytesIO(archive.read(name)));dw=exr.header()['dataWindow']
            width,height=dw.max.x-dw.min.x+1,dw.max.y-dw.min.y+1
            assert (width,height)==native_wh
            d=np.frombuffer(exr.channel('Z',Imath.PixelType(Imath.PixelType.FLOAT)),dtype=np.float32).reshape(height,width).copy()
            exr.close()
            if native_wh!=(256,256):d=cv2.resize(d,(256,256),interpolation=cv2.INTER_NEAREST_EXACT)
            depths.append(d)
    if native_wh!=(256,256):
        sx,sy=256/native_wh[0],256/native_wh[1];K[:,0,0]*=sx;K[:,1,1]*=sy
        K[:,0,2]=(K[:,0,2]+.5)*sx-.5;K[:,1,2]=(K[:,1,2]+.5)*sy-.5
    return camera,K,np.stack(depths)

def plot_depth(scene,rgb,sensor,vipe):
    mask_path=scene/'observed/mask.png'
    mask=cv2.imread(str(mask_path),0)>0 if mask_path.exists() else (rgb[-1,:,:,0]>rgb[-1,:,:,1]*1.35)&(rgb[-1,:,:,0]>rgb[-1,:,:,2]*1.35)
    rows=[];panels=[]
    for i in [0,24,47,49]:
        r,g,bv=rgb[i].transpose(2,0,1).astype(float)
        target=(r>g*1.35)&(r>bv*1.35)&(r>80)
        yy,xx=np.indices(target.shape)
        robot=(xx>140)&(yy<150)&(~target)
        table=((bv>r*1.25)&(bv>g*1.12)&(bv>65))|((yy>175)&(~target)&(~robot))
        regions={'target':mask if i==49 else target,'robot_approximate':robot,'static_table_bin_approximate':table,'background':~(target|robot|table)}
        valid=(sensor[i]>0)&np.isfinite(vipe[i])&(vipe[i]>0)&(vipe[i]<10)
        for name,region in regions.items():
            ok=valid&cv2.erode(region.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            errors=np.abs(sensor[i][ok]-vipe[i][ok]);ratio=vipe[i][ok]/sensor[i][ok]
            rows.append({'local_id':i,'source_index':int(np.load(scene/'observed/source_indices.npy')[i]),'region':name,'valid_pixels':int(ok.sum()),
                         'median_abs_depth_difference_m':float(np.median(errors)) if len(errors) else None,
                         'p90_abs_depth_difference_m':float(np.percentile(errors,90)) if len(errors) else None,
                         'median_vipe_sensor_scale_ratio':float(np.median(ratio)) if len(ratio) else None})
        panels.append((i,valid))
    display_valid=np.isfinite(vipe)&(vipe>0)&(vipe<10)&(sensor>0)
    depth_max=float(np.percentile(np.r_[sensor[display_valid],vipe[display_valid]],99))
    difference_max=float(np.percentile(np.abs(sensor[display_valid]-vipe[display_valid]),99))
    fig,axes=plt.subplots(4,4,figsize=(15,13))
    for row,(i,valid) in enumerate(panels):
        for col,(title,im,cmap) in enumerate([('RGB',rgb[i],None),('sensor Z (scale hypothesis)',sensor[i],'magma'),('ViPE depth',vipe[i],'magma'),('|difference|',np.where(valid,np.abs(sensor[i]-vipe[i]),np.nan),'inferno')]):
            ax=axes[row,col];plot=ax.imshow(im,cmap=cmap,vmin=0 if cmap else None,vmax=depth_max if col in [1,2] else difference_max if col==3 else None);ax.set_title(f'{i}: {title}',fontsize=9);ax.axis('off')
            if cmap:fig.colorbar(plot,ax=ax,fraction=.035,pad=.02,label='m (estimated)')
    fig.tight_layout();fig.savefig(scene/'viz/depth_comparison.png',dpi=140);plt.close(fig)
    # Fixed image-region temporal variation includes camera and occlusion changes; not a material-track statistic.
    variations={}
    for name,region in [('target_t0_fixed_pixels',mask),('whole_image',np.ones(mask.shape,bool))]:
        variations[name]={'sensor_median_pixel_temporal_std_m':float(np.nanmedian(np.nanstd(np.where(sensor[:,region]>0,sensor[:,region],np.nan),axis=0))),
                          'vipe_median_pixel_temporal_std_m':float(np.nanmedian(np.nanstd(vipe[:,region],axis=0)))}
    sequences={name:{'sensor':[],'vipe':[]} for name in ['target','robot_approximate','static_table_bin_approximate','background']}
    for i,frame in enumerate(rgb):
        r,g,bv=frame.transpose(2,0,1).astype(float);yy,xx=np.indices(r.shape)
        target=(r>g*1.35)&(r>bv*1.35)&(r>80);robot=(xx>140)&(yy<150)&(~target)
        table=((bv>r*1.25)&(bv>g*1.12)&(bv>65))|((yy>175)&(~target)&(~robot))
        regions={'target':target,'robot_approximate':robot,'static_table_bin_approximate':table,'background':~(target|robot|table)}
        for name,region in regions.items():
            good=region&(sensor[i]>0)&np.isfinite(vipe[i])&(vipe[i]>0)&(vipe[i]<10)
            for label,z in [('sensor',sensor),('vipe',vipe)]:sequences[name][label].append(float(np.median(z[i][good])) if good.any() else None)
    for name,values in sequences.items():
        variations[name]={'regional_median_depth_temporal_std_m':{label:float(np.nanstd(np.array([np.nan if x is None else x for x in sequence]))) for label,sequence in values.items()},
                          'regional_median_depth_sequence_m':values,'meaning':'Observed RGB-region medians; different pixels/surfaces across moving-camera frames'}
    write(scene/'geometry/depth_comparison.json',{'rows':rows,'temporal_variation':variations,'sensor_scale_confirmed':False,
            'region_segmentation':'Observed RGB color target/board and disclosed coarse robot/background polygons; target t0 uses '+('SAM' if mask_path.exists() else 'approximate RGB mask; native grounding pending'),
            'limitations':'Sensor units and registration are approximate; ViPE agreement is not independent certification. Fixed image-region std includes motion.'})

def diagnose(camera):
    scene=RUN/camera;camera_vipe,K,depth=load_vipe(scene)
    rgb=np.load(scene/'observed/rgb.npy');sensor=np.load(scene/'observed/sensor_z16.npy').astype(np.float32)*1e-4
    robot=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'));n=len(rgb)
    pairs=static_pairs(rgb);write(scene/'geometry/static_correspondences.json',pairs)
    train=np.arange(n-10);test=np.arange(n-10,n)
    x,y,kin,hand=hand_eye(robot,camera_vipe,train,test);write(scene/'geometry/hand_eye.json',hand)
    np.save(scene/'geometry/hand_eye_X.npy',x);np.save(scene/'geometry/hand_eye_Y.npy',y)
    np.save(scene/'geometry/vipe_camera_poses.npy',camera_vipe);np.save(scene/'geometry/tcp_camera_poses_in_vipe_world.npy',kin)
    canonical_kin=kin.copy();selected_x=x.copy();selected_y=y.copy();selected_source='Observed ViPE AX=XB'
    independent_path=scene/'geometry/independent_robot_rgbd_hand_eye.json'
    independent=json.loads(independent_path.read_text()) if independent_path.exists() else None
    profiles=json.loads((RUN/'sources/official_K_candidates.json').read_text())[camera]
    official_prior=np.array(next(p['K_256_resize'] for p in profiles if p['width']==640 and p['height']==480))
    canonical_static=static_reprojection(sensor,official_prior,kin,[row for row in pairs if row['j']>=n-10])
    canonical_usable=hand['finite_physical_plausibility_gate'] and canonical_static['gate']
    if not canonical_usable and independent and independent['gate']:
        selected_x=np.load(scene/'geometry/independent_robot_rgbd_X.npy')
        selected_y=mean_transform([robot[i]@selected_x@np.linalg.inv(camera_vipe[i]) for i in train])
        kin=np.linalg.inv(selected_y)[None]@(robot@selected_x)
        selected_source='Independent observed RGBD + official K + TCP hand-eye; canonical ViPE hand-eye rejected'
    np.save(scene/'geometry/selected_hand_eye_X.npy',selected_x);np.save(scene/'geometry/selected_hand_eye_Y.npy',selected_y)
    write(scene/'geometry/selected_hand_eye.json',{'X_source':selected_source,'X_TCP_from_camera':selected_x.tolist(),
           'future_used':False,'canonical_ViPE_hand_eye_retained':True,'independent_gate':independent['gate'] if independent else None,
           'selection_basis':'Historical physical plausibility and heldout static reprojection; no future measurements',
           'canonical_heldout_static_reprojection':canonical_static,'canonical_usable':bool(canonical_usable),
           'canonical_X_sha256':sha(scene/'geometry/hand_eye_X.npy'),'selected_X_sha256':sha(scene/'geometry/selected_hand_eye_X.npy')})
    np.save(scene/'geometry/vipe_K.npy',K);np.save(scene/'geometry/vipe_depth.npy',depth)
    candidates=json.loads((RUN/'sources/official_K_candidates.json').read_text())[camera]
    results=[]
    for profile in candidates:
        k=np.array(profile['K_256_resize']);audit=static_reprojection(sensor,k,kin,pairs)
        results.append({'profile':profile,**audit})
    # Official capture source sets 640x480 RGB, no crop. Start with matching 640x480 profile,
    # and report others as historical controls instead of selecting an arbitrary better ADE.
    official=next(row for row in results if row['profile']['width']==640 and row['profile']['height']==480)
    official_K=np.array(official['profile']['K_256_resize']);np.save(scene/'geometry/official_K.npy',official_K)
    write(scene/'geometry/official_profile_validation.json',{'profiles':results,'selected_source_profile':official['profile'],
           'effective_RGB_K_confirmed':False,'selection_reason':'Matches official capture 640x480, cv2 resize; rectified vs actual RGB intrinsic correspondence remains checked by observed reprojection',
           'future_used':False})
    branches={'A_vipe_full':(depth,K,camera_vipe),'B_sensor_vipe':(sensor,K,camera_vipe),'C_sensor_tcp_official':(sensor,official_K,kin)}
    diagnostics={}
    for name,(z,k,c2w) in branches.items():
        diagnostics[name]={'static_reprojection':static_reprojection(z,k,c2w,pairs),
                           'heldout_static_reprojection':static_reprojection(z,k,c2w,[row for row in pairs if row['j']>=n-10]),
                           'pose_source':selected_source if name.startswith('C') else 'ViPE observed RGB',
                           'depth_scale_status':'ViPE estimated metric depth' if name.startswith('A') else 'sensor 0.0001 hypothesis'}
    uni_path=scene/'geometry/unidepth_K.npy';uni_median=np.median(np.load(uni_path),axis=0) if uni_path.exists() else np.full((3,3),np.nan)
    diagnostics['K_values']={'official':official_K.tolist(),'vipe_median':np.median(K,axis=0).tolist(),'unidepth_median':uni_median.tolist() if uni_path.exists() else None,
                              'unidepth_pending':not uni_path.exists()}
    write(scene/'geometry/geometry_diagnostics.json',diagnostics)
    write(scene/'geometry/rejected_canonical_vipe_hand_eye_static.json',static_reprojection(sensor,official_K,canonical_kin,pairs))
    for name,(z,k,c2w) in branches.items():
        folder=scene/'geometry'/name;folder.mkdir(exist_ok=True)
        np.save(folder/'depth.npy',z);np.save(folder/'K.npy',k);np.save(folder/'c2w.npy',c2w)
    # Bootstrap on observed training subsets exposes excitation/hand-eye fragility.
    boot=[]
    for seed in range(5):
        rng=np.random.default_rng(seed);ids=np.sort(rng.choice(train,size=32,replace=False))
        bx,_,_,bi=hand_eye(robot,camera_vipe,ids,test)
        boot.append({'seed':seed,'translation_difference_mm':float(np.linalg.norm(bx[:3,3]-x[:3,3])*1000),
                     'rotation_difference_deg':float(np.degrees((Rotation.from_matrix(x[:3,:3]).inv()*Rotation.from_matrix(bx[:3,:3])).magnitude())),
                     'heldout':bi['heldout']})
    write(scene/'geometry/hand_eye_bootstrap.json',boot)
    fig,axes=plt.subplots(2,3,figsize=(13,7));times=np.arange(n)/10
    for axis in range(3):
        axes[0,axis].plot(times,camera_vipe[:,axis,3],label='ViPE');axes[0,axis].plot(times,canonical_kin[:,axis,3],label='TCP + AX=XB X',alpha=.7);axes[0,axis].plot(times,kin[:,axis,3],label='TCP + selected X');axes[0,axis].set(ylabel='XYZ'[axis]+' (m)',xlabel='observed nominal time (s)');axes[0,axis].legend(fontsize=7)
    axes[1,0].plot(times,hand['translation_error_per_frame_mm']);axes[1,0].set(ylabel='AX=XB pose residual (mm)',xlabel='time (s)')
    axes[1,1].plot(times,hand['rotation_error_per_frame_deg']);axes[1,1].set(ylabel='AX=XB rotation residual (deg)',xlabel='time (s)')
    vals=np.array([[v[0,0],v[1,1],v[0,2],v[1,2]] for v in [official_K,np.median(K,axis=0),uni_median]])
    for i,label in enumerate(['fx','fy','cx','cy']):axes[1,2].plot(['official','ViPE','UniDepth'],vals[:,i],'o-',label=label)
    axes[1,2].legend();axes[1,2].set(ylabel='pixels',title='Same 256x256 coordinates')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(scene/'viz/pose_K_comparison.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(projection='3d')
    ax.plot(*camera_vipe[:,:3,3].T,label='ViPE');ax.plot(*kin[:,:3,3].T,label='TCP + X');ax.scatter(*camera_vipe[-1,:3,3],label='t0');ax.legend();ax.set(xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');fig.tight_layout();fig.savefig(scene/'viz/camera_paths_3d.png',dpi=160);plt.close(fig)
    plot_depth(scene,rgb,sensor,depth)
    print(camera,'geometry', {k:v['static_reprojection'] for k,v in diagnostics.items() if k!='K_values'},'handeye',hand['estimated_translation_norm_m'],flush=True)

def export(camera):
    scene=RUN/camera;assert not(scene/'geometry/branch_selection.json').exists(),'Inputs already exported'
    tracks=np.load(scene/'observed/observed_tracks_2d.npz');uv=tracks['tracks'];visible=tracks['visibility'];confidence=tracks['confidence']
    rgb=np.load(scene/'observed/rgb.npy');mask=cv2.imread(str(scene/'observed/mask.png'),0)>0
    hmask=np.load(scene/'observed/historical_masks_h3.npy');inside=np.zeros((3,len(uv[0])),bool)
    for t in range(3):
        points=np.rint(uv[-3+t]).astype(int);bounded=(points>=0).all(-1)&(points<256).all(-1)
        inside[t,bounded]=hmask[t,points[bounded,1],points[bounded,0]]
    margin=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5);points=np.rint(uv[-1]).astype(int)
    border=margin[np.clip(points[:,1],0,255),np.clip(points[:,0],0,255)]
    eligible=inside.all(0)&visible[-3:].all(0)&(border>=2.5)
    names=['A_vipe_full','B_sensor_vipe','C_sensor_tcp_official'];outputs={}
    author=author_filter()
    for name in names:
        folder=scene/'geometry'/name;depth=np.load(folder/'depth.npy');K=np.load(folder/'K.npy');c2w=np.load(folder/'c2w.npy')
        raw,valid,spread,coverage=lift(depth,uv,visible,K,c2w)
        # Author functions receive zero-filled invisible samples, with the same visibility gate.
        work=np.where(valid[...,None],raw,0).astype(np.float32)
        trust,anchors=author.compute_trust_weights(work,valid,K=16)
        keep=~author.filter_tracks_by_trust(trust,valid,z_thresh=2)
        origins=np.array([transform(np.linalg.inv(c2w[-1]),t[:3,3][None])[0] for t in c2w],np.float32)
        smoothed=work.copy()
        smoothed[:,keep]=author.consensus_gated_smooth(work[:,keep],origins,valid[:,keep],trust[:,keep],device='cpu')
        eligible &= valid[-3:].all(0)&keep&(spread[-3:].max(0)<.03)&(smoothed[-3:,:,2]>0).all(0)
        outputs[name]=(raw,smoothed,valid,trust,keep,coverage)
        np.save(folder/'points_3d_raw.npy',raw);np.save(folder/'points_3d_filtered.npy',smoothed)
        np.savez_compressed(folder/'filter_diagnostics.npz',valid=valid,trust=trust,keep=keep,anchors=anchors,depth_spread=spread,depth_coverage=coverage)
    ids=b.spread_select(uv[-1],eligible,24,confidence[-3:].min(0))
    if len(ids)!=24:raise RuntimeError(f'Only {len(ids)} common valid points across A/B/C, cannot silently reduce scope')
    np.save(scene/'observed/selected_point_ids.npy',ids);np.save(scene/'observed/points_2d_history.npy',uv[-3:,ids])
    diag=json.loads((scene/'geometry/geometry_diagnostics.json').read_text());hand=json.loads((scene/'geometry/hand_eye.json').read_text())
    rows={}
    for name,(raw,smoothed,valid,trust,keep,coverage) in outputs.items():
        history=smoothed[-3:,ids];raw_h=raw[-3:,ids]
        a,c=np.triu_indices(24,1);dist=np.linalg.norm(history[:,:,None]-history[:,None],axis=-1)[:,a,c]
        relative=np.ptp(dist,axis=0)/np.maximum(np.median(dist,axis=0),.001)
        rigid=float(np.median(relative));gate=diag[name]['static_reprojection']['gate'] and rigid<=.15
        if name.startswith('C'):
            selected=json.loads((scene/'geometry/selected_hand_eye.json').read_text())
            gate &= bool(hand['finite_physical_plausibility_gate'] or selected['independent_gate'])
        rows[name]={'validated_geometry_gate':bool(gate),'relative_rigidity_median':rigid,
                    'smoothing_max_displacement_m':float(np.max(np.linalg.norm(history-raw_h,axis=-1))),
                    'static_reprojection':diag[name]['static_reprojection']}
        c2w=np.load(scene/'geometry'/name/'c2w.npy');K=np.load(scene/'geometry'/name/'K.npy')
        if K.ndim==2:K=np.repeat(K[None],len(c2w),axis=0)
        roundtrip=[]
        for t in range(3):
            xyz_camera=from_anchor(history[t],c2w[-3+t],c2w[-1]);roundtrip.append(np.linalg.norm(project(xyz_camera,K[-3+t])-uv[-3+t,ids],axis=-1))
        max_error=float(np.max(roundtrip));assert max_error<.02,f'Moving camera ray roundtrip failed {name}: {max_error}'
        rows[name]['moving_camera_roundtrip_max_error_px']=max_error
        dest=scene/'branches'/name;(dest/'geometry').mkdir(parents=True,exist_ok=True);(dest/'groups').mkdir(exist_ok=True);(dest/'predictions').mkdir(exist_ok=True)
        shutil.copytree(scene/'observed',dest/'observed',dirs_exist_ok=True)
        np.save(dest/'observed/points_3d_history.npy',history);np.save(dest/'observed/native_depth.npy',np.load(scene/'geometry'/name/'depth.npy'))
        np.save(dest/'geometry/points_3d_raw.npy',raw);np.save(dest/'geometry/points_3d_filtered.npy',smoothed)
        np.save(dest/'geometry/camera_poses.npy',np.load(scene/'geometry'/name/'c2w.npy'));np.save(dest/'geometry/K.npy',np.load(scene/'geometry'/name/'K.npy'))
        meta=json.loads((scene/'metadata.json').read_text());meta.update(geometry_variant=name,geometry_gate=bool(gate),diagnostic_only=not gate,
            history_timestamps=(np.array(meta['history_source_indices'])/10).tolist(),coordinates='XYZ in camera_t0, OpenCV, estimated meters',
            primary_selection_source='Observed-only static reprojection, rigid history and hand-eye validation')
        write(dest/'metadata.json',meta)
        for g in range(3):
            out=dest/'groups'/f'group_{g:02d}';out.mkdir(exist_ok=True);group_ids=ids[g*8:g*8+8]
            np.save(out/'point_ids.npy',group_ids);np.save(out/'points_2d_at_t0.npy',uv[-1,group_ids]);np.save(out/'points_3d_history.npy',smoothed[-3:,group_ids])
        from berkeley_input_audit import validate_group_payloads
        audit=validate_group_payloads(dest)
        write(dest/'geometry/input_audit.json',{'success':True,'meaning':'Computational payload/coordinate integrity only; physical validation gate separately recorded',
              'geometry_validation':rows[name],'finite_H3':bool(np.isfinite(history).all()),'group_payloads':audit,'future_used':False,'selected_points':24})
    passed=[name for name in names if rows[name]['validated_geometry_gate']]
    primary=min(passed,key=lambda name:diag[name]['heldout_static_reprojection']['median_px'] or float('inf')) if passed else None
    write(scene/'geometry/branch_selection.json',{'branches':rows,'primary':primary,'common_point_ids':ids.tolist(),'future_used':False,
        'selection_reason':'Minimum observed static reprojection among predeclared validated gates; none accepted if all fail',
        'needs_builtin_visual_review_before_inference':True})
    im=b.draw_points(rgb[-1],uv[-1,ids],ids,radius=2);b.save_rgb(scene/'viz/selected_24_points.png',im)
    panels=[]
    for t in [39,44,47,48,49]:
        source_index=int(np.load(scene/'observed/source_indices.npy')[t])
        im=b.draw_points(rgb[t],uv[t,ids],ids,radius=1);im=cv2.resize(im,(512,512));cv2.putText(im,f'{camera} source {source_index}',(10,28),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(im)
    b.save_rgb(scene/'viz/selected_history_tracks.png',np.hstack(panels))
    print(camera,'frozen IDs',ids.tolist(),'primary',primary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','export']);p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);a=p.parse_args()
    for camera in a.cameras:{'diagnose':diagnose,'export':export}[a.stage](camera)
