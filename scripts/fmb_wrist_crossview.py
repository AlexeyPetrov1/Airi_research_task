"""Observed-only wrist/side geometry checks with explicit correspondence limits."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from fmb_wrist_prepare import ROOT,RUN,CAMERAS,write
from fmb_wrist_math import *
import berkeley_preprocess as b

def rgb_masks(rgb):
    r,g,blue=rgb.transpose(2,0,1).astype(float)
    target=(r>g*1.35)&(r>blue*1.35)&(r>80)
    board=(blue>r*1.25)&(blue>g*1.12)&(blue>65)
    return target,board

def scene_points(rgb,depth,K,region):
    vv,uu=np.where(region);uv=np.stack([uu,vv],-1).astype(float)
    z=depth[vv,uu];good=(z>0)&(z<2)&np.isfinite(z)
    return backproject(uv[good],z[good],K)

def align_rigid(a,c):
    aa,cc=a-a.mean(0),c-c.mean(0);u,_,vt=np.linalg.svd(aa.T@cc)
    R=vt.T@u.T
    if np.linalg.det(R)<0:vt[-1]*=-1;R=vt.T@u.T
    out=np.eye(4);out[:3,:3]=R;out[:3,3]=c.mean(0)-R@a.mean(0);return out

def run():
    rows=[];wrist={};sides={};allrgb={};alldepth={}
    for camera in CAMERAS:
        scene=RUN/camera;rgb=np.load(scene/'observed/rgb.npy');z=np.load(scene/'observed/sensor_z16.npy').astype(float)*1e-4
        allrgb[camera]=rgb;alldepth[camera]=z
        target,board=rgb_masks(rgb[-1])
        kdata=json.loads((RUN/'sources/official_K_candidates.json').read_text())[camera]
        k=np.array(next(v for v in kdata if v['width']==640 and v['height']==480)['K_256_resize'])
        rows.append({'camera':camera,'target_t0_visible_red_pixels':int(target.sum()),'board_t0_visible_blue_pixels':int(board.sum()),
                     'target_sensor_valid_fraction':float((z[-1][target]>0).mean()) if target.any() else None,
                     'primary_target_full_contour_visible':None,'K_is_official_profile_prior':True})
        if camera.startswith('wrist'):
            x=np.load(scene/'geometry/selected_hand_eye_X.npy');tcp=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'));pose=tcp@x
            boardcloud=scene_points(rgb[-1],z[-1],k,board);targetcloud=scene_points(rgb[-1],z[-1],k,target)
            wrist[camera]={'board':transform(pose[-1],boardcloud),'target':transform(pose[-1],targetcloud),'pose':pose,'K':k}
        else:sides[camera]={'K':k,'c2w':np.repeat(np.eye(4)[None],len(rgb),axis=0)}
    w1,w2=wrist['wrist_1'],wrist['wrist_2'];a,c=w1['board'],w2['board']
    nearest=cKDTree(c).query(a)[0];reverse=cKDTree(a).query(c)[0]
    agreement={'board_bidirectional_nearest_median_mm':float(np.median(np.r_[nearest,reverse])*1000),
               'board_bidirectional_nearest_p90_mm':float(np.percentile(np.r_[nearest,reverse],90)*1000),
               'target_centroid_difference_mm':float(np.linalg.norm(np.median(w1['target'],0)-np.median(w2['target'],0))*1000),
               'interpretation':'Different surfaces/partial FOV, ambiguous factory RGB K and independently estimated X; nearest-surface agreement is not shared-point accuracy'}
    fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(projection='3d')
    for camera,data in wrist.items():
        for label in ['board','target']:
            cloud=data[label][::10];ax.scatter(*cloud.T,s=2,alpha=.4,label=camera+' '+label)
    ax.set(xlabel='robot base x (m)',ylabel='y (m)',zlabel='z (m)');ax.legend();fig.tight_layout();fig.savefig(RUN/'wrist_crossview_base_clouds.png',dpi=150);plt.close(fig)
    # SIFT correspondences on the static board; no stereo triangulation and no synchronized-frame claim.
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.015)
    camera='wrist_1';rgb=allrgb[camera][-1];_,board=rgb_masks(rgb)
    kp,des=sift.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),board.astype(np.uint8)*255)
    matches_receipts=[]
    for side in ['side_1','side_2']:
        im=allrgb[side][-1];_,region=rgb_masks(im)
        kk,dd=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_RGB2GRAY),region.astype(np.uint8)*255)
        matched=[]
        if des is not None and dd is not None and len(dd)>=2:
            for candidates in cv2.BFMatcher().knnMatch(des,dd,k=2):
                if len(candidates)==2 and candidates[0].distance<.72*candidates[1].distance:matched.append(candidates[0])
        usable=[];objects=[];pixels=[];depthpoints=[]
        for m in matched:
            p=np.array(kp[m.queryIdx].pt)[None];q=np.array(kk[m.trainIdx].pt)[None]
            zw,_,_=sample_z(alldepth[camera][-1],p);zs,_,_=sample_z(alldepth[side][-1],q)
            if np.isfinite(zw).all() and np.isfinite(zs).all():
                objects.append(transform(w1['pose'][-1],backproject(p,zw,w1['K']))[0]);pixels.append(q[0]);depthpoints.append(backproject(q,zs,sides[side]['K'])[0]);usable.append(m)
        receipt={'side_camera':side,'wrist_camera':camera,'matched_static_board_features':len(matched),'valid_depth_matches':len(usable),'future_used':False}
        if len(usable)>=6:
            world=np.array(objects);sidepoints=np.array(depthpoints);candidate=align_rigid(sidepoints,world)
            error=np.linalg.norm(transform(candidate,sidepoints)-world,axis=-1)
            receipt.update(alignment_side_c2base=candidate.tolist(),alignment_median_mm=float(np.median(error)*1000),
                           alignment_p90_mm=float(np.percentile(error,90)*1000),alignment_validation='Observed fit only; repetitive board holes, no held-out independent validation')
            np.save(RUN/side/'geometry/estimated_side_c2base.npy',candidate)
        else:receipt['alignment_status']='INSUFFICIENT_UNAMBIGUOUS_STATIC_CROSSVIEW_MATCHES'
        draw=cv2.drawMatches(rgb,kp,im,kk,usable,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        b.save_rgb(RUN/side/'viz/crossview_board_matches.png',draw)
        # Side cameras are fixed; 3D displacement magnitude needs no absolute camera rotation calibration.
        centers=[];areas=[]
        for frame,depth in zip(allrgb[side],alldepth[side]):
            mask,_=rgb_masks(frame);points=scene_points(frame,depth,sides[side]['K'],mask);areas.append(len(points))
            centers.append(np.median(points,axis=0) if len(points)>=20 else np.full(3,np.nan))
        centers=np.array(centers);np.save(RUN/side/'geometry/observed_target_color_centroids.npy',centers)
        receipt.update(target_valid_centroid_frames=int(np.isfinite(centers).all(-1).sum()),target_t0_valid_depth_pixels=areas[-1],
           target_centroid_limitations='Partial contours, perspective and visible surfaces change; color centroid is an approximate independent control, not the same material IDs')
        matches_receipts.append(receipt)
    # Approximate camera timing offset: fit X from prefix, compare translation residual under integer offsets.
    offsets={}
    for camera in ['wrist_1','wrist_2']:
        scene=RUN/camera;vipe=np.load(scene/'geometry/vipe_camera_poses.npy');robot=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'))
        x=np.load(scene/'geometry/hand_eye_X.npy');y=np.load(scene/'geometry/hand_eye_Y.npy');est=np.linalg.inv(y)[None]@(robot@x)
        values=[]
        for shift in range(-2,3):
            ids=np.arange(max(0,-shift),min(40,40-shift));res=np.linalg.inv(vipe[ids])@est[ids+shift]
            values.append({'robot_offset_frames':shift,'median_translation_mm':float(np.median(np.linalg.norm(res[:,:3,3],axis=-1))*1000)})
        offsets[camera]={'candidate_offsets':values,'chosen_for_geometry':0,'interpretation':'Sensitivity only; correlated pose drift and hand-eye fit confound offset. No timestamp correction asserted.'}
    write(RUN/'crossview_observed_audit.json',{'view_visibility':rows,'wrist_geometry_agreement':agreement,'side_checks':matches_receipts,
         'timing_offset_sensitivity':offsets,'future_used':False,'four_camera_exact_triangulation_used':False,
         'hardware_synchronization_confirmed':False,'requires_builtin_image_review':True})
    print('Observed cross-view audit',agreement,matches_receipts,flush=True)

if __name__=='__main__':run()
