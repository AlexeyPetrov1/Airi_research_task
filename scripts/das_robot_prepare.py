"""Build a paired static-arm/articulated-arm DaS experiment from observed data.

No evaluation directory or future robot state is read. The cup is rigidly
attached to the flange. Bounded nominal UR5 IK projects the frozen Molmo cup
forecast into this coupled control. This is approximate image registration,
not factory-calibrated robot planning.
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from das_prepare_control import backproject,project,splat,save_video,sheet,sha256,write_json,object_depth
from das_repair_background import complete_background
from das_robot_geometry import fk, transform

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup'
OBS=OUT/'observed'
PARTS={'upper_arm':2,'forearm':3,'wrist_low':5,'wrist_high':4,'gripper':6}

def table_plane(depth,k):
    yy,xx=np.indices(depth.shape);sel=(xx>230)&(xx<420)&(yy>310)&(yy<400)&(depth>0)
    pts=backproject(np.c_[xx[sel],yy[sel]],depth[sel],k)[::5]
    rng=np.random.default_rng(42);best=None
    for _ in range(200):
        a,b,c=pts[rng.choice(len(pts),3,replace=False)];n=np.cross(b-a,c-a)
        if np.linalg.norm(n)<1e-7:continue
        n/=np.linalg.norm(n);d=-n@a;inlier=np.abs(pts@n+d)<.005
        if best is None or inlier.sum()>best.sum():best=inlier
    p=pts[best];_,_,v=np.linalg.svd(p-p.mean(0),full_matrices=False);n=v[-1];d=-n@p.mean(0)
    if d<0:n,d=-n,-d
    return n,d,float(best.mean())

def robot_cloud(depth,k,masks,frames_cam):
    labels=np.zeros(depth.shape,np.uint8)
    for part in ['upper_arm','forearm','wrist_high','wrist_low','gripper']:
        labels[masks[part]]=PARTS[part]
    xyzs=[];uvs=[];links=[];estimates=[]
    for link in [2,3,4,5,6]:
        mask=labels==link;y,x=np.nonzero(mask);uv=np.c_[x,y]
        valid=(depth>0)&(depth<2)&mask
        if valid.any():
            nearest=distance_transform_edt(~valid,return_distances=False,return_indices=True)
            z=depth[tuple(nearest)][y,x].astype(float)
            # Filling reflective depth holes is an estimate, recorded separately.
        else:z=np.full(len(uv),frames_cam[link,2,3]-.035)
        z=np.maximum(z,.1)
        xyzs.append(backproject(uv,z,k));uvs.append(uv);links.append(np.full(len(uv),link));estimates.append(~valid[y,x])
    return np.concatenate(xyzs),np.concatenate(uvs),np.concatenate(links),np.concatenate(estimates),labels

def main():
    if any((OUT/name/'generated_molmomotion_seed42.mp4').exists() for name in ['robot_static','robot_coupled','robot_coupled_initial_only']):
        raise FileExistsError('Preserve finished experiments; prepare controls in a fresh checkout with observed inputs only.')
    rgb=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    depth=np.load(SCENE/'observed/native_depth.npy')[-1].astype(float)
    cup_mask=np.array(Image.open(SCENE/'observed/mask.png'))>0
    c=np.load(OBS/'nominal_ur5_camera.npz');k=c['K'];cam=c['camera_from_base'];q0=c['q0'];f0=c['fk0'];fc0=cam@f0
    masks={name:np.array(Image.open(OBS/f'{name}_mask.png'))>0 for name in ['upper_arm','forearm','wrist','gripper']}
    yy=np.indices(depth.shape)[0];masks['wrist_high']=masks['wrist']&(yy<108);masks['wrist_low']=masks['wrist']&(yy>=108)
    # Mask seams overlap in SAM; use a single owner per observed pixel.
    rx,ruv,rlink,restimated,labels=robot_cloud(depth,k,masks,fc0)
    cup=np.load(SCENE/'das_wanfun/object_cloud_t0.npz');cx=cup['xyz'];cu=cup['uv'];sparse=np.load(SCENE/'observed/points_3d_history.npy')[-1]
    robot_union=labels>0;union=robot_union|cup_mask
    # A dense visible surface follows its nominal link, with exact t0 anchoring.
    local=np.empty_like(rx)
    for link in np.unique(rlink):local[rlink==link]=transform(rx[rlink==link],np.linalg.inv(fc0[link]))
    cup_local=transform(cx,np.linalg.inv(fc0[6]));sparse_local=transform(sparse,np.linalg.inv(fc0[6]))
    target=np.load(SCENE/'das_wanfun/rigid_motion_49.npz');times=target['times']
    n,d,inliers=table_plane(depth,k)
    initial_clearance=float((cx@n+d).min());clearance_floor=min(initial_clearance,0.)
    qseq=[q0];pose_errors=[];qs=q0.copy();speed_limit=2.0;dt=1/8
    bounds_low=np.full(6,-2*np.pi);bounds_high=-bounds_low;bounds_low[2]=-np.pi;bounds_high[2]=np.pi
    for i in range(1,17):
        delta=np.eye(4);delta[:3,:3]=target['R'][i];delta[:3,3]=target['t'][i]
        wanted_cam=delta@fc0[6];wanted_base=np.linalg.inv(cam)@wanted_cam
        prev=qs.copy()
        def loss(q):
            actual=fk(q)[6];pos=(actual[:3,3]-wanted_base[:3,3])/.01
            rot=Rotation.from_matrix(wanted_base[:3,:3].T@actual[:3,:3]).as_rotvec()/.10
            pts=transform(cup_local,cam@actual)
            penetration=max(0.,clearance_floor-float((pts@n+d).min()))/.005
            return np.r_[pos,rot,(q-prev)*.15,penetration]
        lo=np.maximum(bounds_low,prev-speed_limit*dt);hi=np.minimum(bounds_high,prev+speed_limit*dt)
        fit=least_squares(loss,np.clip(prev,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=120)
        qs=fit.x;qseq.append(qs.copy());actual=fk(qs)[6]
        pose_errors.append([np.linalg.norm(actual[:3,3]-wanted_base[:3,3]),np.linalg.norm(Rotation.from_matrix(wanted_base[:3,:3].T@actual[:3,:3]).as_rotvec())])
    qseq=np.array(qseq+[qseq[-1]]*32);fkseq=np.array([fk(q) for q in qseq]);fc=cam@fkseq
    cup_xyz=np.array([transform(cup_local,t) for t in fc[:,6]])
    cup_sparse=np.array([transform(sparse_local,t) for t in fc[:,6]])
    robot_xyz=np.empty((49,len(rx),3),np.float32)
    for link in np.unique(rlink):
        for i in range(49):robot_xyz[i,rlink==link]=transform(local[rlink==link],fc[i,link])
    assert np.allclose(cup_xyz[0],cx,atol=1e-8) and np.allclose(robot_xyz[0],rx,atol=1e-7)
    # Representative material points for independent tracking and contact audit.
    manual=np.array([[600,100],[600,275],[599,340],[310,20],[270,48],[228,75],[202,84],
                     [150,76],[164,100],[161,120],[166,158],[159,181],[182,202],[173,223]],float)
    robot_ids=[]
    for x,y in manual:
        distance=np.linalg.norm(ruv-[x,y],axis=1);robot_ids.append(int(np.argmin(distance)))
    robot_ids=np.array(robot_ids)
    np.savez_compressed(OUT/'joint_control.npz',times=times,joints=qseq,flange_camera=fc[:,6],
        fk_camera=fc,cup_sparse_xyz=cup_sparse,robot_sparse_xyz=robot_xyz[:,robot_ids],
        robot_query_uv=ruv[robot_ids],robot_query_link=rlink[robot_ids],robot_query_cloud_ids=robot_ids,
        cup_centroid_xyz=cup_xyz.mean(1),table_normal=n,table_offset=d)
    np.savez_compressed(OUT/'robot_cloud_t0.npz',xyz=rx,uv=ruv,link_ids=rlink,estimated_depth=restimated,local_xyz=local)
    np.savez_compressed(OUT/'robot_dense_motion.npz',xyz=robot_xyz,link_ids=rlink,times=times)
    desired=np.einsum('tij,nj->tni',target['R'],cx)+target['t'][:,None]
    cup_projection_error=np.linalg.norm(project(cup_xyz.reshape(-1,3),k).reshape(49,-1,2)-project(desired.reshape(-1,3),k).reshape(49,-1,2),axis=-1)
    initial_rel=np.linalg.inv(fc0[6]);contact_err=[]
    for i in range(49):contact_err.append(np.linalg.norm(transform(cup_xyz[i],np.linalg.inv(fc[i,6]))-cup_local,axis=1).max())
    clearance=np.min(cup_xyz@n+d,axis=1)
    receipt={'future_used':False,'method':'Frozen Molmo rigid cup forecast -> bounded nominal UR5 6-DOF IK -> FK robot link motion; cup rigidly attached to achieved flange pose',
        'projection_changes_cup_forecast':True,'nominal_ur5_speed_limit_rad_s':float(np.pi),'experiment_speed_limit_rad_s':speed_limit,
        'max_joint_speed_rad_s':np.max(np.abs(np.diff(qseq,axis=0))/dt,axis=0).tolist(),
        'max_joint_acceleration_rad_s2':np.max(np.abs(np.diff(qseq,axis=0,n=2))/dt**2,axis=0).tolist(),
        'max_cup_flange_local_attachment_error_m':float(max(contact_err)),
        'target_flange_position_error_m':[p[0] for p in pose_errors],'target_flange_rotation_error_rad':[p[1] for p in pose_errors],
        'cup_to_frozen_rigid_forecast_mean_3d_error_m':float(np.linalg.norm(cup_xyz[:17]-desired[:17],axis=-1).mean()),
        'cup_to_frozen_rigid_forecast_mean_2d_error_px':float(cup_projection_error[:17].mean()),
        'table_plane_fit_inlier_fraction':inliers,'visible_cup_min_table_clearance_m':clearance.tolist(),
        'robot_points':len(rx),'estimated_robot_depth_points':int(restimated.sum()),'manual_tracking_points':len(manual),
        'pair':'A robot_static vs B robot_coupled; cup motion, background, prompts, seed, model and vacancy algorithm identical',
        'limitations':['Estimated camera registration; visible point clouds only','IK position and velocity bounded; no full CAD self-collision or torque validation','Maximum acceleration descriptive; no public nominal acceleration constraint imposed','Reflective robot depth holes filled from nearest measured link pixels; explicitly estimated','Trajectory projection must be separated from original Molmo forecast accuracy'],
        'speed_limit_source':'https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/rolling/config/ur5/joint_limits.yaml'}
    write_json(OUT/'kinematic_projection.json',receipt)
    overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay)
    for i,(x,y) in enumerate(ruv[robot_ids]):draw.ellipse((x-3,y-3,x+3,y+3),fill='red');draw.text((x+4,y-5),f'R{i}/L{rlink[robot_ids[i]]}',fill='yellow')
    overlay.save(OBS/'robot_tracking_points.png')
    completed,repair=complete_background(depth,union,dilation=6)
    # Repair is behind every removed observed surface, not a foreground clone.
    nearest=distance_transform_edt(~union,return_distances=False,return_indices=True)
    for link in np.unique(rlink):completed[labels==link]=np.maximum(completed[labels==link],np.percentile(rx[rlink==link,2],95)+.05)
    completed[cup_mask]=np.maximum(completed[cup_mask],cx[:,2].max()+.02)
    valid=(depth>0)&(depth<10);inv_lo,inv_hi=np.percentile(1/depth[valid],[2,98])
    def colors(uv,z):return np.stack([uv[:,0]/639*255,uv[:,1]/479*255,np.clip((1/z-inv_lo)/(inv_hi-inv_lo),0,1)*255],axis=1).astype(np.uint8)
    rcolors=colors(ruv,rx[:,2]);ccolors=cup['colors']
    y,x=np.nonzero((valid&~repair)|repair);buv=np.c_[x,y];bxyz=backproject(buv,completed[y,x],k)
    bg=np.zeros_like(rgb);bz=np.full(depth.shape,np.inf);splat(bxyz,colors(buv,completed[y,x]),k,bg,bz,radius=0)
    # Apply generated plate only inside dilated observed moving-object support.
    alpha=np.minimum(distance_transform_edt(repair)/4,1.)
    plate=np.array(Image.open(OBS/'imagegen_empty_robot_plate.png').convert('RGB').resize((640,480),Image.Resampling.LANCZOS))
    clean=np.rint(rgb*(1-alpha[...,None])+plate*alpha[...,None]).astype(np.uint8)
    assert np.array_equal(clean[~repair],rgb[~repair])
    Image.fromarray(clean).save(OBS/'clean_reference_640x480.png')
    Image.fromarray(clean).resize((720,480),Image.Resampling.BILINEAR).save(OBS/'clean_reference_720x480.png')
    Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(OBS/'clean_reference_alpha.png')
    sheet(OBS/'clean_reference_audit.png',[[rgb],[clean]],['OBSERVED START','SYNTHETIC BACKGROUND CONDITION'],indices=(0,))
    inputs=['observed/frame_000063.png','observed/native_depth.npy','observed/mask.png','observed/points_3d_history.npy','observed/points_2d_history.npy','geometry/K_median.npy','das_wanfun/rigid_motion_49.npz',
            'das_robot_cup/observed/robot_state_history.npz','das_robot_cup/observed/nominal_ur5_camera.npz','das_robot_cup/observed/imagegen_empty_robot_plate.png']+[f'das_robot_cup/observed/{name}_mask.png' for name in ['upper_arm','forearm','wrist','gripper']]
    freeze={'future_used':False,'allowed_inputs':inputs,'sha256':{name:sha256(SCENE/name) for name in inputs}}
    write_json(OUT/'control_input_freeze.json',freeze)
    all_controls=[];all_renders=[]
    for mode in ['robot_static','robot_coupled']:
        out=OUT/mode;out.mkdir(exist_ok=True)
        frames=[];renders=[];moving=[];visibility=[]
        for i in range(49):
            rxyz=rx if mode=='robot_static' else robot_xyz[i]
            canvas,z=bg.copy(),bz.copy();splat(rxyz,rcolors,k,canvas,z,radius=1);good=splat(cup_xyz[i],ccolors,k,canvas,z,radius=1)
            frames.append(canvas);visibility.append(float(good.mean()))
            fg=np.zeros_like(rgb);fz=np.full(depth.shape,np.inf)
            splat(rxyz,np.full_like(rcolors,255),k,fg,fz,radius=1);splat(cup_xyz[i],np.full_like(ccolors,255),k,fg,fz,radius=1);moving.append(fg[...,0]>0)
            render=clean.copy();rz=bz.copy();splat(rxyz,rgb[ruv[:,1],ruv[:,0]],k,render,rz,radius=1);splat(cup_xyz[i],rgb[cu[:,1],cu[:,0]],k,render,rz,radius=1)
            renders.append(render)
        resized=[np.array(Image.fromarray(v).resize((720,480),Image.Resampling.BILINEAR)) for v in frames]
        save_video(out/'control_molmomotion_640x480.mp4',frames);save_video(out/'control_molmomotion_720x480.mp4',resized)
        save_video(out/'geometry_preview.mp4',renders)
        sheet(out/'control_contact_sheet.png',[frames,renders],['DENSE MATERIAL-POINT CONTROL','VISIBLE-SURFACE GEOMETRY PREVIEW'])
        np.savez_compressed(out/'background_completion.npz',depth=completed,repair_mask=repair,moving_mask=np.stack(moving),original_mask=union)
        np.savez_compressed(out/'dense_predicted_motion.npz',xyz=cup_xyz.astype(np.float32),sparse_xyz=cup_sparse.astype(np.float32),times=times)
        for name in ['checkpoint_receipt.json','das_wanfun_commit.txt','image_t0_720x480.png','rigid_motion.npz','rigid_fit.json','temporal_mapping.json']:shutil.copyfile(SCENE/'das_wanfun'/name,out/name)
        write_json(out/'control_input_freeze.json',freeze)
        write_json(out/'control_validation.json',{'generation_ready':False,'visual_review_required':True,'future_used':False,
            'variant':mode,'control_sha256':sha256(out/'control_molmomotion_720x480.mp4'),'cup_motion_same_in_pair':True,
            'robot_articulated':mode=='robot_coupled','cup_in_frame_fraction':visibility,'dense_robot_points':len(rx),
            'cup_rigidly_attached_to_achieved_flange':True,'foreground_colors_constant':True,'approximate_camera_registration':True})
        all_controls.append(frames);all_renders.append(renders)
    assert np.array_equal(all_controls[0][0],all_controls[1][0])
    sheet(OUT/'paired_control_review.png',all_controls+all_renders,['A STATIC ARM','B COUPLED ARM','A GEOMETRY PREVIEW','B GEOMETRY PREVIEW'])
    print(receipt)

if __name__=='__main__':main()
