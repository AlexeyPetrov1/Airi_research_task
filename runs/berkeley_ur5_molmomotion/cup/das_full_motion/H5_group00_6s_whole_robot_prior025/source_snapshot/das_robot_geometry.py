"""Nominal UR5 FK and observed-only camera fit for a video-control experiment."""
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PIL import Image, ImageDraw
from das_prepare_control import backproject, project, kabsch, write_json

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup'
A=[0,-.425,-.39225,0,0,0]
D=[.089159,0,0,.10915,.09465,.0823]
ALPHA=[np.pi/2,0,0,np.pi/2,-np.pi/2,0]

def fk(q):
    t=np.eye(4); frames=[t.copy()]
    for theta,a,d,alpha in zip(q,A,D,ALPHA):
        c,s=np.cos(theta),np.sin(theta); ca,sa=np.cos(alpha),np.sin(alpha)
        t=t@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1.]])
        frames.append(t.copy())
    return np.array(frames)

def transform(points,t): return points@t[:3,:3].T+t[:3,3]

def main():
    OUT.mkdir(exist_ok=True)
    state=np.load(OUT/'observed/robot_state_history.npz')
    q=state['joints'][-1].astype(float); frames=fk(q)
    k=np.load(SCENE/'geometry/K_median.npy'); depth=np.load(SCENE/'observed/native_depth.npy')[-1]
    # Visible joint centres, annotated on observed t0 only. Depth measures skin;
    # a 35 mm nominal radius moves the estimate behind its front surface.
    frame_ids=np.array([1,4,5,6])
    uv=np.array([[600,430],[225,75],[162,80],[165,143]],float)
    z=np.array([np.median(p[p>0]) if (p:=depth[int(y)-4:int(y)+5,int(x)-4:int(x)+5]).max()>0 else np.nan for x,y in uv])+.035
    valid_depth=np.isfinite(z)
    camera_xyz=backproject(uv[valid_depth],z[valid_depth],k)
    r,t,rms,_=kabsch(frames[frame_ids[valid_depth],:3,3],camera_xyz)
    initial=np.r_[Rotation.from_matrix(r).as_rotvec(),t]
    def residual(p):
        rr=Rotation.from_rotvec(p[:3]).as_matrix(); xyz=frames[frame_ids,:3,3]@rr.T+p[3:]
        # The camera observes robot skin, not the DH-axis centres. Treat depth
        # as a weak radius-dependent constraint rather than an exact landmark.
        zz=xyz[valid_depth,2];skin=z[valid_depth]-.035
        return np.r_[(project(xyz,k)-uv).ravel()/2,(zz-z[valid_depth])/.15,
                     np.maximum(skin-zz,0)/.015,np.maximum(zz-skin-.10,0)/.03]
    fit=least_squares(residual,initial)
    cam=np.eye(4);cam[:3,:3]=Rotation.from_rotvec(fit.x[:3]).as_matrix();cam[:3,3]=fit.x[3:]
    centers=transform(frames[:,:3,3],cam); repro=project(centers,k)
    np.savez_compressed(OUT/'observed/nominal_ur5_camera.npz', camera_from_base=cam,q0=q,
                        fk0=frames, annotated_uv=uv, annotated_frame_ids=frame_ids, observed_depth=z, K=k)
    view=Image.open(SCENE/'observed/frame_000063.png').convert('RGB'); draw=ImageDraw.Draw(view)
    for i,(x,y) in enumerate(repro):
        if i:draw.line([tuple(repro[i-1]),(x,y)],fill='yellow',width=3)
        draw.ellipse((x-4,y-4,x+4,y+4),fill='red');draw.text((x+5,y),f'J{i}',fill='red')
    for x,y in uv:draw.ellipse((x-4,y-4,x+4,y+4),outline='cyan',width=2)
    view.save(OUT/'observed/robot_joint_camera_overlay.png')
    ee=state['ee_pose'][-1]; native=np.eye(4);native[:3,:3]=Rotation.from_quat(ee[3:]).as_matrix();native[:3,3]=ee[:3]
    # Native pose describes the TCP, while the DH chain ends at the flange.
    tcp=np.linalg.inv(frames[-1])@native
    history_errors=[]
    for joints,ee_history in zip(state['joints'],state['ee_pose']):
        predicted=fk(joints)[-1]@tcp
        native_r=Rotation.from_quat(ee_history[3:]).as_matrix()
        history_errors.append([np.linalg.norm(predicted[:3,3]-ee_history[:3]),
            np.linalg.norm(Rotation.from_matrix(native_r.T@predicted[:3,:3]).as_rotvec())])
    write_json(OUT/'observed/geometry_audit.json',{'future_used':False,'q0_rad':q.tolist(),
        'K':k.tolist(),'camera_from_nominal_base':cam.tolist(),'nominal_DH_a_m':A,'nominal_DH_d_m':D,
        'nominal_DH_alpha_rad':ALPHA,'manual_joint_uv':uv.tolist(),'observed_skin_depth_plus_radius_m':[float(v) if np.isfinite(v) else None for v in z],
        'joint_projected_uv':repro.tolist(),'annotated_frame_ids':frame_ids.tolist(),
        'fit_joint_reprojection_error_px':np.linalg.norm(repro[frame_ids]-uv,axis=1).tolist(),
        'fit_joint_depth_error_m':(centers[frame_ids[valid_depth],2]-z[valid_depth]).tolist(),'kabsch_initial_rms_m':float(rms),
        'native_TCP_from_nominal_flange':tcp.tolist(),
        'observed_history_TCP_validation_position_error_m':[float(v[0]) for v in history_errors],
        'observed_history_TCP_validation_rotation_error_rad':[float(v[1]) for v in history_errors],
        'DH_source':'https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/',
        'calibration_scope':'Approximate video registration from four visible t0 joints and depth; no factory or camera extrinsic calibration; not a hardware execution plan.'})
    print('FK centers',frames[:,:3,3]);print('camera joints uv',repro); print('depth',z)
    print('fit reprojection errors px',np.linalg.norm(repro[frame_ids]-uv,axis=1));print('native flange TCP',tcp)

if __name__=='__main__':main()
