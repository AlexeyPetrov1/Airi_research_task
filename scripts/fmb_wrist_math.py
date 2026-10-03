"""Explicit c2w moving-camera geometry and observed-only AX=XB calibration."""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def tcp_matrices(tcp):
    tcp=np.asarray(tcp,float)
    if tcp.ndim!=2 or tcp.shape[1]!=7:raise ValueError('Expected xyz + xyzw TCP poses')
    out=np.broadcast_to(np.eye(4),(len(tcp),4,4)).copy()
    out[:,:3,:3]=Rotation.from_quat(tcp[:,3:]).as_matrix();out[:,:3,3]=tcp[:,:3]
    return out

def transform(T,xyz):
    return np.asarray(xyz)@T[:3,:3].T+T[:3,3]

def backproject(uv,z,K):
    uv=np.asarray(uv);return np.stack([(uv[...,0]-K[0,2])/K[0,0]*z,(uv[...,1]-K[1,2])/K[1,1]*z,z],axis=-1)

def project(xyz,K):
    xyz=np.asarray(xyz);uv=xyz[...,:2]/xyz[...,2,None]
    return uv*np.array([K[0,0],K[1,1]])+[K[0,2],K[1,2]]

def to_anchor(xyz_camera,c2w):
    """T time,N,3 -> coordinates in the last observed camera."""
    anchor=np.linalg.inv(c2w[-1]);return np.stack([transform(anchor@T,xyz) for T,xyz in zip(c2w,xyz_camera)])

def from_anchor(xyz_anchor,c2w_t,c2w_anchor):
    return transform(np.linalg.inv(c2w_t)@c2w_anchor,xyz_anchor)

def sample_z(depth,uv,size=5):
    uv=np.asarray(uv);z=np.full(len(uv),np.nan);spread=np.full(len(uv),np.inf);coverage=np.zeros(len(uv));half=size//2
    for p,(u,v) in enumerate(uv):
        if not np.isfinite([u,v]).all() or not(half<=u<depth.shape[1]-half and half<=v<depth.shape[0]-half):continue
        x,y=np.rint([u,v]).astype(int);patch=depth[y-half:y+half+1,x-half:x+half+1]
        values=patch[np.isfinite(patch)&(patch>0)&(patch<10)]
        coverage[p]=len(values)/size**2
        if len(values)>=size**2//2+1:z[p]=np.median(values);spread[p]=np.ptp(np.percentile(values,[10,90]))
    return z,spread,coverage

def lift(depths,uv,visible,K,c2w):
    local=[];valid=[];spread=[];coverage=[]
    if np.asarray(K).ndim==2:K=np.repeat(K[None],len(depths),axis=0)
    for depth,points,vis,k in zip(depths,uv,visible,K):
        z,s,c=sample_z(depth,points);good=np.isfinite(z)&vis
        local.append(backproject(points,z,k));valid.append(good);spread.append(s);coverage.append(c)
    return to_anchor(np.stack(local),c2w),np.stack(valid),np.stack(spread),np.stack(coverage)

def mean_transform(transforms):
    out=np.eye(4);out[:3,:3]=Rotation.from_matrix(np.asarray(transforms)[:,:3,:3]).mean().as_matrix()
    out[:3,3]=np.median(np.asarray(transforms)[:,:3,3],axis=0);return out

def hand_eye(robot,camera,train_ids=None,test_ids=None):
    """G_i X = Y C_i, with G and C both camera/EE-to-world transforms."""
    n=len(robot);train_ids=np.asarray(train_ids if train_ids is not None else np.arange(n),int)
    test_ids=np.asarray(test_ids if test_ids is not None else [],int)
    pairs=[(i,j) for offset,i in enumerate(train_ids) for j in train_ids[offset+1:] if 3<=j-i<=25]
    A=np.array([np.linalg.inv(robot[i])@robot[j] for i,j in pairs]);B=np.array([np.linalg.inv(camera[i])@camera[j] for i,j in pairs])
    if len(A)<6:raise ValueError('Not enough calibration pairs')
    def unpack(v):
        x=np.eye(4);x[:3,:3]=Rotation.from_rotvec(v[:3]).as_matrix();x[:3,3]=v[3:];return x
    def residual(v):
        x=unpack(v);errors=np.linalg.inv(x@B)@(A@x)
        return np.concatenate([Rotation.from_matrix(errors[:,:3,:3]).as_rotvec().ravel()/.05,errors[:,:3,3].ravel()/.02])
    # Log(R_A)=R_X Log(R_B) supplies a portable closed-form initialization.
    # Some installed OpenCV builds omit calibrateHandEye; the scipy solve remains complete.
    av=Rotation.from_matrix(A[:,:3,:3]).as_rotvec();bv=Rotation.from_matrix(B[:,:3,:3]).as_rotvec()
    rr,_=Rotation.align_vectors(av,bv)
    translation_system=np.concatenate([a[:3,:3]-np.eye(3) for a in A])
    translation_rhs=np.concatenate([rr.apply(b[:3,3])-a[:3,3] for a,b in zip(A,B)])
    tt=np.linalg.lstsq(translation_system,translation_rhs,rcond=None)[0]
    starts=[np.r_[rr.as_rotvec(),tt]];methods=['scipy_log_rotation_and_linear_translation']
    ct=np.linalg.inv(camera[train_ids])
    for name,method in [('park',cv2.CALIB_HAND_EYE_PARK),('horaud',cv2.CALIB_HAND_EYE_HORAUD),('tsai',cv2.CALIB_HAND_EYE_TSAI)]:
        try:
            r,t=cv2.calibrateHandEye(list(robot[train_ids,:3,:3]),list(robot[train_ids,:3,3]),list(ct[:,:3,:3]),list(ct[:,:3,3]),method=method)
            if np.isfinite(r).all() and np.isfinite(t).all() and np.linalg.det(r)>.9:
                starts.append(np.r_[Rotation.from_matrix(r).as_rotvec(),t.ravel()]);methods.append(name)
        except (cv2.error,ValueError,AttributeError):pass
    if not starts:starts=[np.zeros(6)];methods=['identity_init']
    fits=[least_squares(residual,start,loss='soft_l1',max_nfev=150) for start in starts]
    best=min(range(len(fits)),key=lambda k:np.mean(residual(fits[k].x)**2));fit=fits[best];x=unpack(fit.x)
    y=mean_transform([robot[i]@x@np.linalg.inv(camera[i]) for i in train_ids])
    estimated=np.linalg.inv(y)[None]@(robot@x)
    error=np.linalg.inv(camera)@estimated
    translation=np.linalg.norm(error[:,:3,3],axis=1);rotation=np.degrees(Rotation.from_matrix(error[:,:3,:3]).magnitude())
    singular=np.linalg.svd(np.concatenate([a[:3,:3]-np.eye(3) for a in A]),compute_uv=False)
    def stats(ids):
        return {'frames':ids.tolist(),'median_translation_mm':float(np.median(translation[ids])*1000),
                'p90_translation_mm':float(np.percentile(translation[ids],90)*1000),
                'median_rotation_deg':float(np.median(rotation[ids])),'p90_rotation_deg':float(np.percentile(rotation[ids],90))} if len(ids) else None
    info={'X_TCP_from_camera':x.tolist(),'Y_base_from_vipe_world':y.tolist(),'calibration_pair_count':len(A),
          'translation_system_singular_values':singular.tolist(),'translation_system_condition':float(singular[0]/max(singular[-1],1e-12)),
          'estimated_translation_norm_m':float(np.linalg.norm(x[:3,3])),'train':stats(train_ids),'heldout':stats(test_ids),
          'translation_error_per_frame_mm':(translation*1000).tolist(),'rotation_error_per_frame_deg':rotation.tolist(),
          'optimization_initialization':methods[best],'optimizer_success':bool(fit.success),
          'finite_physical_plausibility_gate':bool(np.isfinite(x).all() and np.linalg.norm(x[:3,3])<=.5),
          'future_used':False,'source_dependency':'Estimated hand-eye X uses observed ViPE poses; TCP camera path subsequently uses robot only'}
    return x,y,estimated,info

def static_pairs(rgb):
    """Observed LK correspondences on blue board/gray background, excluding camera-fixed gripper."""
    rows=[];n=len(rgb)
    grays=[cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) for x in rgb]
    for i in list(range(0,n-5,6))+[n-4,n-3]:
        for gap in [1,5]:
            j=i+gap
            if j>=n:continue
            r,g,b=rgb[i].transpose(2,0,1).astype(float)
            board=(b>r*1.25)&(b>g*1.12)&(b>65)
            # Conservative static background: low-chroma walls lower-left and blue board.
            yy,xx=np.indices(board.shape)
            gray=(np.max(rgb[i],-1)-np.min(rgb[i],-1)<30)&(xx<125)&(yy>65)
            allowed=cv2.erode((board|gray).astype(np.uint8),np.ones((5,5),np.uint8))*255
            p=cv2.goodFeaturesToTrack(grays[i],maxCorners=300,qualityLevel=.008,minDistance=4,mask=allowed)
            if p is None:continue
            q,ok,_=cv2.calcOpticalFlowPyrLK(grays[i],grays[j],p,None,winSize=(21,21),maxLevel=3)
            back,okback,_=cv2.calcOpticalFlowPyrLK(grays[j],grays[i],q,None,winSize=(21,21),maxLevel=3)
            good=ok.ravel().astype(bool)&okback.ravel().astype(bool)&(np.linalg.norm(back-p,axis=-1).ravel()<.7)
            pp,qq=p[:,0][good],q[:,0][good]
            if len(pp)<8:continue
            # Static planar board/walls can have different planes; only reject extreme image-flow outliers.
            H,inliers=cv2.findHomography(pp,qq,cv2.RANSAC,3.)
            if H is None:continue
            good=inliers.ravel().astype(bool);rows.append({'i':i,'j':j,'uv_i':pp[good].tolist(),'uv_j':qq[good].tolist()})
    return rows

def static_reprojection(depths,K,c2w,pairs):
    if np.asarray(K).ndim==2:K=np.repeat(K[None],len(depths),axis=0)
    errors=[];rows=[]
    for row in pairs:
        i,j=row['i'],row['j'];p,q=np.array(row['uv_i']),np.array(row['uv_j'])
        z,spread,coverage=sample_z(depths[i],p);good=np.isfinite(z)&(spread<.05)
        xyz=backproject(p[good],z[good],K[i]);other=transform(np.linalg.inv(c2w[j])@c2w[i],xyz)
        valid=other[:,2]>0;err=np.linalg.norm(project(other[valid],K[j])-q[good][valid],axis=-1)
        if len(err):errors.extend(err);rows.append({'i':i,'j':j,'count':len(err),'median_px':float(np.median(err)),'p90_px':float(np.percentile(err,90))})
    arr=np.asarray(errors)
    return {'pairs':rows,'correspondences':len(arr),'median_px':float(np.median(arr)) if len(arr) else None,
            'p90_px':float(np.percentile(arr,90)) if len(arr) else None,
            'gate':bool(len(arr)>=30 and len(rows)>=3 and np.median(arr)<=4 and np.percentile(arr,90)<=10)}
