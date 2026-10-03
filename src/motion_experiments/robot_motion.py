import numpy as np

from scipy.spatial.transform import Rotation

from .geometry import transform



def forecast_tcp(poses,window,tau,times):
    w=min(window,len(poses));p=poses[-w:]
    dt=np.arange(-w+1,1)/10
    # Least-squares velocity with the last observed pose fixed as intercept.
    v=(dt[:,None]*(p[:,:3,3]-p[-1,:3,3])).sum(0)/(dt@dt)
    relative=Rotation.from_matrix(p[-1,:3,:3].T@p[:,:3,:3]).as_rotvec()
    omega=(dt[:,None]*relative).sum(0)/(dt@dt)
    factor=times if tau is None else tau*(-np.expm1(-times/tau))
    out=np.repeat(p[-1][None],len(times),axis=0)
    out[:,:3,3]+=factor[:,None]*v
    out[:,:3,:3]=p[-1,:3,:3]@Rotation.from_rotvec(factor[:,None]*omega).as_matrix()
    return out

def centroid_forecast(tcp,x,template,window,tau,times):
    w=min(window,len(tcp));poses=tcp[-w:]@x
    dt=np.arange(-w+1,1)/10
    centers=np.stack([np.nanmean(transform(p,template),axis=0) for p in poses])
    velocity=(dt[:,None]*(centers-centers[-1])).sum(0)/(dt@dt)
    factor=times if tau is None else tau*(-np.expm1(-times/tau))
    initial=transform(poses[-1],template)
    world=initial[:,None]+factor[None,:,None]*velocity[None,None]
    return transform(np.linalg.inv(poses[-1]),world)

def predict(r,anchor,window,tau,times,family):
    tcp=r['tcp'][:anchor+1];x=r['X'];template=r['camera_points'][anchor]
    if family=='rigid_centroid_translation':return centroid_forecast(tcp,x,template,window,tau,times)
    p=forecast_tcp(tcp,window,tau,times)
    return np.stack([transform(np.linalg.inv(tcp[-1]@x)@(g@x),template) for g in p],axis=1)
