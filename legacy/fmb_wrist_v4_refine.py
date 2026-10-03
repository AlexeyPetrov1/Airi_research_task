import numpy as np
from .fmb_wrist_math import transform
from .fmb_wrist_v4_prepare import forecast_tcp


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
