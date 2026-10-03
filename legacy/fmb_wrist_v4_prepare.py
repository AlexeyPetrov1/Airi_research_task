import numpy as np
from scipy.spatial.transform import Rotation


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
