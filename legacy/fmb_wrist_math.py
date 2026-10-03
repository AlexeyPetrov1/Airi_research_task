import numpy as np


def transform(T,xyz):
    return np.asarray(xyz)@T[:3,:3].T+T[:3,3]

def project(xyz,K):
    xyz=np.asarray(xyz);uv=xyz[...,:2]/xyz[...,2,None]
    return uv*np.array([K[0,0],K[1,1]])+[K[0,2],K[1,2]]

def from_anchor(xyz_anchor,c2w_t,c2w_anchor):
    return transform(np.linalg.inv(c2w_t)@c2w_anchor,xyz_anchor)
