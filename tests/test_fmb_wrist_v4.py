"""Check body-frame rotation extrapolation and moving-camera attachment geometry."""
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fmb_wrist_v4_prepare import forecast_tcp
from fmb_wrist_math import transform,project
from fmb_wrist_v4_refine import centroid_forecast

def poses():
    times=np.arange(-5,1)/10
    first=Rotation.from_euler('xyz',[.2,-.4,.8]).as_matrix()
    omega=np.array([.2,-.1,.3]);velocity=np.array([.04,.02,-.03])
    out=np.repeat(np.eye(4)[None],6,axis=0)
    out[:,:3,:3]=first@Rotation.from_rotvec(times[:,None]*omega).as_matrix()
    out[:,:3,3]=[.3,.1,.4]+times[:,None]*velocity
    return out,first,omega,velocity

def test_constant_body_twist_rotated_axes():
    observed,r,w,v=poses();times=np.array([.1,.5,2.0])
    forecast=forecast_tcp(observed,6,None,times)
    np.testing.assert_allclose(forecast[:,:3,3],[.3,.1,.4]+times[:,None]*v,atol=1e-12)
    np.testing.assert_allclose(forecast[:,:3,:3],r@Rotation.from_rotvec(times[:,None]*w).as_matrix(),atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(forecast[:,:3,:3]),1,atol=1e-12)

def test_damping_stops_pose_with_valid_rotations():
    observed,r,w,v=poses();forecast=forecast_tcp(observed,6,.8,np.array([0.,100.]))
    np.testing.assert_allclose(forecast[0],observed[-1],atol=1e-12)
    np.testing.assert_allclose(forecast[-1,:3,3]-observed[-1,:3,3],.8*v,atol=1e-12)
    np.testing.assert_allclose(forecast[-1,:3,:3],r@Rotation.from_rotvec(.8*w).as_matrix(),atol=1e-12)

def test_rigid_attachment_and_projection_in_future_camera():
    observed,_,_,_=poses();times=np.array([.1,.5,2.0]);future=forecast_tcp(observed,6,.8,times)
    x=np.eye(4);x[:3,:3]=Rotation.from_euler('xyz',[.7,-.2,.3]).as_matrix();x[:3,3]=[.05,-.08,.02]
    points=np.array([[.01,.02,.2],[-.03,.01,.25],[.02,-.01,.3]])
    camera0=observed[-1]@x;k=np.array([[180.,0,128],[0,210,128],[0,0,1.]])
    for pose in future:
        anchor=transform(np.linalg.inv(camera0)@(pose@x),points)
        returned=transform(np.linalg.inv(pose@x)@camera0,anchor)
        np.testing.assert_allclose(returned,points,atol=1e-12)
        np.testing.assert_allclose(project(returned,k),project(points,k),atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(anchor[:,None]-anchor[None,:],axis=-1),np.linalg.norm(points[:,None]-points[None,:],axis=-1),atol=1e-12)

def test_base_frame_change_does_not_change_anchor_forecast():
    observed,_,_,_=poses();times=np.array([.1,.5,2.])
    base=np.eye(4);base[:3,:3]=Rotation.from_euler('xyz',[-.5,.6,1.]).as_matrix();base[:3,3]=[1,2,3]
    original=forecast_tcp(observed,6,.8,times)
    moved=forecast_tcp(base@observed,6,.8,times)
    np.testing.assert_allclose(moved,base@original,atol=1e-12)

def test_centroid_motion_ignores_ineligible_nan_points_and_preserves_rigidity():
    observed,_,_,_=poses();times=np.array([0.,.5,2.]);x=np.eye(4)
    points=np.array([[.01,.02,.2],[-.03,.01,.25],[np.nan,np.nan,np.nan]])
    result=centroid_forecast(observed,x,points,6,.8,times)
    assert np.isfinite(result[:2]).all() and np.isnan(result[2]).all()
    np.testing.assert_allclose(result[:2,0],points[:2],atol=1e-12)
    np.testing.assert_allclose(result[1]-result[0],np.repeat((points[1]-points[0])[None],3,axis=0),atol=1e-12)
    # The target-centroid velocity is measured from transformed attached points,
    # including rotation about the TCP; it is not just TCP translation velocity.
    relative=result[0,-1]-result[0,0]
    assert np.linalg.norm(relative)>0 and np.isfinite(relative).all()
