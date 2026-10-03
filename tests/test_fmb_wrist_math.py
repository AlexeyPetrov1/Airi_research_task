import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fmb_wrist_math import transform,backproject,project,to_anchor,from_anchor,hand_eye

def test_moving_camera_anchor_and_future_projection():
    k=np.array([[200,0,128],[0,210,127],[0,0,1.]])
    c=np.repeat(np.eye(4)[None],3,axis=0);c[0,:3,3]=[-.1,0,0];c[1,:3,3]=[0,.1,0];c[2,:3,3]=[.2,0,0]
    world=np.array([[0,0,1.],[.1,.2,1.5]])
    local=np.stack([transform(np.linalg.inv(p),world) for p in c])
    anchored=to_anchor(local,c)
    np.testing.assert_allclose(anchored,np.broadcast_to(local[-1],anchored.shape),atol=1e-12)
    future=np.eye(4);future[:3,3]=[.3,-.1,0]
    got=from_anchor(anchored[-1],future,c[-1]);expected=transform(np.linalg.inv(future),world)
    np.testing.assert_allclose(got,expected,atol=1e-12)
    assert np.linalg.norm(project(anchored[-1],k)-project(got,k))>10

def test_metric_hand_eye_recovers_known_transform_without_future():
    rng=np.random.default_rng(17);g=np.repeat(np.eye(4)[None],25,axis=0)
    g[:,:3,:3]=Rotation.from_rotvec(rng.normal(size=(25,3))*.4).as_matrix();g[:,:3,3]=rng.normal(size=(25,3))*.15
    x=np.eye(4);x[:3,:3]=Rotation.from_rotvec([.4,-.7,.2]).as_matrix();x[:3,3]=[.04,.03,.11]
    y=np.eye(4);y[:3,:3]=Rotation.from_rotvec([.2,.1,-.5]).as_matrix();y[:3,3]=[.2,.1,.4]
    camera=np.linalg.inv(y)[None]@(g@x)
    actual,_,estimated,info=hand_eye(g,camera,np.arange(18),np.arange(18,25))
    np.testing.assert_allclose(actual,x,atol=1e-7);np.testing.assert_allclose(estimated,camera,atol=1e-7)
    assert info['heldout']['p90_translation_mm']<1e-4 and info['future_used'] is False

def test_backproject_roundtrip_and_outside_retained():
    k=np.array([[152.162,0,125.7932],[0,202.88267,128.004],[0,0,1]])
    uv=np.array([[130,120],[-100,100],[900,300.]])
    np.testing.assert_allclose(project(backproject(uv,np.array([.2,.3,.4]),k),k),uv,atol=1e-10)

def test_15hz_forecast_is_evaluated_at_physical_10hz_times():
    from fmb_wrist_evaluate import align_prediction
    origin=np.tile([.04,-.1,.5],(24,1));velocity=np.array([.2,-.3,.07])
    forecast=origin[:,None]+np.arange(1,31)[None,:,None]/15*velocity
    sampled=align_prediction(forecast,origin)
    expected=origin[:,None]+np.arange(1,21)[None,:,None]/10*velocity
    np.testing.assert_allclose(sampled,expected,atol=1e-12)
    assert not np.allclose(sampled[:,0],forecast[:,0])
