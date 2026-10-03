"""Check geometry invariants that matter when changing forecast amplitudes."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from berkeley_arc_expansion import expand
from berkeley_evaluate import project
from berkeley_arc_phase import temporal_expand


def test_lift_preserves_constellation_depth_and_t0():
    p0 = np.array([[.12, .16, .7], [-.08, .04, .9]])
    times = np.array([0., 1., 2.])
    raw = p0[:,None] + np.array([[0.,0.,0.], [.2,-.3,.1], [.4,-.1,.2]])[None]
    base = expand(raw, p0, .36, 0., times)
    lifted = expand(raw, p0, .36, .01, times)
    np.testing.assert_array_equal(lifted[:,0], p0)
    np.testing.assert_array_equal(lifted[..., [0,2]], base[..., [0,2]])
    np.testing.assert_allclose(lifted[0]-lifted[1], base[0]-base[1], atol=1e-14)
    K = np.array([[434.5584,0,320],[0,434.5584,240],[0,0,1.]])
    uv_base,uv_lift=project(base,K),project(lifted,K)
    np.testing.assert_allclose(uv_base[...,0],uv_lift[...,0])
    assert np.all(uv_lift[:,1:,1] < uv_base[:,1:,1])
    np.testing.assert_allclose(uv_base[:,-1,1]-uv_lift[:,-1,1], .01*K[1,1]/base[:,-1,2])


def test_expansion_preserves_each_point_origin_and_direction():
    p0=np.array([[.1,.2,1.],[.3,-.2,.6]])
    raw=p0[:,None]+np.array([[[0.,0.,0.],[.2,-.1,.05]],[[0.,0.,0.],[-.1,-.2,.1]]])
    low=expand(raw,p0,1/3,0.,[0,2]);high=expand(raw,p0,.4,0.,[0,2])
    np.testing.assert_array_equal(high[:,0],p0)
    np.testing.assert_allclose(high-p0[:,None],1.2*(low-p0[:,None]),atol=1e-14)


@pytest.mark.parametrize("height,times",[(float('nan'),[0,2]),(-.01,[0,2]),(.01,[0,3]),(.01,[0]),(.01,[0,float('nan')])])
def test_invalid_time_or_lift_is_rejected(height,times):
    with pytest.raises(ValueError):
        expand(np.ones((2,2,3)),np.ones((2,3)),.36,height,times)


def test_late_schedule_preserves_early_motion_and_reaches_shared_endpoint():
    p0=np.array([[.1,.2,1.],[.3,-.2,.6]])
    times=np.array([0.,.2,1.,1.5,2.])
    raw=p0[:,None]+np.array([[[0.,0.,0.],[.2,-.1,.05],[.3,-.2,.1],[.4,-.3,.15],[.5,-.4,.2]]])
    late=temporal_expand(raw,p0,.36,.005,1.,times)
    reference=expand(raw,p0,1/3,0.,times)
    constant=expand(raw,p0,.36,.005,times)
    np.testing.assert_array_equal(late[:,:3],reference[:,:3])
    np.testing.assert_allclose(late[:,-1],constant[:,-1],rtol=0,atol=1e-14)
    np.testing.assert_allclose(late[:,3,2]-p0[:,2],(.5*(1/3+.36))*(raw[:,3,2]-p0[:,2]),atol=1e-14)
