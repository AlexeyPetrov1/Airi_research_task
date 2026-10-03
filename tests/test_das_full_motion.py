"""Regressions for cadence preservation and competing independent forecasts."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from das_full_motion_prepare import sample_motion
from das_full_motion_diagnose import fit_motion


class FullMotionTests(unittest.TestCase):
    def test_stretch_preserves_pose_at_matching_phase_and_delays_hold(self):
        times=np.arange(1,31)/15
        r=Rotation.from_rotvec(np.c_[times*.1,times*.2,times*.3]).as_matrix()
        t=np.c_[times**2,np.sin(times),times]
        variants=[sample_motion(r,t,timing) for timing in ['physical_2s','stretched_4s','stretched_6s']]
        for key in [0,1]:
            np.testing.assert_allclose(variants[0][key][:17],variants[1][key][:33:2],atol=1e-12)
            np.testing.assert_allclose(variants[0][key][:17],variants[2][key][::3],atol=1e-12)
        self.assertTrue(np.all(np.diff(variants[2][2])>0))
        self.assertTrue(np.all(variants[0][2][16:]==2))
        self.assertTrue(np.all(variants[1][2][32:]==2))

    def test_group00_is_independent_of_other_groups(self):
        p0=np.random.default_rng(4).normal(size=(24,3))*.05
        r=Rotation.from_euler('z',10,degrees=True).as_matrix()
        true=p0@r.T+np.array([.1,-.2,.03])
        pred=np.repeat(true[:,None],30,axis=1)
        pred[8:]+=np.array([1.,.5,-.2])
        rr,tt,residual,_=fit_motion(p0,pred,'group00')
        np.testing.assert_allclose(rr,np.repeat(r[None],30,axis=0),atol=1e-12)
        np.testing.assert_allclose(tt,np.tile([.1,-.2,.03],(30,1)),atol=1e-12)
        self.assertLess(residual.max(),1e-12)

    def test_robust_fit_rejects_small_outlier_subset(self):
        p0=np.random.default_rng(6).normal(size=(24,3))*.05
        target=p0+np.array([.12,-.08,.03])
        target[-3:]+=np.array([.4,-.5,.3])
        pred=np.repeat(target[:,None],30,axis=1)
        r,t,_,_=fit_motion(p0,pred,'robust24')
        ar,at,_,_=fit_motion(p0,pred,'all24')
        desired=p0[:-3]+np.array([.12,-.08,.03])
        robust=np.linalg.norm(p0[:-3]@r[0].T+t[0]-desired,axis=1).mean()
        plain=np.linalg.norm(p0[:-3]@ar[0].T+at[0]-desired,axis=1).mean()
        self.assertLess(robust,plain*.3)
        np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)


if __name__=='__main__':unittest.main()
