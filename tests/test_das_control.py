"""Geometric and temporal checks for causal dense trajectory conditioning."""
import sys
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from das_prepare_control import kabsch, interpolate_motion, splat, object_depth


class DenseControlTests(unittest.TestCase):
    def test_known_rigid_motion(self):
        p = np.random.default_rng(2).normal(size=(24,3))
        r = Rotation.from_euler('xyz',[25,70,-8],degrees=True).as_matrix()
        t = np.array([.3,-.8,1.2])
        actual_r,actual_t,residual,_=kabsch(p,p@r.T+t)
        np.testing.assert_allclose(actual_r,r,atol=1e-12)
        np.testing.assert_allclose(actual_t,t,atol=1e-12)
        self.assertLess(residual,1e-12)

    def test_reflection_is_not_allowed(self):
        p=np.random.default_rng(3).normal(size=(24,3))
        r,_,residual,_=kabsch(p,p*np.array([-1,1,1]))
        self.assertAlmostEqual(np.linalg.det(r),1.)
        self.assertGreater(residual,.1)

    def test_physical_time_and_hold(self):
        times=np.arange(1,31)/15
        rotations=Rotation.from_rotvec(np.c_[np.zeros(30),np.zeros(30),times]).as_matrix()
        translations=np.c_[times,np.zeros((30,2))]
        r,t=interpolate_motion(rotations,translations)
        np.testing.assert_allclose(r[0],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(t[:17,0],np.arange(17)/8)
        np.testing.assert_allclose(r[17:],np.repeat(r[16:17],32,axis=0))
        np.testing.assert_allclose(t[17:],np.repeat(t[16:17],32,axis=0))
        np.testing.assert_allclose(np.linalg.det(r),1.,atol=1e-12)

    def test_nearest_depth_wins_and_behind_camera_is_rejected(self):
        xyz=np.array([[2.,2.,1.],[4.,4.,2.],[-2.,-2.,-1.],[100.,100.,1.]])
        color=np.array([[255,0,0],[0,255,0],[0,0,255],[255,255,0]],np.uint8)
        image=np.zeros((5,5,3),np.uint8);depth=np.full((5,5),np.inf)
        splat(xyz,color,np.eye(3),image,depth,radius=0)
        np.testing.assert_array_equal(image[2,2],[255,0,0])
        self.assertEqual(np.count_nonzero(image),1)
        self.assertEqual(depth[2,2],1.)

    def test_invalid_depth_patch_is_rejected(self):
        depth=np.ones((10,10));mask=np.zeros((10,10),bool);mask[5,5]=True
        depth[3:8,3:8]=65.535
        uv,z=object_depth(depth,mask)
        self.assertEqual(len(uv),0)
        depth[3:8,3:8]=.6
        uv,z=object_depth(depth,mask)
        np.testing.assert_allclose(z,[.6])


if __name__=='__main__': unittest.main()
