import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from berkeley_temporal_diagnostics import scale_displacement, oracle_alpha


class TemporalDiagnosticsTests(unittest.TestCase):
    def test_distinct_point_origins_survive_scale(self):
        p0=np.array([[1.,2.,3.],[4.,8.,6.]])
        pred=p0[:,None]+np.array([[[.3,0,0],[.6,0,0]]])
        scaled=scale_displacement(pred,p0,1/3)
        np.testing.assert_allclose(scaled-p0[:,None],(pred-p0[:,None])/3)
        np.testing.assert_allclose(scaled[1]-scaled[0],np.repeat((p0[1]-p0[0])[None],2,0))
        np.testing.assert_array_equal(scale_displacement(pred,p0,0),np.repeat(p0[:,None],2,1))

    def test_oracle_masks_missing_depth_and_recovers_known_scale(self):
        p0=np.array([[1.,2.,3.],[4.,8.,6.]])
        pred=p0[:,None]+np.array([[[.3,.1,.2],[.6,.2,.4]]])
        truth=scale_displacement(pred,p0,.4)
        truth[1,1]=np.nan
        self.assertAlmostEqual(oracle_alpha(pred,truth,p0,np.array([[True,True],[True,False]])),.4)

    def test_zero_motion_cannot_identify_oracle_scale(self):
        p0=np.zeros((2,3)); pred=np.zeros((2,4,3))
        with self.assertRaises(ValueError):
            oracle_alpha(pred,pred,p0,np.ones((2,4),bool))


if __name__=="__main__":
    unittest.main()
