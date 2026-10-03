import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from das_repair_background import complete_background


class BackgroundRepairTests(unittest.TestCase):
    def test_hidden_background_uses_surrounding_plane(self):
        depth=np.full((64,64),1.,np.float32);mask=np.zeros((64,64),bool)
        mask[24:40,24:40]=True;depth[mask]=.5
        fixed,repair=complete_background(depth,mask,dilation=2,min_depth=.55)
        np.testing.assert_allclose(fixed[repair],1.,atol=1e-4)
        np.testing.assert_array_equal(fixed[~repair],depth[~repair])

    def test_missing_measurements_outside_repair_are_not_claimed_measured(self):
        depth=np.full((64,64),1.,np.float32);mask=np.zeros((64,64),bool)
        mask[24:40,24:40]=True;depth[mask]=.5;depth[0:8,0:8]=np.nan
        fixed,repair=complete_background(depth,mask)
        self.assertTrue(np.isnan(fixed[0:8,0:8]).all())
        self.assertTrue(np.isfinite(fixed[repair]).all())

    def test_background_cannot_occlude_initial_foreground(self):
        depth=np.full((64,64),.6,np.float32);mask=np.zeros((64,64),bool)
        mask[24:40,24:40]=True;depth[mask]=.65
        fixed,repair=complete_background(depth,mask,min_depth=.7)
        self.assertTrue((fixed[repair]>=.7).all())
        self.assertTrue((fixed[mask]>depth[mask]).all())


if __name__=='__main__':unittest.main()
