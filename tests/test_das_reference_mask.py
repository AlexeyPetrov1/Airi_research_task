import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from das_reference_mask import reference_alpha

class ReferenceMaskTests(unittest.TestCase):
    def test_dark_blue_rim_is_removed_black_gripper_is_protected(self):
        image=np.full((480,640,3),190,np.uint8);mask=np.zeros((480,640),bool)
        mask[220:240,120:160]=True
        image[216:220,125:135]=[20,45,90]
        image[216:220,145:155]=[25,25,25]
        old,_=reference_alpha(image,mask,'gray-only')
        new,_=reference_alpha(image,mask,'non-blue-dark')
        self.assertEqual(float(old[218,130]),0.)
        self.assertGreater(float(new[218,130]),0.)
        self.assertEqual(float(new[218,150]),0.)

    def test_exact_cup_including_black_print_and_outside_support(self):
        image=np.zeros((480,640,3),np.uint8);mask=np.zeros((480,640),bool)
        mask[220:240,120:160]=True
        alpha,support=reference_alpha(image,mask,'non-blue-dark')
        self.assertTrue((alpha[mask]==1).all())
        self.assertFalse(support[0,0]);self.assertEqual(float(alpha[0,0]),0.)

if __name__=='__main__':unittest.main()
