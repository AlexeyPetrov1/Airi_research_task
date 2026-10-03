import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from das_vacancy_prior import make_latent_masks,VacancyPrior


class VacancyProtectionTests(unittest.TestCase):
    def test_start_frame_preserved_and_disoccluded_site_is_constrained(self):
        alpha=np.zeros((480,640),np.float32);alpha[240:320,120:200]=1
        moving=np.zeros((49,480,640),bool)
        masks=make_latent_masks(alpha,moving)
        self.assertFalse(masks[0].any())
        self.assertGreater(masks[1].sum(),0)
        np.testing.assert_array_equal(masks[1],masks[12])

    def test_foreground_at_any_decoded_frame_protects_entire_latent_block(self):
        alpha=np.ones((480,640),np.float32);moving=np.zeros((49,480,640),bool)
        moving[3,240:320,120:200]=True
        masks=make_latent_masks(alpha,moving)
        self.assertEqual(float(masks[1,31:39,18:27].sum()),0)
        self.assertGreater(float(masks[2,31:39,18:27].sum()),0)

    def test_flow_noise_level_final_background_and_protected_cells(self):
        import torch
        from types import SimpleNamespace
        masks=np.ones((3,2,2),np.float32);masks[0]=0;masks[1,0,0]=0
        original=torch.full((1,1,3,2,2),7.)
        images=[torch.full((1,1,1,2,2),9.),torch.full((1,1,1,2,2),2.)]
        prior=VacancyPrior(masks,1.,9043,images)
        pipeline=SimpleNamespace(scheduler=SimpleNamespace(sigmas=torch.tensor([1.,.5,0.])))
        state=torch.random.get_rng_state().clone()
        middle=prior(pipeline,0,None,{'latents':original.clone()})['latents']
        torch.testing.assert_close(middle[0,0,2],1.+.5*prior.noise[0,0,2])
        final=prior(pipeline,1,None,{'latents':middle})['latents']
        torch.testing.assert_close(final[0,0,0],original[0,0,0])
        self.assertEqual(float(final[0,0,1,0,0]),7.)
        torch.testing.assert_close(final[0,0,2],torch.full((2,2),2.))
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_context_expands_background_without_overwriting_foreground(self):
        alpha=np.zeros((480,640),np.float32);alpha[240:320,120:200]=1
        moving=np.zeros((49,480,640),bool);moving[6,260:280,205:225]=True
        narrow=make_latent_masks(alpha,moving);wide=make_latent_masks(alpha,moving,24)
        self.assertGreater(float(wide[3].sum()),float(narrow[3].sum()))
        self.assertFalse(wide[0].any())
        self.assertEqual(float(wide[2,33:35,30:31].sum()),0)


if __name__=='__main__':unittest.main()
