"""Describe future-latent proximity to the synthetic photographic condition."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from das_full_motion_diagnose import OUT
from das_prepare_control import write_json, sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args()
    out=OUT/a.name;config=json.loads((out/'config.json').read_text())
    assert config['photographic_guide_strength']>0
    prep=OUT/Path(config['preparation_root']).name
    actual=torch.load(out/'generated_latents.pt',map_location='cpu',weights_only=False)['latents'].float()
    guide=torch.load(out/'photographic_guide_latents.pt',map_location='cpu',weights_only=False)['latents'].float()
    mask=torch.from_numpy(np.load(prep/'prior_masks.npz')['weights'])[None,None].float()
    mask=torch.cat([mask[:,:,:1],mask[:,:,1:].reshape(1,1,12,4,480,720).mean(3)],dim=2)
    mask=torch.nn.functional.interpolate(mask,size=(13,60,90),mode='trilinear',align_corners=False)
    def residual(region):
        region=region.expand_as(actual)
        return float(((actual-guide)[region].square().mean()/guide[region].square().mean()).sqrt())
    future=torch.ones_like(mask,dtype=torch.bool);future[:,:,0]=False
    report={'generated_latent_sha256':sha256(out/'generated_latents.pt'),
        'guide_latent_sha256':sha256(out/'photographic_guide_latents.pt'),
        'initial_anchor_excluded':True,'future_all_latent_relative_RMS':residual(future),
        'future_moving_region_relative_RMS':residual(future&(mask>.9)),
        'future_background_region_relative_RMS':residual(future&(mask<.1)),
        'moving_latent_fraction':float((future&(mask>.9)).sum()/future.sum()),
        'interpretation':'RMS(generated-guide)/RMS(guide), latent frames 1..12 only. Describes condition proximity, not realism, trajectory accuracy, or semantic identity.',
        'not_directly_comparable_to_old_F':'Old F used an endpoint future-RGB guide and a different mask/temporal scope.'}
    write_json(out/'prior_latent_audit.json',report);print(report)


if __name__=='__main__':main()
