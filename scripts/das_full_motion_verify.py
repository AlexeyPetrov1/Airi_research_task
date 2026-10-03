"""Validate immutable inputs, timing, actual diffusion, and matched conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from das_full_motion_diagnose import ROOT, SCENE, OUT
from das_prepare_control import sha256, write_json
from das_robot_evaluate import frames


def main():
    p=argparse.ArgumentParser();p.add_argument('--chosen',required=True);a=p.parse_args()
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise AssertionError(name)
    for path,digest in json.loads((OUT/'preserved_baselines.json').read_text()).items():
        check(f'preserved {path}',sha256(path)==digest)
    cpu=json.loads((OUT/'cpu_verification.json').read_text())
    check('all CPU path checks',all(cpu['checks'].values()))
    configs={}
    for video in sorted(OUT.glob('*/generated_seed42.mp4')):
        out=video.parent;name=out.name
        config=json.loads((out/'config.json').read_text());configs[name]=config
        resource=json.loads((out/'resource_usage.json').read_text())
        check(f'{name}: successful actual generation',resource['success'] and resource['output_sha256']==sha256(video))
        check(f'{name}: 49 RGB frames',frames(video).shape==(49,480,720,3))
        check(f'{name}: causal inputs',config['future_used'] is False and config['future_pixels_or_pose_condition_passed_to_model'] is False)
        check(f'{name}: no RGB replacement',config['postprocessed_RGB'] is False)
        prep=OUT/Path(config['preparation_root']).name
        check(f'{name}: immutable preparation',sha256(prep/'preparation.json')==config['preparation_sha256'])
        freeze=json.loads((prep/'input_freeze.json').read_text())['sha256']
        check(f'{name}: immutable source files',all(sha256(SCENE/path)==h for path,h in freeze.items()))
        check(f'{name}: reviewed control',json.loads((prep/'preparation_visual_review.json').read_text(encoding='utf-8-sig'))['accepted'])
        import torch
        latents=torch.load(out/'generated_latents.pt',map_location='cpu',weights_only=False)['latents']
        check(f'{name}: real finite latent output',latents.shape==(1,16,13,60,90) and torch.isfinite(latents).all().item())
        if config['photographic_guide_strength']>0:
            if config.get('cached_guide_from'):
                cache=OUT/config['cached_guide_from']/'photographic_guide_latents.pt'
                check(f'{name}: frozen latent cache bytes',sha256(cache)==config['cached_guide_latents_sha256'])
                cached=torch.load(cache,map_location='cpu',weights_only=False)
                consumed=torch.load(out/'photographic_guide_latents.pt',map_location='cpu',weights_only=False)
                check(f'{name}: exactly identical cached guide',torch.equal(cached['latents'],consumed['latents']) and cached['guide_sha256']==consumed['guide_sha256']==config['guide_sha256'])
            receipt=json.loads((out/'photographic_prior_receipt.json').read_text())
            check(f'{name}: partial prior every step',receipt['applied_steps']==config['num_inference_steps'] and 0<receipt['strength']<1)
            check(f'{name}: correct noise schedule',receipt['sigma_after_each_step'][-1]==0 and np.all(np.diff(receipt['sigma_after_each_step'])<=0))
            check(f'{name}: observed-only prior',receipt['future_used'] is False)
            guide=np.load(prep/'guide_720x480.npz')['frames']
            check(f'{name}: exact first guide RGB',np.array_equal(guide[0],np.array(Image.open(prep/'image_t0_720x480.png'))))
            check(f'{name}: immutable guide',sha256(prep/'guide_720x480.npz')==config['guide_sha256'])
            mask=np.load(prep/'prior_masks.npz')['weights']
            check(f'{name}: spatial mask bounds',mask.shape==(49,480,720) and (mask>=0).all() and (mask<=1).all())
        else:
            check(f'{name}: guide disabled',config['guide_sha256'] is None and not (out/'photographic_prior_receipt.json').exists())
        metrics=json.loads((out/'motion_metrics.json').read_text())
        check(f'{name}: independent measurements match video',metrics['video_sha256']==sha256(video))
    check('selected output exists',a.chosen in configs)
    if 'H3_group00_6s_prior025' in configs:
        h2=configs['H2_group00_6s_no_prior'];h3=configs['H3_group00_6s_prior025']
        keys=['model','checkpoint_revision','das_commit','dtype','seed','num_inference_steps','num_frames',
            'resolution','fps','offload','guidance_scale','teacache','persistent_reference','prompt',
            'trajectory_method','timing','control_sha256','trajectory_control']
        check('H2/H3 matched except appearance prior',all(h2[key]==h3[key] for key in keys))
    if 'H4_no_trajectory_control' in configs:
        h2=configs['H2_group00_6s_no_prior'];h4=configs['H4_no_trajectory_control']
        keys=['model','checkpoint_revision','das_commit','dtype','seed','num_inference_steps','num_frames',
            'resolution','fps','offload','guidance_scale','teacache','persistent_reference','prompt',
            'trajectory_method','timing','photographic_guide_strength']
        check('H2/H4 matched except trajectory control',all(h2[key]==h4[key] for key in keys) and h2['trajectory_control'] and not h4['trajectory_control'])
    review=json.loads((OUT/'visual_review.json').read_text())
    check('selected all-frame semantic review',review['chosen_variant']==a.chosen and review['variants'][a.chosen]['frames_reviewed']==49)
    check('selected full arc accepted',review['variants'][a.chosen]['full_predicted_arc_visible'])
    report={'passed':len(checks),'all_passed':all(checks.values()),'chosen_variant':a.chosen,'checks':checks}
    write_json(OUT/'verification.json',report)
    manifest=[]
    for path in sorted(OUT.rglob('*')):
        if path.is_file() and path.name!='artifact_manifest.json':
            manifest.append({'path':str(path.relative_to(OUT)).replace('\\','/'),'bytes':path.stat().st_size,'sha256':sha256(path)})
    write_json(OUT/'artifact_manifest.json',{'files':manifest,'count':len(manifest),'total_bytes':sum(entry['bytes'] for entry in manifest)})
    print({'checks_passed':len(checks),'artifact_files':len(manifest)})


if __name__=='__main__':main()
