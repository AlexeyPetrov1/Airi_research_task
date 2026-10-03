"""Audit frozen inputs, actual diffusion, matched conditioning, and evidence."""
from pathlib import Path
import argparse, json
import cv2
import numpy as np
from PIL import Image
from das_prepare_control import sha256, write_json
from das_robot_evaluate import frames
from das_evaluate import metrics

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup';OUT=SCENE/'das_reference_repair'
CHOSEN='guided_endpoint_background'

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');p.add_argument('--write-manifest',action='store_true');a=p.parse_args()
    if a.write_manifest:
        files={str(p.relative_to(OUT)).replace('\\','/'):{'sha256':sha256(p),'bytes':p.stat().st_size}
            for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'}
        import hashlib
        sources=sorted((ROOT/'scripts').glob('das_reference_*.py'))+[ROOT/'scripts'/name for name in
            ['das_wanfun_runtime.py','das_generate.py','das_prepare_control.py','das_robot_evaluate.py','das_evaluate.py','das_quality.py']]
        code={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in sources}
        write_json(OUT/'artifact_manifest.json',{'files':files,'source_code_sha256_LF_normalized':code,'total_bytes':sum(v['bytes'] for v in files.values())});print('Manifest written',len(files),'files');return
    if a.manifest_only:
        import hashlib
        manifest=json.loads((OUT/'artifact_manifest.json').read_text())
        assert all(sha256(OUT/name)==v['sha256'] and (OUT/name).stat().st_size==v['bytes'] for name,v in manifest['files'].items())
        actual={str(p.relative_to(OUT)).replace('\\','/') for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}
        assert actual==set(manifest['files'])
        assert all(hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest for name,digest in manifest['source_code_sha256_LF_normalized'].items())
        print('MANIFEST PASS',len(actual),'files',manifest['total_bytes'],'bytes; source hashes match');return
    checks=[];details={}
    def check(label,condition):
        assert condition,label;checks.append(label)
    for prepared in [OUT,OUT/'refined_background']:
        prep=json.loads((prepared/'preparation.json').read_text());review=json.loads((prepared/'preparation_visual_review.json').read_text(encoding='utf-8-sig'))
        check(f'{prepared.name}: explicit future boundary',prep['future_used'] and not prep['intermediate_future_frames_used'] and prep['endpoint_source_index']==73)
        check(f'{prepared.name}: frozen source inputs',all(sha256(SCENE/path)==digest for path,digest in prep['input_sha256'].items()))
        check(f'{prepared.name}: inspected exact guide and control',review['accepted'] and review['control_sha256']==prep['control_sha256']==sha256(prepared/'control_endpoint_720x480.mp4')
            and review['guide_sha256']==prep['guide_sha256']==sha256(prepared/'guide_720x480.npz'))
        guide=np.load(prepared/'guide_720x480.npz')['frames']
        check(f'{prepared.name}: guide shape/type',guide.shape==(49,480,720,3) and guide.dtype==np.uint8)
        source=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
        check(f'{prepared.name}: exact t0 transform',np.array_equal(guide[0],cv2.resize(source,(720,480))))
        check(f'{prepared.name}: exact pose hold after 2 s',np.array_equal(guide[16:],np.broadcast_to(guide[16],guide[16:].shape)))
        control=frames(prepared/'control_endpoint_720x480.mp4');check(f'{prepared.name}: dense control dimensions',control.shape==(49,480,720,3))
        check(f'{prepared.name}: no predominantly empty control',float((control.max(-1)==0).mean())<.005)
    config={};usage={}
    import torch
    for name in ['guided_strength045','endpoint_without_rgb_prior',CHOSEN,'no_trajectory_control']:
        folder=OUT/name;c=json.loads((folder/'config.json').read_text());u=json.loads((folder/'resource_usage.json').read_text())
        config[name]=c;usage[name]=u
        check(f'{name}: actual successful generation',u['success'] and u['stage']=='complete' and u['output_sha256']==sha256(folder/'generated_seed42.mp4'))
        check(f'{name}: trained pinned model',c['checkpoint_revision']=='a116c52b08182f7a5916eae7a47fcd61a9467a04' and c.get('das_commit')=='47dcaf13f78fe3cafd2a4aed20df1e059917599a')
        check(f'{name}: physical time and output',c['fps']==8 and c['num_frames']==49 and frames(folder/'generated_seed42.mp4').shape==(49,480,720,3))
        check(f'{name}: hardware bounds',u['peak_cuda_allocated_bytes']<12*2**30 and u['peak_process_rss_bytes']<24*2**30)
        check(f'{name}: future labeling and no RGB composition',c['future_used'] and not c['causal_MolmoMotion_benchmark'] and not c['postprocessed_RGB'])
        z=torch.load(folder/'generated_latents.pt',map_location='cpu',weights_only=True)['latents'].float()
        check(f'{name}: real generated latent tensor',tuple(z.shape)==(1,16,13,60,90) and torch.isfinite(z).all())
        if name in ['guided_strength045',CHOSEN]:
            prior=json.loads((folder/'photographic_prior_receipt.json').read_text())
            check(f'{name}: partial correctly scheduled diffusion prior',prior['applied_steps']==30 and prior['sigma_after_each_step'][-1]==0 and 0<prior['strength']<.9)
            g=torch.load(folder/'photographic_guide_latents.pt',map_location='cpu',weights_only=True)['latents'].float()
            relative_rms=float(torch.sqrt(torch.mean((z[:,:,1:]-g[:,:,1:])**2))/torch.sqrt(torch.mean(g[:,:,1:]**2)))
            check(f'{name}: model output differs from guide',relative_rms>1e-4)
            check(f'{name}: t0 VAE latent preserved',torch.equal(z[:,:,:1],g[:,:,:1]))
            details[name]={'generated_vs_guide_relative_latent_RMS':relative_rms,'model_output_is_not_guide_copy':True}
    left=config['endpoint_without_rgb_prior'];right=config[CHOSEN]
    shared=['model','checkpoint_revision','das_commit','dtype','seed','num_inference_steps','num_frames','resolution','fps','guidance_scale','teacache','persistent_reference','prompt','control_sha256']
    check('Matched ablation: identical model/control/sampling',all(left[k]==right[k] for k in shared))
    check('Matched ablation: only RGB latent guidance added',left['photographic_guide_strength']==0 and right['photographic_guide_strength']==.45)
    none=config['no_trajectory_control']
    common_no_control=[k for k in shared if k!='control_sha256']
    check('Native control ablation: identical model/text/sampling',all(left[k]==none[k] for k in common_no_control))
    check('Native no-control route: no future pixel condition',none['control_sha256'] is None and not none['trajectory_control'] and not none['future_pixels_or_pose_condition_passed_to_model'] and none['photographic_guide_strength']==0)
    report=json.loads((OUT/'comparison_metrics.json').read_text());m=report['variants'];new=m[CHOSEN];old=m['previous_initial_only']
    check('Withheld intermediate frames explicit',report['withheld_intermediate_evaluation_time_s']==report['time_s'][:-1] and len(report['time_s'])==10)
    for name in m:
        arrays=np.load(OUT/f'aligned_{name}.npz');check(f'{name}: aligned physical time',np.allclose(arrays['times'],np.arange(1,11)/5))
        independent=metrics(arrays['real_xy'][:,:24],arrays['xy'][:,:24],arrays['four_way_mask'][:,:24]&arrays['real_visibility'][:,:24])
        recorded=m[name]['four_way_common_cup_to_real']
        check(f'{name}: independently recomputed shared errors',independent['visible_pairs']==recorded['visible_pairs'] and np.isclose(independent['ADE_px'],recorded['ADE_px']))
    # These are reconstruction checks. They do not claim an improvement in
    # MolmoMotion forecasting, because the endpoint is known to the generator.
    check('Better reconstruction image error',new['image_quality']['shared_ROI_MAE_0_255']<old['image_quality']['shared_ROI_MAE_0_255'])
    check('Better withheld-intermediate image error',new['image_quality']['withheld_intermediate_ROI_MAE_0_255']<old['image_quality']['withheld_intermediate_ROI_MAE_0_255'])
    check('Better shared visible cup reconstruction',new['three_way_common_cup_to_real']['ADE_px']<old['three_way_common_cup_to_real']['ADE_px'])
    visual=json.loads((OUT/'visual_review.json').read_text(encoding='utf-8-sig'))
    check('Actual all-frame semantic inspection',visual['chosen_variant']==CHOSEN and visual['reviewed_frame_indices']==list(range(49)) and visual['accepted'])
    check('Single cup and preserved recognition',visual['chosen_blue_cup_count_by_frame']==[1]*49 and visual['printed_motif_recognizable'] and visual['stationary_old_cup_copy_absent'])
    # Prior causal experiments and their forecast artifacts remain untouched.
    frozen=json.loads((SCENE/'das_robot_cup/control_input_freeze.json').read_text())['sha256']
    check('Original causal experiment inputs unchanged',all(sha256(SCENE/path)==digest for path,digest in frozen.items()))
    write_json(OUT/'verification.json',{'status':'PASS','check_count':len(checks),'checks':checks,'details':details,
        'scope':'One-scene endpoint-assisted DaS synthesis; quality and reconstruction checks, not causal forecast validation'})
    print('VERIFICATION PASS',len(checks),'checks',json.dumps(details))

if __name__=='__main__':main()
