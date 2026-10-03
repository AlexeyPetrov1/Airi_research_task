"""Verify a new causal-control run separately from research quality judgment."""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--variant',type=Path,required=True);a=p.parse_args()
original=a.scene/'das_wanfun';out=a.variant
freeze=json.loads((out/'control_input_freeze.json').read_text())
old=json.loads((original/'config.json').read_text());new=json.loads((out/'config.json').read_text())
review=json.loads((out/'control_validation.json').read_text(encoding='utf-8-sig'))
resource=json.loads((out/'resource_usage.json').read_text())
recovery=json.loads((out/'decode_recovery.json').read_text()) if (out/'decode_recovery.json').exists() else None
checks={'causal_inputs_unchanged':all(sha256(a.scene/n)==h for n,h in freeze['sha256'].items()),
        'no_future_in_control':freeze['future_used'] is False and all(not n.startswith('evaluation/') for n in freeze['sha256']),
        'same_forecast_and_dense_motion':sha256(original/'dense_predicted_motion.npz')==sha256(out/'dense_predicted_motion.npz'),
        'matched_sampling_parameters':all(old[k]==new[k] for k in ['model','checkpoint_revision','git_commit','dtype','seed',
                'num_inference_steps','num_frames','resolution','fps','prompt','image_sha256','guidance_scale','scheduler','teacache_threshold','offload']),
        'reviewed_control':review['generation_ready'] and not review['visual_review_required'],
        'control_receipt_matches':review['control_sha256']==sha256(out/'control_molmomotion_720x480.mp4'),
        'generation_success':resource['success'] is True or bool(recovery and recovery['success'] and
            resource['stage']=='vae_decode' and recovery['latent_sha256']==sha256(out/'generated_latents.pt'))}
videos={}
for name,width in [('control_molmomotion_640x480.mp4',640),('control_molmomotion_720x480.mp4',720),
                   ('generated_molmomotion_seed42.mp4',720),('triple_comparison.mp4',1920)]:
    with imageio.get_reader(str(out/name)) as r:
        metadata=r.get_meta_data();video=[f for f in r]
    count=49 if not name.startswith('triple') else 17
    checks['video_'+name]=len(video)==count and video[0].shape==(480,width,3) and abs(metadata['fps']-8)<.01
    videos[name]={'frames':len(video),'shape':video[0].shape,'fps':metadata['fps'],'sha256':sha256(out/name)}
bg=np.load(out/'background_completion.npz');mask=bg['repair_mask'];depth=np.load(a.scene/'observed/native_depth.npy')[-1]
checks['background_unchanged_outside_estimate']=bool(np.array_equal(bg['depth'][~mask],depth[~mask],equal_nan=True))
checks['background_complete_in_footprint']=bool(np.isfinite(bg['depth'][mask]).all() and (bg['depth'][mask]>0).all())
checks['estimated_background_behind_initial_cup']=bool((bg['depth'][mask]>=review['background_depth_lower_bound_m']-1e-6).all())
data=json.loads((out/'metrics.json').read_text());aligned=np.load(out/'aligned_evaluation.npz')
valid=aligned['shared_mask'];error=np.linalg.norm(aligned['generated']-aligned['real'],axis=-1)
checks['shared_mask_ADE_recomputed']=bool(np.isclose(error[valid].mean(),data['shared_mask_generated_to_real']['ADE_px'],atol=1e-5))
checks['duplicate_audit_all_49_frames']=len(json.loads((out/'duplicate_texture_audit.json').read_text())['videos']['repaired']['NCC_by_frame'])==49
if new.get('vacancy_strength',0):
    from PIL import Image
    from das_vacancy_prior import make_latent_masks
    receipt=json.loads((out/'vacancy_prior_receipt.json').read_text())
    saved=np.load(out/'vacancy_latent_masks.npy')
    alpha=np.array(Image.open(new['command'][new['command'].index('--reference-alpha')+1]).convert('L'))/255.
    checks['vacancy_mask_receipt_recomputed']=bool(np.array_equal(saved,make_latent_masks(alpha,bg['moving_mask'],new.get('vacancy_context_px',0))))
    checks['vacancy_start_latent_protected']=bool(not saved[0].any() and receipt['protected_start_latent'])
    checks['vacancy_all_steps_applied']=receipt['applied_steps']==new['num_inference_steps']
    checks['vacancy_protocol_identified']=receipt['native_DaS'] is False and receipt['postprocessed_RGB'] is False
    checks['vacancy_inputs_match']=new['vacancy_background_bundle_sha256']==sha256(out/'background_completion.npz')
    alpha_path=Path(new['command'][new['command'].index('--reference-alpha')+1])
    reference_path=Path(new['command'][new['command'].index('--reference-image')+1])
    checks['vacancy_reference_files_match']=new['vacancy_reference_alpha_sha256']==sha256(alpha_path) and new['persistent_reference_image_sha256']==sha256(reference_path)
    rgb_receipt=json.loads((reference_path.parent/'clean_reference_receipt.json').read_text())
    checks['vacancy_reference_causal_source']=rgb_receipt['future_used'] is False and rgb_receipt['source_observed_sha256']==sha256(a.scene/'observed/frame_000063.png') and rgb_receipt['source_mask_sha256']==sha256(a.scene/'observed/mask.png')
review_path=out/'qualitative_review.json'
visual=json.loads(review_path.read_text()) if review_path.exists() else None
removal_pass=bool(visual and visual.get('duplicate_removal_pass') is True and
                 visual['video_sha256']==sha256(out/'generated_molmomotion_seed42.mp4') and
                 visual.get('reviewed_frames')==list(range(49)))
result={'technical_status':'PASS' if all(checks.values()) else 'PARTIAL','checks':checks,'videos':videos,
        'duplicate_removal_status':'PASS_MANUAL_REVIEW' if removal_pass else 'FAIL_OR_NOT_REVIEWED',
        'quality_status':'Technical PASS does not certify identity or trajectory accuracy; see manual review and metrics.'}
write_json(out/'verification.json',result);print(json.dumps(result,indent=2))
