"""Verify checksums, causal separation, geometry, and delivered video metadata."""
import argparse
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args()
out=a.scene/'das_wanfun'
freeze=json.loads((out/'control_input_freeze.json').read_text())
checks={'causal_inputs_unchanged':all(sha256(a.scene/n)==h for n,h in freeze['sha256'].items()),
        'control_inputs_contain_no_evaluation':all(not n.startswith('evaluation/') for n in freeze['sha256']),
        'visual_control_approved':json.loads((out/'control_validation.json').read_text())['generation_ready']}
motion=np.load(out/'rigid_motion_49.npz')
checks['proper_rotations']=bool(np.allclose(np.linalg.det(motion['R']),1.,atol=1e-6))
checks['initial_identity']=bool(np.allclose(motion['R'][0],np.eye(3)) and np.allclose(motion['t'][0],0))
checks['final_pose_held']=bool(np.allclose(motion['R'][17:],motion['R'][16]) and np.allclose(motion['t'][17:],motion['t'][16]))
videos={}
for name in ['control_molmomotion_640x480.mp4','control_molmomotion_720x480.mp4',
             'generated_molmomotion_seed42.mp4','generated_no_control_seed42.mp4',
             'triple_comparison.mp4','real_vs_generated.mp4','control_vs_generated.mp4']:
    path=out/name
    if not path.exists():continue
    with imageio.get_reader(str(path)) as reader:
        metadata=reader.get_meta_data(); count=0;shape=None
        for frame in reader:count+=1;shape=frame.shape
    expected=49 if name.startswith(('control_molmo','generated_')) else 17
    checks['video_'+name]=bool(count==expected and abs(metadata['fps']-8)<.01)
    videos[name]={'frames':count,'shape':shape,'fps':metadata['fps'],'sha256':sha256(path)}
required=['generated_molmomotion_seed42.mp4','generated_tracking.npz','metrics.json',
          'generated_contact_sheet.png','triple_comparison.mp4','resource_usage.json']
checks['required_generation_artifacts']=all((out/n).exists() for n in required)
if (out/'resource_usage.json').exists():
    checks['generation_success']=json.loads((out/'resource_usage.json').read_text())['success'] is True
if (out/'metrics.json').exists():
    data=json.loads((out/'metrics.json').read_text())
    checks['four_quantitative_errors']=all(data[k]['ADE_px'] is not None and data[k]['FDE_px'] is not None
        for k in ['molmomotion_to_real','rigid_to_molmomotion','generated_to_control','generated_to_real'])
    checks['evaluation_physical_times']=bool(np.allclose(data['evaluation_times_s'],np.arange(1,11)/5))
    tracking=np.load(out/'generated_tracking.npz')
    checks['generated_tracking_shape']=tracking['tracks'].shape==(17,24,2)
    aligned=np.load(out/'aligned_evaluation.npz')
    valid=aligned['generated_visible']&aligned['real_visible']&np.isfinite(aligned['generated']).all(-1)
    valid&=(aligned['generated'][...,0]>=0)&(aligned['generated'][...,0]<640)&(aligned['generated'][...,1]>=0)&(aligned['generated'][...,1]<480)
    error=np.sqrt(np.sum((aligned['generated']-aligned['real'])**2,axis=-1))
    checks['generated_real_ADE_recomputed']=bool(np.isclose(error[valid].mean(),data['generated_to_real']['ADE_px'],atol=1e-5))
if (out/'generated_no_control_seed42.mp4').exists():
    primary=json.loads((out/'config.json').read_text());ablation=json.loads((out/'config_no_control.json').read_text())
    checks['ablation_parameters_match']=all(primary[k]==ablation[k] for k in
        ['model','checkpoint_revision','git_commit','dtype','seed','num_inference_steps','num_frames','resolution','fps','prompt','image_sha256'])
    checks['ablation_native_no_control']=ablation['control_video'] is None
    checks['ablation_generation_success']=json.loads((out/'resource_usage_no_control.json').read_text())['success'] is True
    checks['ablation_evaluated']=(out/'metrics_no_control.json').exists() and (out/'ablation_metrics.json').exists()
result={'checks':checks,'videos':videos,'status':'PASS' if all(checks.values()) else 'PARTIAL',
        'note':'PASS means pipeline deliverables verified, not that forecast or generated motion is accurate.'}
write_json(out/'verification.json',result)
print(json.dumps(result,indent=2))
