"""Compare same-seed trajectory and native null-control ablations."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from das_evaluate import frames,label
from das_prepare_control import sheet,save_video,write_json

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args()
out=a.scene/'das_wanfun'
controlled=json.loads((out/'metrics.json').read_text())
plain=json.loads((out/'metrics_no_control.json').read_text())
c=frames(out/'generated_molmomotion_seed42.mp4')
n=frames(out/'generated_no_control_seed42.mp4')
native=lambda video:np.stack([cv2.resize(f,(640,480),interpolation=cv2.INTER_LINEAR) for f in video])
c,n=native(c),native(n)
control=frames(out/'control_molmomotion_640x480.mp4')
sheet(out/'ablation_contact_sheet.png',[control,c,n],['DAS CONTROL','WITH TRAJECTORY','NO TRAJECTORY (native None)'])
sheet(out/'ablation_full_duration.png',[c,n],['WITH TRAJECTORY','NO TRAJECTORY'],indices=(0,8,16,24,32,40,48))
save_video(out/'ablation_comparison.mp4',[np.concatenate([label(control[i],f'CONTROL {i/8:.3f}s'),
           label(c[i],f'WITH TRAJECTORY {i/8:.3f}s'),label(n[i],f'NO TRAJECTORY {i/8:.3f}s')],axis=1) for i in range(17)])
aligned_c=np.load(out/'aligned_evaluation.npz');aligned_n=np.load(out/'aligned_evaluation_no_control.npz')
shared=aligned_c['shared_mask']&aligned_n['shared_mask']
result={'same_parameters':['image','prompt','checkpoint','seed=42','steps=25','49 frames','720x480','8 FPS'],
    'control_difference':'prepared tracking video versus official control_video=None, not encoded black video',
    'controlled_to_real':controlled['generated_to_real'],'no_control_to_real':plain['generated_to_real'],
    'controlled_to_requested_motion':controlled['generated_to_control'],
    'no_control_to_requested_motion':plain['generated_to_control'],
    'shared_pair_count':int(shared.sum()),'shared_total':int(shared.size),
    'mean_RGB_difference_between_ablations_0_255_first_2s':float(np.abs(c[:17].astype(float)-n[:17]).mean()),
    'controlled_warp_error':controlled['temporal_consistency_generated_nearest_5Hz_times']['mean_warp_MAE_0_255'],
    'no_control_warp_error':plain['temporal_consistency_generated_nearest_5Hz_times']['mean_warp_MAE_0_255'],
    'limitation':'One episode and one seed; no statistical/general model capability claim.'}
for name,aligned in [('controlled',aligned_c),('no_control',aligned_n)]:
    for target in ['real','control']:
        error=np.linalg.norm(aligned['generated']-aligned[target],axis=-1)
        result[f'{name}_to_{target}_common_ADE_px']=float(error[shared].mean()) if shared.any() else None
        result[f'{name}_to_{target}_common_FDE_px']=float(error[-1,shared[-1]].mean()) if shared[-1].any() else None
write_json(out/'ablation_metrics.json',result)
print(json.dumps(result,indent=2))
