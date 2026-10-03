"""Compare completion variants; keep descriptive scores separate from quality gates."""
import argparse,json
from pathlib import Path
import numpy as np
from das_prepare_control import write_json,sha256
from das_evaluate import metrics

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--variants',nargs='+',type=Path,required=True);a=p.parse_args()
root=a.scene/'das_wanfun';runs={'original':root}
runs.update({v.name:v for v in a.variants})
report={'scope':'One frozen episode, one seed. Variants change conditioning; original no-control is a baseline, not a matched vacancy-prior ablation.',
        'runs':{},'common_mask_comparison':{},'paired_original_comparisons':{}}
aligned={}
for name,folder in runs.items():
    item={'video_sha256':sha256(folder/'generated_molmomotion_seed42.mp4')}
    for filename,key in [('metrics.json','metrics'),('local_quality.json','local_quality'),
                         ('duplicate_texture_audit.json','initial_site_texture'),('resource_usage.json','resources'),
                         ('qualitative_review.json','visual_review')]:
        if (folder/filename).exists():item[key]=json.loads((folder/filename).read_text(encoding='utf-8'))
    report['runs'][name]=item
    if (folder/'aligned_evaluation.npz').exists():aligned[name]=np.load(folder/'aligned_evaluation.npz')
if aligned:
    shared=np.logical_and.reduce([x['shared_mask'] for x in aligned.values()])
    report['common_mask_comparison']={'included_runs':list(aligned),'visible_pairs':int(shared.sum()),
        'total_pairs':int(shared.size),'coverage':float(shared.mean()),'scores':{}}
    for name,data in aligned.items():
        report['common_mask_comparison']['scores'][name]={target:metrics(data[target],data['generated'],shared)
            for target in ['control','real']}
    for name,data in aligned.items():
        if name=='original':continue
        pair_mask=aligned['original']['shared_mask']&data['shared_mask']
        report['paired_original_comparisons'][name]={'visible_pairs':int(pair_mask.sum()),
            'total_pairs':int(pair_mask.size),'coverage':float(pair_mask.mean()),
            'scores':{run:{target:metrics(values[target],values['generated'],pair_mask)
                     for target in ['control','real']} for run,values in [('original',aligned['original']),(name,data)]}}
write_json(root/'variant_comparison.json',report)
print({'all_run_common_pairs':report['common_mask_comparison'].get('visible_pairs'),
       'paired_counts':{name:pair['visible_pairs'] for name,pair in report['paired_original_comparisons'].items()}})
