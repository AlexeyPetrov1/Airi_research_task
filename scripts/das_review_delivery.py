"""Audit selected video, documented numbers, and local report links before delivery."""
import argparse,json,re
from pathlib import Path
from urllib.parse import unquote
import imageio.v2 as imageio
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);a=p.parse_args()
root=a.repo/'runs/berkeley_ur5_molmomotion/cup/das_wanfun';selected=root/'variants/clean_reference_mask_v6'
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
verification=read(selected/'verification.json');quality=read(selected/'qualitative_review.json')
resource=read(selected/'resource_usage.json');metrics=read(selected/'metrics.json')
comparison=read(root/'variant_comparison.json')
report=(a.repo/'report/das_cup_dedup.md').read_text(encoding='utf-8')
checks={'selected_generation_success':resource['success'] and read(selected/'process_exit.json')['returncode']==0,
        'selected_evaluation_success':read(selected/'evaluation_process_exit.json')['returncode']==0,
        'selected_technical_checks':verification['technical_status']=='PASS' and all(verification['checks'].values()),
        'selected_manual_persistent_duplicate_gate':verification['duplicate_removal_status']=='PASS_MANUAL_REVIEW' and quality['duplicate_removal_pass'],
        'selected_all_49_visual_frames':quality['reviewed_frames']==list(range(49)),
        'selected_raw_video_sha_matches':sha256(selected/'generated_molmomotion_seed42.mp4')==quality['video_sha256']==resource['output_sha256'],
        'extension_and_transition_disclosed':'расширение inference' in report and '0.125…0.375' in report and 'не ретушировались' in report,
        'identity_and_visibility_limits_disclosed':'физически корректный перенос' in report and 'пропущены 1.0…1.4' in report,
        'nine_new_tests_documented':'Четыре теста vacancy-prior' in report and 'Два теста RGB-mask' in report and 'Три теста восстановления depth' in report}
values=[metrics[target][measure] for target in ['generated_to_control','generated_to_real'] for measure in ['ADE_px','FDE_px']]
pair=comparison['paired_original_comparisons'][selected.name]
values += [pair['scores'][name][target][measure] for name in ['original',selected.name]
           for target in ['control','real'] for measure in ['ADE_px','FDE_px']]
checks['documented_raw_and_common_mask_numbers']=all(f'{value:.2f}' in report for value in values)
checks['common_mask_count_documented']=pair['visible_pairs']==93 and '93/240' in report
missing=[]
for relative in ['report/das_cup_dedup.md','report/das_wanfun_cup.md',
                 'runs/berkeley_ur5_molmomotion/research_story.md',str(root.relative_to(a.repo)/'README.md')]:
    path=a.repo/relative;text=re.sub(r'```.*?```','',path.read_text(encoding='utf-8'),flags=re.S)
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
        target=link.strip().strip('<>')
        if target.startswith(('http:','https:','#','app:')):continue
        target=unquote(target.split('#')[0])
        if not (path.parent/target).exists():missing.append({'document':relative,'target':target})
checks['all_local_report_links_resolve']=not missing
for name,width in [('before_after_full_duration.mp4',1280),('initial_site_before_after.mp4',768)]:
    with imageio.get_reader(str(selected/name)) as reader:
        meta=reader.get_meta_data();count=0;shapes=set()
        for frame in reader:count+=1;shapes.add(frame.shape)
    checks[name+'_metadata']=count==49 and shapes=={(480,width,3)} and meta['fps']==8
result={'status':'PASS' if all(checks.values()) else 'FAIL','selected_variant':selected.name,
        'checks':checks,'missing_local_links':missing,'video_sha256':quality['video_sha256'],
        'scope':'Persistent duplicate removed after 0.5s and improved evaluation/report delivered; early fade and physical identity limitations remain.',
        'report':'report/das_cup_dedup.md'}
write_json(root/'delivery_verification.json',result);print(json.dumps(result,indent=2))
raise SystemExit(not all(checks.values()))
