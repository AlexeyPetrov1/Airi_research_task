"""Review finished robot/cup control, generation, and evaluation artifacts."""
from pathlib import Path
import json
import numpy as np
import imageio.v2 as imageio
from PIL import Image
from das_prepare_control import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup'

def main():
    checks={}
    frozen=json.loads((OUT/'control_input_freeze.json').read_text())
    checks['frozen_causal_inputs']=not frozen['future_used'] and all(sha256(SCENE/n)==h for n,h in frozen['sha256'].items())
    state=np.load(OUT/'observed/robot_state_history.npz')
    checks['robot_states_observed_only']=np.array_equal(state['source_indices'],np.arange(56,64))
    control=np.load(OUT/'joint_control.npz');q=control['joints'];fc=control['flange_camera']
    checks['49_frames_8_fps_native_2s_horizon']=np.array_equal(control['times'],np.arange(49)/8)
    checks['final_pose_held_after_2s']=np.allclose(q[17:],q[16])
    checks['position_limits']=bool((np.abs(q)<=2*np.pi+1e-8).all() and (np.abs(q[:,2])<=np.pi+1e-8).all())
    checks['speed_bounded']=bool(np.max(np.abs(np.diff(q,axis=0))*8)<=2+1e-8)
    checks['proper_flange_rotations']=bool(np.allclose(np.linalg.det(fc[:,:3,:3]),1,atol=1e-8))
    cup=control['cup_sparse_xyz'];local=np.einsum('tij,tnj->tni',fc[:,:3,:3].transpose(0,2,1),cup-fc[:,None,:3,3])
    checks['cup_attached_to_flange']=bool(np.max(np.abs(local-local[0]))<1e-8)
    nominal=np.load(OUT/'observed/nominal_ur5_camera.npz');robot=np.load(OUT/'robot_dense_motion.npz');cloud=np.load(OUT/'robot_cloud_t0.npz')
    checks['initial_robot_geometry_exact']=bool(np.allclose(robot['xyz'][0],cloud['xyz'],atol=1e-7))
    checks['initial_cup_geometry_exact']=bool(np.allclose(cup[0],np.load(SCENE/'observed/points_3d_history.npy')[-1],atol=1e-8))
    errors=[]
    for link in np.unique(cloud['link_ids']):
        idx=np.flatnonzero(cloud['link_ids']==link)[::20]
        distances=np.linalg.norm(robot['xyz'][:,idx]-robot['xyz'][:,idx[:1]],axis=-1)
        errors.append(float(np.max(np.abs(distances-distances[0]))))
    checks['each_robot_link_preserves_shape']=max(errors)<1e-6
    original=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    clean=np.array(Image.open(OUT/'observed/clean_reference_640x480.png'))
    alpha=np.array(Image.open(OUT/'observed/clean_reference_alpha.png'))
    checks['reference_outside_edit_support_unchanged']=bool(np.array_equal(original[alpha==0],clean[alpha==0]))
    configs={};video_counts={}
    for name in ['robot_static','robot_coupled','robot_coupled_initial_only']:
        out=OUT/name;resource=json.loads((out/'resource_usage.json').read_text());exit=json.loads((out/'process_exit.json').read_text())
        checks[f'{name}_generation_success']=resource['success'] and exit['returncode']==0 and not exit['running']
        config=json.loads((out/'config.json').read_text());configs[name]=config
        path=out/'generated_molmomotion_seed42.mp4'
        checks[f'{name}_output_hash_matches']=resource['output_sha256']==sha256(path)
        checks[f'{name}_control_reviewed']=json.loads((out/'control_validation.json').read_text())['generation_ready']
        with imageio.get_reader(str(path)) as reader:
            dims=[frame.shape for frame in reader]
        video_counts[name]=len(dims);checks[f'{name}_video_format']=len(dims)==49 and all(shape==(480,720,3) for shape in dims)
    common_keys=['model','checkpoint_revision','git_commit','dtype','seed','num_inference_steps','num_frames','resolution','fps','prompt','image_sha256','guidance_scale','scheduler','teacache_threshold','reference_mode','persistent_reference_image_sha256','vacancy_strength','vacancy_context_px','vacancy_reference_alpha_sha256']
    checks['A_B_parameters_matched']=all(configs['robot_static'][key]==configs['robot_coupled'][key] for key in common_keys)
    checks['A_B_cup_control_identical']=sha256(OUT/'robot_static/dense_predicted_motion.npz')==sha256(OUT/'robot_coupled/dense_predicted_motion.npz')
    checks['B_C_control_identical']=configs['robot_coupled']['control_sha256']==configs['robot_coupled_initial_only']['control_sha256']
    metrics=json.loads((OUT/'paired_metrics.json').read_text())
    checks['all_three_variants_evaluated']=set(metrics['variants'])==set(configs)
    review=json.loads((OUT/'qualitative_review.json').read_text())
    checks['all_49_frames_visually_reviewed']=all(row['reviewed_frame_count']==49 for row in review['variants'].values())
    checks['video_comparisons_present']=all((OUT/name).is_file() for name in ['paired_generated_vs_real_2s.mp4','static_vs_coupled_full_6s.mp4','paired_results_exact_times.png','all_three_vs_real_2s.mp4'])
    checks['report_present']=(OUT/'README.md').is_file()
    import re
    missing_links=[]
    for target in re.findall(r'\]\(([^)]+)\)',(OUT/'README.md').read_text(encoding='utf8')):
        if '://' not in target and not (OUT/target.split('#')[0]).exists():missing_links.append(target)
    checks['report_local_links_resolve']=not missing_links
    code=list((ROOT/'scripts').glob('das_robot_*.py'))+[ROOT/'scripts'/name for name in
        ['das_generate.py','das_vacancy_prior.py','das_prepare_control.py','das_evaluate.py','das_quality.py','das_wanfun_runtime.py','probe_fmb_rlds_record.py']]
    import hashlib
    def source_hash(path):return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    write_json(OUT/'source_code_manifest.json',{'scope':'Source code state at delivery; generator/conditioning code used for the actual runs included',
        'hash_method':'SHA256 with CRLF normalized to LF, matching canonical repository source across Windows and Linux',
        'repository_files':{str(path.relative_to(ROOT)):source_hash(path) for path in code},
        'external_alltracker_source_sha256':source_hash(Path('/mnt/f/AIRI_task/molmo-motion/data_generation/third_party/alltracker/nets/alltracker.py'))})
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
        'maximum_link_rigidity_error_m':max(errors),'decoded_video_counts':video_counts,
        'scope':'Artifact/control correctness and actual experiment completion. PASS does not assert that DaS generated a realistic robot/cup video.'}
    write_json(OUT/'verification.json',report);print(json.dumps(report,indent=2));raise SystemExit(not all(checks.values()))

if __name__=='__main__':main()
