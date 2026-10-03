"""Prove completion against actual outputs and the unchanged earlier experiment."""
import json,re
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np,torch
from fmb_wrist_prepare import ROOT,write,sha
from fmb_wrist_math import tcp_matrices
from fmb_v2_infer import strict_parse
from fmb_wrist_v4_diagnose import BASE,OUT
from fmb_wrist_v4_prepare import forecast_tcp,MODEL_TIMES
from fmb_wrist_v4_refine import predict
from fmb_wrist_v4_evaluate import align

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def run():
    checks=[];calls=[]
    def check(name,condition,evidence):
        checks.append({'requirement':name,'pass':bool(condition),'evidence':evidence});assert condition,f'{name}: {evidence}'
    state=read(OUT/'inference_status.json');check('All planned native variants finished',state['status']=='COMPLETE' and state['successful_P8_calls']==12,'inference_status.json')
    prereg=read(OUT/'preregistration.json');selection=read(OUT/'robot_observed_model_selection.json');amendment=read(OUT/'motion_policy_amendment_v2.json')
    check('Observed policy selection stays before source127',selection['validation_targets_max_source_index']==126 and not selection['future_127_146_used'] and not amendment['new_parameters_selected_from_future'],'Observed selectors and their source bounds')
    check('Two motion families actually validated',len(amendment['candidates'])==50 and {r['family'] for r in amendment['candidates']}=={'full_TCP_pose','rigid_centroid_translation'} and all(r['valid_observed_validation_samples']>0 and np.isfinite(r['observed_target_ADE_m']) for r in amendment['candidates']),'motion_policy_amendment_v2.json; NaN-empty comparison rejected')
    check('Winner selected by observed target error',amendment['winner']==min(amendment['candidates'],key=lambda r:r['observed_target_ADE_m']),'Observed ranking, no future ADE selection')
    check('Exploratory prior future knowledge disclosed',prereg['known_future_results'] and amendment['first_stage_future_scores_already_known'],'Preregistration and amendment limitations')
    review=read(OUT/'builtin_visual_review.json')
    check('Built-in image assessment, no external assessment VLM',review['external_HF_VLM_used'] is False and len(review['reviewed_images'])>=7,'builtin_visual_review.json')
    check('Actual reviewed images preserved',all(sha(OUT/r['image'])==r['sha256'] for r in review['reviewed_images']),'Reviewed image SHA256')
    tests=read(OUT/'coordinate_tests.json');check('Five meaningful coordinate tests pass',tests['exit_code']==0 and '5 passed' in tests['output'],'coordinate_tests.json')
    check('Tested source code unchanged',bool(tests.get('source_sha256')) and all(sha(ROOT/p)==digest for p,digest in tests['source_sha256'].items()),'Test/code hashes')
    script_hashes={sha(ROOT/'scripts/fmb_wrist_v4_infer.py'),sha(OUT/'code_snapshots/fmb_wrist_v4_infer_initial.py')}
    for camera in ['wrist_2','wrist_1']:
        old=BASE/camera;dest=OUT/camera;observed=np.load(dest/'observed_reference.npz')
        check(camera+'only50 observed TCP poses',np.array_equal(observed['tcp'],tcp_matrices(np.load(old/'observed/tcp_pose_xyzw.npy'))) and len(observed['tcp'])==50 and np.array_equal(np.load(old/'observed/source_indices.npy'),np.arange(77,127)),'Observed archive compared to original prefix')
        check(camera+'same X/K/24 point IDs',np.array_equal(observed['X'],np.load(old/'geometry/selected_hand_eye_X.npy')) and np.array_equal(observed['K'],np.load(old/'geometry/official_K.npy')) and np.array_equal(observed['ids'],np.load(old/'observed/selected_point_ids.npy')),'Fixed original geometry and query set')
        w=amendment['winner'];recomputed=predict(observed,49,w['window'],w['damping_tau_s'],MODEL_TIMES,w['family']).astype(np.float32)
        check(camera+'forecast reconstructs from observed poses alone',np.array_equal(recomputed,np.load(dest/'selected_motion_policy_v2_15hz.npy')) and sha(dest/'selected_motion_policy_v2_15hz.npy')==amendment['forecast_hashes'][camera],'Replayed frozen motion policy; no future TCP passed')
        for name,h,f in [('H3_rigid_observed',3,30),('H1_native',1,32)]:
            folder=dest/'variants'/name;spec=read(folder/'input_spec.json');hist=np.load(folder/'history_xyz.npy');rgb=np.load(folder/'history_rgb.npy')
            check(camera+'/'+name+'frozen inputs unchanged',all(sha(folder/p)==digest for p,digest in spec['sha256'].items()) and not spec['future_used'],'input_spec.json')
            check(camera+'/'+name+'genuine unchanged RGB frames',np.array_equal(rgb,np.load(old/'observed/history_rgb.npy')[-h:]) and spec['source_history_indices']==[124,125,126][-h:],'Original source RGB, no guessed future image')
            check(camera+'/'+name+'same point order',np.array_equal(np.load(folder/'point_ids.npy'),observed['ids']) and hist.shape==(h,24,3) and np.isfinite(hist).all(),'24 frozen IDs and finite H1/H3')
            combined=[]
            for group in range(3):
                d=folder/f'group_{group:02d}';r=read(d/'model_run.json');p=np.load(d/'future_3d.npy');a=np.load(d/'anchor.npy')
                raw=(d/'raw_model_output.txt').read_text();parsed=strict_parse(raw,h,f)
                batch=torch.load(d/'processor_inputs.pt',map_location='cpu',weights_only=False)
                valid=r['success'] and r['generation_started'] and r['future_input_used'] is False and r['dtype']=='bfloat16' and r['seed']==0
                valid &= p.shape==(8,f,3) and np.isfinite(p).all() and np.allclose(p,parsed+a,atol=1e-4)
                valid &= np.array_equal(a,hist[-1,group*8]) and np.array_equal(batch['anchor_3d'].float().numpy().reshape(3),a)
                valid &= r['processor_inputs_sha256']==sha(d/'processor_inputs.pt') and r['raw_sha256']==sha(d/'raw_model_output.txt') and r['prediction_sha256']==sha(d/'future_3d.npy')
                valid &= r['script_sha256'] in script_hashes and r['input_spec_sha256']==sha(folder/'input_spec.json')
                checkpoint=ROOT/'data/checkpoints'/r['model_id']
                valid &= r['config_sha256']==sha(checkpoint/'config.yaml') and (checkpoint/'model.pt').is_file()
                check(camera+'/'+name+f'/group{group} actual strict output',valid,str(d.relative_to(OUT)))
                combined.append(p);calls.append({'camera':camera,'variant':name,'group':group,'seconds':r['prediction_seconds'],'peak_cuda_allocated_gib':r['peak_cuda_allocated_gib']})
            check(camera+'/'+name+'combined native output exact',np.array_equal(np.concatenate(combined),np.load(folder/'future_3d.npy')),folder.name)
        metrics=read(dest/'metrics.json');oldm=read(old/'evaluation/metrics.json');data=np.load(dest/'predictions_and_projections.npz');ref=np.load(old/'evaluation/future_reference.npz')
        check(camera+'unchanged reference and masks',metrics['reference_sha256']==sha(old/'evaluation/future_reference.npz') and metrics['common_valid_3d_samples']==int(ref['common_mask3d'].sum()) and metrics['common_valid_2d_samples']==int(ref['common_mask2d'].sum()),'Exact original reference SHA/masks')
        for name,original in [('v3_Molmo_H3','C_sensor_tcp_official'),('v3_Static','C_sensor_tcp_official/Static'),('v3_CV','C_sensor_tcp_official/CV')]:
            check(camera+'/'+name+'original metrics reproduced',all(abs(metrics['methods'][name][key][field]-oldm['methods'][original][key][field])<1e-9 for key in ['3D_est_m','2D_px'] for field in ['ADE','FDE']),'Recomputed original baseline metrics')
        check(camera+'every method keeps original masks and off-image errors',not metrics['offimage_predictions_excluded'] and all(r['3D_est_m']['valid_samples']==metrics['common_valid_3d_samples'] and r['2D_px']['valid_samples']==metrics['common_valid_2d_samples'] for r in metrics['methods'].values()),'No masks narrowed for improved numbers')
        check(camera+'practical forecast improves over both old model and Static',metrics['methods']['Robot_attached_observed_selected']['3D_est_m']['ADE']<min(metrics['methods']['v3_Molmo_H3']['3D_est_m']['ADE'],metrics['methods']['v3_Static']['3D_est_m']['ADE']) and metrics['methods']['Robot_attached_observed_selected']['2D_px']['ADE']<metrics['methods']['v3_Molmo_H3']['2D_px']['ADE'],'Numerical improvement on the same reference, not attributed to neural retraining')
        correction=np.load(dest/'hybrid_correction.npy');cap=read(OUT/'observed_diagnosis.json')[camera]['camera_relative_target_drift_last10_p90_mm']/1000
        check(camera+'bounded neural residual uses observed radius',np.linalg.norm(correction,axis=-1).max()<=cap+1e-9 and np.allclose(data['Robot_plus_bounded_Molmo'],data['Robot_attached_observed_selected']+correction[None]),'Observed clipping rule and exact hybrid decomposition')
        check(camera+'no future camera poses used in forecast',not metrics['future_TCP_in_forecast'] and metrics['future_camera_poses_evaluation_only'],'metrics.json and reconstructed prediction above')
        sensitivity=read(dest/'calibration_ranking_sensitivity.json');check(camera+'seven observed-fit reference sensitivity',len(sensitivity['rows'])==7 and not sensitivity['future_used_for_refitting'] and sensitivity['forecasts_fixed'],'calibration_ranking_sensitivity.json')
        video=dest/'viz/forecast_comparison.mp4';capv=cv2.VideoCapture(str(video));count=capv.get(cv2.CAP_PROP_FRAME_COUNT);fps=capv.get(cv2.CAP_PROP_FPS);ok,_=capv.read();capv.set(cv2.CAP_PROP_POS_FRAMES,19);last_ok,_=capv.read();capv.release()
        check(camera+'new comparison video plays to final frame',count==20 and abs(fps-10)<.01 and ok and last_ok,{'sha256':sha(video),'frames':count,'fps':fps})
        check(camera+'complete30/32 step plots exist',all((dest/'viz'/p).exists() for p in ['comparison_01.png','comparison_10.png','comparison_20.png','errors.png','full_neural_outputs.png']),'Complete native trajectories shown, no clipped coordinate range')
    preserved=read(OUT/'preserved_v3.json')
    check('Entire prior run and report preserved',all(sha(ROOT/p)==digest for p,digest in preserved['sha256'].items()) and sha(ROOT/'report/fmb_wrist_v3.md')==preserved['report_sha256'],{'preserved_files':len(preserved['sha256'])})
    check('12 complete native P8 calls',len(calls)==12,'Two neural variants, two cameras, three groups each')
    interrupted=read(OUT/'runtime_interruption_00/receipt.json');check('Incomplete extra call disclosed, no outside process killed',interrupted['generation_started'] and interrupted['partial_group_archived'] and not interrupted['completed_group_discarded'] and not interrupted['external_process_terminated'],'Runtime interruption ledger')
    report=ROOT/'report/fmb_wrist_v4_improvement.md';body=report.read_text(encoding='utf-8')
    links=[p for p in re.findall(r'\]\(([^)]+)\)',body) if not p.startswith('http')]
    check('Final report and all evidence/video links resolve',all((report.parent/p).resolve().is_file() for p in links) and 'Промежуточный отчёт' not in body,'Completed full report')
    check('Reproduction commands and limitations documented',(OUT/'REPRODUCE.md').is_file(),'REPRODUCE.md')
    gallery=read(OUT/'media_receipt.json');check('Local video gallery present and hash verified',sha(OUT/'index.html')==gallery['gallery_sha256'],'index.html')
    write(OUT/'completion_audit.json',{'success':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'native_calls':calls,
          'successful_generations':12,'interrupted_generations':1,'initialization_failures_before_new_generations':2,
          'report_sha256':sha(report),'audit_script_sha256':sha(Path(__file__)),
          'experimental_unit':'One previously studied episode, two views sharing the robot stream',
          'result':'Actual practical forecast improved on unchanged estimated reference; native ablations and negative results retained, improvement attribution separated from neural model changes.',
          'physical_geometry_certified':False})
    print('Completion audit passed:',len(checks),'checks,12 complete native calls',flush=True)

if __name__=='__main__':run()
