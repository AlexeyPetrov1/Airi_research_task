"""Verify requested artifacts, actual native calls, hashes, coordinates and evaluation."""
import json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
import cv2,numpy as np,torch
from fmb_wrist_prepare import ROOT,RUN,write,sha
from fmb_wrist_math import project,from_anchor
from fmb_v2_infer import strict_parse
from berkeley_input_audit import validate_group_payloads

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def audit():
    checks=[];calls=[]
    def check(name,condition,evidence):
        checks.append({'requirement':name,'pass':bool(condition),'evidence':evidence})
        assert condition,f'{name}: {evidence}'
    provenance=read(RUN/'source_episode_5201_provenance.json')
    check('Source-matched FMB / ShareRobot5201',provenance['matches_independently_recovered_v1_source'] and provenance['new_raw_source_sha256']==sha(Path(read(RUN/'wrist_2/metadata.json')['source_path'])),'source_episode_5201_provenance.json')
    ranking=read(RUN/'selection/geometry_ranking.json')
    candidates=read(RUN/'selection/candidates.json')
    check('Observed episode selection with alternatives and target depth support',bool(ranking) and len(candidates)==10 and all(not row['future_pixel_accessed'] and not row['selection_future_poses_accessed'] for row in candidates) and (RUN/'selection/target_depth_support.json').exists(),'selection/candidates.json; geometry_ranking.json; target_depth_support.json')
    observed=read(RUN/'builtin_visual_review_observed.json');future=read(RUN/'builtin_visual_review_future.json')
    check('Built-in image assessment replaces HF VLM',observed['external_HF_VLM_used'] is False and future['external_HF_VLM_used'] is False,'builtin_visual_review_*.json')
    blind=read(RUN/'builtin_blinded_review.json')
    check('First blinded judgments preserved before map/metric disclosure',
          sha(RUN/'builtin_blinded_review.json')==future['first_blinded_review']['sha256'] and
          not blind['method_map_read_before_record'] and not blind['quantitative_metrics_read_before_record'] and
          all(len(blind['point_reviews'][camera])==24 for camera in ['wrist_2','wrist_1']),
          'builtin_blinded_review.json; immutable first assessment and per-point flags')
    tests=read(RUN/'coordinate_test_receipt.json')
    check('Four meaningful coordinate/time tests passed',tests['exit_code']==0 and '4 passed' in tests['output'] and all(sha(ROOT/name)==digest for name,digest in tests['source_sha256'].items()),'coordinate_test_receipt.json')
    for review in [observed,future]:
        check('Reviewed image hashes unchanged',all(sha(RUN/row['image'])==row['sha256'] for row in review['reviewed_images']),'Image-level visual review receipts')
    check('Observed geometry unchanged since visual freeze',all(sha(RUN/name)==digest for name,digest in observed['frozen_geometry_sha256'].items()),'builtin_visual_review_observed.json:frozen_geometry_sha256')
    sources=read(RUN/'sources/source_receipts.json')
    check('Pinned official wrist K and capture evidence',len([row for row in sources if row['role']=='calibration'])==4 and all(len(row['revision'])==40 for row in sources),'sources/source_receipts.json')
    check('Pinned official source file hashes unchanged',all(sha(RUN/'sources'/row['role']/row['path'])==row['sha256'] for row in sources),'sources/source_receipts.json')
    check('Actual H3 checkpoint weight hash verified',sha(ROOT/'data/checkpoints/MolmoMotion-4B-H3-F30/model.pt')=='506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205','Pinned native checkpoint SHA256 from verified prior Hub download')
    for camera in ['wrist_2','wrist_1']:
        scene=RUN/camera;meta=read(scene/'metadata.json');indices=np.load(scene/'observed/source_indices.npy')
        check(camera+'50 distinct observed frames and H3 boundary',np.array_equal(indices,np.arange(77,127)) and meta['history_source_indices']==[124,125,126] and not meta['future_used_for_model_input'],'metadata.json; observed/source_indices.npy')
        check(camera+'native and aspect-restored default ViPE',all(read(scene/name)['status']=='COMPLETE' for name in ['vipe_default_receipt.json','vipe_rectified_default_receipt.json']),'vipe_*_receipt.json')
        rect=read(scene/'vipe_rectified_default_receipt.json')
        check(camera+'no temporal resampling or altered model RGB',not rect['temporal_resampling'] and not rect['original_MolmoMotion_RGB_changed'] and rect['source_real_frame_count']==50,'vipe_rectified_default_receipt.json')
        check(camera+'observed grounding/SAM/AllTracker and UniDepth',all((scene/name).exists() for name in ['observed/molmopoint_grounding.json','observed/mask.png','observed/historical_masks_h3.npy','observed/observed_tracks_2d.npz','geometry/unidepth_K.npy']) and np.load(scene/'geometry/unidepth_K.npy').shape==(8,3,3),'Observed author model outputs')
        check(camera+'calibration estimates and heldout uncertainty disclosed',all((scene/'geometry'/name).exists() for name in ['hand_eye.json','hand_eye_bootstrap.json','independent_robot_rgbd_hand_eye.json','independent_robot_rgbd_bootstrap.json','native_square_control.json']),'geometry/ hand-eye and control audits')
        depth=read(scene/'geometry/depth_comparison.json')
        check(camera+'regional depth comparison and temporal variation',len(depth['rows'])==16 and all(name in depth['temporal_variation'] for name in ['target','robot_approximate','static_table_bin_approximate','background']),'geometry/depth_comparison.json')
        selection=read(scene/'geometry/branch_selection.json');ids=np.load(scene/'observed/selected_point_ids.npy')
        check(camera+'24 same deterministic points across A/B/C',len(ids)==24 and len(set(ids))==24 and selection['common_point_ids']==ids.tolist(),'observed/selected_point_ids.npy; geometry/branch_selection.json')
        tracking=np.load(scene/'observed/observed_tracks_2d.npz');masks=np.load(scene/'observed/historical_masks_h3.npy')
        inside=[]
        for t in range(3):
            xy=np.rint(tracking['tracks'][-3+t,ids]).astype(int);inside.append(masks[t,xy[:,1],xy[:,0]].astype(bool))
        filters=[]
        for name in selection['branches']:
            f=np.load(scene/'geometry'/name/'filter_diagnostics.npz')
            filters.append(f['valid'][-3:,ids].all() and f['keep'][ids].all() and (f['depth_spread'][-3:,ids]<.03).all() and (f['depth_coverage'][-3:,ids]>=13/25).all())
        check(camera+'all selected H3 points supported by masks/visibility/depth/author filters',np.array(inside).all() and tracking['visibility'][-3:,ids].all() and all(filters),'Historical SAM and filter_diagnostics.npz for all three geometries')
        check(camera+'primary selected before future with physical limitations',selection['primary']=='C_sensor_tcp_official' and selection['future_used'] is False and not selection['physical_geometry_certified'],'branch_selection.json; builtin observed review')
        branch_names=list(selection['branches']) if camera=='wrist_2' else ['C_sensor_tcp_official']
        for name in branch_names:
            branch=scene/'branches'/name;groupaudit=validate_group_payloads(branch)
            hist=np.load(branch/'observed/points_3d_history.npy');K=np.load(branch/'geometry/K.npy');poses=np.load(branch/'geometry/camera_poses.npy')
            if K.ndim==2:K=np.repeat(K[None],50,axis=0)
            uv=np.load(branch/'observed/points_2d_history.npy')
            err=max(np.linalg.norm(project(from_anchor(hist[t],poses[-3+t],poses[-1]),K[-3+t])-uv[t],axis=-1).max() for t in range(3))
            check(camera+'/'+name+'correct moving H3 coordinates',hist.shape==(3,24,3) and np.isfinite(hist).all() and (hist[...,2]>0).all() and err<.02,'moving-camera roundtrip and group payload audit; implementation integrity only')
            freeze=read(branch/'predictions/input_freeze.json')
            check(camera+'/'+name+'model inputs unchanged',all(sha(branch/key)==digest for key,digest in freeze['sha256'].items()),'predictions/input_freeze.json')
            run=read(branch/'predictions/model_run.json')
            check(camera+'/'+name+'three complete native H3 calls',run['success'] and run['successful_chunks']==3 and run['attempted_chunks']==3 and run['prediction_shape']==[24,30,3] and run['seed']==0 and run['dtype']=='bfloat16' and run['checkpoint_revision']=='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629','predictions/model_run.json')
            combined=[]
            for g in range(3):
                folder=branch/'predictions'/f'group_{g:02d}';r=read(folder/'model_run.json');h=np.load(branch/'groups'/folder.name/'points_3d_history.npy')
                raw=(folder/'raw_model_output.txt').read_text();parsed=strict_parse(raw,3,30);stored=np.load(folder/'future_3d.npy');anchor=np.load(folder/'anchor.npy')
                inputs=read(folder/'input_hashes.json');payload=torch.load(folder/'processor_inputs.pt',map_location='cpu',weights_only=False)
                valid=r['success'] and r['generation_started'] and r['prediction_seconds']>0 and r['parse_status']=='FULL_8x30x3'
                valid &= np.array_equal(anchor,h[-1,0]) and np.allclose(stored,parsed+anchor,atol=1e-4) and stored.shape==(8,30,3) and np.isfinite(stored).all()
                valid &= np.array_equal(payload['anchor_3d'].float().numpy().reshape(3),anchor) and inputs['processor_inputs_sha256']==sha(folder/'processor_inputs.pt')
                valid &= inputs['source_history_indices']==[124,125,126] and not inputs['future_used'] and np.array_equal(np.array(inputs['actual_xyz'],dtype=np.float32),h)
                valid &= r['raw_output_sha256']==sha(folder/'raw_model_output.txt') and r['prediction_sha256']==sha(folder/'future_3d.npy')
                check(camera+'/'+name+'/'+folder.name+'strict raw parser and actual payload',valid,str(folder.relative_to(RUN)))
                calls.append({'camera':camera,'geometry':name,'group':folder.name,'seconds':r['prediction_seconds'],'peak_cuda_allocated_gib':r['peak_cuda_allocated_gib'],'raw_sha256':r['raw_output_sha256']});combined.append(stored)
            check(camera+'/'+name+'combined24x30 exact',np.array_equal(np.concatenate(combined),np.load(branch/'predictions/future_3d.npy')),'predictions/future_3d.npy')
        metrics=read(scene/'evaluation/metrics.json');reference=np.load(scene/'evaluation/future_reference.npz');tracker=np.load(scene/'evaluation/future_alltracker.npz')
        export=read(scene/'evaluation/export_receipt.json')
        check(camera+'same future IDs opened after complete primary forecasts',np.array_equal(tracker['selected_ids'],ids) and export['source_indices']==list(range(126,147)) and all(sha(Path(row['path']))==row['sha256'] for row in export['prediction_receipts']),'evaluation/export_receipt.json; future_alltracker.npz')
        check(camera+'moving future camera and nominal XYZ time alignment',metrics['future_camera_motion_used_for_evaluation_only'] and np.allclose(reference['times'],np.arange(1,21)/10) and metrics['common_reference_geometry']=='C_sensor_tcp_official','evaluation/metrics.json; future_reference.npz')
        check(camera+'common masks and no off-image removal',metrics['outside_predictions_excluded'] is False and len(set(row['3D_est_m']['valid_samples'] for row in metrics['methods'].values()))==1 and len(set(row['2D_px']['valid_samples'] for row in metrics['methods'].values()))==1,'evaluation/metrics.json')
        check(camera+'ADE/FDE/time/perpoint/baselines/uncertainty',all(row['3D_est_m']['ADE'] is not None and len(row['3D_est_m']['per_time_mean'])==20 and len(row['3D_est_m']['per_point_mean'])==24 for row in metrics['methods'].values()) and any(name.endswith('/Static') for name in metrics['methods']) and any(name.endswith('/CV') for name in metrics['methods']) and (scene/'evaluation/hand_eye_reference_sensitivity.json').exists(),'evaluation/metrics.json and hand_eye_reference_sensitivity.json')
        for filename in ['geometry_forecasts_side_by_side.mp4','primary_prediction_vs_real.mp4']:
            path=scene/'viz'/filename;video=cv2.VideoCapture(str(path));count=int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=video.get(cv2.CAP_PROP_FPS);ok,first=video.read();video.set(cv2.CAP_PROP_POS_FRAMES,19);last_ok,last=video.read();video.release()
            check(camera+'/'+filename+'complete playable video',count==20 and abs(fps-10)<.01 and ok and last_ok,{'frames':count,'fps':fps,'sha256':sha(path)})
        check(camera+'required report figures exist',all((scene/'viz'/name).exists() for name in ['observed_prefix_review.png','registration_review.png','pose_K_comparison.png','camera_paths_3d.png','depth_comparison.png','candidates_to_selected.png','selected_history_tracks.png','history_3d_clouds.png','intrinsics_observed_frames.png','reference_all24_review.png','forecast_errors.png','full_30step_xyz.png','reference_hand_eye_sensitivity.png']),'viz/*.png')
    cross=read(RUN/'crossview_observed_audit.json');side=read(RUN/'side_future_crosscheck.json')
    check('Two wrist and two side cross-checks disclose disagreement',len(cross['view_visibility'])==4 and len(side['rows'])==2 and not cross['four_camera_exact_triangulation_used'] and not cross['hardware_synchronization_confirmed'],'crossview_observed_audit.json; side_future_crosscheck.json')
    check('Independent wrist world-motion replication',read(RUN/'wrist_future_world_motion.json')['future_used_for_evaluation_only'] and (RUN/'wrist_future_world_motion.png').exists(),'wrist_future_world_motion.json')
    gallery=read(RUN/'video_gallery_receipt.json')
    check('Local video gallery ready with unchanged HTML',
          sha(RUN/'index.html')==gallery['index_sha256'] and gallery['real_future_includes_t0'] and gallery['source_indices']==list(range(126,147)),
          'index.html; video_gallery_receipt.json')
    for camera in ['wrist_2','wrist_1','side_1','side_2']:
        path=RUN/camera/'viz/real_future.mp4';video=cv2.VideoCapture(str(path))
        count=int(video.get(cv2.CAP_PROP_FRAME_COUNT));fps=video.get(cv2.CAP_PROP_FPS);ok,first=video.read()
        video.set(cv2.CAP_PROP_POS_FRAMES,20);last_ok,last=video.read();video.release()
        check(camera+'real future video t0 through2s playable',count==21 and abs(fps-10)<.01 and ok and last_ok,
              {'frames':count,'fps':fps,'sha256':sha(path)})
    prior=read(ROOT/'runs/fmb_v2_berkeley_matched/preserved_v1.json')
    check('FMBv1 914 preserved files',len(prior['sha256'])==914 and all(sha(ROOT/name)==digest for name,digest in prior['sha256'].items()),'runs/fmb_v2_berkeley_matched/preserved_v1.json')
    preservation=read(RUN/'preserved_prior_runs.json')
    check('FMBv2 preserved audit and report',sha(ROOT/'runs/fmb_v2_berkeley_matched/completion_audit.json')==preservation['v2_completion_audit_sha256'] and sha(ROOT/'report/fmb_v2_berkeley_matched.md')==preservation['v2_report_sha256'],'preserved_prior_runs.json')
    report=ROOT/'report/fmb_wrist_v3.md';text=report.read_text(encoding='utf-8')
    local_links=[url for url in re.findall(r'\]\(([^)]+)\)',text) if not url.startswith(('http','https'))]
    check('Report figures/videos/evidence links resolve',all((report.parent/url).resolve().is_file() for url in local_links),'report/fmb_wrist_v3.md local links')
    check('12 actual complete P8 model calls',len(calls)==12,{'main_wrist2':9,'replication_wrist1':3})
    interrupted=RUN/'wrist_2/branches/A_vipe_full/predictions/interrupted_resource_attempt_00'
    interruption=read(interrupted/'interruption_receipt.json')
    check('Resource-interrupted extra attempt disclosed and preserved',interruption['generation_started'] and interruption['retry_same_frozen_inputs'] and not interruption['external_process_terminated'] and not(interrupted/'group_00/raw_model_output.txt').exists(),'predictions/interrupted_resource_attempt_00/')
    write(RUN/'completion_audit.json',{'success':True,'completed_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'native_model_calls':calls,
          'total_generation_attempts':13,'successful_generation_attempts':12,'interrupted_attempts':1,
          'physical_geometry_certified':False,'experimental_unit':'one source-matched episode; 24 point paths are not independent episodes',
          'interpretation':'Requested controlled experiment completed, including failed geometry/cross-view validations; no calibrated physical GT or successful four-view fusion claimed.',
          'report_sha256':sha(report),'audit_script_sha256':sha(Path(__file__))})
    write(RUN/'requirements_progress.json',{'status':'COMPLETE','report_sha256':sha(report),'completion_audit_required':True,'completion_audit_passed':True,
          'successful_native_calls':12,'built_in_image_reviews_complete':True,'physical_geometry_certified':False})
    pipeline=read(RUN/'forecast_pipeline_status.json');pipeline['builtin_future_review_pending']=False
    pipeline['completion_audit_passed']=True;write(RUN/'forecast_pipeline_status.json',pipeline)
    print('COMPLETE audit:',len(checks),'checks,',len(calls),'complete actual H3 model calls',flush=True)

if __name__=='__main__':audit()
