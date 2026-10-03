"""Verify the full amplitude/phase experiment independently from its renderer."""
from datetime import datetime, timezone
import json
import re
import cv2
import numpy as np
from berkeley_arc_expansion import ROOT, DEST, BASE, OUT, sha, snapshot, scene_data
from berkeley_evaluate import ALIGNMENT, TIMES, project
from berkeley_baseline_comparison import scores
from berkeley_temporal_diagnostics import write


def main():
    main_protocol=json.loads((DEST/'protocol.json').read_text())
    phase_protocol=json.loads((DEST/'phase_schedule/protocol.json').read_text())
    for protocol in [main_protocol,phase_protocol]:
        assert protocol['source_sha256']=={**snapshot(BASE),**snapshot(OUT)}
    for relative,digest in phase_protocol['previous_run_sha256'].items():
        assert sha(ROOT/relative)==digest,relative
    assert sha(DEST/'phase_schedule/executed_experiment.py')==phase_protocol['executed_script_sha256']
    primary=json.loads((DEST/'results.json').read_text())
    secondary=json.loads((DEST/'phase_schedule/results.json').read_text())
    selection=json.loads((DEST/'selected/metadata.json').read_text())
    review=json.loads((DEST/'visual_review.json').read_text())
    assert review['completed'] and review['shared_candidate']==selection['variant']
    all_variants={**main_protocol['variants'],**phase_protocol['variants']}
    pairs=0;videos=0; early_checks=[];selected_metrics={}
    common_K=None
    for name in ('cup','bottle'):
        s,K,history,raw,_=scene_data(name)
        if common_K is not None:np.testing.assert_array_equal(K,common_K)
        common_K=K
        full={**dict(np.load(DEST/name/'full_30_steps.npz')),**dict(np.load(DEST/'phase_schedule'/name/'full_30_steps.npz'))}
        uv={**dict(np.load(DEST/name/'projected_2d.npz')),**dict(np.load(DEST/'phase_schedule'/name/'projected_2d.npz'))}
        for key,params in all_variants.items():
            steps=np.arange(1,31)/15
            if 'onset_s' in params:
                q=np.maximum(0,np.minimum(1,(steps-params['onset_s'])/(2-params['onset_s'])))
                blend=3*q**2-2*q**3
                alpha=1/3+(params['alpha_end']-1/3)*blend
            else:
                q=steps/2;blend=3*q**2-2*q**3
                alpha=np.full(30,params['alpha'])
            expected=history[-1,:,None]+alpha[None,:,None]*(raw-history[-1,:,None])
            expected[:,:,1]-=params['lift_m']*blend[None]
            np.testing.assert_allclose(full[key],expected,rtol=0,atol=1e-14)
            assert full[key].shape==(24,30,3) and np.isfinite(full[key]).all() and (full[key][...,2]>0).all()
            np.testing.assert_allclose(uv[key],project(expected[:,ALIGNMENT],K),rtol=0,atol=1e-12)
            recomputed=scores(uv[key],s,expected[:,ALIGNMENT],history[-1],K)
            saved=secondary[name][key] if key in secondary[name] else primary[name]['methods'][key]
            for metric in ['ADE_2D_px','FDE_2D_px','reference_visible_pair_coverage','prediction_pair_coverage']:
                np.testing.assert_allclose(recomputed[metric],saved[metric],rtol=0,atol=1e-10)
            pairs+=240
        key=selection['variant']
        np.testing.assert_array_equal(full[key][:,:15],full['reference_0333'][:,:15])
        np.testing.assert_allclose(full[key][:,-1],full['scale_036_lift005'][:,-1],atol=1e-14,rtol=0)
        np.testing.assert_array_equal(np.load(DEST/'selected'/name/'postprocessed_future_3d.npy'),full[key])
        np.testing.assert_array_equal(np.load(DEST/'selected'/name/'point_ids.npy'),s['ids'])
        np.testing.assert_array_equal(np.load(DEST/'selected'/name/'K.npy'),K)
        # Prove the actual late 2D projections, not only XYZ norms, are farther and higher.
        reference_uv=uv['reference_0333'];selected_uv=uv[key]
        baseline_distance=np.linalg.norm(reference_uv-s['uv0'][:,None],axis=-1)
        selected_distance=np.linalg.norm(selected_uv-s['uv0'][:,None],axis=-1)
        assert np.all(selected_uv[:,5:,1]<reference_uv[:,5:,1])
        assert np.all(selected_distance[:,5:]>baseline_distance[:,5:])
        np.testing.assert_array_equal(selected_uv[:,:5],reference_uv[:,:5])
        early_checks.append(name)
        selected_metrics[name]={k:secondary[name][key][k] for k in ['ADE_2D_px','FDE_2D_px']}
        for folder in [DEST/name,DEST/'phase_schedule'/name]:
            for video in folder.glob('*.mp4'):
                cap=cv2.VideoCapture(str(video));count=0
                fps=cap.get(cv2.CAP_PROP_FPS)
                while cap.read()[0]:count+=1
                cap.release()
                assert count==10 and abs(fps-5)<1e-6,video
                videos+=1
    assert videos==30 and pairs==7200
    # Validate the viewer's actual embedded coordinates and every referenced image.
    html=(DEST/'gallery_phase.html').read_text(encoding='utf8')
    match=re.search(r'const data=(.*);\nconst variant=',html)
    viewer=json.loads(match.group(1))
    assert len(viewer['variants'])==18 and viewer['default_variant']==selection['variant']
    for name in ('cup','bottle'):
        np.testing.assert_array_equal(viewer['scenes'][name]['ids'],selection['scenes'][name]['point_ids'])
        np.testing.assert_array_equal(viewer['scenes'][name]['uv'][selection['variant']],np.load(DEST/'selected'/name/'projected_2d.npy'))
        for t in range(10):
            assert cv2.imread(str(DEST/name/'frames'/f'{t:02d}.jpg')).shape==(480,640,3)
    write(DEST/'final_verification.json',dict(success=True,source_files_unchanged=len(main_protocol['source_sha256']),
        prior_constant_run_files_unchanged=len(phase_protocol['previous_run_sha256']),shared_K_exact=True,
        all_30_postprocessed_forecasts_reconstructed=True,forecast_shape=[24,30,3],candidate_reference_pairs_scored=pairs,
        all_secondary_ADE_FDE_recomputed=True,selected_early_frames_identical=early_checks,
        selected_late_240_pairs_farther_and_higher=True,selected_endpoint_same_as_constant_control=True,
        all_30_movies_fully_decoded=True,frames_per_movie=10,fps=5,viewer_arrays_and_20_real_images_verified=True,
        visual_review_complete=True,ML_calls=0,selected_metrics=selected_metrics,
        finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({'success':True,'movies':videos,'evaluated_pairs':pairs,'selected_metrics':selected_metrics}),flush=True)


if __name__=='__main__':main()
