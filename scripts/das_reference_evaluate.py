"""Compare actual reference-assisted DaS outputs, a matched ablation, and real.

Intermediate real frames are opened only here, after successful generation.
Metrics measure reconstruction with a known endpoint, not predictive accuracy.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from das_prepare_control import sha256, write_json, save_video, sheet
from das_evaluate import interpolate, metrics
from das_quality import temporal_warp_error, cup_appearance
from das_robot_evaluate import frames, tracker, valid, tiled_all, labeled

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_reference_repair';OLD=SCENE/'das_robot_cup'

def preview(name,video):
    tiled_all(OUT/name/'all_49_generated_frames.png',video)
    sheet(OUT/name/'generated_contact_sheet.png',[video],['ACTUAL DaS DIFFUSION OUTPUT'])
    Image.fromarray(video[16]).save(OUT/name/'generated_at_2s.png')
    for first in range(0,49,7):
        sheet(OUT/name/f'review_frames_{first:02d}_{min(first+6,48):02d}.png',[video],['ACTUAL GENERATED'],indices=tuple(range(first,min(first+7,49))),tile=(400,300))

def main():
    p=argparse.ArgumentParser();p.add_argument('--preview-only',action='store_true');a=p.parse_args()
    names=[p.parent.name for p in sorted(OUT.glob('*/generated_seed42.mp4'))]
    assert names,'No actual generated clips'
    gen={}
    for name in names:
        assert json.loads((OUT/name/'resource_usage.json').read_text())['success']
        f=frames(OUT/name/'generated_seed42.mp4');assert f.shape==(49,480,720,3)
        gen[name]=np.array([cv2.resize(v,(640,480)) for v in f]);preview(name,gen[name])
    if a.preview_only:return
    chosen='guided_endpoint_background';names=['endpoint_without_rgb_prior',chosen,'no_trajectory_control']
    assert all(name in gen for name in names),'Matched ablation required'
    gen={name:gen[name] for name in names}
    prepared=OUT/'refined_background'
    prep=json.loads((prepared/'preparation.json').read_text());assert prep['future_used'] and not prep['intermediate_future_frames_used']
    assert sha256(prepared/'guide_720x480.npz')==prep['guide_sha256'] and sha256(prepared/'control_endpoint_720x480.mp4')==prep['control_sha256']
    assert prep['control_sha256']==json.loads((OUT/'preparation.json').read_text())['control_sha256'],'Ablation must use identical control'
    assert all(sha256(SCENE/path)==digest for path,digest in prep['input_sha256'].items())
    old_path=OLD/'robot_coupled_initial_only/generated_molmomotion_seed42.mp4'
    old=frames(old_path);gen['previous_initial_only']=np.array([cv2.resize(f,(640,480)) for f in old])
    reference=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    control=np.load(OLD/'joint_control.npz');cup_uv=np.load(SCENE/'observed/points_2d_history.npy')[-1].astype(float)
    uv=np.concatenate([cup_uv,control['robot_query_uv']]);n_cup=len(cup_uv)
    tracked={}
    for name in names:
        path=OUT/name/'joint_tracking.npz'
        result=tracker(np.concatenate([reference[None],gen[name][:17]]),uv,path,
            {'video_sha256':sha256(OUT/name/'generated_seed42.mp4'),'future_reference_used_in_generation':True,
             'independent_tracker':True,'extra_observed_anchor':True})
        tracked[name]={'xy':result['tracks'][1:],'vis':result['visibility'][1:]}
        write_json(OUT/name/'cup_appearance.json',cup_appearance(gen[name][:17],tracked[name]['xy'][:,:n_cup],tracked[name]['vis'][:,:n_cup],reference,
            np.array(Image.open(SCENE/'observed/mask.png'))>0))
    old_tracking=np.load(OLD/'robot_coupled_initial_only/joint_tracking.npz')
    assert np.allclose(old_tracking['query_uv'],uv)
    tracked['previous_initial_only']={'xy':old_tracking['tracks'][1:],'vis':old_tracking['visibility'][1:]}
    real_track=np.load(OLD/'real_joint_tracking.npz');assert np.allclose(real_track['query_uv'],uv)
    real_xy=real_track['tracks'][1:];real_vis=real_track['visibility'][1:]
    future=np.load(SCENE/'evaluation/future_rgb.npy');assert future.shape==(10,480,640,3)
    real=np.concatenate([reference[None],future]);times=np.arange(1,11)/5
    flow=np.load(prepared/'endpoint_flow.npz');rounded=np.rint(uv).astype(int);deltas=flow['flow0'][rounded[:,1],rounded[:,0]]
    # Right arm queries are static; only the segmented moving left chain moves.
    moving=flow['fg0'][rounded[:,1],rounded[:,0]];deltas[~moving]=0
    phase=np.interp(times,flow['times'],flow['phase']);target=uv[None]+phase[:,None,None]*deltas[None]
    lo=np.floor(times*8).astype(int);hi=np.ceil(times*8).astype(int)
    samples={name:{'xy':interpolate(t['xy'],np.arange(17)/8,times),'vis':t['vis'][lo]&t['vis'][hi]} for name,t in tracked.items()}
    pair_common=valid(target).copy()
    for name in [chosen,'endpoint_without_rgb_prior']:pair_common&=samples[name]['vis']&valid(samples[name]['xy'])
    all_common=pair_common&samples['previous_initial_only']['vis']&valid(samples['previous_initial_only']['xy'])
    native_common=valid(target).copy()
    for name in ['endpoint_without_rgb_prior','no_trajectory_control']:native_common&=samples[name]['vis']&valid(samples[name]['xy'])
    four_common=all_common&samples['no_trajectory_control']['vis']&valid(samples['no_trajectory_control']['xy'])
    roi0=flow['fg0'];moving_old=np.load(OLD/'robot_coupled/background_completion.npz')['moving_mask']
    masks=[]
    yy,xx=np.indices(roi0.shape,dtype=np.float32)
    for a,b,ph in zip(lo,hi,phase):
        # Conservative union includes the old forecast, new trajectory, and
        # original cup site. Every video is measured on the identical ROI.
        current=cv2.warpAffine(roi0.astype(np.uint8),np.array([[1,0,prep['cup_shift_px'][0]*ph],[0,1,prep['cup_shift_px'][1]*ph]],np.float32),(640,480))>0
        mask=roi0|current|flow['fg1']|moving_old[a]|moving_old[b]
        masks.append(cv2.dilate(mask.astype(np.uint8),np.ones((11,11),np.uint8))>0)
    masks=np.array(masks);report={'future_used':True,'chosen_variant':chosen,'interpretation':'Endpoint-assisted synthesis/reconstruction; not causal forecast accuracy',
        'time_s':times.tolist(),'endpoint_seen_at_s':2.,'withheld_intermediate_evaluation_time_s':times[:-1].tolist(),
        'coordinate_system':'Native 640x480; linear sample of 8Hz generated tracks to native 5Hz',
        'tracker_limitation':'Visibility and low ADE do not certify semantic identity; all-frame visual review is required',
        'three_way_mask_variants':['endpoint_without_rgb_prior',chosen,'previous_initial_only'],
        'native_control_ablation_variants':['endpoint_without_rgb_prior','no_trajectory_control'],
        'shared_ROI_fraction':float(masks.mean()),'variants':{}}
    for name,video in gen.items():
        xy=samples[name]['xy'];v=samples[name]['vis']&valid(xy)
        aligned=np.array([(video[a].astype(float)*(1-(t*8-a))+video[b].astype(float)*(t*8-a)).astype(np.uint8) for t,a,b in zip(times,lo,hi)])
        error=np.abs(aligned.astype(float)-future.astype(float));mse=((aligned.astype(float)-future.astype(float))**2).mean()
        desc={'cup_to_endpoint_control':metrics(target[:,:n_cup],xy[:,:n_cup],v[:,:n_cup]&valid(target[:,:n_cup])),
            'cup_to_real_individual_visibility':metrics(real_xy[:,:n_cup],xy[:,:n_cup],v[:,:n_cup]&real_vis[:,:n_cup]),
            'three_way_common_cup_to_real':metrics(real_xy[:,:n_cup],xy[:,:n_cup],all_common[:,:n_cup]&real_vis[:,:n_cup]),
            'three_way_common_cup_to_endpoint_control':metrics(target[:,:n_cup],xy[:,:n_cup],all_common[:,:n_cup]),
            'three_way_common_robot_to_real':metrics(real_xy[:,n_cup:],xy[:,n_cup:],all_common[:,n_cup:]&real_vis[:,n_cup:]),
            'image_quality':{'full_frame_PSNR_dB':float(10*np.log10(255**2/max(mse,1e-10))),'full_frame_MAE_0_255':float(error.mean()),
                'shared_ROI_MAE_0_255':float(error[masks].mean()),
                'withheld_intermediate_ROI_MAE_0_255':float(error[:-1][masks[:-1]].mean()),
                'withheld_intermediate_full_frame_PSNR_dB':float(10*np.log10(255**2/max(((aligned[:-1].astype(float)-future[:-1].astype(float))**2).mean(),1e-10))),
                'laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for f in aligned])),
                'temporal_5Hz_shared_ROI_warp':temporal_warp_error(aligned,masks)},
            'source_video_sha256':sha256(old_path if name=='previous_initial_only' else OUT/name/'generated_seed42.mp4')}
        desc['withheld_intermediate_three_way_cup_to_real']=metrics(real_xy[:-1,:n_cup],xy[:-1,:n_cup],all_common[:-1,:n_cup]&real_vis[:-1,:n_cup])
        if name in [chosen,'endpoint_without_rgb_prior']:
            desc['matched_pair_common_cup_to_control']=metrics(target[:,:n_cup],xy[:,:n_cup],pair_common[:,:n_cup])
            desc['matched_pair_common_cup_to_real']=metrics(real_xy[:,:n_cup],xy[:,:n_cup],pair_common[:,:n_cup]&real_vis[:,:n_cup])
        if name in ['endpoint_without_rgb_prior','no_trajectory_control']:
            desc['native_control_ablation_cup_to_target']=metrics(target[:,:n_cup],xy[:,:n_cup],native_common[:,:n_cup])
            desc['native_control_ablation_cup_to_real']=metrics(real_xy[:,:n_cup],xy[:,:n_cup],native_common[:,:n_cup]&real_vis[:,:n_cup])
        if name=='no_trajectory_control':
            for key in [k for k in desc if 'three_way' in k]:del desc[key]
        desc['four_way_common_cup_to_real']=metrics(real_xy[:,:n_cup],xy[:,:n_cup],four_common[:,:n_cup]&real_vis[:,:n_cup])
        desc['four_way_common_cup_to_endpoint_control']=metrics(target[:,:n_cup],xy[:,:n_cup],four_common[:,:n_cup])
        np.savez_compressed(OUT/f'aligned_{name}.npz',times=times,xy=xy,visibility=v,endpoint_control=target,real_xy=real_xy,real_visibility=real_vis,
            three_way_mask=all_common,matched_pair_mask=pair_common,native_control_ablation_mask=native_common,four_way_mask=four_common)
        report['variants'][name]=desc
    report['real_image_quality']={'laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for f in future])),
        'temporal_5Hz_shared_ROI_warp':temporal_warp_error(future,masks)}
    report['provenance']={'real_rgb_sha256':sha256(SCENE/'evaluation/future_rgb.npy'),'real_tracking_sha256':sha256(OLD/'real_joint_tracking.npz'),
        'old_tracking_sha256':sha256(OLD/'robot_coupled_initial_only/joint_tracking.npz'),'endpoint_flow_sha256':sha256(prepared/'endpoint_flow.npz')}
    write_json(OUT/'comparison_metrics.json',report)
    columns=[gen['previous_initial_only'],gen['endpoint_without_rgb_prior'],gen[chosen]]
    labels=['PREVIOUS: MolmoMotion control','ENDPOINT: without RGB prior','ENDPOINT + RGB prior: actual DaS']
    comparison=[]
    for i in range(17):
        ri=int(np.clip(np.rint(i/8*5),0,10))
        comparison.append(np.concatenate([np.concatenate([labeled(columns[0][i],labels[0]),labeled(columns[1][i],labels[1])],axis=1),
            np.concatenate([labeled(columns[2][i],labels[2]),labeled(real[ri],f'REAL nearest {ri/5:.2f}s')],axis=1)],axis=0))
    save_video(OUT/'old_ablation_new_real_2s.mp4',comparison)
    save_video(OUT/'before_after_full_6s.mp4',[np.concatenate([labeled(a,f'PREVIOUS {i/8:.3f}s'),labeled(b,f'NEW with endpoint reference {i/8:.3f}s')],axis=1)
        for i,(a,b) in enumerate(zip(columns[0],columns[2]))])
    sheet(OUT/'comparison_exact_times.png',[c[[0,8,16]] for c in columns]+[real[[0,5,10]]],labels+['REAL'],indices=(0,1,2),fps=1,tile=(640,480))
    save_video(OUT/'native_control_vs_no_control_2s.mp4',[np.concatenate([labeled(gen['endpoint_without_rgb_prior'][i],f'DaS endpoint CONTROL {i/8:.3f}s'),
        labeled(gen['no_trajectory_control'][i],f'DaS NO CONTROL {i/8:.3f}s')],axis=1) for i in range(17)])
    sheet(OUT/'native_control_ablation_exact_times.png',[gen['endpoint_without_rgb_prior'][[0,8,16]],gen['no_trajectory_control'][[0,8,16]],real[[0,5,10]]],
        ['NATIVE DaS WITH ENDPOINT CONTROL','NATIVE DaS WITHOUT CONTROL','REAL'],indices=(0,1,2),fps=1,tile=(640,480))
    print(json.dumps({n:{'quality':v['image_quality'],'common_cup':v['four_way_common_cup_to_real']} for n,v in report['variants'].items()},indent=2))

if __name__=='__main__':main()
