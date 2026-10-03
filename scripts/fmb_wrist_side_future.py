"""Evaluation-only side controls: partial-contour caveats and motion magnitude."""
import json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import RUN,write
from fmb_wrist_crossview import rgb_masks,scene_points
from fmb_wrist_evaluate import prediction_gate,TIMES
from fmb_wrist_math import transform
import berkeley_preprocess as b

def run():
    prediction_gate();rows=[];magnitudes={};camera='wrist_2'
    ref=np.load(RUN/camera/'evaluation/future_reference.npz');gt=ref['GT_3D_est'];valid=ref['common_mask3d']
    # Reference displacement is relative to sensor-supported observed t0 points, not a model forecast.
    future=np.load(RUN/camera/'evaluation/future_alltracker.npz');depth=np.load(RUN/camera/'evaluation/future_sensor_depth.npy')
    from fmb_wrist_math import sample_z,backproject
    z,_,_=sample_z(depth[0],future['tracks'][0]);start=backproject(future['tracks'][0],z,ref['K'])
    wrist_amp=np.array([np.nanmedian(np.linalg.norm(gt[:,t]-start,axis=-1)[valid[:,t]]) for t in range(20)])
    for side in ['side_1','side_2']:
        scene=RUN/side;rgb=np.load(scene/'evaluation/future_rgb.npy');depth=np.load(scene/'evaluation/future_sensor_depth.npy')
        candidate=json.loads((RUN/'sources/official_K_candidates.json').read_text())[side]
        K=np.array(next(row for row in candidate if row['width']==640 and row['height']==480)['K_256_resize'])
        centers=[];areas=[];panels=[];touching=[]
        for t,(frame,z) in enumerate(zip(rgb,depth)):
            target,_=rgb_masks(frame);pts=scene_points(frame,z,K,target);areas.append(len(pts))
            centers.append(np.median(pts,0) if len(pts)>=20 else np.full(3,np.nan))
            touching.append(bool(target[0].any() or target[-1].any() or target[:,0].any() or target[:,-1].any()))
            if t in [0,1,10,20]:
                tile=frame.copy();tile[target]=(tile[target]*.6+np.array([30,255,50])*.4).astype(np.uint8);tile=cv2.resize(tile,(512,512))
                source_index=int(np.load(scene/'evaluation/source_indices.npy')[t])
                cv2.putText(tile,f'{side} source {source_index}, pixels {len(pts)}',(8,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2);panels.append(tile)
        centers=np.array(centers);disp=centers[1:]-centers[0];magnitude=np.linalg.norm(disp,axis=-1);magnitudes[side]=magnitude
        np.save(scene/'evaluation/target_color_centroids.npy',centers)
        b.save_rgb(scene/'viz/side_future_color_control.png',np.hstack(panels))
        alignment=scene/'geometry/estimated_side_c2base.npy'
        if alignment.exists():
            world=transform(np.load(alignment),centers);np.save(scene/'evaluation/estimated_base_target_centroids.npy',world)
        rows.append({'camera':side,'valid_centroid_frames':int(np.isfinite(centers).all(-1).sum()),'visible_valid_target_pixels':areas,
                     'mask_touches_border_per_frame':touching,'future_endpoint_displacement_m':float(magnitude[-1]) if np.isfinite(magnitude[-1]) else None,
                     'median_abs_motion_magnitude_disagreement_with_wrist_m':float(np.nanmedian(np.abs(magnitude-wrist_amp))),
                     'interpretation':'Approximate whole-visible-color centroid, not same material points. Partial contours, surface bias, official-K linkage and asynchronous frames affect agreement.',
                     'future_used_for_evaluation_only':True,'calibration_fit_uses_future':False})
    fig,ax=plt.subplots(figsize=(9,5));ax.plot(TIMES,wrist_amp*1000,'k-',label='wrist_2 fixed point IDs: median displacement')
    for side,magnitude in magnitudes.items():ax.plot(TIMES,magnitude*1000,label=side+' visible-color centroid')
    ax.set(xlabel='future nominal time (s)',ylabel='3D_est displacement magnitude (mm)');ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(RUN/'side_future_motion_crosscheck.png',dpi=160);plt.close(fig)
    write(RUN/'side_future_crosscheck.json',{'rows':rows,'exact_triangulation_used':False,'hardware_synced':False,
          'reference_bias':'Wrist material point trajectories versus side visible-surface centroids; descriptive cross-view control only'})
    print('Side future controls',rows,flush=True)

if __name__=='__main__':run()
