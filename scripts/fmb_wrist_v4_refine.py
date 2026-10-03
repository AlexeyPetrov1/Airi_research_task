"""Compare rigid-centroid translation with full TCP-pose extrapolation on observed data."""
import json
from datetime import datetime,timezone
import numpy as np
from fmb_wrist_prepare import write,sha
from fmb_wrist_math import transform
from fmb_wrist_v4_diagnose import OUT
from fmb_wrist_v4_prepare import forecast_tcp,MODEL_TIMES

def centroid_forecast(tcp,x,template,window,tau,times):
    w=min(window,len(tcp));poses=tcp[-w:]@x
    dt=np.arange(-w+1,1)/10
    centers=np.stack([np.nanmean(transform(p,template),axis=0) for p in poses])
    velocity=(dt[:,None]*(centers-centers[-1])).sum(0)/(dt@dt)
    factor=times if tau is None else tau*(-np.expm1(-times/tau))
    initial=transform(poses[-1],template)
    world=initial[:,None]+factor[None,:,None]*velocity[None,None]
    return transform(np.linalg.inv(poses[-1]),world)

def predict(r,anchor,window,tau,times,family):
    tcp=r['tcp'][:anchor+1];x=r['X'];template=r['camera_points'][anchor]
    if family=='rigid_centroid_translation':return centroid_forecast(tcp,x,template,window,tau,times)
    p=forecast_tcp(tcp,window,tau,times)
    return np.stack([transform(np.linalg.inv(tcp[-1]@x)@(g@x),template) for g in p],axis=1)

def run():
    assert not (OUT/'motion_policy_amendment_v2.json').exists(),'Keep the corrected follow-up policy selection immutable'
    refs={camera:np.load(OUT/camera/'observed_reference.npz') for camera in ['wrist_2','wrist_1']}
    times=np.arange(1,21)/10;candidates=[]
    for family in ['full_TCP_pose','rigid_centroid_translation']:
        for window in [3,5,8,12,20]:
            for tau in [.15,.4,.8,1.6,None]:
                errors=[]
                for camera,r in refs.items():
                    for anchor in range(19,30):
                        pred=predict(r,anchor,window,tau,times,family)
                        truth=np.stack([transform(np.linalg.inv(r['tcp'][anchor]@r['X'])@(r['tcp'][i]@r['X']),r['camera_points'][i])
                                         for i in range(anchor+1,anchor+21)],axis=1)
                        valid=r['valid'][anchor][:,None]&r['valid'][anchor+1:anchor+21].T
                        dist=np.linalg.norm(pred-truth,axis=-1);valid &= np.isfinite(dist)
                        errors.extend(dist[valid].tolist())
                assert errors,'A candidate has no valid observed validation samples; do not treat this as a completed comparison'
                candidates.append({'family':family,'window':window,'damping_tau_s':tau,'observed_target_ADE_m':float(np.mean(errors)),'valid_observed_validation_samples':len(errors)})
    winner=min(candidates,key=lambda r:r['observed_target_ADE_m'])
    for camera,r in refs.items():
        forecast=predict(r,49,winner['window'],winner['damping_tau_s'],MODEL_TIMES,winner['family'])
        np.save(OUT/camera/'selected_motion_policy_v2_15hz.npy',forecast.astype(np.float32))
    write(OUT/'motion_policy_amendment_v2.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'corrects_preserved_attempt':'motion_policy_amendment.json: centroid averaging propagated NaNs from ineligible historical points, so that family had no valid validation samples; use finite observed members, retain original receipt, and require nonempty validation for every candidate',
          'reason':'First built-in provisional comparison still shows endpoint drift despite lower ADE. Compare a simpler rigid-centroid motion forecast against full TCP rotation/lever-arm extrapolation.',
          'first_stage_future_scores_already_known':True,'new_parameters_selected_from_future':False,
          'selection_data':'Only observed77..126 target points and TCP, fixed original K/X; no127..146 files opened by this selector',
          'candidates':candidates,'winner':winner,'validation_anchors_source_indices':list(range(96,107)),
          'caveat':'Exploratory iteration after seeing stage1 outcomes. Overlapping observed validation and the fixed full-prefix calibration do not constitute a new independent test.',
          'forecast_hashes':{camera:sha(OUT/camera/'selected_motion_policy_v2_15hz.npy') for camera in refs}})
    print('Observed target-motion policy selection',winner,flush=True)

if __name__=='__main__':run()
