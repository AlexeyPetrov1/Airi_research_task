"""Freeze neural ablations and observed-selected robot-aware forecasts before scoring."""
import json
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
from fmb_wrist_prepare import ROOT,write,sha
from fmb_wrist_math import transform
from fmb_wrist_v4_diagnose import BASE,OUT

MODEL_TIMES=np.arange(1,31)/15

def forecast_tcp(poses,window,tau,times):
    w=min(window,len(poses));p=poses[-w:]
    dt=np.arange(-w+1,1)/10
    # Least-squares velocity with the last observed pose fixed as intercept.
    v=(dt[:,None]*(p[:,:3,3]-p[-1,:3,3])).sum(0)/(dt@dt)
    relative=Rotation.from_matrix(p[-1,:3,:3].T@p[:,:3,:3]).as_rotvec()
    omega=(dt[:,None]*relative).sum(0)/(dt@dt)
    factor=times if tau is None else tau*(-np.expm1(-times/tau))
    out=np.repeat(p[-1][None],len(times),axis=0)
    out[:,:3,3]+=factor[:,None]*v
    out[:,:3,:3]=p[-1,:3,:3]@Rotation.from_rotvec(factor[:,None]*omega).as_matrix()
    return out

def run():
    assert not (OUT/'preregistration.json').exists(),'Never replace a frozen improvement experiment'
    ref=np.load(OUT/'wrist_2/observed_reference.npz');tcp=ref['tcp']
    candidates=[]
    # Every pseudo-future used for selection is already in observed77..126.
    times=np.arange(1,21)/10
    for w in [3,5,8,12,20]:
        for tau in [.15,.4,.8,1.6,None]:
            errors=[]
            for anchor in range(19,30):
                pred=forecast_tcp(tcp[:anchor+1],w,tau,times)
                truth=tcp[anchor+1:anchor+21]
                translation=np.linalg.norm(pred[:,:3,3]-truth[:,:3,3],axis=-1)
                rotation=Rotation.from_matrix(pred[:,:3,:3].transpose(0,2,1)@truth[:,:3,:3]).magnitude()
                errors.extend((translation+.1*rotation).tolist())
            candidates.append({'window':w,'damping_tau_s':tau,'observed_score_m':float(np.mean(errors))})
    winner=min(candidates,key=lambda x:x['observed_score_m'])
    write(OUT/'robot_observed_model_selection.json',{'candidates':candidates,'winner':winner,
          'validation_anchors_source_indices':list(range(96,107)),
          'validation_targets_max_source_index':126,'future_127_146_used':False,
          'score':'mean translation error + 0.1m times angular error in radians; 11 overlapping observed validation rollouts, not independent samples'})
    for camera in ['wrist_2','wrist_1']:
        scene=BASE/camera;dest=OUT/camera
        r=np.load(dest/'observed_reference.npz');x=r['X'];template=r['camera_points'][-1]
        future_tcp=forecast_tcp(r['tcp'],winner['window'],winner['damping_tau_s'],MODEL_TIMES)
        forecast=np.stack([transform(np.linalg.inv(r['tcp'][-1]@x)@(g@x),template) for g in future_tcp],axis=1)
        np.save(dest/'robot_attached_forecast_15hz.npy',forecast.astype(np.float32))
        # All native variants retain the same24 IDs and three original P8 groups.
        rgb=np.load(scene/'observed/history_rgb.npy');xy=np.load(scene/'observed/points_2d_history.npy')[-1]
        original=np.load(scene/'branches/C_sensor_tcp_official/observed/points_3d_history.npy')
        clean=np.load(dest/'denoised_rigid_H3.npy')
        action=json.loads((scene/'metadata.json').read_text())['instruction']
        for name,h,f,xyz in [('H3_rigid_observed',3,30,clean),('H1_native',1,32,original[-1:])]:
            folder=dest/'variants'/name;folder.mkdir(parents=True)
            np.save(folder/'history_xyz.npy',xyz);np.save(folder/'history_rgb.npy',rgb[-h:])
            np.save(folder/'query_uv.npy',xy);np.save(folder/'point_ids.npy',r['ids'])
            write(folder/'input_spec.json',{'camera':camera,'variant':name,'history':h,'horizon':f,
                  'action':action,'source_history_indices':[124,125,126][-h:],
                  'coordinates':'camera_t0; estimated meters; unchanged official K/frozen observed hand-eye X',
                  'input_change':'Replace noisy per-frame target coordinates by the same t0 rigid camera-attached template transformed with observed TCP' if h==3 else 'Official H1 checkpoint with original t0 XYZ and RGB; no historical velocity',
                  'future_used':False,'same_point_ids_and_group_order_as_v3':True,
                  'sha256':{p.name:sha(p) for p in folder.glob('*.npy')}})
    write(OUT/'preregistration.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),
          'objective':'Improve the actual point forecasts and distinguish native MolmoMotion changes from robot-aware hybrid gains',
          'cameras':['wrist_2','wrist_1'],'baseline':'Frozen v3 C native H3, Static and CV',
          'neural_variants':['H3_rigid_observed','H1_native'],'planned_native_P8_calls':12,
          'neural_model_selection_by_future_ADE':False,
          'robot_aware_forecast':winner,'robot_forecasts_use_future_TCP':False,
          'hybrid_rule':'Robot attached forecast plus shared median Molmo displacement residual, hard-clipped to the observed last10 camera-relative target p90 drift; preserves rigid point configuration. Report robot-only ablation and disclose that improvement may come entirely from kinematics.',
          'reference':'Reuse exactly frozen v3 independent AllTracker/sensor/TCP reference and common masks; no calibration or reference changes',
          'known_future_results':'v3 future already reviewed; this is an exploratory improvement study, not a new blind test',
          'quality_assessment':'Built-in assistant image inspection; no external Hugging Face assessment VLM',
          'scope_limits':'One previously studied episode, two viewpoints, overlapping observed validation; no claim of calibrated physical GT or generalization'})
    print('Frozen robot hyperparameters',winner,flush=True)
    print('Two native variants per view,12 calls; no future files read for preparation',flush=True)

if __name__=='__main__':run()
