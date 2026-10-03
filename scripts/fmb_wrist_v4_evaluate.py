"""Evaluate all frozen improvement variants on the unchanged v3 masks/reference."""
import argparse,json
from datetime import datetime,timezone
import numpy as np
from fmb_wrist_prepare import write,sha
from fmb_wrist_math import from_anchor,project
from fmb_wrist_evaluate import scores
from fmb_wrist_v4_diagnose import BASE,OUT

TIMES=np.arange(1,21)/10

def align(future,initial):
    source=np.r_[0,np.arange(1,future.shape[1]+1)/15]
    values=np.concatenate([initial[:,None],future],axis=1)
    return np.array([[np.interp(TIMES,source,values[n,:,a]) for a in range(3)] for n in range(24)]).transpose(0,2,1)

def run(robot_only=False):
    assert (OUT/'preregistration.json').exists()
    if not robot_only:assert json.loads((OUT/'inference_status.json').read_text())['status']=='COMPLETE'
    diagnosis=json.loads((OUT/'observed_diagnosis.json').read_text())
    for camera in ['wrist_2','wrist_1']:
        scene=BASE/camera;dest=OUT/camera
        refpath=scene/'evaluation/future_reference.npz';ref=np.load(refpath)
        mask3=ref['common_mask3d'];mask2=ref['common_mask2d'];gt3=ref['GT_3D_est'];gt2=ref['GT_2D_est'];poses=ref['camera_c2w'];k=ref['K']
        initial=np.load(scene/'branches/C_sensor_tcp_official/observed/points_3d_history.npy')[-1]
        methods={};starts={}
        for label,name in [('v3_Molmo_H3','C_sensor_tcp_official'),('v3_Static','C_sensor_tcp_official/Static'),('v3_CV','C_sensor_tcp_official/CV')]:
            if label=='v3_Molmo_H3':p=align(np.load(scene/'branches/C_sensor_tcp_official/predictions/future_3d.npy'),initial)
            elif label=='v3_Static':p=np.repeat(initial[:,None],20,axis=1)
            else:
                h=np.load(scene/'branches/C_sensor_tcp_official/observed/points_3d_history.npy')
                p=initial[:,None]+((h[-1]-h[0])/.2)[:,None]*TIMES[None,:,None]
            methods[label]=p;starts[label]=initial
        robot_init=np.load(dest/'observed_reference.npz')['camera_points'][-1]
        robot=align(np.load(dest/'robot_attached_forecast_15hz.npy'),robot_init)
        methods['Robot_attached_observed_selected']=robot;starts['Robot_attached_observed_selected']=robot_init
        if (OUT/'motion_policy_amendment_v2.json').exists():
            selected=align(np.load(dest/'selected_motion_policy_v2_15hz.npy'),robot_init)
            methods['Observed_selected_motion_policy']=selected;starts['Observed_selected_motion_policy']=robot_init
        if not robot_only:
            for name in ['H3_rigid_observed','H1_native']:
                folder=dest/'variants'/name;spec=json.loads((folder/'input_spec.json').read_text())
                assert all(sha(folder/p)==digest for p,digest in spec['sha256'].items())
                start=np.load(folder/'history_xyz.npy')[-1];p=align(np.load(folder/'future_3d.npy'),start)
                methods[name]=p;starts[name]=start
            # Original neural forecast retained; constrained correction is explicitly small.
            correction=np.median(methods['v3_Molmo_H3']-initial[:,None],axis=0)
            cap=diagnosis[camera]['camera_relative_target_drift_last10_p90_mm']/1000
            correction*=np.minimum(1,cap/np.maximum(np.linalg.norm(correction,axis=-1),1e-12))[:,None]
            methods['Robot_plus_bounded_Molmo']=robot+correction[None];starts['Robot_plus_bounded_Molmo']=robot_init
            if 'Observed_selected_motion_policy' in methods:
                methods['Selected_policy_plus_bounded_Molmo']=selected+correction[None];starts['Selected_policy_plus_bounded_Molmo']=robot_init
            np.save(dest/'hybrid_correction.npy',correction)
        projections={name:np.stack([project(from_anchor(p[:,t],poses[t+1],poses[0]),k) for t in range(20)],axis=1) for name,p in methods.items()}
        assert all(np.isfinite(p).all() for p in methods.values()),'Never conceal invalid predictions by narrowing reference masks'
        assert all(np.isfinite(p).all() for p in projections.values()),'Report nonfinite projections before comparisons'
        rows={}
        for name,p in methods.items():
            uv=projections[name]
            rows[name]={'3D_est_m':scores(p,gt3,mask3),'2D_px':scores(uv,gt2,mask2),
                        'outside_fraction':float(np.mean((uv<0).any(-1)|(uv>=256).any(-1))),
                        'median_endpoint_amplitude_m':float(np.median(np.linalg.norm(p[:,-1]-starts[name],axis=-1)))}
        prefix=('provisional_policy_' if (OUT/'motion_policy_amendment_v2.json').exists() else 'provisional_robot_') if robot_only else ''
        write(dest/(prefix+'metrics.json'),{'created_utc':datetime.now(timezone.utc).isoformat(),'methods':rows,
            'reference_path':str(refpath),'reference_sha256':sha(refpath),'unchanged_v3_masks':True,
            'common_valid_3d_samples':int(mask3.sum()),'common_valid_2d_samples':int(mask2.sum()),
            'future_TCP_in_forecast':False,'future_camera_poses_evaluation_only':True,'offimage_predictions_excluded':False,
            'preregistration_sha256':sha(OUT/'preregistration.json'),'reference_not_physical_GT':True,
            'no_future_model_or_hyperparameter_selection':True})
        np.savez_compressed(dest/(prefix+'predictions_and_projections.npz'),**{name:p for name,p in methods.items()},
                            **{name+'_uv':p for name,p in projections.items()})
        print(camera,{name:{'ADE_mm':round(row['3D_est_m']['ADE']*1000,3),'FDE_mm':round(row['3D_est_m']['FDE']*1000,3),'ADE_px':round(row['2D_px']['ADE'],3)} for name,row in rows.items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--robot-only',action='store_true');run(p.parse_args().robot_only)
