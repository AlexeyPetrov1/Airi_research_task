"""Propagate frozen observed calibration bootstrap into evaluation-only references."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import RUN,write
from fmb_wrist_math import tcp_matrices,backproject,sample_z,transform
from fmb_wrist_evaluate import prediction_gate,align_prediction,scores

def run(camera):
    prediction_gate();scene=RUN/camera;dest=scene/'evaluation'
    stored=np.load(dest/'future_reference.npz');K=stored['K'];main=stored['GT_3D_est'];mask=stored['common_mask3d']
    track=np.load(dest/'future_alltracker.npz')['tracks'];depth=np.load(dest/'future_sensor_depth.npy');robot=tcp_matrices(np.load(dest/'future_tcp_pose_xyzw.npy'))
    local=np.stack([backproject(uv,sample_z(z,uv)[0],K) for z,uv in zip(depth,track)])
    boot=json.loads((scene/'geometry/independent_robot_rgbd_bootstrap.json').read_text())['fits'];rows=[];differences=[]
    for fit in boot:
        x=np.array(fit['X_TCP_from_camera']);poses=robot@x
        gt=np.stack([transform(np.linalg.inv(poses[0])@T,xyz) for T,xyz in zip(poses,local)])[1:].transpose(1,0,2)
        diff=np.linalg.norm(gt-main,axis=-1);differences.append(diff)
        row={'observed_bootstrap_seed':fit['seed'],'heldout_observed_p90_px':fit['heldout_reprojection']['p90_px'],
             'median_reference_difference_mm':float(np.nanmedian(diff[mask])*1000),'p90_reference_difference_mm':float(np.nanpercentile(diff[mask],90)*1000),
             'endpoint_reference_difference_mm':float(np.nanmedian(diff[:,-1][mask[:,-1]])*1000),'methods':{}}
        for branch in sorted((scene/'branches').glob('*')):
            if not(branch/'predictions/future_3d.npy').exists():continue
            h=np.load(branch/'observed/points_3d_history.npy');future=np.load(branch/'predictions/future_3d.npy')
            methods={branch.name:align_prediction(future,h[-1]),branch.name+'/Static':np.repeat(h[-1,:,None],20,axis=1),
                     branch.name+'/CV':h[-1,:,None]+((h[2]-h[0])/.2)[:,None]*stored['times'][None,:,None]}
            row['methods']={**row['methods'],**{name:scores(pred,gt,mask) for name,pred in methods.items()}}
        rows.append(row)
    diff=np.array(differences);rankings={}
    for name in rows[0]['methods']:
        if '/' in name:continue
        rankings[name]={'bootstrap_fits':len(rows),'observed_heldout_gate_pass_fits':sum(row['heldout_observed_p90_px']<=10 for row in rows),
            'model_lower_ADE_than_static_fits':sum(row['methods'][name]['ADE']<row['methods'][name+'/Static']['ADE'] for row in rows),
            'model_lower_ADE_than_CV_fits':sum(row['methods'][name]['ADE']<row['methods'][name+'/CV']['ADE'] for row in rows)}
    write(dest/'hand_eye_reference_sensitivity.json',{'rows':rows,'future_fitted':False,'calibration_bootstrap_source':'Observed training-pair subsets only',
          'ranking_sensitivity':rankings,
          'interpretation':'Descriptive sensitivity envelope, not a confidence interval; correlated geometry and depth errors remain.',
          'max_bootstrap_p90_reference_difference_mm':max(row['p90_reference_difference_mm'] for row in rows)})
    fig,ax=plt.subplots(figsize=(9,5))
    for row,d in zip(rows,diff):ax.plot(stored['times'],[np.nanmedian(d[:,t][mask[:,t]])*1000 for t in range(20)],alpha=.65,label=f'Observed bootstrap {row["observed_bootstrap_seed"]}')
    ax.set(xlabel='future nominal time (s)',ylabel='Reference change from frozen X (mm)');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(scene/'viz/reference_hand_eye_sensitivity.png',dpi=150);plt.close(fig)
    print(camera,'future calibration sensitivity complete',flush=True)

if __name__=='__main__':
    for camera in ['wrist_2','wrist_1']:run(camera)
