"""Assess fixed-forecast rankings under seven frozen observed calibration fits."""
import json,numpy as np
from fmb_wrist_prepare import write,sha
from fmb_wrist_math import tcp_matrices,backproject,sample_z,transform
from fmb_wrist_evaluate import scores
from fmb_wrist_v4_diagnose import BASE,OUT

for camera in ['wrist_2','wrist_1']:
    scene=BASE/camera;dest=OUT/camera;data=np.load(dest/'predictions_and_projections.npz')
    reference=np.load(scene/'evaluation/future_reference.npz');k=reference['K'];mask=reference['common_mask3d']
    uv=np.load(scene/'evaluation/future_alltracker.npz')['tracks'];depth=np.load(scene/'evaluation/future_sensor_depth.npy')
    tcp=tcp_matrices(np.load(scene/'evaluation/future_tcp_pose_xyzw.npy'))
    local=np.stack([backproject(p,sample_z(z,p)[0],k) for z,p in zip(depth,uv)])
    bootpath=scene/'geometry/independent_robot_rgbd_bootstrap.json';fits=json.loads(bootpath.read_text())['fits'];rows=[]
    methods=[name for name in data.files if not name.endswith('_uv')]
    for fit in fits:
        poses=tcp@np.array(fit['X_TCP_from_camera'])
        gt=np.stack([transform(np.linalg.inv(poses[0])@pose,p) for pose,p in zip(poses,local)])[1:].transpose(1,0,2)
        rows.append({'seed':fit['seed'],'observed_heldout_p90_px':fit['heldout_reprojection']['p90_px'],
                     'methods':{name:scores(data[name],gt,mask) for name in methods}})
    ranking={name:{'fits':len(rows),'lower_ADE_than_v3_Molmo':sum(r['methods'][name]['ADE']<r['methods']['v3_Molmo_H3']['ADE'] for r in rows),
              'lower_ADE_than_Static':sum(r['methods'][name]['ADE']<r['methods']['v3_Static']['ADE'] for r in rows),
              'lower_ADE_than_CV':sum(r['methods'][name]['ADE']<r['methods']['v3_CV']['ADE'] for r in rows),
              'min_ADE_mm':min(r['methods'][name]['ADE']*1000 for r in rows),'max_ADE_mm':max(r['methods'][name]['ADE']*1000 for r in rows)} for name in methods}
    write(dest/'calibration_ranking_sensitivity.json',{'rows':rows,'ranking':ranking,'observed_bootstrap_source_sha256':sha(bootpath),
          'future_used_for_refitting':False,'forecasts_fixed':True,'reference_common_mask_unchanged':True,
          'interpretation':'Descriptive reference-only sensitivity, not a confidence interval or full joint calibration/prediction uncertainty. Robot forecasts retain their originally frozen X; no new neural inference or calibration fitting.'})
    print(camera,'seven-fit reference sensitivity complete',flush=True)
