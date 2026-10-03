"""Descriptive world-frame motion replication, with distinct visible point sets."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fmb_wrist_prepare import RUN,write
from fmb_wrist_math import tcp_matrices,transform,backproject,sample_z
from fmb_wrist_evaluate import prediction_gate,TIMES

prediction_gate();motions={};rows=[]
for camera in ['wrist_2','wrist_1']:
    scene=RUN/camera;ref=np.load(scene/'evaluation/future_reference.npz');tracker=np.load(scene/'evaluation/future_alltracker.npz')
    rgbdepth=np.load(scene/'evaluation/future_sensor_depth.npy');K=ref['K']
    start=backproject(tracker['tracks'][0],sample_z(rgbdepth[0],tracker['tracks'][0])[0],K)
    poses=tcp_matrices(np.load(scene/'evaluation/future_tcp_pose_xyzw.npy'))@np.load(scene/'geometry/selected_hand_eye_X.npy')
    origin=transform(poses[0],start);world=transform(poses[0],ref['GT_3D_est'])
    displacement=world-origin[:,None];mask=ref['common_mask3d']&np.isfinite(displacement).all(-1)
    motion=np.array([np.nanmedian(displacement[:,t][mask[:,t]],axis=0) for t in range(20)])
    motions[camera]=motion;np.save(scene/'evaluation/estimated_world_point_displacements.npy',displacement)
    rows.append({'camera':camera,'median_point_displacement_base_m':motion.tolist(),'point_ids':tracker['selected_ids'].tolist(),
                 'visible_point_set_independently_selected':True,'same_material_points_across_cameras_confirmed':False})
diff=np.linalg.norm(motions['wrist_1']-motions['wrist_2'],axis=-1)
fig,axes=plt.subplots(1,3,figsize=(14,4))
for axis,ax in enumerate(axes):
    for camera,motion in motions.items():ax.plot(TIMES,motion[:,axis]*1000,label=camera)
    ax.set(xlabel='future nominal time (s)',ylabel='base '+ 'xyz'[axis]+' displacement (mm)');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(RUN/'wrist_future_world_motion.png',dpi=160);plt.close(fig)
write(RUN/'wrist_future_world_motion.json',{'rows':rows,'median_displacement_vector_disagreement_mm':float(np.nanmedian(diff)*1000),
      'endpoint_displacement_vector_disagreement_mm':float(diff[-1]*1000),'future_used_for_evaluation_only':True,
      'interpretation':'Estimated base-frame point displacement, not calibrated motion GT. Different surfaces/point sets, rotation, frozen uncertain X/K/scale and asynchronous capture confound replication.'})
print('Independent wrist future world-motion cross-check complete',flush=True)
