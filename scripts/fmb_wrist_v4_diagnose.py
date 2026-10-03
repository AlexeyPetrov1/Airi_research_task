"""Review v3 and preregister improvement experiments using observed inputs only."""
import json
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
from fmb_wrist_prepare import ROOT, write, sha
from fmb_wrist_math import tcp_matrices, transform, sample_z, backproject

BASE=ROOT/'runs/fmb_wrist_v3'
OUT=ROOT/'runs/fmb_wrist_v4_improvement'

def run():
    OUT.mkdir(exist_ok=True)
    if not (OUT/'preserved_v3.json').exists():
        files=[p for p in BASE.rglob('*') if p.is_file()]
        write(OUT/'preserved_v3.json', {'recorded_utc':datetime.now(timezone.utc).isoformat(),
              'sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},
              'report_sha256':sha(ROOT/'report/fmb_wrist_v3.md')})
    rows={}
    for camera in ['wrist_2','wrist_1']:
        scene=BASE/camera
        ids=np.load(scene/'observed/selected_point_ids.npy')
        hist=np.load(scene/'branches/C_sensor_tcp_official/observed/points_3d_history.npy')
        uv=np.load(scene/'observed/observed_tracks_2d.npz')['tracks'][:,ids]
        vis=np.load(scene/'observed/observed_tracks_2d.npz')['visibility'][:,ids]
        z=np.load(scene/'observed/sensor_z16.npy').astype(float)*1e-4
        k=np.load(scene/'geometry/official_K.npy')
        tcp=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'))
        x=np.load(scene/'geometry/selected_hand_eye_X.npy')
        local=[];valid=[]
        for depth,p,v in zip(z,uv,vis):
            zz,spread,c=sample_z(depth,p)
            local.append(backproject(p,zz,k));valid.append(v&np.isfinite(zz)&(spread<.03))
        local=np.array(local);valid=np.array(valid)
        # Camera and grasped target share the same rigid EE mount; inspect rather than assume.
        drift=np.linalg.norm(local-local[-1],axis=-1)
        template=local[-1]
        clean=np.stack([transform(np.linalg.inv(tcp[-1]@x)@(g@x),template) for g in tcp[-3:]])
        q=np.rint((hist-hist[-1,0])*1000).astype(int)
        rows[camera]={
            'camera_relative_target_drift_last10_median_mm':float(np.median(drift[-10:][valid[-10:]])*1000),
            'camera_relative_target_drift_last10_p90_mm':float(np.percentile(drift[-10:][valid[-10:]],90)*1000),
            'historical_camera_relative_motion_last10_px_median':float(np.median(np.linalg.norm(uv[-10:]-uv[-1],axis=-1))),
            'original_H3_point_velocity_mm_s':((hist[-1]-hist[0])/.2*1000).tolist(),
            'rigid_camera_attached_H3_velocity_mm_s':((clean[-1]-clean[0])/.2*1000).tolist(),
            'original_H3_quantized_stationary_point_fraction':float(np.mean(np.all(q[0]==q[-1],axis=-1))),
            'tcp_last3_velocity_base_mm_s':((tcp[-1,:3,3]-tcp[-3,:3,3])/.2*1000).tolist(),
            'tcp_velocity_by_window_mm_s':{str(w):((tcp[-1,:3,3]-tcp[-w,:3,3])/((w-1)/10)*1000).tolist() for w in [3,5,8,12,20]},
            'future_files_accessed':False,
        }
        dest=OUT/camera;dest.mkdir(exist_ok=True)
        np.savez_compressed(dest/'observed_reference.npz',camera_points=local,valid=valid,uv=uv,tcp=tcp,X=x,K=k,ids=ids)
        np.save(dest/'denoised_rigid_H3.npy',clean.astype(np.float32))
    write(OUT/'observed_diagnosis.json',rows)
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':run()
