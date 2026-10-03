"""Independent observed RGBD/TCP hand-eye control, without ViPE poses/depth."""
from __future__ import annotations
import argparse,json
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from fmb_wrist_prepare import RUN,write
from fmb_wrist_math import tcp_matrices,static_pairs,sample_z,backproject,project,transform

def run(camera):
    scene=RUN/camera;rgb=np.load(scene/'observed/rgb.npy');depth=np.load(scene/'observed/sensor_z16.npy').astype(float)*1e-4
    robot=tcp_matrices(np.load(scene/'observed/tcp_pose_xyzw.npy'));candidates=json.loads((RUN/'sources/official_K_candidates.json').read_text())[camera]
    profile=next(row for row in candidates if row['width']==640 and row['height']==480);K=np.array(profile['K_256_resize'])
    rows=static_pairs(rgb);data=[]
    for row in rows:
        i,j=row['i'],row['j'];p,q=np.array(row['uv_i']),np.array(row['uv_j'])
        z,spread,_=sample_z(depth[i],p);zz,sj,_=sample_z(depth[j],q)
        good=np.isfinite(z)&np.isfinite(zz)&(spread<.03)&(sj<.03)
        if good.sum()<6:continue
        # Cap each temporal pair equally, keeping fixed image-order subsampling.
        ids=np.flatnonzero(good)[::max(1,int(good.sum()//50))][:50]
        data.append({'i':i,'j':j,'xyz_i':backproject(p[ids],z[ids],K),'xyz_j':backproject(q[ids],zz[ids],K),
                     'uv_j':q[ids],'relative_robot':np.linalg.inv(robot[j])@robot[i]})
    train=[row for row in data if row['j']<40];test=[row for row in data if row['j']>=40]
    if len(train)<3 or len(test)<2:raise RuntimeError('Insufficient independent static RGBD pairs')
    def unpack(v):
        x=np.eye(4);x[:3,:3]=Rotation.from_rotvec(v[:3]).as_matrix();x[:3,3]=v[3:];return x
    def residual(v,rows):
        x=unpack(v);ix=np.linalg.inv(x);values=[]
        for row in rows:
            moved=transform(ix@row['relative_robot']@x,row['xyz_i'])
            # Keep all observed correspondences; behind-camera/near-zero Z is penalized.
            uv=project(np.where(np.abs(moved[:,2,None])>1e-6,moved,moved+np.array([0,0,1e-6])),K)
            values.append((uv-row['uv_j']).ravel()/3)
            values.append((moved-row['xyz_j']).ravel()/.02)
            values.append(np.minimum(moved[:,2]-.005,0)*100)
        return np.concatenate(values)
    starts=[np.r_[r.as_rotvec(),[0,0,.08]] for r in Rotation.create_group('O')]
    fits=[]
    for start in starts:
        fit=least_squares(lambda v:residual(v,train),start,bounds=(np.r_[[-6,-6,-6],[-.5,-.5,-.5]],np.r_[[6,6,6],[.5,.5,.5]]),loss='soft_l1',max_nfev=100)
        fits.append(fit)
    best=min(fits,key=lambda fit:fit.cost);x=unpack(best.x)
    def metrics(rows,estimate=None):
        estimate=x if estimate is None else estimate
        errors=[];space=[];per=[]
        for row in rows:
            moved=transform(np.linalg.inv(estimate)@row['relative_robot']@estimate,row['xyz_i']);error=np.linalg.norm(project(moved,K)-row['uv_j'],axis=-1);metric=np.linalg.norm(moved-row['xyz_j'],axis=-1)
            errors.extend(error);space.extend(metric);per.append({'i':row['i'],'j':row['j'],'count':len(error),'median_px':float(np.median(error))})
        return {'pairs':per,'correspondences':len(errors),'median_px':float(np.median(errors)),'p90_px':float(np.percentile(errors,90)),
                'median_3d_correspondence_mm':float(np.median(space)*1000),'p90_3d_correspondence_mm':float(np.percentile(space,90)*1000)}
    tr,te=metrics(train),metrics(test)
    bootstrap=[]
    bounds=(np.r_[[-6,-6,-6],[-.5,-.5,-.5]],np.r_[[6,6,6],[.5,.5,.5]])
    basepath=np.linalg.inv(robot[-1]@x)[None]@(robot@x)
    for seed in range(7):
        rng=np.random.default_rng(seed);subset=[train[i] for i in sorted(rng.choice(len(train),size=max(3,int(len(train)*.7)),replace=False))]
        fit=least_squares(lambda v:residual(v,subset),best.x,bounds=bounds,loss='soft_l1',max_nfev=150)
        bx=unpack(fit.x);pose=np.linalg.inv(robot[-1]@bx)[None]@(robot@bx)
        pathdifference=np.linalg.inv(basepath)@pose
        bootstrap.append({'seed':seed,'X_TCP_from_camera':bx.tolist(),
              'translation_difference_mm':float(np.linalg.norm(bx[:3,3]-x[:3,3])*1000),
              'rotation_difference_deg':float(np.degrees((Rotation.from_matrix(x[:3,:3]).inv()*Rotation.from_matrix(bx[:3,:3])).magnitude())),
              'heldout_reprojection':metrics(test,bx),
              'observed_relative_path_median_translation_difference_mm':float(np.median(np.linalg.norm(pathdifference[:,:3,3],axis=-1))*1000),
              'observed_relative_path_p90_translation_difference_mm':float(np.percentile(np.linalg.norm(pathdifference[:,:3,3],axis=-1),90)*1000)})
    singular=np.linalg.svd(best.jac,compute_uv=False)
    write(scene/'geometry/independent_robot_rgbd_bootstrap.json',{'fits':bootstrap,'jacobian_singular_values':singular.tolist(),
          'jacobian_condition_mixed_parameter_units':float(singular[0]/max(singular[-1],1e-12)),
          'interpretation':'Static pairs subsampled on training prefix only; hand-eye parameter spread and relative-path spread reported separately. Future cannot calibrate X.'})
    gate=te['median_px']<=4 and te['p90_px']<=10 and np.linalg.norm(x[:3,3])<=.5
    write(scene/'geometry/independent_robot_rgbd_hand_eye.json',{'X_TCP_from_camera':x.tolist(),'train':tr,'heldout':te,
          'observed_only':True,'ViPE_used':False,'static_matches_source':'Observed conservative board/gray-background LK FB and homography filters',
          'calibration_source':'Official factory 640x480 rectified prior mapped to 256x256, sensor Z16 x1e-4, measured robot TCP',
          'metric_scale_confirmed':False,'translation_norm_m':float(np.linalg.norm(x[:3,3])),'gate':gate,
          'multistart_fits':len(fits),'optimizer_success':bool(best.success),'initial_prior_rotation':'24 cube rotations',
          'limitations':'Strong RGB/depth registration and static-match assumptions. Limited rotation diversity; factory RGB K and units remain uncertain. This estimated X is an independent observed control, not measured hand-eye.'})
    np.save(scene/'geometry/independent_robot_rgbd_X.npy',x)
    print(camera,'independent RGBD/TCP X',x,'heldout',te,'gate',gate,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);a=p.parse_args()
    for c in a.cameras:run(c)
