import numpy as np
from .berkeley_evaluate import velocity
THRESHOLDS=(5,10,20,40)
TIMES=np.arange(1,11)/5


def scores(uv,s,xyz=None,p0=None,K=None):
    gt=s['gt'];mask=s['mask'];error=np.linalg.norm(uv-gt,axis=-1)
    if uv.shape!=gt.shape or not np.isfinite(error[mask]).all():
        raise ValueError('Prediction is incomplete on the common reference mask')
    result=dict(ADE_2D_px=float(error[mask].mean()),
        FDE_2D_px=float(error[:,-1][mask[:,-1]].mean()),
        median_2D_px=float(np.median(error[mask])),P90_2D_px=float(np.percentile(error[mask],90)),
        ADE_2D_image_diagonal_fraction=float(error[mask].mean()/800),
        PWT_native_px={str(t):float((error[mask]<t).mean()) for t in THRESHOLDS},
        reference_visible_pair_coverage=float(mask.mean()),prediction_pair_coverage=1.,
        error_by_horizon_px=[float(error[:,t][mask[:,t]].mean()) for t in range(10)])
    actual=np.concatenate([s['uv0'][:,None],gt],axis=1)
    forecast=np.concatenate([s['uv0'][:,None],uv],axis=1)
    segment_mask=mask & np.concatenate([np.ones((len(mask),1),bool),mask[:,:-1]],axis=1)
    vg=np.diff(actual,axis=1);vp=np.diff(forecast,axis=1)
    ng=np.linalg.norm(vg,axis=-1);npred=np.linalg.norm(vp,axis=-1)
    moving=segment_mask & (ng>1.);direction_valid=moving & (npred>1.)
    cosine=(vg*vp).sum(-1)/np.maximum(ng*npred,1e-12)
    result['moving_step_direction_coverage']=float(direction_valid.sum()/moving.sum()) if moving.any() else None
    result['direction_error_median_degrees']=float(np.median(np.degrees(np.arccos(np.clip(cosine[direction_valid],-1,1))))) if direction_valid.any() else None
    result['projected_speed_MAE_px_per_s']=float(np.abs(npred-ng)[segment_mask].mean()/.2)
    full=segment_mask.all(1);length_gt=ng[full].sum(1);length_pred=npred[full].sum(1)
    keep=length_gt>1.
    result['median_projected_path_length_ratio']=float(np.median(length_pred[keep]/length_gt[keep])) if keep.any() else None
    displacement_gt=np.linalg.norm(gt[:,-1]-s['uv0'],axis=-1)
    displacement_pred=np.linalg.norm(uv[:,-1]-s['uv0'],axis=-1)
    final=mask[:,-1] & (displacement_gt>1.)
    result['median_endpoint_displacement_ratio']=float(np.median(displacement_pred[final]/displacement_gt[final])) if final.any() else None
    if xyz is not None:
        if K is None:raise ValueError('3D consistency requires an explicit geometry basis')
        gt3=s['gt3']@(np.linalg.inv(K.astype(float))@s['K'].astype(float)).T
        mask3=s['mask3'];error3=np.linalg.norm(xyz-gt3,axis=-1)
        if not np.isfinite(error3[mask3]).all():raise ValueError('Incomplete conditional 3D error')
        result['conditional_3D_est']={
            'ADE_m':float(error3[mask3].mean()),'FDE_m':float(error3[:,-1][mask3[:,-1]].mean()),
            'PWT_m':{str(t):float((error3[mask3]<t).mean()) for t in (.01,.02,.05,.1,.2)},
            'valid_pair_coverage':float(mask3.mean()),
            'reference':'same native future sensor Z and tracker pixels, reexpressed with the method candidate K; not calibrated 3D ground truth'}
        pairs=np.triu_indices(len(xyz),1)
        initial=np.linalg.norm(p0[pairs[0]]-p0[pairs[1]],axis=-1)
        future=np.linalg.norm(xyz[pairs[0]]-xyz[pairs[1]],axis=-1)
        drift=np.abs(future-initial[:,None])*1000
        within=pairs[0]//8==pairs[1]//8
        result['pair_distance_drift_3D_mm']=dict(median=float(np.median(drift)),
            P90=float(np.percentile(drift,90)),within_P8_median=float(np.median(drift[within])),
            across_P8_median=float(np.median(drift[~within])))
    result['groups_P8']=[dict(group=g,ADE_px=float(error[g*8:(g+1)*8][mask[g*8:(g+1)*8]].mean())) for g in range(3)]
    return result

def controls(history,p0,times):
    vel=velocity(history,times)
    return {'static':np.repeat(p0[:,None],10,axis=1),
            'CV_3D':p0[:,None]+vel[:,None]*TIMES[None,:,None],
            'object_translation_3D':p0[:,None]+np.median(vel,axis=0)[None,None]*TIMES[None,:,None]}
