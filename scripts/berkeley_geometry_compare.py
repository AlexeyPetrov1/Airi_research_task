"""Visual-first comparison of actual geometry-case predictions and frozen baseline.

Pilot references are the first fixed eight IDs; complete cases use all 24.
2D reference and visibility are reused exactly. 3D_est uses the candidate K and
same measured future sensor Z, so it is explicitly conditional on calibration.
"""
import argparse
import json
import numpy as np
from berkeley_temporal_diagnostics import (BASE,OUT,load_scene,render,evaluate,
    scale_displacement,write,fingerprint,velocity,TIMES)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True)
    p.add_argument('--scene',choices=['cup','bottle'],default='cup');a=p.parse_args()
    case=OUT/a.case/a.scene
    status=json.loads((case/'predictions/model_run.json').read_text())
    if status.get('success') is not True:raise ValueError('No complete real model output')
    prediction=np.load(case/'predictions/future_3d.npy')
    n=len(prediction)
    s=load_scene(a.scene)
    oldK=s['K'];K=np.load(case/'geometry/K_median.npy')
    basis=np.linalg.inv(K)@oldK
    for key in ('ids','pred','p0','uv0','gt','mask','gt3','mask3'):
        s[key]=s[key][:n]
    s['gt3']=s['gt3']@basis.T
    original=s['pred']@basis.T
    originalp0=s['p0']@basis.T
    history=np.load(case/'observed/points_3d_history.npy')[:,:n]
    s['p0']=history[-1];s['history']=history;s['K']=K;s['name']=f'{a.case}/{a.scene}'
    baseline=original[:,np.arange(2,30,3)]
    actual=prediction[:,np.arange(2,30,3)]
    fixed=scale_displacement(prediction,s['p0'],1/3)[:,np.arange(2,30,3)]
    variants={'BASELINE-U':baseline,f'{a.case}_actual':actual,f'{a.case}_one_third':fixed}
    controls={'static':np.repeat(s['p0'][:,None],10,axis=1),
              'constant_velocity':s['p0'][:,None]+velocity(history,s['times'])[:,None]*TIMES[None,:,None]}
    before=fingerprint();render(s,variants,'geometry')
    if n==24:
        for g in (1,2):
            subset=slice(g*8,(g+1)*8)
            group_s=dict(s)
            for key in ('ids','pred','p0','uv0','gt','mask','gt3','mask3'):
                group_s[key]=s[key][subset]
            group_s['history']=history[:,subset]
            render(group_s,{name:xyz[subset] for name,xyz in variants.items()},f'geometry_group_{g:02d}')
    if before!=fingerprint():raise ValueError('Baseline changed')
    write(case/'comparison.json',dict(mode=status.get('mode'),point_count=n,point_ids=s['ids'].tolist(),
        methods={name:evaluate(s,xyz) for name,xyz in dict(variants,**controls).items()},
        secondary_metrics=True,reference_2D_unchanged=True,
        visual_groups_saved=[0,1,2] if n==24 else [0],
        reference_3D_est='same measured future sensor Z, rays interpreted with candidate case K',
        caveat='AugE/learned K are calibration hypotheses, not measured ground truth',
        baseline_3D_coordinates_reexpressed_for_common_candidate_basis=True))
    print(json.dumps(dict(case=a.case,scene=a.scene,points=n,success=True)),flush=True)


if __name__=='__main__':main()
