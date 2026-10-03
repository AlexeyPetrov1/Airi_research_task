"""Compare all saved Berkeley forecasts and causal baselines without ML calls.

Scores native 640x480 projections on one frozen tracker reference and physical
timebase. 3D shape consistency is a diagnostic, not calibrated 3D accuracy.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from berkeley_temporal_diagnostics import (BASE, OUT, load_scene, scale_displacement,
                                          fingerprint, render)
from berkeley_evaluate import project, velocity, ALIGNMENT, TIMES

DEST=OUT/'baseline_comparison'
THRESHOLDS=(5,10,20,40)


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


def overview(scene,s,variants):
    (DEST/scene).mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(18,10));colors=('#4aa3ff','#ff9518','#f240b8')
    for ax,(method,uv) in zip(axes.flat,variants.items()):
        ax.imshow(s['rgb'][-1])
        for p in range(24):
            actual=np.vstack([s['uv0'][p],s['gt'][p]]);actual[1:][~s['mask'][p]]=np.nan
            predicted=np.vstack([s['uv0'][p],uv[p]])
            ax.plot(*actual.T,color='#22c05a',lw=1,alpha=.6)
            ax.plot(*predicted.T,color=colors[p//8],lw=1,alpha=.8)
            ax.scatter(*uv[p,-1],color=colors[p//8],s=12,marker='x')
        ax.set(title=method,xlim=(0,640),ylim=(480,0),aspect='equal')
    ax=axes.flat[-1]
    for method,uv in variants.items():ax.plot(TIMES,np.linalg.norm(uv-s['gt'],axis=-1).mean(0),label=method)
    ax.set(title='Mean physical-time error',xlabel='future seconds',ylabel='pixels');ax.legend(fontsize=8)
    fig.suptitle(f'{scene}: all 24 frozen IDs; green=reference; blue/orange/pink=P8 groups',fontsize=13)
    fig.tight_layout();fig.savefig(DEST/scene/'all24_baselines.png',dpi=120);plt.close(fig)


def main():
    before=fingerprint();DEST.mkdir(exist_ok=True);all_results={}
    for name in ('cup','bottle'):
        s=load_scene(name);methods={};geometry_controls={};selected=None
        raw=s['pred'][:,ALIGNMENT];corrected=scale_displacement(s['pred'],s['p0'],1/3)[:,ALIGNMENT]
        for key,xyz in {'MolmoMotion_original':raw,'MolmoMotion_original_x1_3':corrected}.items():
            methods[key]=scores(project(xyz,s['K']),s,xyz,s['p0'],s['K'])
        methods['step_equals_frame_diagnostic']=scores(project(s['pred'][:,:10],s['K']),s)
        for key,xyz in controls(s['history'],s['p0'],s['times']).items():
            methods[key+'_BASELINE_U']=scores(project(xyz,s['K']),s,xyz,s['p0'],s['K'])
        uvh=np.load(BASE/name/'observed/points_2d_history.npy').astype(float)
        cv2=uvh[-1,:,None]+velocity(uvh,s['times'])[:,None]*TIMES[None,:,None]
        methods['CV_2D_no_K']=scores(cv2,s)
        for case in ('CASE-AUGE','CASE-NOK','CASE-NOK-STRICT'):
            folder=OUT/case/name;K=np.load(folder/'geometry/K_median.npy')
            history=np.load(folder/'observed/points_3d_history.npy').astype(float);p0=history[-1]
            full=np.load(folder/'predictions/future_3d.npy').astype(float)
            xyzs={case+'_actual':full[:,ALIGNMENT],
                  case+'_x1_3':scale_displacement(full,p0,1/3)[:,ALIGNMENT]}
            saved=json.loads((folder/'comparison.json').read_text())['methods']
            for key,xyz in xyzs.items():
                methods[key]=scores(project(xyz,K),s,xyz,p0,K)
                oldkey=key.replace('_x1_3','_one_third')
                np.testing.assert_allclose(methods[key]['ADE_2D_px'],saved[oldkey]['2D_px']['ADE_2D_px'],atol=.001,rtol=0)
            geometry_controls[case]={key:scores(project(xyz,K),s,xyz,p0,K) for key,xyz in controls(history,p0,s['times']).items()}
            for key,oldkey in (('static','static'),('CV_3D','constant_velocity')):
                for metric,saved_key in (('ADE_2D_px','ADE_2D_px'),('FDE_2D_px','FDE_2D_px')):
                    np.testing.assert_allclose(geometry_controls[case][key][metric],saved[oldkey]['2D_px'][saved_key],atol=.001,rtol=0)
            if case==('CASE-NOK-STRICT' if name=='cup' else 'CASE-NOK'):
                selected=(K,p0,xyzs[case+'_x1_3'],controls(history,p0,s['times']),history,case)
        K,p0,best,control,history,case=selected
        overview(name,s,dict(static=project(control['static'],K),
            CV_3D=project(control['CV_3D'],K),object_translation_3D=project(control['object_translation_3D'],K),
            CV_2D_no_K=cv2,**{case+'_x1_3':project(best,K)}))
        view=dict(s,name=f'baseline_comparison/{name}',K=K,p0=p0,history=history)
        view['pred']=s['pred']@(np.linalg.inv(K)@s['K']).T
        view['gt3']=s['gt3']@(np.linalg.inv(K)@s['K']).T
        for g in range(3):
            subset=slice(g*8,(g+1)*8);group=dict(view)
            for key in ('ids','pred','p0','uv0','gt','mask','gt3','mask3'):group[key]=view[key][subset]
            group['history']=history[:,subset]
            render(group,{'static':control['static'][subset],'constant_velocity':control['CV_3D'][subset],case+'_x1_3':best[subset]},f'baselines_group_{g:02d}')
        all_results[name]=dict(point_ids=s['ids'].tolist(),methods=methods,geometry_matched_controls=geometry_controls,
                              visually_selected_model=case+'_x1_3')
    report=dict(protocol=dict(width=640,height=480,image_diagonal_px=800,points_per_scene=24,
        real_future_timestamps_s=TIMES.tolist(),common_mask='frozen independent AllTracker future visibility',
        PWT_comparison='strict error < threshold in native pixels; this is not the official TAP-Vid score',
        direction_motion_threshold='GT and predicted displacement >1 native pixel per 0.2 s; nonmoving predictions reduce direction coverage',
        rigidity='pair-distance drift from each method own initial 3D constellation; a consistency diagnostic, not ground-truth 3D accuracy',
        candidates_selected_after_inspection=True,independent_test_set=False,
        ML_calls=0,original_oracle_excluded=True),scenes=all_results)
    (DEST/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    lines=['# Сравнение траекторий с причинными бейзлайнами','',
           'Все 24 фиксированных ID, десять настоящих будущих моментов +0,2…+2,0 с, один общий tracker reference. Будущее не использовано для построения бейзлайнов. Выбор лучших существующих веток после просмотра этих сцен — исследовательский выбор, не результат на новом test split.','',
           '| Сцена | Метод | ADE px ↓ | FDE px ↓ | PWT@10px ↑ | PWT@20px ↑ | Длина пути / reference |','|---|---|---:|---:|---:|---:|---:|']
    for scene,values in all_results.items():
        for method,m in values['methods'].items():
            lines.append(f"| {scene} | {method} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} | {100*m['PWT_native_px']['10']:.1f}% | {100*m['PWT_native_px']['20']:.1f}% | {m['median_projected_path_length_ratio']:.3f} |")
    lines+=['','CV_2D_no_K — линейная экстраполяция исходных 2D H3 tracks. CV_3D — линейная экстраполяция XYZ по настоящим timestamps с последующей проекцией. object_translation_3D — одна медианная observed скорость всех точек, что сохраняет исходные попарные 3D расстояния. Статика использует t0. H1 и future-fitted oracle имеют иной контракт и остаются в [предыдущей сводке](../summary.md).','',
        'Для строгой MoGe-ветки CV пересчитан по тем же её входам; полные результаты для каждой геометрии в JSON. Нативные PWT пороги и диагностическая метрика длины пути не являются официальным TAP-Vid протоколом.','',
        '| Сцена | Контроль в геометрии лучшей визуальной ветки | ADE px ↓ | FDE px ↓ | PWT@20px ↑ |','|---|---|---:|---:|---:|']
    for scene,values in all_results.items():
        case=values['visually_selected_model'].removesuffix('_x1_3')
        for key,m in values['geometry_matched_controls'][case].items():
            lines.append(f"| {scene} | {case}/{key} | {m['ADE_2D_px']:.3f} | {m['FDE_2D_px']:.3f} | {100*m['PWT_native_px']['20']:.1f}% |")
    lines+=['','Новые contact sheets и MP4 сопоставляют static / CV в той же геометрии / выбранную модель ×1/3 на настоящих RGB; зелёный — tracker reference. Все три группы сохранены.','']
    for scene in ('cup','bottle'):
        lines.append(f'- {scene}: [все 24 точки пяти методов и ошибка по времени]({scene}/all24_baselines.png).')
        for g in range(3):lines.append(f'- {scene}, группа {g:02d}: [PNG]({scene}/baselines_group_{g:02d}_contact_sheet.png), [MP4]({scene}/baselines_group_{g:02d}_comparison.mp4).')
    lines+=['','[Полные метрики, покрытие, горизонты, группы и 3D согласованность](results.json).']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    if before!=fingerprint():raise ValueError('Baseline changed')
    print(json.dumps({name:{k:round(v['ADE_2D_px'],3) for k,v in row['methods'].items()} for name,row in all_results.items()}),flush=True)


if __name__=='__main__':main()
