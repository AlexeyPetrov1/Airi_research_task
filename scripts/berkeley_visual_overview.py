"""Show all 24 frozen points at common full plot bounds, colored by P8 group."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from berkeley_temporal_diagnostics import BASE,OUT,load_scene,scale_displacement,fingerprint
from berkeley_evaluate import project

COLORS=['#4aa3ff','#ff9518','#f240b8']


def main():
    before=fingerprint();receipts=[]
    for name in ('cup','bottle'):
        s=load_scene(name);raw={'BASELINE-U':(s['pred'],s['K'],s['p0'])}
        for case in ('CASE-AUGE','CASE-NOK','CASE-NOK-STRICT'):
            folder=OUT/case/name;receipt=folder/'predictions/model_run.json'
            if not receipt.exists():continue
            status=json.loads(receipt.read_text())
            if not status.get('success') or status.get('mode')!='full_fixed_24':continue
            pred=np.load(folder/'predictions/future_3d.npy')
            if pred.shape!=(24,30,3):raise ValueError('Not a complete case')
            raw[case]=(pred,np.load(folder/'geometry/K_median.npy'),np.load(folder/'observed/points_3d_history.npy')[-1])
        for corrected in (False,True):
            values={case:project(scale_displacement(xyz,p0,1/3) if corrected else xyz,K)
                    for case,(xyz,K,p0) in raw.items()}
            bounds=np.concatenate([s['gt'].reshape(-1,2),*[x.reshape(-1,2) for x in values.values()]])
            lo=np.minimum(np.nanmin(bounds,axis=0)-20,[0,0]);hi=np.maximum(np.nanmax(bounds,axis=0)+20,[640,480])
            fig,axes=plt.subplots(1,len(values),figsize=(6*len(values),5),squeeze=False)
            for ax,(case,uv) in zip(axes[0],values.items()):
                ax.imshow(s['rgb'][-1],extent=(0,640,480,0))
                for idx in range(24):
                    gt=np.vstack([s['uv0'][idx],s['gt'][idx]])
                    gt[1:][~s['mask'][idx]]=np.nan
                    prediction=np.vstack([s['uv0'][idx],uv[idx]])
                    ax.plot(*gt.T,color='#22c05a',lw=.8,alpha=.6)
                    ax.plot(*prediction.T,color=COLORS[idx//8],lw=1.,alpha=.7)
                    ax.scatter(*uv[idx,-1],s=12,color=COLORS[idx//8],marker='x')
                ax.set(title=case+(' | own-point displacement x1/3' if corrected else ' | actual raw output'),
                       xlim=(lo[0],hi[0]),ylim=(hi[1],lo[1]),aspect='equal')
            fig.suptitle(f'{name}: all 24 IDs, all 30 model steps; blue/orange/pink = P8 groups; green = tracked future',fontsize=11)
            fig.tight_layout();p=OUT/name/f'all24_{"one_third" if corrected else "actual"}.png'
            fig.savefig(p,dpi=150);plt.close(fig)
            receipts.append(dict(path=str(p.relative_to(OUT)),cases=list(values),all_24_points=True,
                prediction_steps=30,reference_real_steps=10,common_axis_bounds=[lo.tolist(),hi.tolist()],
                visual_review_pending=True))
    if before!=fingerprint():raise ValueError('Original evidence changed')
    (OUT/'all24_visual_manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')
    print(json.dumps(dict(figures=len(receipts))),flush=True)


if __name__=='__main__':main()
