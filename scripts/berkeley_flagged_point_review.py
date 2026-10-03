"""Display the retained strict-geometry outlier without changing frozen inputs."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from berkeley_temporal_diagnostics import OUT, load_scene, scale_displacement, fingerprint
from berkeley_evaluate import project, ALIGNMENT, TIMES


def main():
    before=fingerprint();folder=OUT/'CASE-NOK-STRICT/bottle';s=load_scene('bottle')
    ids=s['ids'];index=int(np.flatnonzero(ids==30)[0]);K=np.load(folder/'geometry/K_median.npy')
    raw=np.load(folder/'geometry/points_3d_raw.npy')[-3:,30]
    smooth=np.load(folder/'geometry/points_3d_filtered.npy')[-3:,30]
    p0=np.load(folder/'observed/points_3d_history.npy')[-1]
    pred=np.load(folder/'predictions/future_3d.npy')
    variants={'actual':project(pred,K)[index,ALIGNMENT],
              'own-point x1/3':project(scale_displacement(pred,p0,1/3),K)[index,ALIGNMENT]}
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    for ax,(title,uv) in zip(axes[:2],variants.items()):
        ax.imshow(s['rgb'][-1]);ax.plot(*np.vstack([s['uv0'][index],s['gt'][index]]).T,'g.-',label='tracked future')
        ax.plot(*np.vstack([s['uv0'][index],uv]).T,'.-',color='#f240b8',label='prediction')
        ax.set(title='Retained ID 30: '+title,xlim=(0,640),ylim=(480,0));ax.legend(fontsize=7)
    axes[2].plot(s['times']-s['times'][-1],raw[:,2],'o-',label='fresh native-Z lift')
    axes[2].plot(s['times']-s['times'][-1],smooth[:,2],'x-',label='fresh smoothed Z')
    axes[2].set(title='ID 30 input H3',xlabel='observed seconds',ylabel='sensor Z (m)');axes[2].legend(fontsize=7)
    for title,uv in variants.items():axes[3].plot(TIMES,np.linalg.norm(uv-s['gt'][index],axis=-1),'o-',label=title)
    axes[3].set(title='ID 30 secondary error',xlabel='future seconds',ylabel='2D distance (px)');axes[3].legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'bottle/strict_retained_id30.png',dpi=140);plt.close(fig)
    result=dict(point_id=30,selected_point_offset=index,P8_group=index//8,excluded=False,
                history_timestamps_relative_s=(s['times']-s['times'][-1]).tolist(),
                raw_H3_xyz=raw.tolist(),smoothed_H3_xyz=smooth.tolist(),
                raw_H3_projection=project(raw,K).tolist(),smoothed_H3_projection=project(smooth,K).tolist(),
                H3_smoothing_displacement_m=np.linalg.norm(raw-smooth,axis=-1).tolist(),
                input_change=False,figure='bottle/strict_retained_id30.png',
                caveat='The filter flag is retained; visual inspection does not certify an accurate depth or remove eligibility concerns.')
    (OUT/'bottle/strict_retained_id30.json').write_text(json.dumps(result,indent=2)+'\n')
    if before!=fingerprint():raise ValueError('Baseline changed')


if __name__=='__main__':main()
