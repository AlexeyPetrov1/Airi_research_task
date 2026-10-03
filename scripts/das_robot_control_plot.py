"""Visualize the frozen robot points and joint trajectory, without future data."""
from pathlib import Path
import numpy as np
import json
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_prepare_control import project,save_video
from das_robot_evaluate import frames,labeled

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup'
c=np.load(OUT/'joint_control.npz');k=np.load(SCENE/'geometry/K_median.npy')
rgb=np.array(Image.open(SCENE/'observed/frame_000063.png'))
xy=project(c['robot_sparse_xyz'][:17].reshape(-1,3),k).reshape(17,-1,2)
colors=plt.get_cmap('tab20')(np.linspace(0,1,xy.shape[1]))
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].imshow(rgb)
for i,color in enumerate(colors):
    axes[0].plot(xy[:,i,0],xy[:,i,1],color=color,lw=1.5)
    axes[0].scatter(*xy[0,i],color=color,s=15);axes[0].text(*xy[0,i],f'R{i}',color=color,fontsize=8)
axes[0].set(xlim=(0,640),ylim=(480,0),title='Robot material-point trajectories: nominal IK/FK, 0..2 s')
for i in range(6):axes[1].plot(c['times'][:17],c['joints'][:17,i]-c['joints'][0,i],label=f'joint {i}')
axes[1].set(xlabel='Physical time (s)',ylabel='Joint angle change from t0 (rad)',title='Bounded UR5 joint motion; cup attached to flange')
axes[1].grid(alpha=.3);axes[1].legend()
fig.tight_layout();fig.savefig(OUT/'robot_point_and_joint_trajectories.png',dpi=150);plt.close(fig)

if (OUT/'paired_metrics.json').exists():
    m=json.loads((OUT/'paired_metrics.json').read_text());times=m['physical_evaluation_times_s']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,data in m['variants'].items():
        axes[0].plot(times,[np.nan if p is None else p for p in data['shared_three_robot_to_requested_control']['error_by_time_px']],marker='o',label=name)
        if name!='robot_coupled_initial_only':
            axes[1].plot(times,[np.nan if p is None else p for p in data['common_cup_gripper_relative_motion']['per_time_offset_error_px']],marker='o',label=name)
    axes[0].set(xlabel='Physical time (s)',ylabel='Robot requested-motion error (px)',title='Shared A/B/C visibility intersection')
    axes[1].set(xlabel='Physical time (s)',ylabel='Cup/gripper relative-motion error (px)',title='Matched A/B visibility intersection')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'paired_motion_metrics.png',dpi=150);plt.close(fig)
    clips={name:frames(OUT/name/'generated_molmomotion_seed42.mp4')[:17] for name in m['variants']}
    reference=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    real=np.concatenate([reference[None],np.load(SCENE/'evaluation/future_rgb.npy')])
    grids=[]
    for i in range(17):
        row=[]
        for name,label in [('robot_static','A STATIC ARM'),('robot_coupled','B COUPLED ARM'),('robot_coupled_initial_only','C COUPLED, INITIAL ONLY')]:
            native=np.array(Image.fromarray(clips[name][i]).resize((640,480)))
            row.append(labeled(native,f'{label}: {i/8:.3f}s'))
        ri=int(np.clip(np.rint(i/8*5),0,10));row.append(labeled(real[ri],f'REAL: measured {ri/5:.3f}s'))
        grids.append(np.concatenate([np.concatenate(row[:2],axis=1),np.concatenate(row[2:],axis=1)],axis=0))
    save_video(OUT/'all_three_vs_real_2s.mp4',grids)
