"""Geometry comparisons on observed t0 and frozen predictions only."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_full_motion_diagnose import ROOT, SCENE, OUT
from das_prepare_control import project, save_video, sheet, sha256, write_json
from das_robot_evaluate import frames, labeled


def main():
    rgb = np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    p0 = np.load(SCENE/'observed/points_3d_history.npy')[-1]
    pred = np.load(SCENE/'predictions/future_3d.npy')
    k = np.load(SCENE/'geometry/K_median.npy')
    uv0 = project(p0, k)
    uv = project(pred.transpose(1,0,2).reshape(-1,3),k).reshape(30,24,2)
    displacement = np.array([np.median(uv[:,g*8:(g+1)*8]-uv0[None,g*8:(g+1)*8],axis=1) for g in range(3)])
    consensus = np.median(displacement,axis=0)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(rgb,alpha=.4)
    for name, delta in list(zip(['group_00','group_01','group_02'],displacement))+[('componentwise median',consensus)]:
        path = uv0[:8].mean(0)+np.concatenate([np.zeros((1,2)),delta])
        ax.plot(*path.T, '-o',markersize=3,label=name)
    ax.set(xlim=(0,640),ylim=(480,-100),xlabel='u (px)',ylabel='v (px)',title='Independent P8 groups: different global motion')
    ax.legend();fig.tight_layout();fig.savefig(OUT/'group_paths.png',dpi=160);plt.close(fig)
    np.savez_compressed(OUT/'group_displacements.npz',group_displacements=displacement,consensus=consensus,raw_projected_points=uv)
    methods=['group00','all24','robust24']
    clips=[frames(OUT/f'{m}_stretched_6s/control_720x480.mp4') for m in methods]
    rows=[np.array([cv2.resize(f,(640,480)) for f in clip]) for clip in clips]
    sheet(OUT/'rigid_methods_contact_sheet.png',rows,['GROUP00 rigid','ALL24 Kabsch','ALL24 robust'],indices=(0,12,24,36,48))
    comparison=[]
    for i in range(49):
        comparison.append(np.concatenate([labeled(rows[j][i],methods[j]) for j in range(3)],axis=1))
    save_video(OUT/'rigid_methods_comparison.mp4',comparison)
    timing={name:np.load(OUT/f'group00_{name}/motion.npz') for name in ['physical_2s','stretched_4s','stretched_6s']}
    temporal=[]
    for frame in range(49):
        row=[]
        for name,motion in timing.items():
            view=rgb.copy()
            path=project(motion['sparse_xyz'][:frame+1].reshape(-1,3),k).reshape(frame+1,8,2)
            rawt=np.linspace(0,30,49) if name=='stretched_6s' else np.minimum(np.arange(49)*30/(16 if name=='physical_2s' else 32),30)
            for point in range(8):
                pts=np.rint(path[:,point]).astype(int)
                cv2.polylines(view,[pts],False,(255,50,170),1)
                cv2.circle(view,tuple(pts[-1]),2,(255,50,170),-1)
            row.append(labeled(view,f'{name} | frame {frame} | step {rawt[frame]:.2f}'))
        temporal.append(np.concatenate(row,axis=1))
    save_video(OUT/'timing_control_comparison.mp4',temporal)
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        assert value,name
    for name,motion in timing.items():
        check(f'{name}: initial identity',np.allclose(motion['R'][0],np.eye(3),atol=1e-9) and np.allclose(motion['t'][0],0))
        check(f'{name}: final forecast pose',np.allclose(motion['R'][-1],motion['fit_R'][-1]) and np.allclose(motion['t'][-1],motion['fit_t'][-1]))
        prep=json.loads((OUT/f'group00_{name}/preparation.json').read_text())
        check(f'{name}: no future input',prep['future_used'] is False and prep['forecast_geometry_changed'] is False)
        check(f'{name}: frozen inputs',all(sha256(SCENE/p)==h for p,h in json.loads((OUT/f'group00_{name}/input_freeze.json').read_text())['sha256'].items()))
    for i in range(17):
        check(f'same path at source sample {i}', all(np.allclose(timing['physical_2s'][key][i],timing['stretched_4s'][key][2*i]) and
            np.allclose(timing['physical_2s'][key][i],timing['stretched_6s'][key][3*i]) for key in ['R','t','sparse_xyz']))
    preserved={}
    for root in [Path('/mnt/f/AIRI_task/molmo-motion'),Path('/mnt/f/AIRI_task/das_reference_repair_20261003')]:
        for rel in ['runs/berkeley_ur5_improvement_v1/cup/fixed_comparison.mp4',
            'runs/berkeley_ur5_molmomotion/cup/predictions/future_3d.npy',
            'runs/berkeley_ur5_molmomotion/cup/das_reference_repair/guided_endpoint_background/generated_seed42.mp4']:
            path=root/rel
            if path.exists():preserved[str(path)] = sha256(path)
    write_json(OUT/'preserved_baselines.json',preserved)
    write_json(OUT/'cpu_verification.json',{'passed':len(checks),'checks':checks,'future_used':False})
    print({'checks_passed':len(checks),'output':str(OUT)})


if __name__=='__main__':
    main()
