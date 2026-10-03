"""Observed-only controlled geometry cases with frozen Berkeley point identities.

CASE-AUGE isolates ray calibration; native depth and author-smoothed Z stay fixed.
CASE-NOK estimates rays from genuine MoGe-2 RGB inference with fov_x=None,
then uses those rays with the same measured sensor Z. No K/FOV is supplied.
DELTA default calibration is audited separately, since it hardcodes f=W.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/berkeley_ur5_molmomotion'
OUT=ROOT/'runs/berkeley_ur5_improvement_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def prepare(name,case,K,receipt,strict_geometry=False):
    import torch
    from berkeley_input_audit import audit
    src=BASE/name; dst=OUT/case/name
    if (dst/'geometry/case_provenance.json').exists():
        raise FileExistsError(f'Prepared case already exists: {dst}')
    dst.mkdir(parents=True,exist_ok=True)
    for folder in ('observed','geometry','groups'):
        shutil.copytree(src/folder,dst/folder,dirs_exist_ok=True)
    shutil.copy2(src/'metadata.json',dst/'metadata.json')
    (dst/'viz').mkdir(exist_ok=True)
    if strict_geometry:
        # No original K/XYZ/trust weights are read to construct new geometry.
        from berkeley_preprocess import robust_lift,load_author_filter
        tracks=np.load(src/'observed/observed_tracks_2d.npz')
        depth=np.load(src/'observed/native_depth.npy')
        xyz,valid,depth_fraction,spread=robust_lift(depth,tracks['tracks'],tracks['visibility'],K,patch_size=5)
        author=load_author_filter()
        trust,anchors=author.compute_trust_weights(xyz,valid,K=16)
        drop=author.filter_tracks_by_trust(trust,valid,z_thresh=2.)
        # Frozen query IDs are retained by design; report changed eligibility.
        filtered=xyz.copy()
        usable=np.isfinite(xyz[-3:]).all((0,2)) & valid[-3:].all(0)
        filtered[:,usable]=author.consensus_gated_smooth(xyz[:,usable],np.zeros((len(xyz),3),np.float32),
            valid[:,usable],trust[:,usable],device='cpu')
        np.save(dst/'geometry/points_3d_raw.npy',xyz)
        np.save(dst/'geometry/points_3d_filtered.npy',filtered)
        ids=np.load(src/'observed/selected_point_ids.npy')
        np.savez_compressed(dst/'geometry/filter_diagnostics.npz',trust=trust,anchor_ids=anchors,
            keep=~drop,eligible=usable,depth_valid_fraction=depth_fraction,depth_patch_spread_m=spread)
        write(dst/'geometry/strict_geometry_audit.json',dict(original_K_read_for_geometry=False,
            original_XYZ_read_for_geometry=False,original_trust_weights_reused=False,
            measured_sensor_depth_unchanged=True,point_selection_frozen_by_design=True,
            selected_ids_flagged_by_new_outlier_filter=ids[drop[ids]].tolist(),
            smoothing='official consensus_gated_smooth applied to all H3-valid tracks; fixed selected identities retained'))
        write(dst/'geometry/filter_metadata.json',dict(
            geometry_source='native sensor depth lifted with observed-only MoGe-2 estimated rays',
            author_functions=['compute_trust_weights', 'filter_tracks_by_trust', 'consensus_gated_smooth'],
            trust_anchor_count=16,outlier_z_threshold=2.,
            original_K_used=False,original_filtered_XYZ_used=False,original_trust_used=False,
            selected_point_ids_frozen=ids.tolist(),
            selected_ids_flagged_by_new_outlier_filter=ids[drop[ids]].tolist(),
            flagged_selected_ids_retained=True,
            smoothing_applied_to='all tracks with valid finite H3; point selection not repeated'))
    else:
        oldK=np.load(src/'geometry/K_median.npy')
        change=np.linalg.inv(K)@oldK
        for filename in ('points_3d_raw.npy','points_3d_filtered.npy'):
            values=np.load(src/'geometry'/filename)
            new=(values@change.T).astype(np.float32)
            if not np.allclose(new[...,2],values[...,2],atol=1e-7,equal_nan=True):
                raise ValueError('Measured/smoothed sensor Z changed')
            np.save(dst/'geometry'/filename,new)
    filtered=np.load(dst/'geometry/points_3d_filtered.npy')
    ids=np.load(src/'observed/selected_point_ids.npy')
    hist=filtered[-3:,ids]
    np.save(dst/'observed/points_3d_history.npy',hist)
    np.save(dst/'geometry/K_median.npy',K.astype(np.float32))
    np.save(dst/'geometry/K_per_frame.npy',np.repeat(K[None],len(filtered),axis=0).astype(np.float32))
    for i in range(3):
        group=dst/f'groups/group_{i:02d}'
        values=hist[:,i*8:(i+1)*8]
        np.save(group/'points_3d_history.npy',values)
        torch.save(torch.from_numpy(values),group/'points_3d_history.pt')
    meta=json.loads((dst/'metadata.json').read_text())
    meta['K_source']=receipt['K_source'];meta['geometry_case']=case
    meta['warnings']=list(meta.get('warnings',[]))+[receipt['limitation']]
    write(dst/'metadata.json',meta)
    depth=json.loads((dst/'geometry/depth_source.json').read_text())
    depth['K_source']=receipt['K_source'];write(dst/'geometry/depth_source.json',depth)
    provenance=dict(**receipt,K=K.tolist(),future_used=False,
        baseline='../../../../berkeley_ur5_molmomotion',selected_point_ids=ids.tolist(),
        only_ray_basis_changed=not strict_geometry,author_filter_not_reexecuted=not strict_geometry,
        observed_sensor_depth_sha256=sha(dst/'observed/native_depth.npy'),
        unchanged=['RGB','mask','2D tracks','all 24 point IDs','sensor depth','camera poses','instruction']
            + ([] if strict_geometry else ['smoothed Z']))
    # Stale UniDepth estimates are retained only as historical source evidence.
    audit(dst)
    write(dst/'geometry/case_provenance.json',provenance)
    print(json.dumps(dict(scene=name,case=case,success=True,K=K.tolist())),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=['CASE-AUGE','CASE-NOK','CASE-NOK-STRICT','CASE-H1'],required=True)
    p.add_argument('--scenes',nargs='+',default=['cup'],choices=['cup','bottle']);a=p.parse_args()
    if a.case=='CASE-AUGE':
        config=OUT/'sources/auge_config.py'
        # Parse the source as text; do not execute third-party configuration.
        text=config.read_text(); start=text.index('"berkeley_autolab_ur5":'); part=text[start:start+1500]
        import re
        fov=float(re.search(r'"camera_fov":\s*([0-9.]+)',part).group(1))
        focal=480/2/np.tan(np.radians(fov)/2)
        K=np.array([[focal,0,320],[0,focal,240],[0,0,1]],float)
        receipt=dict(K_source='OXE-AugE Berkeley tuned simulation vertical FOV; centered pinhole derived at 640x480',
            fov_y_deg=fov,formula='f=H/(2*tan(fovy/2)), cx=W/2, cy=H/2',
            source_commit='0added8b6645fef1c7de2ad5d5bd7e9c1bde0d2e',source_config_sha256=sha(config),
            limitation='AugE tuned simulation camera is a candidate prior, not measured sensor calibration.')
        for n in a.scenes:prepare(n,a.case,K,receipt)
    elif a.case=='CASE-NOK-STRICT':
        for n in a.scenes:
            maps=OUT/'CASE-NOK'/n/'geometry/moge_observed_outputs.npz'
            values=np.load(maps)
            K=np.median(values['intrinsics_per_frame'],axis=0).astype(float)
            receipt=json.loads((OUT/'CASE-NOK'/n/'geometry/case_provenance.json').read_text())
            for key in ('K','future_used','baseline','selected_point_ids','only_ray_basis_changed',
                        'author_filter_not_reexecuted','observed_sensor_depth_sha256','unchanged'):
                receipt.pop(key,None)
            receipt['limitation']='Independent MoGe rays; all raw XYZ and author trust/smoothing recomputed from native depth and unchanged 2D tracks. Fixed IDs retained for comparability.'
            receipt['moge_snapshot_sha256']=sha(maps)
            prepare(n,a.case,K,receipt,strict_geometry=True)
            shutil.copy2(maps,OUT/a.case/n/'geometry/moge_observed_outputs.npz')
    elif a.case=='CASE-H1':
        for n in a.scenes:
            K=np.load(BASE/n/'geometry/K_median.npy').astype(float)
            prepare(n,a.case,K,dict(K_source='Same frozen BASELINE-U UniDepth intrinsics',
                limitation='H1 changes both history conditioning and checkpoint; it is a control, not a causal isolation of FPS.'))
    else:
        import torch
        sys.path.insert(0,str(ROOT.parent/'third_party/MoGe'))
        from moge.model.v2 import MoGeModel
        modelpath=ROOT.parent/'models/moge-2-vitl/model.pt'
        expected='3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5'
        if sha(modelpath)!=expected:raise ValueError('Unexpected MoGe-2 checkpoint')
        model=MoGeModel.from_pretrained(modelpath).cuda().eval()
        for n in a.scenes:
            rgb=np.load(BASE/n/'observed/rgb.npy')
            matrices=[];depths=[]
            started=__import__('time').monotonic()
            for index,frame in enumerate(rgb):
                tensor=torch.from_numpy(frame.copy()).permute(2,0,1).cuda().float()/255
                with torch.inference_mode():
                    output=model.infer(tensor,num_tokens=1200,use_fp16=True,apply_mask=False,fov_x=None)
                K=output['intrinsics'].float().cpu().numpy()
                K[0]*=640;K[1]*=480
                matrices.append(K);depths.append(output['depth'].float().cpu().numpy())
                print('MoGe no supplied FOV',n,index,float(K[0,0]),flush=True)
            K=np.median(np.stack(matrices),axis=0)
            receipt=dict(K_source='MoGe-2 independently inferred camera rays from observed RGB; no supplied K or FOV',
                supplied_intrinsics=None,supplied_fov=None,checkpoint_sha256=expected,
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'third_party/MoGe',text=True).strip(),
                model_calls=len(rgb),runtime_s=__import__('time').monotonic()-started,
                limitation='Learned focal estimate from RGB; sensor metric Z retained. Independent geometry, not calibration ground truth.')
            prepare(n,a.case,K,receipt)
            np.savez_compressed(OUT/a.case/n/'geometry/moge_observed_outputs.npz',
                intrinsics_per_frame=np.stack(matrices),model_depth=np.stack(depths))


if __name__=='__main__':main()
