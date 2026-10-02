"""Pre-inference quality/fence audit, using observed data only."""
from pathlib import Path
import argparse,json
import cv2
import numpy as np


def validate_group_payloads(scene):
    """Reject stale/reordered model payloads before any expensive forward call."""
    scene = Path(scene)
    observed = scene / 'observed'
    ids = np.load(observed / 'selected_point_ids.npy', allow_pickle=False)
    history2d = np.load(observed / 'points_2d_history.npy', allow_pickle=False)
    history3d = np.load(observed / 'points_3d_history.npy', allow_pickle=False)
    if (ids.ndim != 1 or len(ids) not in (8, 16, 24) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= 100))):
        raise ValueError('Selected IDs must be 8/16/24 unique integer candidate IDs')
    if history2d.shape != (3, len(ids), 2) or history3d.shape != (3, len(ids), 3):
        raise ValueError('Canonical observed history shapes do not match selected IDs')
    if not np.isfinite(history2d).all() or not np.isfinite(history3d).all() or (history3d[..., 2] <= 0).any():
        raise ValueError('Canonical model history contains nonfinite points or nonpositive depth')
    groups = sorted(path for path in (scene / 'groups').glob('group_*') if path.is_dir())
    if [path.name for path in groups] != [f'group_{i:02d}' for i in range(len(ids)//8)]:
        raise ValueError('Model groups must be contiguous, ordered chunks of eight, capped at three')
    for index, group in enumerate(groups):
        subset = slice(index*8, (index+1)*8)
        expectations = {'point_ids.npy': ids[subset],
                        'points_2d_at_t0.npy': history2d[-1, subset],
                        'points_3d_history.npy': history3d[:, subset]}
        for name, expected in expectations.items():
            actual = np.load(group / name, allow_pickle=False)
            if not np.array_equal(actual, expected):
                raise ValueError(f'Stale or reordered group payload: {group.name}/{name}')
    return {'group_count': len(groups), 'group_names': [group.name for group in groups],
            'selected_point_ids': ids.tolist(), 'all_group_payloads_equal_canonical_history': True}


def audit(scene):
    assert not (scene/'predictions/input_freeze.json').exists(), 'Audit must precede inference freeze'
    meta=json.loads((scene/'metadata.json').read_text())
    obs=scene/'observed'
    rgb=np.load(obs/'rgb.npy')
    idx=np.load(obs/'source_indices.npy')
    hidx=np.load(obs/'history_source_indices.npy')
    mask=cv2.imread(str(obs/'mask.png'),cv2.IMREAD_GRAYSCALE)>0
    depth=np.load(obs/'native_depth.npy')
    tracks=np.load(obs/'observed_tracks_2d.npz')
    ids=np.load(obs/'selected_point_ids.npy')
    raw=np.load(scene/'geometry/points_3d_raw.npy')
    filtered=np.load(scene/'geometry/points_3d_filtered.npy')
    K=np.load(scene/'geometry/K_median.npy')
    motion=json.loads((scene/'geometry/camera_motion_audit.json').read_text())
    grounding=json.loads((obs/'grounding_metadata.json').read_text())
    assert idx[-1]==meta['t0_source_frame'] and (np.diff(idx)==1).all()
    assert np.array_equal(idx[-3:],hidx) and np.array_equal(rgb[-3:],np.load(obs/'history_rgb.npy'))
    assert np.array_equal(depth,np.load(scene/'geometry/depth_observed.npy'))
    assert motion['fixed_camera_supported']
    assert grounding['segmentation']=='official Meta SAM2.1 Hiera-Large'
    mask_valid=float(((depth[-1]>0)&(depth[-1]<10)&np.isfinite(depth[-1]))[mask].mean())
    assert mask_valid>.5, f'Most target depth must be valid, found {mask_valid}'
    xyz=filtered[-3:,ids]
    assert np.array_equal(xyz, np.load(obs/'points_3d_history.npy')), 'Canonical 3D history differs from filtered selected points'
    assert np.array_equal(tracks['tracks'][-3:,ids], np.load(obs/'points_2d_history.npy')), 'Canonical 2D history differs from selected observed tracks'
    group_audit = validate_group_payloads(scene)
    uv=xyz[...,:2]/xyz[...,2,None]*[K[0,0],K[1,1]]+[K[0,2],K[1,2]]
    reproj=np.linalg.norm(uv-tracks['tracks'][-3:,ids],axis=-1)
    assert np.isfinite(xyz).all() and (xyz[...,2]>0).all() and reproj.max()<.01
    assert len(ids) in (8,16,24)
    assert tracks['visibility'][-3:,ids].all()
    result={'success':True,'future_used':False,'selected_points':len(ids),
        'group_payload_audit':group_audit,
        'source_indices':idx.tolist(),'history_indices':hidx.tolist(),
        'mask_area_px':int(mask.sum()),'mask_valid_depth_fraction_t0':mask_valid,
        'selected_H3_depth_range_m':[float(xyz[...,2].min()),float(xyz[...,2].max())],
        'selected_H3_smoothing_displacement_m_max':float(np.linalg.norm(filtered[-3:,ids]-raw[-3:,ids],axis=-1).max()),
        'ray_preservation_max_reprojection_error_px':float(reproj.max()),
        'background_median_displacement_px_max':motion['max_pair_median_displacement_px'],
        'depth_is_native_array_exactly':True,
        'visual_mask_review_required':'Root separately inspects saved mask_overlay.png before inference'}
    (scene/'geometry/input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir',type=Path,nargs='+',required=True)
    for scene in p.parse_args().scene_dir:audit(scene)
