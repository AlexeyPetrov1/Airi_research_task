"""Frozen group_00 forecast -> dense control, without real future input.

The t0 photographic guide is a separate diffusion condition, never an output.
Only local wrist/gripper surfaces follow the cup; the proximal arm is free.
"""
import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from scipy.spatial.transform import Rotation, Slerp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_prepare_control import project, backproject, splat, save_video, sheet, sha256, write_json
from das_full_motion_diagnose import ROOT, SCENE, OUT, fit_motion
from das_full_motion_safety import forbid_real_future


def sample_motion(r, t, timing):
    source = np.arange(31)/15
    duration = {'physical_2s':2., 'stretched_4s':4., 'stretched_6s':6.}[timing]
    query = np.minimum(np.arange(49)/8/duration*2, 2.)
    rr = Slerp(source, Rotation.from_matrix(np.concatenate([np.eye(3)[None], r])))(query).as_matrix()
    tt = np.stack([np.interp(query, source, np.r_[0., t[:, c]]) for c in range(3)], axis=-1)
    return rr, tt, query


def load_mask(path):
    return np.array(Image.open(path).convert('L')) > 0


def color_surface(uv, z, limits):
    lo, hi = limits
    return np.stack([uv[:, 0]/639*255, uv[:, 1]/479*255,
        np.clip((1/z-lo)/(hi-lo), 0, 1)*255], axis=1).astype(np.uint8)


def main():
    forbid_real_future()
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['group00', 'all24', 'robust24'], default='group00')
    parser.add_argument('--timing', choices=['physical_2s', 'stretched_4s', 'stretched_6s'], default='stretched_6s')
    args = parser.parse_args()
    out = OUT/f'{args.method}_{args.timing}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'preparation.json').exists():
        raise FileExistsError('Prepared conditioning is immutable; choose a new experiment.')
    names = ['predictions/future_3d.npy', 'predictions/model_run.json',
        'observed/points_3d_history.npy', 'observed/points_2d_history.npy',
        'observed/selected_point_ids.npy', 'observed/frame_000063.png',
        'observed/mask.png', 'observed/native_depth.npy', 'geometry/K_median.npy',
        'das_wanfun/object_cloud_t0.npz', 'das_robot_cup/observed/gripper_mask.png',
        'das_robot_cup/observed/wrist_mask.png', 'das_robot_cup/observed/forearm_mask.png',
        'das_robot_cup/observed/clean_reference_640x480.png',
        'das_robot_cup/observed/imagegen_receipt.json']
    frozen = {name: sha256(SCENE/name) for name in names}
    write_json(out/'input_freeze.json', {'future_used': False, 'sha256': frozen})
    rgb = np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    depth = np.load(SCENE/'observed/native_depth.npy')[-1].astype(float)
    k = np.load(SCENE/'geometry/K_median.npy').astype(float)
    pred = np.load(SCENE/'predictions/future_3d.npy').astype(float)
    p0 = np.load(SCENE/'observed/points_3d_history.npy')[-1].astype(float)
    assert pred.shape == (24, 30, 3) and p0.shape == (24, 3)
    assert np.array_equal(pred[:8], np.load(SCENE/'predictions/group_00/future_3d.npy'))
    r, t, residual, weights = fit_motion(p0, pred, args.method)
    rr, tt, query = sample_motion(r, t, args.timing)
    cup = np.load(SCENE/'das_wanfun/object_cloud_t0.npz')
    cup_xyz, cup_uv = cup['xyz'].astype(float), cup['uv'].astype(int)
    cup_mask = load_mask(SCENE/'observed/mask.png')
    obs = SCENE/'das_robot_cup/observed'
    wrist = load_mask(obs/'wrist_mask.png')
    gripper = load_mask(obs/'gripper_mask.png')
    forearm = load_mask(obs/'forearm_mask.png')
    local = (wrist | gripper) & ~cup_mask
    ry, rx = np.nonzero(local)
    robot_uv = np.c_[rx, ry]
    robot_z = np.empty(len(rx))
    valid = np.isfinite(depth) & (depth > 0) & (depth < 10)
    estimated = np.zeros(len(rx), bool)
    for part in [gripper, wrist & ~gripper]:
        sel = part[ry, rx]
        good = valid & part
        if not good.any():
            raise ValueError('No measured local robot depth')
        nearest = distance_transform_edt(~good, return_distances=False, return_indices=True)
        robot_z[sel] = depth[tuple(nearest)][ry[sel], rx[sel]]
        estimated[sel] = ~valid[ry[sel], rx[sel]]
    robot_xyz = backproject(robot_uv, robot_z, k)
    xyz = np.r_[cup_xyz, robot_xyz]
    uv0 = np.r_[cup_uv, robot_uv]
    limits = np.percentile(1/depth[valid], [2, 98])
    colors = color_surface(uv0, xyz[:, 2], limits)
    texture = rgb[uv0[:, 1], uv0[:, 0]]
    dense = np.einsum('tij,nj->tni', rr, xyz)+tt[:, None]
    sparse = np.einsum('tij,nj->tni', rr, p0[:8])+tt[:, None]
    assert np.allclose(dense[0], xyz, atol=1e-10)
    assert np.allclose(dense[-1], xyz@r[-1].T+t[-1])
    assert np.allclose(np.linalg.norm(dense[:, 1:]-dense[:, :1], axis=-1),
        np.linalg.norm(xyz[1:]-xyz[:1], axis=-1), atol=1e-9)
    if args.timing == 'physical_2s':
        assert np.allclose(dense[16:], dense[16])
    np.savez_compressed(out/'motion.npz', R=rr, t=tt, xyz=dense.astype(np.float32),
        sparse_xyz=sparse.astype(np.float32), source_times=query, times=np.arange(49)/8,
        cup_points=len(cup_xyz), uv0=uv0, initial_xyz=xyz,
        fit_R=r, fit_t=t, fit_residual=residual, fit_weights=weights)
    # Exclude the proximal left arm from static control: it must be free to move.
    exclusion = cv2.dilate((cup_mask|local|forearm).astype(np.uint8), np.ones((3,3), np.uint8)) > 0
    by, bx = np.nonzero(valid & ~exclusion)
    bg = np.zeros_like(rgb)
    bg[by, bx] = color_surface(np.c_[bx, by], depth[by, bx], limits)
    bg_depth = np.full(depth.shape, np.inf)
    bg_depth[by, bx] = depth[by, bx]
    plate = np.array(Image.open(obs/'clean_reference_640x480.png').convert('RGB'))
    photographic_bg = rgb.copy()
    vacated = cv2.dilate((cup_mask|local).astype(np.uint8), np.ones((9,9), np.uint8)) > 0
    photographic_bg[vacated] = plate[vacated]
    control, overlays, guides, priormasks, coverage, centers = [], [], [], [], [], []
    for i, framexyz in enumerate(dense):
        canvas, zbuffer = bg.copy(), bg_depth.copy()
        splat(framexyz, colors, k, canvas, zbuffer, radius=1)
        control.append(np.array(Image.fromarray(canvas).resize((720,480), Image.Resampling.BILINEAR)))
        cupframe = framexyz[:len(cup_xyz)]
        cupuv = project(cupframe, k)
        centers.append(cupuv.mean(0))
        coverage.append(float(((cupuv[:,0]>=0)&(cupuv[:,0]<640)&(cupuv[:,1]>=0)&(cupuv[:,1]<480)).mean()))
        view = rgb.copy()
        for x, y in np.rint(cupuv[::10]).astype(int):
            if 0<=x<640 and 0<=y<480:
                cv2.circle(view, (x,y), 1, (255,0,255), -1)
        # Brown rim center annotated only on observed t0; not a path endpoint.
        cv2.drawMarker(view, (295,217), (255,200,0), cv2.MARKER_CROSS, 15, 2)
        cx, cy = centers[-1]
        if 0<=cx<640 and 0<=cy<480:
            cv2.drawMarker(view, tuple(np.rint(centers[-1]).astype(int)), (0,255,255), cv2.MARKER_CROSS, 11, 2)
        overlays.append(view)
        photo = np.zeros_like(rgb)
        photodepth = np.full(depth.shape, np.inf)
        splat(framexyz, texture, k, photo, photodepth, radius=1)
        support = np.isfinite(photodepth)
        filled = binary_fill_holes(cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))>0)
        if support.any():
            nearest = distance_transform_edt(~support, return_distances=False, return_indices=True)
            photo[filled] = photo[tuple(nearest)][filled]
        guide = photographic_bg.copy()
        guide[filled] = photo[filled]
        prior = np.full(depth.shape, .03, np.float32)
        prior[vacated] = .5
        prior[cv2.dilate(filled.astype(np.uint8), np.ones((5,5), np.uint8))>0] = 1.
        if i == 0:
            guide = rgb.copy()
            prior[:] = 1.
        guides.append(np.array(Image.fromarray(guide).resize((720,480), Image.Resampling.BILINEAR)))
        priormasks.append(cv2.resize(prior, (720,480), interpolation=cv2.INTER_LINEAR))
    save_video(out/'control_720x480.mp4', control)
    save_video(out/'photographic_guide.mp4', guides)
    np.savez_compressed(out/'guide_720x480.npz', frames=np.array(guides))
    np.savez_compressed(out/'prior_masks.npz', weights=np.array(priormasks))
    Image.fromarray(guides[0]).save(out/'image_t0_720x480.png')
    sheet(out/'control_contact_sheet.png', [control, overlays, guides],
        ['Dense tracking', 'Cup control on t0 (magenta)', 'T0-only photographic guide'], indices=(0,12,24,36,48))
    centers = np.array(centers)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.imshow(rgb, alpha=.45)
    ax.plot(*centers.T, '-o', markersize=3, label=f'{args.method}, {args.timing}')
    ax.scatter(295,217,c='brown',s=80,label='Observed brown rim')
    ax.set(xlim=(0,640),ylim=(480,-100),xlabel='Native u (px)',ylabel='Native v (px)')
    ax.legend();fig.tight_layout();fig.savefig(out/'center_path.png',dpi=150);plt.close(fig)
    shutil.copy2(SCENE/'das_reference_repair/checkpoint_receipt.json', out/'checkpoint_receipt.json')
    prep = {'future_used':False, 'forecast_geometry_changed':False, 'method':args.method,
        'timing':args.timing, 'source_horizon_s':2, 'motion_duration_s':{'physical_2s':2,'stretched_4s':4,'stretched_6s':6}[args.timing],
        'timing_description':'physical time then hold' if args.timing=='physical_2s' else 'time-stretched visualization of the predicted spatial trajectory',
        'rgb_reference':'t0 only; no frame73 or future RGB', 'proximal_robot':'no trajectory control; synthesis under prompt',
        'local_robot':'t0 wrist + gripper follow the exact same SE(3) as cup',
        'local_robot_estimated_depth_fraction':float(estimated.mean()), 'dense_cup_points':len(cup_xyz),
        'dense_robot_points':len(robot_xyz), 'background_plate':'existing t0-only synthetic clean plate',
        'mean_fit_residual_m':float(residual.mean()), 'cup_in_frame_fraction':coverage,
        'cup_centers_uv':centers.tolist(), 'initial_brown_rim_uv':[295,217],
        'source_point_ids':np.load(SCENE/'observed/selected_point_ids.npy')[:8].tolist(),
        'control_sha256':sha256(out/'control_720x480.mp4'), 'guide_sha256':sha256(out/'guide_720x480.npz'),
        'prior_masks_sha256':sha256(out/'prior_masks.npz'), 'background_mode':'observed-only synthetic vacancy',
        'postprocessed_generated_RGB':False, 'generation_ready':False, 'visual_review_required':True}
    write_json(out/'preparation.json', prep)
    assert frozen == {name:sha256(SCENE/name) for name in names}
    print({'preparation':str(out), 'center_start':centers[0].tolist(), 'center_end':centers[-1].tolist(),
        'min_in_frame_fraction':min(coverage), 'cup_points':len(cup_xyz), 'robot_points':len(robot_xyz)})


if __name__ == '__main__':
    main()
