"""Causal MolmoMotion -> rigid object -> dense DaS RGB tracking video.

This module never opens evaluation/ or any real future frames.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def kabsch(p0, target):
    a, b = p0.mean(0), target.mean(0)
    u, singular, vt = np.linalg.svd((p0-a).T @ (target-b))
    correction = np.diag([1., 1., np.linalg.det(vt.T @ u.T)])
    r = vt.T @ correction @ u.T
    t = b-r@a
    fit = p0@r.T+t
    return r, t, np.sqrt(np.mean(np.sum((fit-target)**2, axis=1))), singular


def interpolate_motion(rotations, translations, fps=8, frames=49):
    times = np.arange(31)/15
    query = np.minimum(np.arange(frames)/fps, 2.)
    r = Slerp(times, Rotation.from_matrix(np.concatenate([np.eye(3)[None], rotations])))(query).as_matrix()
    t = np.stack([np.interp(query, times, np.r_[0., translations[:, c]]) for c in range(3)], axis=-1)
    return r, t


def project(xyz, k):
    uvw = xyz@k.T
    return uvw[:, :2]/uvw[:, 2:3]


def backproject(uv, z, k):
    rays = np.c_[uv, np.ones(len(uv))]@np.linalg.inv(k).T
    return rays*z[:, None]


def object_depth(depth, mask):
    """Same observed Berkeley 5x5 finite 0<depth<10 patch-median rule."""
    y, x = np.nonzero(mask)
    patches = np.lib.stride_tricks.sliding_window_view(np.pad(depth, 2, constant_values=np.nan), (5, 5))[y, x]
    valid = np.isfinite(patches)&(patches>0)&(patches<10)
    # Mask is away from the image boundary; clipped-window rule is nevertheless retained.
    footprint = np.lib.stride_tricks.sliding_window_view(np.pad(np.ones_like(depth, bool), 2), (5, 5))[y, x]
    keep = valid.sum((1, 2)) >= np.maximum(3, (footprint.sum((1, 2))+1)//2)
    z = np.nanmedian(np.where(valid[keep], patches[keep], np.nan), axis=(1, 2))
    # A valid central measured pixel is also required for dense geometry.
    center_valid = np.isfinite(depth[y[keep], x[keep]])&(depth[y[keep], x[keep]]>0)&(depth[y[keep], x[keep]]<10)
    return np.c_[x[keep], y[keep]][center_valid], z[center_valid]


def splat(xyz, colors, k, canvas, zbuffer, radius=1):
    uv = project(xyz, k)
    good = np.isfinite(xyz).all(1)&np.isfinite(uv).all(1)&(xyz[:, 2]>0)
    good &= (uv[:, 0]>=-radius)&(uv[:, 0]<canvas.shape[1]+radius)&(uv[:, 1]>=-radius)&(uv[:, 1]<canvas.shape[0]+radius)
    indices = np.flatnonzero(good)
    xy = np.rint(uv[good]).astype(int)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            xx, yy = xy[:, 0]+dx, xy[:, 1]+dy
            ok = (xx>=0)&(xx<canvas.shape[1])&(yy>=0)&(yy<canvas.shape[0])
            ids, xx, yy = indices[ok], xx[ok], yy[ok]
            linear = yy*canvas.shape[1]+xx
            order = np.lexsort((xyz[ids, 2], linear))
            unique = np.r_[True, np.diff(linear[order])!=0]
            selected = order[unique]
            ids, xx, yy = ids[selected], xx[selected], yy[selected]
            front = xyz[ids, 2] < zbuffer[yy, xx]
            ids, xx, yy = ids[front], xx[front], yy[front]
            canvas[yy, xx] = colors[ids]
            zbuffer[yy, xx] = xyz[ids, 2]
    return good


def save_video(path, frames, fps=8):
    with imageio.get_writer(str(path), fps=fps, codec='libx264', quality=9,
                            macro_block_size=1, ffmpeg_params=['-pix_fmt', 'yuv420p']) as writer:
        for frame in frames:
            writer.append_data(frame)


def sheet(path, rows, labels, indices=(0, 4, 8, 12, 16), fps=8, tile=(320, 240)):
    w, h = tile
    canvas = Image.new('RGB', (w*len(indices), (h+30)*len(rows)), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (frames, name) in enumerate(zip(rows, labels)):
        for col, i in enumerate(indices):
            canvas.paste(Image.fromarray(frames[i]).resize(tile), (col*w, row*(h+30)+30))
            draw.text((col*w+8, row*(h+30)+8), f'{name}: {i/fps:.3f} s', fill='black')
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--das-root', type=Path, required=True)
    a = p.parse_args()
    scene, out = a.scene, a.scene/'das_wanfun'
    out.mkdir(parents=True, exist_ok=True)
    names = ['predictions/future_3d.npy', 'predictions/model_run.json', 'observed/points_3d_history.npy',
             'observed/points_2d_history.npy', 'observed/selected_point_ids.npy', 'observed/mask.png',
             'observed/native_depth.npy', 'observed/native_depth_metadata.json', 'geometry/K_median.npy',
             'observed/frame_000063.png', 'metadata.json', 'geometry/filter_metadata.json']
    frozen = {n: sha256(scene/n) for n in names}
    write_json(out/'control_input_freeze.json', {'future_used': False, 'sha256': frozen,
                                               'allowed_inputs': names})
    pred = np.load(scene/'predictions/future_3d.npy').astype(float)
    p0 = np.load(scene/'observed/points_3d_history.npy')[-1].astype(float)
    ids = np.load(scene/'observed/selected_point_ids.npy')
    meta = json.loads((scene/'metadata.json').read_text(encoding='utf8'))
    assert meta['source_episode_id']==10 and meta['t0']==63 and meta['source_fps']==5
    assert meta['instruction']=='Pick up the blue cup and put it into the brown cup.'
    receipt = json.loads((scene/'predictions/model_run.json').read_text(encoding='utf8'))
    assert receipt['success'] and pred.shape == (24, 30, 3) and p0.shape == (24, 3)
    assert np.isfinite(pred).all() and np.isfinite(p0).all()
    k = np.load(scene/'geometry/K_median.npy').astype(float)
    depth_receipt = json.loads((scene/'observed/native_depth_metadata.json').read_text(encoding='utf8'))
    assert depth_receipt['measured_metric_depth'] and depth_receipt['units']=='meters'
    depth = np.load(scene/'observed/native_depth.npy')[-1]
    rgb = np.array(Image.open(scene/'observed/frame_000063.png').convert('RGB'))
    mask = np.array(Image.open(scene/'observed/mask.png').convert('L'))>0
    assert rgb.shape==(480, 640, 3) and depth.shape==mask.shape==(480, 640)
    fits = [kabsch(p0, pred[:, i]) for i in range(30)]
    rotations, translations, residual, singular = map(np.array, zip(*fits))
    rigid = np.einsum('tij,nj->nti', rotations, p0)+translations[None]
    np.savez_compressed(out/'rigid_motion.npz', R=rotations, t=translations, residual=residual)
    write_json(out/'rigid_fit.json', {'method':'Kabsch SO(3); no scale or reflection',
        'future_used':False, 'points':len(p0), 'rms_residual_m':residual.tolist(),
        'mean_rms_residual_m':float(residual.mean()), 'max_rms_residual_m':float(residual.max()),
        'rotation_determinants':np.linalg.det(rotations).tolist(), 'singular_values':singular.tolist(),
        'source_points':'observed/points_3d_history.npy[-1], filtered MolmoMotion input',
        'dense_cloud_source':'native depth patch median at t0; may differ from filtered P0',
        'limitation':'Rigid motion projects a possibly nonrigid model forecast onto one rigid cup.'})
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, 31)/15, residual*1000, marker='o')
    ax.set(xlabel='Future time (s)', ylabel='Rigid fit RMS residual (mm)', title='MolmoMotion: consistency with one rigid cup')
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out/'rigid_fit_residual.png', dpi=160); plt.close(fig)
    fig = plt.figure(figsize=(10, 5))
    for col, (paths, title) in enumerate([(pred, 'Original MolmoMotion'), (rigid, 'Rigid approximation')], 1):
        ax = fig.add_subplot(1, 2, col, projection='3d')
        for trajectory in paths:
            ax.plot(*trajectory.T, alpha=.65)
        ax.scatter(*p0.T, c='black', s=7)
        ax.set(xlabel='Camera X (m)', ylabel='Camera Y (m)', zlabel='Camera Z (m)', title=title)
        combined = np.concatenate([pred.reshape(-1, 3), rigid.reshape(-1, 3), p0])
        for setter, d in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], range(3)):
            setter(combined[:, d].min(), combined[:, d].max())
    fig.tight_layout(); fig.savefig(out/'molmo_vs_rigid_3d.png', dpi=160); plt.close(fig)
    uv, z = object_depth(depth, mask)
    xyz = backproject(uv, z, k)
    scene_valid = np.isfinite(depth)&(depth>0)&(depth<10)
    inv = 1/depth[scene_valid]
    p2, p98 = np.percentile(inv, [2, 98])
    colors = np.stack([uv[:, 0]/639*255, uv[:, 1]/479*255,
                        np.clip((1/z-p2)/(p98-p2), 0, 1)*255], axis=1).astype(np.uint8)
    np.savez_compressed(out/'object_cloud_t0.npz', xyz=xyz, uv=uv, depth=z, colors=colors, K=k)
    overlay = rgb.copy()
    overlay[uv[:, 1], uv[:, 0]] = np.round(.45*rgb[uv[:, 1], uv[:, 0]]+.55*np.array([0, 255, 100])).astype(np.uint8)
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121); ax.imshow(overlay); ax.set_title(f't0 cup surface: {len(xyz)} measured points'); ax.axis('off')
    ax = fig.add_subplot(122, projection='3d'); ax.scatter(*xyz[::3].T, c=rgb[uv[::3, 1], uv[::3, 0]]/255, s=2)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Observed visible surface only')
    fig.tight_layout(); fig.savefig(out/'object_cloud_t0.png', dpi=160); plt.close(fig)
    r49, t49 = interpolate_motion(rotations, translations)
    dense = np.einsum('tij,nj->tni', r49, xyz)+t49[:, None]
    sparse49 = np.einsum('tij,nj->tni', r49, p0)+t49[:, None]
    assert np.allclose(dense[0], xyz, atol=1e-10)
    assert np.allclose(r49[17:], r49[16]) and np.allclose(t49[17:], t49[16])
    assert np.allclose(np.linalg.norm(dense[:, 1:]-dense[:, :1], axis=-1),
                       np.linalg.norm(xyz[1:]-xyz[:1], axis=-1), atol=1e-9)
    np.savez_compressed(out/'rigid_motion_49.npz', R=r49, t=t49, times=np.arange(49)/8)
    np.savez_compressed(out/'dense_predicted_motion.npz', xyz=dense.astype(np.float32),
                        sparse_xyz=sparse49.astype(np.float32), point_ids=ids, times=np.arange(49)/8)
    write_json(out/'temporal_mapping.json', {'molmo_future_times_s':(np.arange(1,31)/15).tolist(),
        'das_times_s':(np.arange(49)/8).tolist(), 'fps':8, 'num_frames':49, 'last_frame_time_s':6,
        'container_duration_s':49/8, 'prediction_horizon_s':2,
        'hold_final_pose_from_frame':17, 'translation':'linear', 'rotation':'quaternion SLERP',
        'image_transform':'PIL bilinear stretch from 640x480 to 720x480; identical RGB and control',
        'K_720':(np.diag([720/640,1,1])@k).tolist()})
    # Fixed background from measured t0 depth; remove original target before moving it.
    yy, xx = np.nonzero(scene_valid&~mask)
    bg_uv = np.c_[xx, yy]
    bg_xyz = backproject(bg_uv, depth[yy, xx], k)
    bg_colors = np.stack([xx/639*255, yy/479*255,
                         np.clip((1/depth[yy,xx]-p2)/(p98-p2),0,1)*255],axis=1).astype(np.uint8)
    bg_canvas = np.zeros_like(rgb); bg_z = np.full(mask.shape, np.inf)
    splat(bg_xyz, bg_colors, k, bg_canvas, bg_z, radius=0)
    native_frames, resized_frames, overlays, coverage = [], [], [], []
    for i in range(49):
        canvas, buffer = bg_canvas.copy(), bg_z.copy()
        good = splat(dense[i], colors, k, canvas, buffer, radius=1)
        native_frames.append(canvas)
        resized_frames.append(np.array(Image.fromarray(canvas).resize((720,480), Image.Resampling.BILINEAR)))
        view = rgb.copy()
        uv_i = project(dense[i], k)
        for x, y in np.rint(uv_i[good][::12]).astype(int):
            if 0<=x<640 and 0<=y<480:
                cv2.circle(view, (x,y), 1, (255,0,255), -1)
        overlays.append(view)
        coverage.append(float(good.mean()))
    save_video(out/'control_molmomotion_640x480.mp4', native_frames)
    save_video(out/'control_molmomotion_720x480.mp4', resized_frames)
    sheet(out/'control_contact_sheet.png', [native_frames, overlays], ['DaS dense tracking', 'Projected cup on t0 RGB'])
    Image.fromarray(rgb).resize((720,480),Image.Resampling.BILINEAR).save(out/'image_t0_720x480.png')
    import subprocess
    commit = subprocess.check_output(['git','-C',str(a.das_root),'rev-parse','HEAD'], text=True).strip()
    (out/'das_wanfun_commit.txt').write_text(commit+'\n', encoding='utf8')
    check = {'future_used':False, 'source_prediction_shape':list(pred.shape), 'dense_points':len(xyz),
             'mask_pixels':int(mask.sum()), 'initial_pose_identity':True, 'rigid_shape_preserved':True,
             'colors_constant_per_physical_point':True, 'depth_encoding':'t0 inverse depth p2/p98 over valid measured scene',
             'zbuffer':True, 'static_background':True, 'object_in_frame_fraction':coverage,
             'generation_ready':False, 'visual_review_required':True,
             'gripper':'fixed observed geometry, no independently predicted arm/gripper motion',
             'native_depth_only':True, 'observed_P0_projection_offset_px_mean':float(np.linalg.norm(project(p0,k)-np.load(scene/'observed/points_2d_history.npy')[-1],axis=1).mean()),
             'control_sha256':sha256(out/'control_molmomotion_720x480.mp4')}
    write_json(out/'control_validation.json', check)
    print(json.dumps({'dense_points':len(xyz),'rigid_rms_mean_m':float(residual.mean()),
                      'in_frame_at_2s':coverage[16], 'artifacts':str(out)},indent=2))


if __name__=='__main__':
    main()
