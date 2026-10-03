"""Post-generation motion/appearance assessment against the frozen forecast.

Stretched clips are compared at equal forecast phase; they are not evaluated
against real 6-second continuations. Real future access remains evaluation-only.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_full_motion_diagnose import SCENE, OUT
from das_prepare_control import project, sheet, save_video, sha256, write_json
from das_robot_evaluate import frames, tracker, valid, tiled_all, labeled
from das_evaluate import interpolate, metrics
from das_quality import temporal_warp_error, cup_appearance


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--preview-only',action='store_true')
    p.add_argument('--name')
    p.add_argument('--include-real',action='store_true')
    a=p.parse_args()
    paths=sorted(OUT.glob('*/generated_seed42.mp4'))
    if a.name:paths=[path for path in paths if path.parent.name==a.name]
    assert paths,'No complete generated video'
    reference=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    uv0=np.load(SCENE/'observed/points_2d_history.npy')[-1,:8].astype(float)
    k=np.load(SCENE/'geometry/K_median.npy')
    p0=np.load(SCENE/'observed/points_3d_history.npy')[-1,:8].astype(float)
    raw=np.load(SCENE/'predictions/future_3d.npy')[:8].astype(float)
    rawuv=np.stack([project(p0,k)]+[project(raw[:,i],k) for i in range(30)])
    videos={};reports={}
    for path in paths:
        out=path.parent
        resource=json.loads((out/'resource_usage.json').read_text())
        assert resource['success'] and resource['output_sha256']==sha256(path)
        config=json.loads((out/'config.json').read_text())
        prep=OUT/Path(config['preparation_root']).name
        frozen=json.loads((prep/'input_freeze.json').read_text())['sha256']
        assert all(sha256(SCENE/name)==digest for name,digest in frozen.items())
        assert config['future_used'] is False
        clip=frames(path);assert clip.shape==(49,480,720,3)
        video=np.stack([cv2.resize(f,(640,480)) for f in clip])
        videos[out.name]=video
        tiled_all(out/'all_49_generated_frames.png',video)
        sheet(out/'generated_contact_sheet.png',[video],['ACTUAL DIFFUSION OUTPUT'],indices=(0,8,16,24,32,40,48))
        for first in range(0,49,7):
            sheet(out/f'review_{first:02d}_{min(first+6,48):02d}.png',[video],['ACTUAL OUTPUT'],
                indices=tuple(range(first,min(first+7,49))),tile=(400,300))
        Image.fromarray(video[-1]).save(out/'final_generated_frame.png')
        if a.preview_only:continue
        result=tracker(np.concatenate([reference[None],video]),uv0,out/'cup_tracking.npz',
            {'video_sha256':sha256(path),'future_real_used':False,'extra_observed_anchor':True,'tracked_frames':49})
        xy=result['tracks'][1:];vis=result['visibility'][1:]&valid(xy)
        motion=np.load(prep/'motion.npz')
        query=motion['source_times']
        target=interpolate(rawuv,np.arange(31)/15,query)
        rigid=project(motion['sparse_xyz'].reshape(-1,3),k).reshape(49,8,2)
        expected=valid(target)
        available=vis&expected
        desc={'video_sha256':sha256(path),'future_used_in_generation':False,'timing':config['timing'],
            'trajectory_control_passed_to_model':config['trajectory_control'],
            'coordinate_system':'native 640x480 pixels; generated image stretch is undone',
            'raw_forecast_to_rigid_control':metrics(target,rigid,np.ones_like(expected)),
            'generated_to_raw_forecast':metrics(target,xy,available),
            'generated_to_rigid_control':metrics(rigid,xy,vis&valid(rigid)),
            'target_in_frame_pairs':int(expected.sum()),'target_in_frame_fraction':float(expected.mean()),
            'usable_fraction_of_target_in_frame':float(available.sum()/max(1,expected.sum())),
            'tracker_anchor_error_px':float(np.linalg.norm(xy[0]-uv0,axis=-1).mean()),
            'motion_duration_s':json.loads((prep/'preparation.json').read_text())['motion_duration_s'],
            'whole_clip_temporal_warp':temporal_warp_error(video),
            'limitation':'Tracker can lose/reidentify an off-screen or deformed cup. Low error is not semantic identity certification; review all 49 frames.',
            'source_times':query.tolist()}
        np.savez_compressed(out/'motion_measurements.npz',generated_xy=xy,generated_visibility=vis,
            raw_forecast_xy=target,rigid_control_xy=rigid,comparison_mask=available,source_times=query)
        write_json(out/'cup_appearance.json',cup_appearance(video,xy,result['visibility'][1:],reference,
            np.array(Image.open(SCENE/'observed/mask.png'))>0))
        if a.include_real and config['timing']=='physical_2s':
            # Only physical timing is meaningful for real-continuation errors.
            real=np.load(SCENE/'evaluation/evaluation_tracks_2d.npz')
            times=np.arange(1,11)/5
            sampled=interpolate(xy,np.arange(49)/8,times)
            lo=np.floor(times*8).astype(int);hi=np.ceil(times*8).astype(int)
            mask=vis[lo]&vis[hi]&real['visibility'][1:,:8]&valid(sampled)
            desc['generated_to_real_physical_2s']=metrics(real['tracks'][1:,:8],sampled,mask)
            raw_sample=interpolate(rawuv,np.arange(31)/15,times)
            desc['raw_forecast_to_real_full_reference']=metrics(real['tracks'][1:,:8],raw_sample,real['visibility'][1:,:8])
            desc['raw_forecast_to_real_shared_with_generation']=metrics(real['tracks'][1:,:8],raw_sample,mask)
            desc['real_comparison_scope']='evaluation only; physical 0.2..2.0s; no stretched timing compared to real 6s'
            future=np.load(SCENE/'evaluation/future_rgb.npy')
            compared=[]
            for j,time in enumerate(times):
                index=int(round(time*8))
                compared.append(np.concatenate([labeled(video[index],f'Generated physical: sampled {index/8:.3f}s'),
                    labeled(future[j],f'Real future: {time:.3f}s')],axis=1))
            save_video(OUT/'physical_generated_vs_real_2s.mp4',compared,fps=5)
        overlay=[]
        for i,frame in enumerate(video):
            view=frame.copy()
            for point in range(8):
                x,y=np.rint(target[i,point]).astype(int)
                if expected[i,point]:cv2.drawMarker(view,(x,y),(255,30,170),cv2.MARKER_CROSS,9,1)
                if vis[i,point]:cv2.circle(view,tuple(np.rint(xy[i,point]).astype(int)),2,(30,255,60),-1)
            overlay.append(labeled(view,'pink: raw Molmo | green: independent AllTracker'))
        save_video(out/'generated_vs_raw_tracking.mp4',overlay)
        fig,ax=plt.subplots(figsize=(8,5))
        ax.imshow(reference,alpha=.35)
        ax.plot(*target.mean(1).T,color='magenta',label='raw forecast centroid')
        center=np.array([points[keep].mean(0) if keep.any() else [np.nan,np.nan] for points,keep in zip(xy,available)])
        ax.plot(*center.T,'o-',color='green',markersize=3,label='tracked centroid on usable points')
        ax.set(xlim=(0,640),ylim=(480,-100));ax.legend();fig.tight_layout();fig.savefig(out/'measured_motion.png',dpi=150);plt.close(fig)
        write_json(out/'motion_metrics.json',desc);reports[out.name]=desc
        print(out.name,desc['generated_to_raw_forecast'],flush=True)
    if len(videos)>1:
        names=list(videos)
        combined=[np.concatenate([labeled(videos[name][i],name) for name in names],axis=1) for i in range(49)]
        save_video(OUT/'generated_variants_comparison.mp4',combined)
    if reports:write_json(OUT/'comparison_metrics.json',{'variants':reports,'future_used_in_generation':False})


if __name__=='__main__':main()
