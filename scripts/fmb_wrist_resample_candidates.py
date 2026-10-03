"""Observed-only depth-supported author KMeans retry; preserve the initial 100."""
import json,shutil
import cv2,numpy as np
from fmb_wrist_prepare import RUN,write,sha
from fmb_wrist_math import sample_z
import berkeley_preprocess as b

def run(camera):
    scene=RUN/camera;obs=scene/'observed';receipt=obs/'candidate_resampling.json'
    if receipt.exists():return
    assert not(scene/'geometry/branch_selection.json').exists(),'Cannot change frozen inputs'
    assert not any((scene/'branches').glob('*/predictions/model_run.json')),'Cannot change after prediction'
    mask=cv2.imread(str(obs/'mask.png'),0)>0;margin=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    yy,xx=np.where(mask&(margin>=2.5));uv=np.stack([xx,yy],-1).astype(float)
    vipe=np.load(scene/'geometry/vipe_depth.npy')[-1];sensor=np.load(obs/'sensor_z16.npy')[-1].astype(float)*1e-4
    supported=np.ones(len(uv),bool)
    for depth in [sensor,vipe]:
        z,spread,c=sample_z(depth,uv);supported &= np.isfinite(z)&(spread<.03)
    subset=np.zeros(mask.shape,bool);subset[yy[supported],xx[supported]]=True
    assert subset.sum()>=100,f'Insufficient observed depth-supported surface: {subset.sum()}'
    queries=b.official_kmeans(subset,int(np.load(obs/'source_indices.npy')[-1]))
    backup=scene/'geometry/initial_100_candidates';backup.mkdir(exist_ok=True)
    names=['query_points_100.npy','observed_tracks_2d.npz','alltracker_execution.json']
    for name in names:
        if (obs/name).exists():shutil.copy2(obs/name,backup/name)
    for name in ['candidate_history_tracks.png','mask_and_100_queries.png']:
        if (scene/'viz'/name).exists():shutil.copy2(scene/'viz'/name,backup/name)
    rgb=np.load(obs/'rgb.npy');xy,confidence,visible,probability,info=b.alltracker_tracks(rgb,queries,reverse=True,max_side=256)
    np.save(obs/'query_points_100.npy',queries)
    np.savez_compressed(obs/'observed_tracks_2d.npz',tracks=xy,confidence=confidence,visibility=visible,visibility_probability=probability,source_indices=np.load(obs/'source_indices.npy'))
    write(obs/'alltracker_execution.json',{**info,'query_count':100,'future_used':False,'query_mask':'SAM intersect observed t0 sensor/ViPE local depth support'})
    cv2.imwrite(str(obs/'depth_supported_sampling_mask.png'),subset.astype(np.uint8)*255)
    write(receipt,{'future_used':False,'old_query_sha256':sha(backup/'query_points_100.npy'),'new_query_sha256':sha(obs/'query_points_100.npy'),
          'reason':'Initial 100 produced only 13 common eligible tracks on wrist_2. Retain 24-point requirement and all quality thresholds.',
          'sampling_rule':'Author KMeans seed0,100 points on SAM interior margin>=2.5 px and 5x5 sensor and ViPE t0 depth finite, >=13 valid, p90-p10<0.03m',
          'support_pixels':int(subset.sum()),'source_t0':int(np.load(obs/'source_indices.npy')[-1]),'original_candidates_preserved':str(backup)})
    print(camera,'depth-supported candidates',subset.sum(),flush=True)

if __name__=='__main__':
    for camera in ['wrist_2','wrist_1']:run(camera)
