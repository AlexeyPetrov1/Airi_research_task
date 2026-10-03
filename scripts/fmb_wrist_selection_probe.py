"""Explain common observed eligibility without changing gates."""
import cv2,json
import numpy as np
from fmb_wrist_prepare import RUN,write

for camera in ['wrist_2','wrist_1']:
    scene=RUN/camera;uv=np.load(scene/'observed/observed_tracks_2d.npz')['tracks'];vis=np.load(scene/'observed/observed_tracks_2d.npz')['visibility']
    masks=np.load(scene/'observed/historical_masks_h3.npy').astype(bool);inside=np.ones(len(uv[0]),bool)
    for t in range(3):
        xy=np.rint(uv[-3+t]).astype(int);inside &= masks[t,np.clip(xy[:,1],0,255),np.clip(xy[:,0],0,255)]
    margin=cv2.distanceTransform((cv2.imread(str(scene/'observed/mask.png'),0)>0).astype(np.uint8),cv2.DIST_L2,5)
    xy=np.rint(uv[-1]).astype(int);eligible=inside&vis[-3:].all(0)&(margin[xy[:,1],xy[:,0]]>=2.5)
    out={'mask_visible_margin':int(eligible.sum()),'branches':{}}
    for name in ['A_vipe_full','B_sensor_vipe','C_sensor_tcp_official']:
        p=scene/'geometry'/name/'filter_diagnostics.npz'
        if not p.exists():continue
        f=np.load(p);smooth=np.load(p.parent/'points_3d_filtered.npy')
        conditions={'H3_depth_valid':f['valid'][-3:].all(0),'author_trust_keep':f['keep'],'depth_spread_lt_3cm':f['depth_spread'][-3:].max(0)<.03,'positive_Z':(smooth[-3:,:,2]>0).all(0)}
        out['branches'][name]={'individual_counts':{k:int(v.sum()) for k,v in conditions.items()}}
        for k,v in conditions.items():eligible &= v
        out['branches'][name]['cumulative_common']=int(eligible.sum())
    out['common_ids']=np.flatnonzero(eligible).tolist();write(scene/'geometry/initial_100_eligibility.json',out);print(camera,out,flush=True)
