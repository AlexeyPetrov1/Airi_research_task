"""Independent visible-link motion and cup/gripper coupling measurement."""
import argparse
import json
import cv2
import numpy as np
from PIL import Image
from das_full_motion_diagnose import SCENE, OUT
from das_prepare_control import project, sha256, write_json, save_video
from das_robot_evaluate import frames, tracker, valid, labeled
from das_evaluate import metrics


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='H5_group00_6s_whole_robot_prior025');a=p.parse_args()
    prep=OUT/'group00_stretched_6s_whole_robot_v1'
    articulated=np.load(prep/'articulated_robot.npz');motion=np.load(prep/'motion.npz')
    k=np.load(SCENE/'geometry/K_median.npy')
    body_uv=articulated['robot_query_uv'];body_target=articulated['robot_target_uv']
    # Additional observed local material points; never real future observations.
    manual=np.array([[160.,55.],[164.,100.],[166.,159.],[150.,186.],[179.,203.]])
    count=int(motion['cup_points']);uv0=motion['uv0'][count:]
    ids=np.linalg.norm(uv0[:,None]-manual[None],axis=-1).argmin(0)+count
    local_uv=motion['uv0'][ids].astype(float)
    local_target=project(motion['xyz'][:,ids].reshape(-1,3),k).reshape(49,len(ids),2)
    queries=np.r_[body_uv,local_uv]
    target=np.concatenate([body_target,local_target],axis=1)
    reference=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    names=['H2_group00_6s_no_prior','H3_group00_6s_prior025',a.name]
    measured={};clips={}
    for name in names:
        out=OUT/name;path=out/'generated_seed42.mp4'
        state=json.loads((out/'resource_usage.json').read_text());assert state['success'] and state['output_sha256']==sha256(path)
        clip=np.stack([cv2.resize(f,(640,480)) for f in frames(path)])
        result=tracker(np.concatenate([reference[None],clip]),queries,out/'robot_tracking.npz',
            {'video_sha256':sha256(path),'future_real_used':False,'extra_observed_anchor':True,
             'tracked_frames':49,'target_sha256':sha256(prep/'articulated_robot.npz')})
        xy=result['tracks'][1:];vis=result['visibility'][1:]&valid(xy)
        measured[name]=(xy,vis);clips[name]=clip
        np.savez_compressed(out/'robot_motion_measurements.npz',generated_xy=xy,generated_visibility=vis,
            target_xy=target,query_uv=queries,body_link=articulated['robot_query_link'])
    common=valid(target)
    for xy,vis in measured.values():common&=vis
    report={'coordinate_system':'native 640x480 pixels','future_real_used':False,
        'target_preparation':prep.name,'target_sha256':sha256(prep/'articulated_robot.npz'),
        'query_count':len(queries),'proximal_query_count':len(body_uv),'local_query_count':len(local_uv),
        'joint_control':'Exact cup/local rigid motion; nominal IK proximal endpoints with observed silhouette mapping',
        'limitation':'Visible material-point tracks and projected cup/gripper offsets; no semantic identity or physical contact certification',
        'variants':{}}
    for name,(xy,vis) in measured.items():
        mask=vis&valid(target)
        def displacement(data,keep,anchors):
            values=np.linalg.norm(data-anchors[None],axis=-1)
            return {'mean_px':float(values[keep].mean()) if keep.any() else None,
                'max_px':float(values[keep].max()) if keep.any() else None,'usable_pairs':int(keep.sum())}
        body_slice=slice(0,len(body_uv));local_slice=slice(len(body_uv),None)
        desc={'proximal_to_requested_motion':metrics(target[:,body_slice],xy[:,body_slice],mask[:,body_slice]),
            'local_to_requested_motion':metrics(target[:,local_slice],xy[:,local_slice],mask[:,local_slice]),
            'shared_three_proximal_to_requested_motion':metrics(target[:,body_slice],xy[:,body_slice],common[:,body_slice]),
            'shared_three_proximal_displacement':displacement(xy[:,:len(body_uv)],common[:,:len(body_uv)],body_uv),
            'per_proximal_query':[]}
        for point in range(len(body_uv)):
            desc['per_proximal_query'].append({'uv':queries[point].tolist(),'link':int(articulated['robot_query_link'][point]),
                'to_control':metrics(target[:,point:point+1],xy[:,point:point+1],mask[:,point:point+1]),
                'tracked_displacement':displacement(xy[:,point:point+1],mask[:,point:point+1],queries[point:point+1]),
                'target_max_displacement_px':float(np.linalg.norm(target[:,point]-queries[point],axis=-1)[valid(target)[:,point]].max())})
        cup=np.load(OUT/name/'motion_measurements.npz')
        cupxy=cup['generated_xy'];cupvis=cup['generated_visibility']&valid(cupxy)
        cuptarget=cup['rigid_control_xy'];offsets=[]
        grip=np.arange(len(body_uv)+2,len(queries))
        for i in range(49):
            ci=np.flatnonzero(cupvis[i]&valid(cuptarget[i]));gi=grip[mask[i,grip]]
            if len(ci)<3 or len(gi)<2:offsets.append(None);continue
            actual=cupxy[i,ci].mean(0)-xy[i,gi].mean(0)
            expected=cuptarget[i,ci].mean(0)-target[i,gi].mean(0)
            offsets.append(float(np.linalg.norm(actual-expected)))
        present=[v for v in offsets if v is not None]
        desc['cup_gripper_relative_motion']={'mean_offset_error_px':float(np.mean(present)) if present else None,
            'valid_times':len(present),'per_frame_error_px':offsets}
        report['variants'][name]=desc
        rendered=[]
        for i,frame in enumerate(clips[name]):
            view=frame.copy()
            for j in range(len(queries)):
                if valid(target)[i,j]:cv2.drawMarker(view,tuple(np.rint(target[i,j]).astype(int)),(255,30,170),cv2.MARKER_CROSS,9,1)
                if vis[i,j]:cv2.circle(view,tuple(np.rint(xy[i,j]).astype(int)),3,(30,255,60),-1)
            rendered.append(labeled(view,'pink: whole-arm target | green: independent tracking'))
        save_video(OUT/name/'robot_tracking_overlay.mp4',rendered)
    report['shared_three_target_proximal_displacement']=displacement(target[:,:len(body_uv)],common[:,:len(body_uv)],body_uv)
    write_json(OUT/f'{a.name}_robot_comparison.json',report)
    combined=[np.concatenate([labeled(clips[n][i],n) for n in names],axis=1) for i in range(49)]
    save_video(OUT/f'{a.name}_body_comparison.mp4',combined)
    print({name:d['proximal_to_requested_motion'] for name,d in report['variants'].items()},flush=True)


if __name__=='__main__':main()
