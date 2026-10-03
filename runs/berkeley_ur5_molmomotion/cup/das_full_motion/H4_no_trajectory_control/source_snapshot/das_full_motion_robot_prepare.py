"""Dense articulated whole-arm condition; preserve the H2 cup/gripper arc.

Nominal UR5 IK supplies the proximal joints. Complete observed link silhouettes
are mapped between projected joint endpoints. The forearm's distal endpoint is
anchored to the unchanged rigid wrist motion, rather than changing the cup to
an achieved IK pose. This is a video-control approximation, not robot planning.
"""
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from das_full_motion_diagnose import SCENE, OUT
from das_full_motion_prepare import load_mask, color_surface
from das_prepare_control import project, save_video, sheet, sha256, write_json, splat
from das_full_motion_safety import forbid_real_future
from das_robot_evaluate import tiled_all


def endpoint_map(start, end):
    """Orientation-preserving similarity with exact endpoint correspondence."""
    a, b = start; c, d = end
    v=b-a; w=d-c
    scale=np.linalg.norm(w)/np.linalg.norm(v)
    angle=np.arctan2(w[1],w[0])-np.arctan2(v[1],v[0])
    linear=scale*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return np.c_[linear,c-linear@a]


def main():
    forbid_real_future()
    out=OUT/'group00_stretched_6s_whole_robot_v1'
    if out.exists():raise FileExistsError('Preserve whole-arm preparations')
    out.mkdir()
    source=OUT/'group00_stretched_6s_causal_prior_v3'
    prep=json.loads((source/'preparation.json').read_text())
    freeze=json.loads((source/'input_freeze.json').read_text())
    extra=['das_robot_cup/observed/upper_arm_mask.png',
        'das_robot_cup/observed/nominal_ur5_camera.npz',
        'das_robot_cup/observed/robot_state_history.npz']
    freeze['sha256'].update({name:sha256(SCENE/name) for name in extra})
    write_json(out/'input_freeze.json',freeze)
    for name in ['motion.npz','image_t0_720x480.png','checkpoint_receipt.json']:
        shutil.copy2(source/name,out/name)
    motion=np.load(source/'motion.npz')
    kin=np.load(OUT/'whole_robot_ik_diagnostic/kinematics.npz')
    observed=np.load(SCENE/'das_robot_cup/observed/nominal_ur5_camera.npz')
    fc=kin['fk_camera'];k=observed['K']
    rgb=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    depth=np.load(SCENE/'observed/native_depth.npy')[-1].astype(float)
    obs=SCENE/'das_robot_cup/observed'
    cup=load_mask(SCENE/'observed/mask.png')
    local=(load_mask(obs/'wrist_mask.png')|load_mask(obs/'gripper_mask.png'))&~cup
    forearm=load_mask(obs/'forearm_mask.png')&~(local|cup)
    upper=load_mask(obs/'upper_arm_mask.png')&~(local|cup|forearm)
    union=cup|local|forearm|upper
    clear=cv2.dilate(union.astype(np.uint8),np.ones((9,9),np.uint8))>0
    plate=np.array(Image.open(obs/'clean_reference_640x480.png').convert('RGB'))
    bg=rgb.copy();bg[clear]=plate[clear]
    valid=np.isfinite(depth)&(depth>0)&(depth<10)
    limits=np.percentile(1/depth[valid],[2,98])
    control_bg=np.zeros_like(rgb)
    y,x=np.nonzero(valid&~clear)
    control_bg[y,x]=color_surface(np.c_[x,y],depth[y,x],limits)
    depth_bg=np.full(depth.shape,np.inf);depth_bg[y,x]=depth[y,x]
    # Complete masks give dense material UV control even on reflective metal.
    yy,xx=np.indices(depth.shape)
    source_control=np.stack([xx/639*255,yy/479*255,
        np.clip((1/np.maximum(depth,.1)-limits[0])/(limits[1]-limits[0]),0,1)*255],axis=-1).astype(np.float32)
    for mask in [upper,forearm]:
        z=np.median(depth[valid&mask]);source_control[mask,2]=np.clip((1/z-limits[0])/(limits[1]-limits[0]),0,1)*255
    uv0=motion['uv0'].astype(float)
    dense_colors=color_surface(uv0,motion['initial_xyz'][:,2],limits)
    cup_count=int(motion['cup_points'])
    sparse0=project(motion['sparse_xyz'][0],k)
    sparse_design=np.c_[sparse0,np.ones(len(sparse0))]
    local_design=np.c_[uv0[cup_count:],np.ones(len(uv0)-cup_count)]
    joint_uv=project(fc[:,:,:3,3].reshape(-1,3),k).reshape(49,7,2)
    # Exact rigid movement of the wrist junction, distinct from nominal joint 4.
    wrist_xyz=np.einsum('tij,j->ti',motion['R'],fc[0,4,:3,3])+motion['t']
    wrist_uv=project(wrist_xyz,k)
    controls,guides,weights,maps=[],[],[],[]
    point_queries=np.array([[600.,100.],[600,275],[599,340],[310,20],[270,48],[228,75]])
    point_links=np.array([2,2,2,3,3,3])
    target_queries=[]
    endpoint_errors=[]
    for i in range(49):
        upper_map=endpoint_map(joint_uv[0,[1,2]],joint_uv[i,[1,2]])
        forearm_map=endpoint_map(joint_uv[0,[2,4]],np.array([joint_uv[i,2],wrist_uv[i]]))
        cup_map=np.linalg.lstsq(sparse_design,project(motion['sparse_xyz'][i],k),rcond=None)[0].T
        local_map=np.linalg.lstsq(local_design,project(motion['xyz'][i,cup_count:],k),rcond=None)[0].T
        frame_maps=[upper_map,forearm_map,local_map,cup_map]
        if i==0:frame_maps=[np.array([[1.,0,0],[0,1.,0]]) for _ in frame_maps]
        guide=bg.astype(float).copy();control=control_bg.astype(float).copy()
        weight=np.full(depth.shape,.02,np.float32);weight[clear]=.65
        for region,affine in zip([upper,forearm,local,cup],frame_maps):
            alpha=cv2.warpAffine(region.astype(np.float32),affine,(640,480),flags=cv2.INTER_LINEAR)
            color=cv2.warpAffine(rgb.astype(np.float32)*region[...,None],affine,(640,480),flags=cv2.INTER_LINEAR)
            guide=guide*(1-alpha[...,None])+color
            if region is upper or region is forearm:
                encoded=cv2.warpAffine(source_control*region[...,None],affine,(640,480),flags=cv2.INTER_LINEAR)
                control=control*(1-alpha[...,None])+encoded
            support=cv2.dilate((alpha>.05).astype(np.uint8),np.ones((7,7),np.uint8))>0
            weight[support]=1.
        # Preserve exact dense H2 cup and wrist control, including depth order.
        canvas=np.clip(np.rint(control),0,255).astype(np.uint8)
        splat(motion['xyz'][i],dense_colors,k,canvas,depth_bg.copy(),radius=1)
        resized=np.array(Image.fromarray(canvas).resize((720,480),Image.Resampling.BILINEAR))
        if i==0:
            guide=rgb.copy();weight[:]=1.
        photo=cv2.resize(np.clip(np.rint(guide),0,255).astype(np.uint8),(720,480),interpolation=cv2.INTER_LINEAR)
        if i==0:photo=np.array(Image.open(out/'image_t0_720x480.png').convert('RGB'))
        controls.append(resized);guides.append(photo)
        weights.append(cv2.resize(weight,(720,480),interpolation=cv2.INTER_LINEAR));maps.append(frame_maps)
        q=np.c_[point_queries,np.ones(len(point_queries))]
        target_queries.append(np.array([(q[j]@frame_maps[0 if link==2 else 1].T) for j,link in enumerate(point_links)]))
        predicted=np.c_[joint_uv[0,[2,4]],np.ones(2)]@forearm_map.T
        endpoint_errors.append(np.linalg.norm(predicted-np.array([joint_uv[i,2],wrist_uv[i]]),axis=1))
    save_video(out/'control_720x480.mp4',controls)
    save_video(out/'photographic_guide.mp4',guides)
    np.savez_compressed(out/'guide_720x480.npz',frames=np.array(guides))
    np.savez_compressed(out/'prior_masks.npz',weights=np.array(weights))
    np.savez_compressed(out/'articulated_robot.npz',joints=kin['joints'],fk_camera=fc,affine=np.array(maps),
        robot_query_uv=point_queries,robot_query_link=point_links,robot_target_uv=np.array(target_queries),
        wrist_junction_uv=wrist_uv,forearm_endpoint_errors_px=np.array(endpoint_errors))
    sheet(out/'full_body_contact_sheet.png',[guides,controls],['Whole-arm t0-only guide','Dense whole-arm material control'],indices=(0,8,16,24,32,40,48))
    tiled_all(out/'all_49_guide_frames.png',guides)
    prep.update(source_preparation=source.name,control_sha256=sha256(out/'control_720x480.mp4'),
        guide_sha256=sha256(out/'guide_720x480.npz'),prior_masks_sha256=sha256(out/'prior_masks.npz'),
        proximal_robot='nominal UR5 IK plus dense joint-endpoint silhouette mapping; fixed base',
        photographically_guided_proximal_arm=True,whole_robot_control=True,
        photographic_method='Complete t0 upper arm and forearm silhouettes mapped between moving projected joint endpoints; unchanged H3 cup/wrist maps',
        background_mode='t0-only synthetic clean plate in all vacated robot and cup support',
        cup_motion_sha256=sha256(out/'motion.npz'),cup_and_local_motion_identical_to_H2=True,
        max_nominal_flange_position_error_m=float(kin['errors'][:,0].max()),
        max_forearm_endpoint_error_px=float(np.max(endpoint_errors)),
        limitations=['Nominal UR5 camera registration is approximate',
            'Visible surfaces use 2D similarities; newly exposed robot surfaces are synthesized',
            'Wrist follows the exact H2 rigid transform, not individual nominal wrist-joint articulation',
            'Fixed mounted base remains stationary; this is a video control, not an execution trajectory'],
        generation_ready=False,visual_review_required=True)
    write_json(out/'preparation.json',prep)
    assert sha256(out/'motion.npz')==sha256(OUT/'group00_stretched_6s/motion.npz')
    assert np.array_equal(guides[0],np.array(Image.open(out/'image_t0_720x480.png')))
    assert prep['max_forearm_endpoint_error_px']<1e-8
    print({'preparation':str(out),'joint_errors_m':prep['max_nominal_flange_position_error_m'],
        'unchanged_cup_motion':True,'max_robot_endpoint_error_px':prep['max_forearm_endpoint_error_px']})


if __name__=='__main__':main()
