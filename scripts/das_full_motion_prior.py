"""Complete t0 RGB silhouettes aligned with the unchanged 3D control.

Per-surface affine maps approximate projection of the frozen SE(3) transform.
The original control and trajectory are copied byte-for-byte, so a prior/no-prior
comparison isolates the latent appearance condition at identical motion.
"""
import argparse
import json
import shutil
import cv2
import numpy as np
from PIL import Image
from das_full_motion_diagnose import SCENE, OUT
from das_full_motion_prepare import load_mask
from das_prepare_control import project, save_video, sheet, sha256, write_json
from das_full_motion_safety import forbid_real_future


def main():
    forbid_real_future()
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='group00_stretched_6s')
    p.add_argument('--name',default='group00_stretched_6s_silhouette')
    args=p.parse_args()
    source=(OUT/args.source).resolve();out=(OUT/args.name).resolve()
    assert source.is_relative_to(OUT.resolve()) and out.is_relative_to(OUT.resolve())
    if out.exists():raise FileExistsError('Preserve prepared conditioning')
    out.mkdir()
    for name in ['motion.npz','control_720x480.mp4','image_t0_720x480.png','checkpoint_receipt.json','input_freeze.json']:
        shutil.copy2(source/name,out/name)
    prep=json.loads((source/'preparation.json').read_text())
    motion=np.load(source/'motion.npz')
    k=np.load(SCENE/'geometry/K_median.npy')
    rgb=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    obs=SCENE/'das_robot_cup/observed'
    cup=load_mask(SCENE/'observed/mask.png')
    wrist=load_mask(obs/'wrist_mask.png');grip=load_mask(obs/'gripper_mask.png')
    arm=load_mask(obs/'forearm_mask.png')
    local=(wrist|grip)&~cup
    plate=np.array(Image.open(obs/'clean_reference_640x480.png').convert('RGB'))
    bg=rgb.copy()
    clear=cv2.dilate((cup|local|arm).astype(np.uint8),np.ones((9,9),np.uint8))>0
    bg[clear]=plate[clear]
    vacancy=cv2.dilate((cup|local).astype(np.uint8),np.ones((9,9),np.uint8))>0
    count=int(motion['cup_points'])
    uv0=motion['uv0'].astype(float)
    sparse0=project(motion['sparse_xyz'][0].astype(float),k)
    pieces=[(cup,slice(0,count)),(local,slice(count,None))]
    guides,masks,errors,matrices,anchor_errors=[],[],[],[],[]
    for i in range(49):
        guide=bg.astype(float).copy()
        mask=np.full(cup.shape,.02,np.float32)
        mask[clear]=0 # free proximal arm region
        mask[vacancy]=.5
        frame_errors=[];frame_matrices=[]
        for region,subset in pieces:
            # Use the eight forecast anchors. Measured depth on a silhouette
            # edge may belong to background; it must not distort the cutout.
            sparse_future=project(motion['sparse_xyz'][i].astype(float),k)
            sparse_design=np.c_[sparse0,np.ones(len(sparse0))]
            uv=uv0[subset]
            future=project(motion['xyz'][i,subset].astype(float),k)
            design=np.c_[uv,np.ones(len(uv))]
            # The wrist/gripper can have different observed depth from the cup.
            # Preserve local control alignment with its own projected surface.
            affine=np.linalg.lstsq(sparse_design,sparse_future,rcond=None)[0].T if subset.start==0 else np.linalg.lstsq(design,future,rcond=None)[0].T
            if i==0:affine=np.array([[1.,0,0],[0,1.,0]])
            if subset.start==0:anchor_errors.append(float(np.linalg.norm(sparse_design@affine.T-sparse_future,axis=1).mean()))
            frame_errors.append(float(np.linalg.norm(design@affine.T-future,axis=1).mean()))
            frame_matrices.append(affine)
            alpha=cv2.warpAffine(region.astype(np.float32),affine,(640,480),flags=cv2.INTER_LINEAR)
            color=cv2.warpAffine(rgb.astype(np.float32)*region[...,None],affine,(640,480),flags=cv2.INTER_LINEAR)
            guide=guide*(1-alpha[...,None])+color
            support=cv2.dilate((alpha>.05).astype(np.uint8),np.ones((7,7),np.uint8))>0
            mask[support]=1.
        if i==0:guide=rgb.copy();mask[:]=1.
        resized=cv2.resize(np.clip(np.rint(guide),0,255).astype(np.uint8),(720,480),interpolation=cv2.INTER_LINEAR)
        if i==0:resized=np.array(Image.open(out/'image_t0_720x480.png').convert('RGB'))
        guides.append(resized)
        masks.append(cv2.resize(mask,(720,480),interpolation=cv2.INTER_LINEAR))
        errors.append(frame_errors);matrices.append(frame_matrices)
    np.savez_compressed(out/'guide_720x480.npz',frames=np.array(guides))
    np.savez_compressed(out/'prior_masks.npz',weights=np.array(masks))
    np.savez_compressed(out/'silhouette_maps.npz',affine=np.array(matrices),mean_projection_residual_px=np.array(errors),anchor_projection_residual_px=np.array(anchor_errors))
    save_video(out/'photographic_guide.mp4',guides)
    sheet(out/'photographic_contact_sheet.png',[guides],['T0-only complete silhouette guide'],indices=(0,8,16,24,32,40,48))
    prep.update(guide_sha256=sha256(out/'guide_720x480.npz'),prior_masks_sha256=sha256(out/'prior_masks.npz'),
        source_preparation=args.source,photographic_method='t0 complete RGB silhouette affine-warped using the eight projected SE(3) forecast anchors',
        photographic_map_mean_residual_px=np.array(errors).mean(0).tolist(),
        photographic_map_max_frame_mean_residual_px=np.array(errors).max(0).tolist(),
        photographic_anchor_mean_error_px=float(np.mean(anchor_errors)),
        photographically_guided_proximal_arm=False,
        background_mode='t0-only clean plate in dilated left foreground; spatial prior zero on vacated proximal arm')
    write_json(out/'preparation.json',prep)
    assert sha256(out/'control_720x480.mp4')==prep['control_sha256']
    print({'output':str(out),'affine_mean_residual_px':prep['photographic_map_mean_residual_px']})


if __name__=='__main__':main()
