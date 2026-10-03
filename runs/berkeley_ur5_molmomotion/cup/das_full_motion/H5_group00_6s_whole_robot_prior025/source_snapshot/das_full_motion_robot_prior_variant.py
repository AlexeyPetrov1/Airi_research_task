"""Optional stronger proximal prior while retaining 0.25 cup/wrist weights."""
import json
import shutil
import cv2
import numpy as np
from das_full_motion_diagnose import SCENE,OUT
from das_full_motion_prepare import load_mask
from das_prepare_control import sha256,write_json
from das_full_motion_safety import forbid_real_future


def main():
    forbid_real_future()
    source=OUT/'group00_stretched_6s_whole_robot_v1'
    out=OUT/'group00_stretched_6s_whole_robot_body045'
    if out.exists():raise FileExistsError('Keep existing conditioning immutable')
    out.mkdir()
    for name in ['motion.npz','articulated_robot.npz','control_720x480.mp4',
        'guide_720x480.npz','photographic_guide.mp4','image_t0_720x480.png',
        'checkpoint_receipt.json','input_freeze.json','full_body_contact_sheet.png','all_49_guide_frames.png']:
        shutil.copy2(source/name,out/name)
    weights=np.load(source/'prior_masks.npz')['weights'].copy()
    maps=np.load(source/'articulated_robot.npz')['affine']
    cup=load_mask(SCENE/'observed/mask.png')
    obs=SCENE/'das_robot_cup/observed'
    local=(load_mask(obs/'wrist_mask.png')|load_mask(obs/'gripper_mask.png'))&~cup
    # The global callback strength is 0.45; local mask scaling preserves 0.25.
    for i in range(1,49):
        support=np.zeros(cup.shape,bool)
        for region,matrix in zip([local,cup],maps[i,2:]):
            alpha=cv2.warpAffine(region.astype(np.float32),matrix,(640,480),flags=cv2.INTER_LINEAR)
            support|=cv2.dilate((alpha>.05).astype(np.uint8),np.ones((7,7),np.uint8))>0
        resized=cv2.resize(support.astype(np.float32),(720,480),interpolation=cv2.INTER_LINEAR)
        weights[i]*=1-resized*(1-.25/.45)
    np.savez_compressed(out/'prior_masks.npz',weights=weights)
    prep=json.loads((source/'preparation.json').read_text())
    prep.update(source_preparation=source.name,prior_masks_sha256=sha256(out/'prior_masks.npz'),
        photographic_guide_strength_recommended=.45,
        spatial_strength_description='Proximal moving body 0.45; cup and local wrist/gripper interiors 0.25; feathered boundaries; initial latent 1.0',
        global_strength_required=.45)
    write_json(out/'preparation.json',prep)
    review=json.loads((source/'preparation_visual_review.json').read_text())
    review.update(prior_masks_sha256=prep['prior_masks_sha256'],
        review_notes=review['review_notes']+['Identical reviewed guide/control; only spatial prior weights changed, retaining cup/wrist interior strength 0.25.'])
    write_json(out/'preparation_visual_review.json',review)
    print({'preparation':str(out),'same_guide_and_control':True,'body_strength':.45,'cup_interior_strength':.25})


if __name__=='__main__':main()
