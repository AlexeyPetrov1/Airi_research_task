"""Apply an imagegen background estimate only inside the observed cup footprint."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from das_prepare_control import write_json,sha256,sheet
from das_reference_mask import reference_alpha

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True)
p.add_argument('--plate',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--gripper-protection',choices=['gray-only','non-blue-dark'],default='gray-only');a=p.parse_args()
out=a.output;out.mkdir(parents=True,exist_ok=True)
ref=np.array(Image.open(a.scene/'observed/frame_000063.png').convert('RGB'))
plate=np.array(Image.open(a.plate).convert('RGB').resize((640,480),Image.Resampling.LANCZOS))
mask=np.array(Image.open(a.scene/'observed/mask.png').convert('L'))>0
alpha,expanded=reference_alpha(ref,mask,a.gripper_protection)
clean=np.rint(ref*(1-alpha[...,None])+plate*alpha[...,None]).astype(np.uint8)
assert np.array_equal(clean[~expanded],ref[~expanded])
Image.fromarray(clean).save(out/'clean_reference_640x480.png')
Image.fromarray(clean).resize((720,480),Image.Resampling.BILINEAR).save(out/'clean_reference_720x480.png')
Image.fromarray(np.rint(alpha*255).astype(np.uint8)).save(out/'clean_reference_alpha.png')
sheet(out/'reference_audit.png',[[ref],[clean]],['OBSERVED START (UNCHANGED)','ESTIMATED EMPTY REFERENCE'],indices=(0,))
write_json(out/'clean_reference_receipt.json',{'source_observed_sha256':sha256(a.scene/'observed/frame_000063.png'),
    'source_mask_sha256':sha256(a.scene/'observed/mask.png'),'imagegen_plate_sha256':sha256(a.plate),
    'clean_reference_sha256':sha256(out/'clean_reference_720x480.png'),'future_used':False,
    'edited_support_pixels':int(expanded.sum()),'outside_support_exactly_unchanged':True,
    'gripper_protection':a.gripper_protection,
    'method':'Built-in imagegen cup removal from observed t0, resized to native geometry; composited only in dilated cup support, with selected gripper protection and 4px feather',
    'role':'Synthetic persistent background condition; start_image and CLIP remain original observed t0',
    'limitation':'Occluded background is estimated, not measured; this input changes the original inference protocol.'})
