"""SAM2.1 masks prompted by user-requested manual points on observed t0."""
import sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from das_prepare_control import sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup/observed'
sys.path.insert(0,'/mnt/f/AIRI_task/third_party/sam2')
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

prompts={
 'forearm':{'box':[175,0,421,120],'positive':[[225,75],[309,22]],'negative':[[159,180],[500,100],[152,265]]},
 'wrist':{'box':[124,36,201,148],'positive':[[155,82],[163,120]],'negative':[[225,76],[164,178],[154,261]]},
 'gripper':{'box':[127,138,214,243],'positive':[[163,170],[178,209]],'negative':[[152,267],[229,74],[100,213]]},
 'upper_arm':{'box':[557,0,639,390],'positive':[[600,100],[600,280]],'negative':[[528,210],[599,440],[470,380]]}}

def main():
    start=time.monotonic();rgb=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/f/AIRI_task/models/sam2.1_hiera_large.pt',device='cuda')
    predictor=SAM2ImagePredictor(model); overlays=[]; summary={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(rgb)
        for name,p in prompts.items():
            pts=np.array(p['positive']+p['negative'],np.float32)
            labels=np.array([1]*len(p['positive'])+[0]*len(p['negative']),np.int32)
            masks,scores,logits=predictor.predict(point_coords=pts,point_labels=labels,box=np.array(p['box'],np.float32),multimask_output=True)
            chosen=int(np.argmax(scores));mask=masks[chosen].astype(bool)
            Image.fromarray(mask.astype(np.uint8)*255).save(OUT/f'{name}_mask.png')
            np.savez_compressed(OUT/f'{name}_sam_candidates.npz',masks=masks,scores=scores)
            view=rgb.copy();view[mask]=(.55*view[mask]+.45*np.array([40,230,70])).astype(np.uint8)
            image=Image.fromarray(view);draw=ImageDraw.Draw(image);draw.text((5,5),f'{name}: {mask.sum()} px; SAM score {scores[chosen]:.3f}',fill='white')
            for x,y in p['positive']:draw.ellipse((x-3,y-3,x+3,y+3),fill='red')
            image.save(OUT/f'{name}_mask_overlay.png');overlays.append(image)
            summary[name]={'prompt':p,'scores':scores.tolist(),'chosen':chosen,'pixels':int(mask.sum())}
    canvas=Image.new('RGB',(1280,960))
    for i,v in enumerate(overlays):canvas.paste(v,(i%2*640,i//2*480))
    canvas.save(OUT/'robot_masks_contact_sheet.png')
    write_json(OUT/'robot_segmentation_receipt.json',{'future_used':False,'manual_points_authorized_by':'User requested points on robot and a robot trajectory',
        'source_sha256':sha256(SCENE/'observed/frame_000063.png'),'method':'Official SAM2.1 Hiera Large; t0 manual positive/negative points and boxes',
        'checkpoint_sha256':sha256(Path('/mnt/f/AIRI_task/models/sam2.1_hiera_large.pt')),'parts':summary,'wall_seconds':time.monotonic()-start})
    print(summary)

if __name__=='__main__':main()
