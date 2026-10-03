"""Frozen author models on the observed wrist prefix; no future file loading."""
from __future__ import annotations
import argparse,gc,json,sys
from pathlib import Path
import cv2,numpy as np
from fmb_wrist_prepare import ROOT,RUN,write,sha
import berkeley_preprocess as b

def sam(camera):
    scene=RUN/camera
    if (scene/'observed/mask.png').exists():return
    b.grounding(scene,None,ROOT.parent/'models/sam2.1_hiera_large.pt',pointing_json=scene/'observed/molmopoint_grounding.json')

def track(camera):
    scene=RUN/camera
    if not(scene/'observed/observed_tracks_2d.npz').exists():b.track_observed(scene,max_side=256)
    rgb=np.load(scene/'observed/rgb.npy');idx=np.load(scene/'observed/source_indices.npy')
    tracks=np.load(scene/'observed/observed_tracks_2d.npz')['tracks']
    panels=[]
    for i in [0,9,19,29,39,47,48,49]:
        im=b.draw_points(rgb[i],tracks[i],radius=1);im=cv2.resize(im,(384,384))
        cv2.putText(im,f'{camera} source {idx[i]}',(8,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2);panels.append(im)
    b.save_rgb(scene/'viz/candidate_history_tracks.png',np.vstack([np.hstack(panels[:4]),np.hstack(panels[4:])]))

def unidepth(camera):
    scene=RUN/camera;dest=scene/'geometry'
    if (dest/'unidepth_K.npy').exists():return
    import torch
    sys.path.insert(0,str(ROOT.parent/'third_party/UniDepth'))
    from unidepth.models import UniDepthV2
    rgb=np.load(scene/'observed/rgb.npy');locals=np.unique(np.r_[np.linspace(0,46,5).astype(int),[47,48,49]])
    model=UniDepthV2.from_pretrained(str(ROOT.parent/'models/unidepth-v2-vits14')).cuda().eval()
    model.resolution_level=0
    matrices=[];depths=[]
    for i in locals:
        tensor=torch.from_numpy(rgb[i].copy()).permute(2,0,1).cuda()
        with torch.inference_mode():out=model.infer(tensor)
        matrices.append(out['intrinsics'][0].float().cpu().numpy());depths.append(out['depth'][0,0].float().cpu().numpy())
    np.save(dest/'unidepth_K.npy',np.array(matrices));np.save(dest/'unidepth_depth.npy',np.array(depths));np.save(dest/'unidepth_local_ids.npy',locals)
    write(dest/'unidepth_receipt.json',{'observed_only':True,'local_ids':locals.tolist(),'source_indices':np.load(scene/'observed/source_indices.npy')[locals].tolist(),
           'model':'lpiccinelli/unidepth-v2-vits14','checkpoint_config_sha256':sha(ROOT.parent/'models/unidepth-v2-vits14/config.json'),
           'K_source':'Independent UniDepth observed RGB estimates; sensor depth is not replaced in hybrids'})
    del model;gc.collect();torch.cuda.empty_cache()

def h3_masks(camera):
    scene=RUN/camera;path=scene/'observed/historical_masks_h3.npy'
    if path.exists():return
    import torch
    sys.path.insert(0,str(ROOT.parent/'third_party/sam2'))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(ROOT.parent/'models/sam2.1_hiera_large.pt'),device='cuda')
    predictor=SAM2ImagePredictor(model)
    rgb=np.load(scene/'observed/rgb.npy');tracks=np.load(scene/'observed/observed_tracks_2d.npz')['tracks']
    point=np.array(json.loads((scene/'observed/molmopoint_grounding.json').read_text())['point_xy'])
    query=np.load(scene/'observed/query_points_100.npy');nearest=np.argmin(np.linalg.norm(query-point,axis=-1));masks=[]
    for i in [47,48]:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(rgb[i]);choices,scores,_=predictor.predict(point_coords=tracks[i,nearest][None],point_labels=np.ones(1),multimask_output=True)
        masks.append(choices[np.argmax(scores)])
    masks.append(cv2.imread(str(scene/'observed/mask.png'),0)>0);np.save(path,np.array(masks))
    del predictor,model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sam','track','unidepth','h3-masks']);p.add_argument('--cameras',nargs='+',default=['wrist_2','wrist_1']);a=p.parse_args()
    for camera in a.cameras:
        {'sam':sam,'track':track,'unidepth':unidepth,'h3-masks':h3_masks}[a.stage](camera)
