"""Endpoint-assisted dense image-space control and photographic latent guide.

This is explicitly NOT the causal MolmoMotion benchmark. Only the endpoint RGB
(source frame 73) is opened from evaluation during preparation. Intermediate
future frames are reserved for evaluation. The forecast supplies timing only.
"""
from pathlib import Path
import argparse, sys, json, time
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from das_prepare_control import save_video, sheet, sha256, write_json

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_reference_repair'
OLD=SCENE/'das_robot_cup'

def load_image(path): return np.array(Image.open(path).convert('RGB'))
def load_mask(path): return np.array(Image.open(path).convert('L'))>0

def endpoint_masks():
    import torch
    sys.path.insert(0,'/mnt/f/AIRI_task/third_party/sam2')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    rgb=load_image(SCENE/'evaluation/frame_000073.png')
    model=build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml','/mnt/f/AIRI_task/models/sam2.1_hiera_large.pt',device='cuda')
    predictor=SAM2ImagePredictor(model)
    prompts={
      'cup':{'box':[93,128,184,250],'positive':[[131,165],[145,207]],'negative':[[156,93],[96,225],[184,201]]},
      'robot':{'box':[69,0,226,146],'positive':[[148,13],[157,79],[183,108]],'negative':[[137,184],[204,173],[259,79]]}}
    result={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(rgb)
        for name,p in prompts.items():
            masks,scores,_=predictor.predict(point_coords=np.array(p['positive']+p['negative'],np.float32),
                point_labels=np.array([1]*len(p['positive'])+[0]*len(p['negative'])),box=np.array(p['box'],np.float32),multimask_output=True)
            choice=int(np.argmax(scores)); mask=masks[choice].astype(bool)
            Image.fromarray(mask.astype(np.uint8)*255).save(OUT/f'endpoint_{name}_mask.png')
            overlay=rgb.copy();overlay[mask]=(.5*overlay[mask]+.5*np.array([40,240,80])).astype(np.uint8)
            Image.fromarray(overlay).save(OUT/f'endpoint_{name}_overlay.png')
            result[name]={'prompt':p,'scores':scores.tolist(),'choice':choice,'pixels':int(mask.sum())}
    write_json(OUT/'endpoint_segmentation.json',{'future_used':True,'frame':73,'parts':result,
        'checkpoint_sha256':sha256('/mnt/f/AIRI_task/models/sam2.1_hiera_large.pt')})

def splat2d(xy, rgb, shape):
    """Bilinear forward splat, retaining covered pixels without background bleed."""
    h,w=shape; sums=np.zeros((h*w,3),np.float32);weights=np.zeros(h*w,np.float32)
    base=np.floor(xy).astype(int); frac=xy-base
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        x,y=base[:,0]+dx,base[:,1]+dy
        weight=(frac[:,0] if dx else 1-frac[:,0])*(frac[:,1] if dy else 1-frac[:,1])
        ok=(x>=0)&(x<w)&(y>=0)&(y<h)&(weight>0)
        ids=y[ok]*w+x[ok]; ww=weight[ok].astype(np.float32)
        np.add.at(weights,ids,ww)
        for c in range(3):np.add.at(sums[:,c],ids,rgb[ok,c]*ww)
    out=sums/np.maximum(weights[:,None],1e-6)
    return out.reshape(h,w,3),weights.reshape(h,w)

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--masks-only',action='store_true');p.add_argument('--replace-prepared',action='store_true')
    p.add_argument('--output-subdir',default='');p.add_argument('--background',choices=['initial','endpoint'],default='initial');options=p.parse_args()
    base=OUT.resolve();OUT=(OUT/options.output_subdir).resolve()
    assert OUT.is_relative_to(base),'Output must stay inside this experiment'
    OUT.mkdir(parents=True,exist_ok=True)
    if options.masks_only: endpoint_masks();return
    if (OUT/'guide_720x480.npz').exists() and not options.replace_prepared:raise FileExistsError('Refusing to replace prepared guide')
    if options.replace_prepared:
        for config_path in base.glob('*/config.json'):
            config=json.loads(config_path.read_text(encoding='utf-8-sig'))
            used_root=Path(config.get('preparation_root',str(base))).resolve()
            if used_root==OUT:
                raise RuntimeError('Cannot change conditioning consumed by an existing generation')
    rgb0=load_image(SCENE/'observed/frame_000063.png');rgb1=load_image(SCENE/'evaluation/frame_000073.png')
    cup0=load_mask(SCENE/'observed/mask.png');cup1=load_mask(OUT/'endpoint_cup_mask.png')
    robot0=np.logical_or.reduce([load_mask(OLD/f'observed/{name}_mask.png') for name in ['forearm','wrist','gripper']])
    robot1=load_mask(OUT/'endpoint_robot_mask.png')
    fg0=cup0|robot0;fg1=cup1|robot1
    # Add only small segmentation seam gaps. Whole RGB foreground avoids holes
    # caused by invalid reflective robot depth in the former 3D point renderer.
    fg0=cv2.morphologyEx(fg0.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    fg1=cv2.morphologyEx(fg1.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))>0
    h,w=fg0.shape;y,x=np.indices((h,w));uv=np.stack([x,y],-1).astype(np.float32)
    cent0=np.array(np.nonzero(cup0))[::-1].mean(1);cent1=np.array(np.nonzero(cup1))[::-1].mean(1);shift=cent1-cent0
    seed0=np.zeros((h,w,2),np.float32);seed0[fg0]=shift
    seed1=np.zeros_like(seed0);seed1[fg1]=-shift
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(0);dis.setGradientDescentIterations(50);dis.setVariationalRefinementIterations(10)
    g0=cv2.cvtColor(rgb0,cv2.COLOR_RGB2GRAY);g1=cv2.cvtColor(rgb1,cv2.COLOR_RGB2GRAY)
    flow0=dis.calc(g0,g1,seed0);flow1=dis.calc(g1,g0,seed1)
    # Appearance changes can send DIS cup pixels into background. Restrict cup
    # correspondence to a robust affine endpoint map fitted to mask moments.
    for source,target,flow in [(cup0,cup1,flow0),(cup1,cup0,flow1)]:
        s=np.array(np.nonzero(source))[::-1].T.astype(float);t=np.array(np.nonzero(target))[::-1].T.astype(float)
        scale=np.clip(t.std(0)/s.std(0),.8,1.25)
        mapped=(s-s.mean(0))*scale+t.mean(0)
        flow[source]=mapped-s
    # One shared lift for the visible left robot chain. Unconstrained DIS on
    # reflective metal tore the rendered robot into translucent fragments.
    # This image-space approximation is not calibrated 3D robot kinematics.
    flow0[robot0]=shift;flow1[robot1]=-shift
    # Background is static. Keep t0 RGB except newly exposed pixels for which
    # the endpoint is a real photographic empty-site observation. Where both
    # endpoints hide background, reuse the previous, already saved clean plate.
    bg=rgb0.copy();clear=fg0&~cv2.dilate(fg1.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    bg[clear]=rgb1[clear]
    overlap=fg0&~clear
    plate=load_image(OLD/'observed/clean_reference_640x480.png');bg[overlap]=plate[overlap]
    if options.background=='endpoint':
        # The old frame also contains the cup's cast shadow and small uncovered
        # bracket/cable fragments. The photographed endpoint background clears
        # these without a new image editor or generated-background model.
        bg=rgb1.copy()
        end_region=cv2.dilate(fg1.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        start_region=cv2.dilate(fg0.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
        clear_end=end_region&~start_region;bg[clear_end]=rgb0[clear_end]
        overlap_end=end_region&~clear_end;bg[overlap_end]=plate[overlap_end]
    # Canonical DaS XY/inverse-depth point colors, attached to t0 pixels.
    d=np.load(SCENE/'observed/native_depth.npy')[-1];valid=np.isfinite(d)&(d>0)&(d<10)
    inv=np.zeros_like(d);inv[valid]=1/d[valid];lo,hi=np.percentile(inv[valid],[1,99])
    color0=np.stack([x/(w-1),y/(h-1),np.clip((inv-lo)/(hi-lo),0,1)],-1)*255
    color0[~valid,2]=127.5
    canonical1=uv+flow1
    color1=cv2.remap(color0.astype(np.float32),canonical1[...,0],canonical1[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    pred=np.load(SCENE/'predictions/future_3d.npy');p0=np.load(SCENE/'observed/points_3d_history.npy')[-1]
    centroids=np.concatenate([p0.mean(0)[None],pred.mean(0)])
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(centroids,axis=0),axis=1))];arc/=arc[-1]
    times=np.arange(49)/8;phase=np.interp(np.minimum(times,2),np.arange(31)/15,arc)
    guides=[];controls=[];supports=[]
    bg_control=color0.copy()
    # Replace old foreground colors with a smooth stationary background XY map.
    nearest=distance_transform_edt(fg0,return_distances=False,return_indices=True)
    bg_control[fg0,2]=color0[nearest[0][fg0],nearest[1][fg0],2]
    for i,a in enumerate(phase):
        r0,w0=splat2d((uv+flow0*a)[fg0],rgb0[fg0].astype(float),(h,w))
        r1,w1=splat2d((uv+flow1*(1-a))[fg1],rgb1[fg1].astype(float),(h,w))
        c0,_=splat2d((uv+flow0*a)[fg0],color0[fg0],(h,w))
        c1,_=splat2d((uv+flow1*(1-a))[fg1],color1[fg1],(h,w))
        # Smooth endpoint crossfade inside the moving foreground only. The
        # endpoint mask never remains pasted at its source location in middle.
        ww0=w0*(1-a);ww1=w1*a;total=ww0+ww1
        # Crossfade appearance, never opacity: a foreground visible in one
        # endpoint remains solid rather than fading into the background.
        alpha=np.clip(w0+w1,0,1);rgb=(r0*ww0[...,None]+r1*ww1[...,None])/np.maximum(total[...,None],1e-6)
        colors=(c0*ww0[...,None]+c1*ww1[...,None])/np.maximum(total[...,None],1e-6)
        guide=bg*(1-alpha[...,None])+rgb*alpha[...,None]
        control=bg_control*(1-alpha[...,None])+colors*alpha[...,None]
        if i==0:guide=rgb0;control=color0;alpha=fg0.astype(float)
        if a==1:
            guide=bg.copy().astype(float);guide[fg1]=rgb1[fg1];alpha=fg1.astype(float)
            control=bg_control.copy();control[fg1]=color1[fg1]
        guides.append(np.clip(guide,0,255).astype(np.uint8));controls.append(np.clip(control,0,255).astype(np.uint8));supports.append(alpha)
    guides=np.array(guides);controls=np.array(controls)
    guide720=np.array([cv2.resize(f,(720,480)) for f in guides]);ctrl720=np.array([cv2.resize(f,(720,480)) for f in controls])
    np.savez_compressed(OUT/'guide_720x480.npz',frames=guide720)
    np.savez_compressed(OUT/'endpoint_flow.npz',flow0=flow0,flow1=flow1,phase=phase,times=times,fg0=fg0,fg1=fg1,cup0=cup0,cup1=cup1,robot0=robot0,robot1=robot1)
    save_video(OUT/'photographic_guide.mp4',guide720);save_video(OUT/'control_endpoint_720x480.mp4',ctrl720)
    sheet(OUT/'preparation_contact_sheet.png',[guides,controls],['Photographic GUIDE (not generation)','Dense endpoint CONTROL'])
    Image.fromarray(guide720[0]).save(OUT/'image_t0_720x480.png');Image.fromarray(cv2.resize(rgb1,(720,480))).save(OUT/'endpoint_reference_720x480.png')
    Image.fromarray(bg).save(OUT/'background_reference.png')
    prior_checkpoint=OLD/'robot_coupled_initial_only/checkpoint_receipt.json'
    (OUT/'checkpoint_receipt.json').write_bytes(prior_checkpoint.read_bytes())
    write_json(OUT/'preparation.json',{'future_used':True,'endpoint_source_index':73,'t0_source_index':63,
        'intermediate_future_frames_used':False,'source_fps':5,'output_fps':8,'output_frames':49,'forecast_horizon_s':2,
        'motion':'Endpoint RGB dense correspondence; monotone progress from original MolmoMotion centroid arc length; endpoint held after 2 s.',
        'NOT_causal_MolmoMotion_prediction':True,'forecast_geometry_changed':True,'cup_shift_px':shift.tolist(),
        'new_image_editing_used':False,'background_overlap_source':'Reuse existing clean_reference_640x480.png from prior published experiment',
        'background_mode':options.background,'output_subdir':options.output_subdir,
        'input_sha256':{str(p.relative_to(SCENE)):sha256(p) for p in [SCENE/'observed/frame_000063.png',SCENE/'evaluation/frame_000073.png',SCENE/'predictions/future_3d.npy',OLD/'observed/clean_reference_640x480.png']},
        'guide_sha256':sha256(OUT/'guide_720x480.npz'),'control_sha256':sha256(OUT/'control_endpoint_720x480.mp4')})
    print('Prepared endpoint-assisted guide; cup endpoint displacement:',shift,'px',flush=True)

if __name__=='__main__':main()
