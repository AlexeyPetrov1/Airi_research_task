"""Independent generated/real tracking after the paired DaS runs finish."""
from pathlib import Path
import gc, json, sys, time
import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from das_prepare_control import project,sha256,write_json,save_video,sheet
from das_evaluate import metrics,interpolate
from das_quality import temporal_warp_error,cup_appearance

ROOT=Path(__file__).resolve().parents[1]
SCENE=ROOT/'runs/berkeley_ur5_molmomotion/cup'
OUT=SCENE/'das_robot_cup'

def frames(path):
    with imageio.get_reader(str(path)) as reader:return np.stack([frame for frame in reader])

def tracker(clip,uv,path,receipt):
    if path.exists():return np.load(path)
    import torch
    import torch.nn.functional as F
    sys.path.insert(0,'/mnt/f/AIRI_task/molmo-motion/data_generation/third_party/alltracker')
    from nets.alltracker import Net
    checkpoint=Path('/mnt/f/AIRI_task/.cache/torch/hub/checkpoints/alltracker.pth')
    model=Net(16);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'],strict=True)
    model=model.cuda().eval();torch.set_num_threads(4)
    scaled=np.stack([cv2.resize(f,(512,384),interpolation=cv2.INTER_LINEAR) for f in clip])
    rgbs=torch.from_numpy(scaled.copy()).permute(0,3,1,2)[None].float().cuda()
    pts=(uv*.8).astype(np.float32);norm=pts/np.array([511,383],np.float32)*2-1
    grid=torch.from_numpy(norm).cuda()[None,None].expand(len(clip),-1,-1,-1)
    start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        flow,scores,_,_=model(rgbs,iters=4,sw=None,is_training=False)
        delta=F.grid_sample(flow[0],grid,align_corners=True)[:,:,0].permute(0,2,1).cpu().numpy()
        conf=F.grid_sample(scores[0],grid,align_corners=True)[:,:,0].permute(0,2,1).cpu().numpy()
    tracks=(delta+pts[None])/.8
    np.savez_compressed(path,tracks=tracks,visibility=conf[...,0]>.5,visibility_probability=conf[...,0],confidence=conf[...,1],query_uv=uv)
    receipt.update(tracker='official AllTracker Net(16)',checkpoint_sha256=sha256(checkpoint),iterations=4,
        resolution=[512,384],reported_coordinate_system='native 640x480 pixels',seconds=time.monotonic()-start,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        limitation='Tracker confidence is not semantic identity certification; manual review required')
    write_json(path.with_suffix('.json'),receipt)
    del model,rgbs,flow,scores,grid;gc.collect();torch.cuda.empty_cache()
    return np.load(path)

def valid(xy):return np.isfinite(xy).all(-1)&(xy[...,0]>=0)&(xy[...,0]<640)&(xy[...,1]>=0)&(xy[...,1]<480)

def labeled(frame,label):
    im=Image.fromarray(frame.copy());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,26),fill='white');draw.text((7,7),label,fill='black');return np.array(im)

def tiled_all(path,clip):
    tile=(256,192);canvas=Image.new('RGB',(7*256,7*214),'white');draw=ImageDraw.Draw(canvas)
    for i,f in enumerate(clip):
        x=i%7*256;y=i//7*214;canvas.paste(Image.fromarray(f).resize(tile),(x,y+22));draw.text((x+5,y+5),f'frame {i}, {i/8:.3f}s',fill='black')
    canvas.save(path)

def main():
    if '--preview-only' in sys.argv:
        for name in ['robot_static','robot_coupled','robot_coupled_initial_only']:
            out=OUT/name
            if not (out/'generated_molmomotion_seed42.mp4').exists():continue
            assert json.loads((out/'resource_usage.json').read_text())['success']
            video=frames(out/'generated_molmomotion_seed42.mp4')
            native=np.stack([cv2.resize(f,(640,480)) for f in video])
            tiled_all(out/'all_49_generated_frames.png',native)
            sheet(out/'generated_contact_sheet.png',[native],['GENERATED; PHYSICAL TIME'])
        return
    pair=json.loads((OUT/'paired_generation_receipt.json').read_text());assert pair['success']
    freeze=json.loads((OUT/'control_input_freeze.json').read_text())['sha256']
    assert all(sha256(SCENE/name)==digest for name,digest in freeze.items())
    control=np.load(OUT/'joint_control.npz');k=np.load(SCENE/'geometry/K_median.npy')
    cup_uv=np.load(SCENE/'observed/points_2d_history.npy')[-1].astype(float)
    uv=np.concatenate([cup_uv,control['robot_query_uv']]);n_cup=len(cup_uv)
    reference=np.array(Image.open(SCENE/'observed/frame_000063.png').convert('RGB'))
    cup_mask=np.array(Image.open(SCENE/'observed/mask.png'))>0
    gen={};tracked={}
    for name in ['robot_static','robot_coupled','robot_coupled_initial_only']:
        out=OUT/name;assert json.loads((out/'resource_usage.json').read_text())['success']
        path=out/'generated_molmomotion_seed42.mp4';video=frames(path);assert video.shape==(49,480,720,3)
        gen[name]=np.stack([cv2.resize(f,(640,480),interpolation=cv2.INTER_LINEAR) for f in video])
        clip=np.concatenate([reference[None],gen[name][:17]])
        result=tracker(clip,uv,out/'joint_tracking.npz',{'video_sha256':sha256(path),'future_real_used':False,'extra_observed_anchor':True})
        tracked[name]={'xy':result['tracks'][1:],'vis':result['visibility'][1:]}
        tiled_all(out/'all_49_generated_frames.png',gen[name])
        sheet(out/'generated_contact_sheet.png',[gen[name]],['GENERATED; PHYSICAL TIME'])
        write_json(out/'cup_appearance.json',cup_appearance(gen[name][:17],tracked[name]['xy'][:,:n_cup],tracked[name]['vis'][:,:n_cup],reference,cup_mask))
    # The real continuation is first read only after successful generation and
    # frozen control verification. It is never fed back to geometry or inference.
    future=np.load(SCENE/'evaluation/future_rgb.npy');assert len(future)==10
    real=np.concatenate([reference[None],future])
    rt=tracker(real,uv,OUT/'real_joint_tracking.npz',{'future_real_used':'evaluation only','extra_observed_anchor':False,
        'future_rgb_sha256':sha256(SCENE/'evaluation/future_rgb.npy')})
    real_xy=rt['tracks'][1:];real_vis=rt['visibility'][1:];times=np.arange(1,11)/5
    def projection(x):return project(x.reshape(-1,3),k).reshape(x.shape[0],x.shape[1],2)
    intended=np.concatenate([projection(control['cup_sparse_xyz'][:17]),projection(control['robot_sparse_xyz'][:17])],axis=1)
    target=interpolate(intended,np.arange(17)/8,times)
    own_static=target.copy();own_static[:,n_cup:]=intended[0,n_cup:]
    lo=np.floor(times*8).astype(int);hi=np.ceil(times*8).astype(int)
    samples={}
    for name,t in tracked.items():samples[name]={'xy':interpolate(t['xy'],np.arange(17)/8,times),'vis':t['vis'][lo]&t['vis'][hi]}
    common=samples['robot_static']['vis']&samples['robot_coupled']['vis']&valid(target)
    common&=valid(samples['robot_static']['xy'])&valid(samples['robot_coupled']['xy'])
    common_three=common&samples['robot_coupled_initial_only']['vis']&valid(samples['robot_coupled_initial_only']['xy'])
    eval_masks=np.load(OUT/'robot_coupled/background_completion.npz')['moving_mask']
    eval_masks=np.stack([cv2.dilate((eval_masks[a]|eval_masks[b]).astype(np.uint8),np.ones((21,21),np.uint8))>0 for a,b in zip(lo,hi)])
    report={'physical_evaluation_times_s':times.tolist(),'coordinate_system':'native 640x480 pixels',
        'sampling':'Linear interpolation of generated 8Hz point tracks to 5Hz real times; both endpoint visibility required',
        'motion_control_sha256':sha256(OUT/'joint_control.npz'),'future_real_used':'evaluation only after pair generation',
        'robot_query_count':len(uv)-n_cup,'cup_query_count':n_cup,
        'tracker_semantic_limitation':'Tracking errors can include background substitution; low ADE cannot certify cup identity/contact',
        'variants':{}}
    for name in gen:
        xy=samples[name]['xy'];v=samples[name]['vis']&valid(xy)
        own=own_static if name=='robot_static' else target
        desc={
            'cup_to_control':metrics(target[:,:n_cup],xy[:,:n_cup],v[:,:n_cup]&valid(target[:,:n_cup])),
            'robot_to_requested_coupled_control':metrics(target[:,n_cup:],xy[:,n_cup:],v[:,n_cup:]&valid(target[:,n_cup:])),
            'robot_to_own_control':metrics(own[:,n_cup:],xy[:,n_cup:],v[:,n_cup:]&valid(own[:,n_cup:])),
            'cup_to_real':metrics(real_xy[:,:n_cup],xy[:,:n_cup],v[:,:n_cup]&real_vis[:,:n_cup]),
            'robot_to_real':metrics(real_xy[:,n_cup:],xy[:,n_cup:],v[:,n_cup:]&real_vis[:,n_cup:]),
            'common_cup_to_control':metrics(target[:,:n_cup],xy[:,:n_cup],common[:,:n_cup]),
            'common_robot_to_requested_control':metrics(target[:,n_cup:],xy[:,n_cup:],common[:,n_cup:]),
            'common_cup_to_real':metrics(real_xy[:,:n_cup],xy[:,:n_cup],common[:,:n_cup]&real_vis[:,:n_cup]),
            'common_robot_to_real':metrics(real_xy[:,n_cup:],xy[:,n_cup:],common[:,n_cup:]&real_vis[:,n_cup:])}
        if name=='robot_coupled_initial_only':
            # C has different conditioning. Keep A/B's exact pair comparison;
            # compare C using an explicitly shared three-way intersection.
            for key in [key for key in desc if key.startswith('common_')]:del desc[key]
        desc['shared_three_cup_to_control']=metrics(target[:,:n_cup],xy[:,:n_cup],common_three[:,:n_cup])
        desc['shared_three_robot_to_requested_control']=metrics(target[:,n_cup:],xy[:,n_cup:],common_three[:,n_cup:])
        contact_mask=common if name in ['robot_static','robot_coupled'] else common_three
        # Match the same surviving cup/gripper material-point subset in both
        # variants. This measures projected relative-motion error, not force,
        # 3D contact, or semantic object identity.
        grip=np.flatnonzero(control['robot_query_link']==6)+n_cup;contact=[];counts=[]
        for i in range(10):
            ci=np.flatnonzero(contact_mask[i,:n_cup]);gi=grip[contact_mask[i,grip]];counts.append([len(ci),len(gi)])
            if len(ci)<3 or len(gi)<2:contact.append(None);continue
            actual=xy[i,ci].mean(0)-xy[i,gi].mean(0)
            expected=target[i,ci].mean(0)-target[i,gi].mean(0)
            contact.append(float(np.linalg.norm(actual-expected)))
        present=[value for value in contact if value is not None]
        desc['common_cup_gripper_relative_motion']={'mean_offset_error_px':float(np.mean(present)) if present else None,
            'per_time_offset_error_px':contact,'valid_times':len(present),'pairs_by_time':counts,
            'interpretation':'2D cup-minus-gripper centroid-vector residual on shared visible material points; not certified 3D or physical contact'}
        aligned=np.stack([(gen[name][a].astype(float)*(1-(t*8-a))+gen[name][b].astype(float)*(t*8-a)).astype(np.uint8) for t,a,b in zip(times,lo,hi)])
        mae=np.abs(aligned.astype(float)-future.astype(float));mse=((aligned.astype(float)-future.astype(float))**2).mean()
        desc['image_quality']={'full_frame_PSNR_dB':float(10*np.log10(255**2/max(mse,1e-10))),
            'full_frame_MAE':float(mae.mean()),'shared_requested_motion_ROI_MAE':float(mae[eval_masks].mean()),
            'shared_requested_motion_ROI_fraction':float(eval_masks.mean()),
            'laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for f in aligned])),
            'temporal_5Hz_local_warp':temporal_warp_error(aligned,eval_masks),
            'temporal_5Hz_full_warp':temporal_warp_error(aligned),
            'scope':'Descriptive one-clip comparison. Generated RGB linearly sampled at 5Hz; ROI identical across variants. No perceptual/FVD claim.'}
        desc['contact_mask_scope']='A/B intersection' if name in ['robot_static','robot_coupled'] else 'A/B/C intersection; do not compare to A/B contact mean directly'
        report['variants'][name]=desc
        np.savez_compressed(OUT/name/'aligned_joint_evaluation.npz',times=times,generated_xy=xy,generated_visibility=v,
            requested_xy=target,real_xy=real_xy,real_visibility=real_vis,shared_mask=common)
    report['real_image_quality']={'laplacian_variance':float(np.mean([cv2.Laplacian(cv2.cvtColor(f,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for f in future])),
        'temporal_5Hz_local_warp':temporal_warp_error(future,eval_masks)}
    write_json(OUT/'paired_metrics.json',report)
    with imageio.get_reader(str(OUT/'robot_coupled/control_molmomotion_640x480.mp4')) as reader:controls=np.stack([frame for frame in reader])
    # These exact times are shared by the native 5Hz and generated 8Hz grids.
    real_matching=[real[0],real[5],real[10]]
    static_matching=gen['robot_static'][[0,8,16]];coupled_matching=gen['robot_coupled'][[0,8,16]]
    sheet(OUT/'paired_results_exact_times.png',[controls[[0,8,16]],static_matching,coupled_matching,real_matching],
        ['COUPLED DENSE CONTROL','A STATIC ROBOT; GENERATED','B COUPLED ROBOT; GENERATED','REAL CONTINUATION'],indices=(0,1,2),fps=1,tile=(640,480))
    comparison=[]
    for i in range(17):
        ri=int(np.clip(np.rint(i/8*5),0,10))
        row=[labeled(gen['robot_static'][i],f'A STATIC ARM: generated {i/8:.3f}s'),
             labeled(gen['robot_coupled'][i],f'B COUPLED ARM: generated {i/8:.3f}s'),
             labeled(real[ri],f'REAL: nearest measured {ri/5:.3f}s')]
        comparison.append(np.concatenate(row,axis=1))
    save_video(OUT/'paired_generated_vs_real_2s.mp4',comparison)
    save_video(OUT/'static_vs_coupled_full_6s.mp4',[np.concatenate([labeled(a,f'A STATIC ARM {i/8:.3f}s'),labeled(b,f'B COUPLED ARM {i/8:.3f}s')],axis=1) for i,(a,b) in enumerate(zip(gen['robot_static'],gen['robot_coupled']))])
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in gen:
        m=report['variants'][name]
        axes[0].plot(times,[np.nan if p is None else p for p in m['shared_three_robot_to_requested_control']['error_by_time_px']],marker='o',label=name)
        if name!='robot_coupled_initial_only':
            axes[1].plot(times,[np.nan if p is None else p for p in m['common_cup_gripper_relative_motion']['per_time_offset_error_px']],marker='o',label=name)
    axes[0].set(ylabel='Robot requested-motion error (px)',xlabel='Physical time (s)')
    axes[1].set(ylabel='Cup/gripper relative-motion error (px)',xlabel='Physical time (s)')
    for ax in axes:ax.grid(alpha=.3);ax.legend()
    fig.tight_layout();fig.savefig(OUT/'paired_motion_metrics.png',dpi=150);plt.close(fig)
    print(json.dumps({name:{k:v for k,v in desc.items() if k.startswith('common')} for name,desc in report['variants'].items()},indent=2))

if __name__=='__main__':main()
