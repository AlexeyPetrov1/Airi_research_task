"""Data-only native RGB-D and selection audit, independent of model outputs."""
from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from berkeley_data_extract_native import ROOT,SCENES
RUN=Path(__file__).resolve().parents[1]/'runs/berkeley_ur5_molmomotion'
def main():
    report={'selection_used_model_predictions':False,'selection_frozen_before_model_inference':True,'camera':'observation.images.image','candidate_screening':'First four source-index episodes of each exact task with >=40 frames, then visual inspection; no model output inspected','all_candidates_file':str(ROOT/'candidates_all.json'),'top_candidates':{'cup':[{'episode_id':10,'rank':1,'reason':'Printed motif supplies local texture; blue body exposed and separate from destination at first lift.'},{'episode_id':7,'rank':2,'reason':'Fully exposed and separate but side mostly textureless.'},{'episode_id':12,'rank':3,'reason':'Visible body and separation but smooth side gives weaker tracking texture.'},{'episode_id':4,'rank':4,'reason':'Blue cup initially behind brown cup; initial overlap reduces useful visible area.'}],'bottle':[{'episode_id':9,'rank':1,'reason':'White body remains exposed at lift, before pot occlusion; rotation increases difficulty but interpretable.'},{'episode_id':5,'rank':2,'reason':'Body substantially obscured by gripper during lift.'},{'episode_id':6,'rank':3,'reason':'Body behind gripper at lift; cap dominates the exposed object.'},{'episode_id':2,'rank':4,'reason':'Gripper strongly obscures body and reaches pot soon.'}]},'scenes':{}}
    for scene,ep,t0,start,rec in SCENES:
        folder=ROOT/'native'/scene/'observed';rgb=np.array(Image.open(folder/f'rgb_{t0:06d}.png'));depth=np.load(folder/f'depth_{t0:06d}.npy')
        valid=np.isfinite(depth)&(depth>.05)&(depth<5)
        # Full image diagnostic and object crop selected from t0 only, not future.
        bounds=(110,220,210,340)if scene=='cup'else (110,270,220,385)
        x0,y0,x1,y1=bounds
        crop=depth[y0:y1,x0:x1]
        fig,axes=plt.subplots(2,3,figsize=(15,8));axes[0,0].imshow(rgb);axes[0,0].set_title(f'{scene} episode {ep} t0={t0}: native RGB')
        show=axes[0,1].imshow(np.where(valid,depth,np.nan),cmap='turbo',vmin=.3,vmax=1.2);axes[0,1].set_title('Native depth, meters (invalid transparent)');fig.colorbar(show,ax=axes[0,1],fraction=.046)
        axes[0,2].imshow(rgb);axes[0,2].imshow(np.where(valid,depth,np.nan),cmap='turbo',vmin=.3,vmax=1.2,alpha=.45);axes[0,2].set_title('Pixel-grid alignment overlay')
        for j,arr in enumerate([rgb[y0:y1,x0:x1],crop,np.where((crop>.05)&(crop<5),crop,np.nan)]):
            axes[1,j].imshow(arr,cmap='turbo'if j else None,vmin=.3 if j else None,vmax=1.2 if j else None)
            axes[1,j].set_title(['T0 object neighborhood RGB','Native depth including holes','Native valid depth'][j])
        for ax in axes.flat:ax.axis('off')
        fig.tight_layout();fig.savefig(ROOT/f'native_alignment_{scene}.png',dpi=140);plt.close(fig)
        stats={'episode_id':ep,'t0_source_frame':t0,'t0_seconds':t0/5,'history_indices':list(range(t0-2,t0+1)),'history_seconds':[(t0-2)/5,(t0-1)/5,t0/5],'future_indices':list(range(t0+1,t0+11)),'gripper_closed_frame':52 if scene=='cup'else 36,'postgrasp_robot_motion_start':60 if scene=='cup'else 43,'t0_reason':'Early clear lift with exposed object, before destination occlusion.','native_depth_whole_frame_valid_fraction':float(valid.mean()),'camera_pose_assumption':'Fixed rig to be confirmed separately by observed background optical flow','depth_alignment_image':str(ROOT/f'native_alignment_{scene}.png'),'alignment_audit':'Native depth silhouettes correspond visually to robot, table, and target in external RGB; holes cluster at boundaries and in thin stripes. No image resize or pixel-grid remapping is applied.','capture_code_reference':'https://github.com/yunliangchen/ur5bc/blob/a4285610da52cb30215951c7ac37454fcfae01c8/ur5/robot_env.py','capture_depth_units_evidence_lines':[109,145,150,154,360]}
        mask_path=RUN/scene/'observed/mask.png'
        if mask_path.exists():
            mask=np.array(Image.open(mask_path))>0
            if mask.ndim==3:mask=mask.any(axis=-1)
            values=depth[mask&valid]
            stats.update(object_mask_area=int(mask.sum()),object_valid_native_depth_fraction=float(valid[mask].mean()),object_depth_percentiles_m=np.percentile(values,[0,5,50,95,100]).tolist())
        # Round-trip float32 byte storage verifies interpretation; no scale fitting.
        packed=np.asarray(Image.open(folder/f'depth_float_bits_{t0:06d}.png'),dtype=np.uint8)
        stats['float32_bitcast_roundtrip_exact']=bool(np.array_equal(depth.view(np.uint8).reshape(480,640,4),packed))
        report['scenes'][scene]=stats
    (ROOT/'selection_and_depth_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['scenes'],indent=2),flush=True)
if __name__=='__main__':main()
