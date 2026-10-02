"""Independent artifact/content audit of the completed two-scene experiment."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import cv2
from berkeley_infer import strict_parse

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/berkeley_ur5_molmomotion'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify(run_root=RUN):
    run_root=Path(run_root)
    result={}
    for scene,ep,t0 in [('cup',10,63),('bottle',9,47)]:
        folder=run_root/scene
        meta=json.loads((folder/'metadata.json').read_text())
        metrics=json.loads((folder/'metrics.json').read_text())
        status=json.loads((folder/'predictions/model_run.json').read_text())
        freeze=json.loads((folder/'predictions/input_freeze.json').read_text())
        assert meta['source_episode_id']==ep and meta['t0_source_frame']==t0
        assert meta['camera']=='observation.images.image'
        assert meta['history_source_indices']==[t0-2,t0-1,t0]
        assert meta['future_source_indices']==list(range(t0+1,t0+11))
        assert not meta['future_used_for_model_input']
        grounding=json.loads((folder/'observed/grounding_metadata.json').read_text())
        pointing=json.loads((folder/'observed/molmopoint_grounding.json').read_text())
        assert grounding['segmentation']=='official Meta SAM2.1 Hiera-Large'
        assert 'sam2.1_hiera_large.pt' in grounding['checkpoint']
        assert grounding['source_frame']==t0 and grounding['future_used'] is False
        assert pointing['success'] and pointing['model_id']=='allenai/MolmoPoint-Vid-4B'
        assert pointing['input_source_indices']==[t0] and pointing['future_used'] is False
        assert pointing['no_manual_point_or_segmentation_selection'] is True
        assert (folder/'observed/molmopoint_raw_output.txt').stat().st_size>0
        assert np.load(folder/'observed/molmopoint_generated_token_ids.npy').size>0
        assert all(sha(folder/path)==digest for path,digest in freeze['sha256'].items())
        ids=np.load(folder/'observed/selected_point_ids.npy')
        history=np.load(folder/'observed/points_3d_history.npy')
        pred=np.load(folder/'predictions/future_3d.npy')
        assert len(ids) in [8,16,24] and len(np.unique(ids))==len(ids)
        assert history.shape==(3,len(ids),3) and np.isfinite(history).all()
        assert pred.shape==(len(ids),30,3) and np.isfinite(pred).all() and not np.all(pred==0)
        assert status['success'] and status['successful_chunks']==len(ids)//8<=3
        for i,g in enumerate(status['groups']):
            out=folder/'predictions'/g['group']
            group=folder/'groups'/g['group']
            assert g['success'] and g['parsed_point_times']==240
            delta=strict_parse((out/'raw_model_output.txt').read_text())
            saved=np.load(out/'future_3d.npy')
            assert np.allclose(saved,pred[i*8:(i+1)*8])
            # One shared model anchor is constant over every point/time pair.
            recovered=saved-delta
            assert np.allclose(recovered,recovered[0,0],atol=1e-4)
            assert np.array_equal(np.load(group/'point_ids.npy'),ids[i*8:(i+1)*8])
        assert metrics['number_of_successful_model_chunks']==len(ids)//8
        assert metrics['timing']['model'] is not None
        assert metrics['GPU_memory']['model_peak_gib']>0
        evaldata=np.load(folder/'evaluation/evaluation_results.npz')
        mask=evaldata['common_visibility_mask']
        assert np.array_equal(evaldata['prediction_indices'],[2,5,8,11,14,17,20,23,26,29])
        assert np.array_equal(evaldata['prediction_3d_at_5hz'],pred[:,[2,5,8,11,14,17,20,23,26,29]])
        for name,key in [('MolmoMotion','prediction_2d'),('static','static_2d'),('constant_velocity','constant_velocity_2d')]:
            error=np.linalg.norm(evaldata[key]-evaldata['ground_truth_2d'],axis=-1)
            ade=float(error[mask].mean()) if mask.any() else None
            fde=float(error[:,-1][mask[:,-1]].mean()) if mask[:,-1].any() else None
            row=metrics['methods'][name]
            assert (ade is None and row['ADE_2D_px'] is None) or np.isclose(ade,row['ADE_2D_px'])
            assert (fde is None and row['FDE_2D_px'] is None) or np.isclose(fde,row['FDE_2D_px'])
        required_png=['history_contact_sheet','mask_and_100_queries','selected_24_points','depth_t0',
                      'intrinsics_stability','trajectory_3d','final_overlay_t0','error_by_time']
        for name in required_png:
            image=cv2.imread(str(folder/'viz'/f'{name}.png'))
            assert image is not None and min(image.shape[:2])>100
        videos={}
        for name in ['pred_vs_gt_2d','side_by_side']:
            cap=cv2.VideoCapture(str(folder/'viz'/f'{name}.mp4'))
            frame_count=0
            while True:
                ok,frame=cap.read()
                if not ok:break
                frame_count+=1
            assert frame_count==10
            assert abs(cap.get(cv2.CAP_PROP_FPS)-5)<.01
            cap.release()
            videos[name]=frame_count
        native=np.load(folder/'observed/native_depth.npy')
        assert np.array_equal(np.load(folder/'geometry/depth_observed.npy'),native)
        assert json.loads((folder/'observed/native_depth_metadata.json').read_text())['measured_metric_depth']
        assert metrics['3D_est'] is not None
        result[scene]={'passed':True,'prediction_shape':list(pred.shape),'input_hashes_verified':len(freeze['sha256']),
                       'ADE_2D_px':metrics['ADE_2D_px'],'FDE_2D_px':metrics['FDE_2D_px'],
                       'valid_pairs':int(mask.sum()),'videos_decoded':videos}
    (run_root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=RUN)
    args=parser.parse_args()
    verify(args.run_root)
