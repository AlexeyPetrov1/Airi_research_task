"""Rank observed candidates by independently held-out RGBD/TCP calibration."""
import json
from pathlib import Path
import numpy as np
import fmb_wrist_robot_rgbd_calibration as calibration
from fmb_wrist_prepare import ROOT,RUN,DATA,write

def main():
    specs=json.loads((RUN/'selection/candidates.json').read_text());results=[]
    root=RUN/'selection/geometry_candidates';root.mkdir(exist_ok=True)
    for spec in specs:
        dest=root/Path(spec['source_file']).stem;dest.mkdir(exist_ok=True)
        # Source profiles are shared pinned assets. This is not a camera extrinsic shared between recordings.
        (dest/'sources').mkdir(exist_ok=True)
        (dest/'sources/official_K_candidates.json').write_bytes((RUN/'sources/official_K_candidates.json').read_bytes())
        d=np.load(DATA/spec['source_file'],allow_pickle=True).item();ids=np.array(spec['observed_indices']);row={'source_file':spec['source_file'],'t0':spec['candidate_t0'],'future_used':False,'views':{}}
        for camera in ['wrist_2','wrist_1']:
            scene=dest/camera;(scene/'observed').mkdir(parents=True,exist_ok=True);(scene/'geometry').mkdir(exist_ok=True)
            np.save(scene/'observed/rgb.npy',d['obs/'+camera][ids][...,::-1].copy());np.save(scene/'observed/sensor_z16.npy',d['obs/'+camera+'_depth'][ids]);np.save(scene/'observed/tcp_pose_xyzw.npy',d['obs/tcp_pose'][ids])
            calibration.RUN=dest
            try:
                calibration.run(camera)
                info=json.loads((scene/'geometry/independent_robot_rgbd_hand_eye.json').read_text());boot=json.loads((scene/'geometry/independent_robot_rgbd_bootstrap.json').read_text())['fits']
                row['views'][camera]={'gate':info['gate'],'heldout_median_px':info['heldout']['median_px'],'heldout_p90_px':info['heldout']['p90_px'],
                    'translation_norm_m':info['translation_norm_m'],'bootstrap_median_X_translation_difference_mm':float(np.median([r['translation_difference_mm'] for r in boot])),
                    'bootstrap_p90_relative_path_difference_mm':float(np.percentile([r['observed_relative_path_p90_translation_difference_mm'] for r in boot],90))}
            except (RuntimeError,ValueError) as e:row['views'][camera]={'gate':False,'error':str(e)}
        row['both_views_gate']=all(v['gate'] for v in row['views'].values())
        row['score']=sum(v.get('heldout_p90_px',100) for v in row['views'].values())+sum(v.get('bootstrap_p90_relative_path_difference_mm',1000)/10 for v in row['views'].values())
        results.append(row);write(RUN/'selection/geometry_ranking.json',{'results':results,'future_used':False,'ranked_source_files':[r['source_file'] for r in sorted(results,key=lambda r:(not r['both_views_gate'],r['score']))]})
        print('Candidate result',row,flush=True)
    write(RUN/'selection/geometry_ranking.json',{'results':results,'future_used':False,'ranked_source_files':[r['source_file'] for r in sorted(results,key=lambda r:(not r['both_views_gate'],r['score']))]})

if __name__=='__main__':main()
