"""Compare timing at equal phase and prior ablations on shared visibility."""
import json
import argparse
import cv2
import numpy as np
from PIL import Image
from das_full_motion_diagnose import SCENE, OUT
from das_prepare_control import save_video, sheet, write_json
from das_robot_evaluate import frames, labeled, valid
from das_evaluate import metrics


def before_after(chosen):
    old=frames(SCENE/'das_reference_repair/guided_endpoint_background/generated_seed42.mp4')
    new=frames(OUT/chosen/'generated_seed42.mp4')
    beforeafter=[]
    for i in range(49):
        a=cv2.resize(old[i],(640,480));b=cv2.resize(new[i],(640,480))
        beforeafter.append(np.concatenate([labeled(a,'OLD F: real endpoint, lift only (2s + hold)'),
            labeled(b,f'NEW {chosen}: full raw arc (6s, observed-only)')],axis=1))
    save_video(OUT/'before_after_full_arc.mp4',beforeafter)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--chosen',default='H2_group00_6s_no_prior')
    parser.add_argument('--before-after-only',action='store_true')
    args=parser.parse_args()
    if args.before_after_only:
        before_after(args.chosen)
        return
    names=['H1_group00_2s_no_prior','H2_group00_6s_no_prior']
    data={name:np.load(OUT/name/'motion_measurements.npz') for name in names}
    indices={names[0]:np.arange(17),names[1]:np.arange(17)*3}
    target=data[names[0]]['raw_forecast_xy'][:17]
    assert np.allclose(target,data[names[1]]['raw_forecast_xy'][::3])
    common=valid(target)
    for name in names:
        common&=data[name]['generated_visibility'][indices[name]]
    timing={'interpretation':'Same 17 forecast phases, source t=0..2s. Excludes the 2s variant hold tail.',
        'source_times_s':(np.arange(17)/8).tolist(),'common_pairs':int(common.sum()),'variants':{},
        'individual_visible_coverage_at_same_phases':{}}
    for name in names:
        timing['variants'][name]=metrics(target,data[name]['generated_xy'][indices[name]],common)
        own=valid(target)&data[name]['generated_visibility'][indices[name]]
        timing['individual_visible_coverage_at_same_phases'][name]={'usable_pairs':int(own.sum()),
            'total_pairs':int(own.size),'per_frame':own.sum(1).tolist(),
            'to_forecast':metrics(target,data[name]['generated_xy'][indices[name]],own)}
    report={'matched_timing':timing}
    prior='H3_group00_6s_prior025'
    if (OUT/prior/'motion_measurements.npz').exists():
        native=data[names[1]]
        guided=np.load(OUT/prior/'motion_measurements.npz')
        assert np.allclose(native['raw_forecast_xy'],guided['raw_forecast_xy'])
        target=native['raw_forecast_xy']
        common=valid(target)&native['generated_visibility']&guided['generated_visibility']
        report['matched_prior']={'interpretation':'Same trajectory, time, seed, prompt and sampler; common target/track visibility.',
            'common_pairs':int(common.sum()),'variants':{
                names[1]:metrics(target,native['generated_xy'],common),prior:metrics(target,guided['generated_xy'],common)}}
    uncontrolled='H4_no_trajectory_control'
    if (OUT/uncontrolled/'motion_measurements.npz').exists():
        native=data[names[1]];other=np.load(OUT/uncontrolled/'motion_measurements.npz')
        target=native['raw_forecast_xy']
        common=valid(target)&native['generated_visibility']&other['generated_visibility']
        report['matched_control']={'interpretation':'Same seed, prompt, initial image and native sampler; no prior in either; control_video=None in H4.',
            'common_pairs':int(common.sum()),'variants':{names[1]:metrics(target,native['generated_xy'],common),
                uncontrolled:metrics(target,other['generated_xy'],common)}}
    videos={name:np.array([cv2.resize(frame,(640,480)) for frame in frames(OUT/name/'generated_seed42.mp4')]) for name in names}
    phaseclip=[]
    for i in range(17):
        phaseclip.append(np.concatenate([labeled(videos[name][indices[name][i]],f'{name}: equal forecast phase {i}/16') for name in names],axis=1))
    save_video(OUT/'timing_comparison_equal_phase.mp4',phaseclip,fps=8)
    before_after(args.chosen)
    write_json(OUT/'matched_comparisons.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
