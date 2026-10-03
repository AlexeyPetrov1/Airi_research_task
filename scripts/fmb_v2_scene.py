"""Independent FMB v2 export. Future export requires complete H3 and H1 receipts."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import numpy as np
import cv2
from berkeley_preprocess import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/fmb_v2_berkeley_matched'
SPECS = {
    'episode_5201': {'source': ROOT/'runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy',
                     't0': 126, 'share_robot_id': '57_fmb#episode_5201'},
    'fmb_control_n3': {'source': ROOT.parent/'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy',
                       't0': 130, 'share_robot_id': None},
}
ACTION = 'Insert the red rectangular peg into the matching hole on the blue board.'

def preserved_v1():
    receipt = RUN/'preserved_v1.json'
    if receipt.exists():
        return
    folders = [ROOT/'runs/sharerobot_fmb_episode_5201', ROOT/'runs/fmb_second_example_1_M_L_3_vertical_n_3']
    paths = [p for base in folders for p in base.rglob('*') if p.is_file() and p.suffix != '.pyc']
    paths += [ROOT/'report'/name for name in ['fmb_quantitative_2d_episode_5201.md',
              'fmb_quantitative_2d_second_trial.md', 'fmb_geometry_forecast_study.md']]
    write_json(receipt, {'label': 'FMB v1 — geometry sensitivity study',
                        'sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}})

def export(name, future=False):
    spec = SPECS[name]
    scene = RUN/name
    t0 = spec['t0']
    if future:
        for model in ['predictions', 'predictions_h1'] if spec['share_robot_id'] else ['predictions']:
            status = json.loads((scene/model/'model_run.json').read_text())
            assert status['success'] and status['successful_chunks'] == 3
        folder = scene/'evaluation'
        assert not (folder/'future_rgb.npy').exists(), 'Future already exported'
        idx = np.arange(t0+1, t0+21)
    else:
        assert not (scene/'metadata.json').exists(), 'Refusing to overwrite selected scene'
        preserved_v1()
        idx = np.arange(t0-7, t0+1)
        folder = scene/'observed'
    folder.mkdir(parents=True, exist_ok=True)
    # Trusted, previously audited local official NPY dictionary; pickle is intrinsic to source format.
    source = np.load(spec['source'], allow_pickle=True).item()
    raw = source['obs/side_1'][idx]
    rgb = raw[..., ::-1].copy()  # Official FMB format is BGR; models require RGB.
    z16 = source['obs/side_1_depth'][idx]
    assert rgb.shape == (len(idx),256,256,3) and rgb.dtype == np.uint8
    assert z16.shape == (len(idx),256,256) and z16.dtype == np.uint16
    np.save(folder/('future_rgb.npy' if future else 'rgb.npy'), rgb)
    np.save(folder/'source_indices.npy', idx)
    np.save(folder/'timestamps.npy', idx.astype(float)/10)
    np.save(folder/'sensor_z16.npy', z16)
    np.save(folder/('native_depth_future.npy' if future else 'native_depth.npy'), z16.astype(np.float32)*1e-4)
    for i, frame in zip(idx, rgb):
        cv2.imwrite(str(folder/f'frame_{i:06d}.png'), frame[..., ::-1])
    write_json(folder/'native_depth_metadata.json', {
        'measured_metric_depth': True, 'measurement': 'RealSense sensor Z16, not monocular depth',
        'units': 'meters', 'depth_scale_m_per_raw_unit': 1e-4,
        'depth_scale_status': 'SUPPORTED_HYPOTHESIS_NOT_RECORD_CONFIRMED',
        'scale_evidence': 'Prior independent CAD and TCP checks support 0.0001; no recorded sensor depth_units metadata found',
        'registration_status': 'published pixel geometry requires separate observed-only audit',
        'source_indices': idx.tolist(), 'future_used': future,
        'source': 'FMB obs/side_1_depth', 'source_sha256': sha256(spec['source'])})
    if future:
        write_json(folder/'export_receipt.json', {'prediction_gated': True, 'source_indices': idx.tolist(),
                   'source_sha256': sha256(spec['source']), 'nominal_hz': 10, 'hardware_timestamps_available': False})
        return
    for top in ['geometry', 'groups', 'predictions', 'predictions_h1', 'viz']:
        (scene/top).mkdir(exist_ok=True)
    np.save(folder/'history_rgb.npy', rgb[-3:])
    np.save(folder/'history_source_indices.npy', idx[-3:])
    np.save(folder/'history_timestamps.npy', idx[-3:].astype(float)/10)
    if name == 'episode_5201':
        provenance = ROOT/'runs/sharerobot_fmb_episode_5201/preflight.json'
        old = json.loads(provenance.read_text())
        assert old['source_npy_sha256'] == sha256(spec['source'])
        assert old['source_mapping_status'] == 'PASS_SOURCE_EPISODE_RECOVERED'
        shutil.copyfile(provenance, scene/'provenance_v1_source_audit.json')
    write_json(scene/'metadata.json', {
        'experiment': 'FMB v2 — Berkeley-matched pipeline', 'scene': name,
        'share_robot_id': spec['share_robot_id'], 'source_file': spec['source'].name,
        'source_path': str(spec['source']), 'source_sha256': sha256(spec['source']),
        'camera': 'side_1', 'instruction': ACTION, 'task': ACTION,
        'action_origin': 'Fixed English paraphrase of source insert primitive and observed object; same action as v1',
        'target_object': 'red rectangular peg', 'source_object_info': source['object_info'],
        'source_primitive_observed': list(map(str, source['primitive'][idx])),
        't0': t0, 't0_source_frame': t0, 'observed_source_indices': idx.tolist(),
        'history_source_indices': idx[-3:].tolist(), 'history_timestamps': (idx[-3:]/10).tolist(),
        'future_source_indices': list(range(t0+1,t0+21)), 'future_timestamps': (np.arange(t0+1,t0+21)/10).tolist(),
        'source_fps': 10, 'model_training_fps': 15, 'hardware_timestamps_available': False,
        'time_basis': 'nominal source 10 Hz and model 15 Hz',
        'color_transform': 'Official BGR source -> RGB; differs from ShareRobot byte-preserving PNG color interpretation',
        'depth_scale_status': 'SUPPORTED_HYPOTHESIS_NOT_RECORD_CONFIRMED', 'depth_scale': 1e-4,
        'selection_before_prediction': True, 'primary_example': name == 'episode_5201',
        'selection_reason': 'Pre-existing v1 window retained for independent protocol comparison; no v2 outcomes used',
        'future_used_for_model_input': False,
        'second_scene_scope': 'FMB control only; no verified ShareRobot mapping' if not spec['share_robot_id'] else None})
    tiles=[]
    for index,frame in zip(idx,rgb):
        tile=cv2.resize(frame,(384,384),interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile,f'Observed frame {index}',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        tiles.append(tile)
    cv2.imwrite(str(scene/'viz/observed_8_frames.png'), np.vstack([np.hstack(tiles[:4]),np.hstack(tiles[4:])])[...,::-1])
    print(name, 'future' if future else 'observed', rgb.shape, flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene',choices=[*SPECS,'all'])
    p.add_argument('--future',action='store_true')
    args=p.parse_args()
    for name in SPECS if args.scene == 'all' else [args.scene]: export(name,args.future)
