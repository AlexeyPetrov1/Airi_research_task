"""Export Berkeley observations before inference; future export is prediction-gated."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
import av
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/'data/berkeley_ur5_two_scenes'
RUN = ROOT/'runs/berkeley_ur5_molmomotion'
SCENES = {
    'cup': {'episode': 10, 'instruction': 'Pick up the blue cup and put it into the brown cup.',
            'object': 'blue cup', 'reason': 'Printed motif gives visible texture for persistent points; exposed cup separated from brown cup; ep4 initial overlap, ep7/12 plain sides.'},
    'bottle': {'episode': 9, 'instruction': 'Put the ranch bottle into the pot.',
               'object': 'ranch bottle', 'reason': 'White bottle body exposed through lift; ep2/5/6 have substantially stronger gripper occlusion of body.'},
}


def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def attach_native(scene, run_root=RUN):
    out = Path(run_root)/scene
    assert not (out/'predictions/input_freeze.json').exists(), 'Cannot modify frozen model inputs'
    indices = np.load(out/'observed/source_indices.npy')
    files = [DATA/'native'/scene/'observed'/f'depth_{i:06d}.npy' for i in indices]
    depths = np.stack([np.load(p) for p in files])
    assert depths.dtype == np.float32 and depths.shape == (len(indices),480,640)
    np.save(out/'observed/native_depth.npy', depths)
    receipt = DATA/'native_observed_receipt.json'
    write(out/'observed/native_depth_metadata.json', {
        'measured_metric_depth': True, 'source': 'TFDS steps.observation.image_with_depth',
        'units': 'meters', 'dtype': 'float32', 'source_indices': indices.tolist(),
        'source_receipt_path': str(receipt), 'source_receipt_sha256': hashlib.sha256(receipt.read_bytes()).hexdigest(),
        'source_files_sha256': {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'future_used':False,
    })
    meta=json.loads((out/'metadata.json').read_text())
    meta.update(depth_source='RealSense measured aligned metric depth (TFDS float32)',
                K_source='UniDepthV2 estimated intrinsics, observed robust median')
    write(out/'metadata.json', meta)
    print(scene, 'attached native measured depth', depths.shape, flush=True)


def export(scene, t0, future=False, run_root=RUN):
    spec = SCENES[scene]
    out = Path(run_root)/scene
    video = DATA/'videos/chunk-000/observation.images.image'/f"episode_{spec['episode']:06d}.mp4"
    parquet = DATA/'data/chunk-000'/f"episode_{spec['episode']:06d}.parquet"
    df = pd.read_parquet(parquet)
    timestamps = df.timestamp.to_numpy()
    assert 7 <= t0 < len(df)-10
    if future:
        status = json.loads((out/'predictions/model_run.json').read_text())
        assert status['success'], 'Actual completed predictions required before evaluation export'
        saved = json.loads((out/'metadata.json').read_text())
        assert saved['t0_source_frame'] == t0
        indices = list(range(t0+1, t0+11))
        folder = out/'evaluation'
    else:
        assert not (out/'predictions/input_freeze.json').exists(), 'Cannot modify frozen model input'
        indices = list(range(t0-7,t0+1))
        folder = out/'observed'
    folder.mkdir(parents=True, exist_ok=True)
    frames, pts = [], []
    with av.open(video) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in indices:
                frames.append(frame.to_ndarray(format='rgb24'))
                pts.append({'index':index, 'pts':frame.pts, 'time_base':str(frame.time_base), 'video_seconds':float(frame.time)})
            if index >= indices[-1]:
                break
    rgb = np.stack(frames)
    assert len(rgb) == len(indices)
    np.save(folder/('future_rgb.npy' if future else 'rgb.npy'), rgb)
    np.save(folder/'source_indices.npy', np.array(indices))
    np.save(folder/'timestamps.npy', timestamps[indices])
    for i, frame in zip(indices, rgb):
        Image.fromarray(frame).save(folder/f'frame_{i:06d}.png')
    if not future:
        np.save(folder/'history_rgb.npy', rgb[-3:])
        np.save(folder/'history_source_indices.npy', np.array(indices[-3:]))
        np.save(folder/'history_timestamps.npy', timestamps[indices[-3:]])
        meta = dict(scene=scene, source_episode_id=spec['episode'], episode_id=spec['episode'],
            instruction=spec['instruction'], task=spec['instruction'], target_object=spec['object'],
            source_task_text=df.task_index.iloc[0].item() if hasattr(df.task_index.iloc[0], 'item') else int(df.task_index.iloc[0]),
            source_repo='lerobot/berkeley_autolab_ur5', source_revision='6306aa91c00b9b0de28831b7fceaa42b8096bb4f',
            camera='observation.images.image', source_fps=5, model_training_fps=15,
            t0=t0, t0_source_frame=t0, t0_seconds=float(timestamps[t0]),
            observed_source_indices=indices, history_source_indices=indices[-3:],
            history_timestamps=timestamps[indices[-3:]].tolist(),
            future_source_indices=list(range(t0+1,t0+11)),
            future_timestamps=timestamps[t0+1:t0+11].tolist(),
            prediction_evaluation_indices=[2,5,8,11,14,17,20,23,26,29],
            source_video=str(video), source_parquet=str(parquet),
            source_sha256={'video':hashlib.sha256(video.read_bytes()).hexdigest(),
                           'parquet':hashlib.sha256(parquet.read_bytes()).hexdigest()},
            selection_reason=spec['reason'],
            selection_policy='Data quality only; episode/t0 screening precedes predictions. Future screening used only to select episode/t0, never to form model inputs.',
            future_used_for_model_input=False,
            warnings=['H3 uses real adjacent 5 FPS frames [-0.4,-0.2,0] seconds; temporal distribution shift from training at 15 FPS.',
                      'Future tracking is a tracker-based reference, not manually measured physical point ground truth.'],
            created_utc=datetime.now(timezone.utc).isoformat())
        write(out/'metadata.json',meta)
        for name in ['evaluation','geometry','groups','predictions','viz']:
            (out/name).mkdir(exist_ok=True)
    write(folder/'source_frames.json', {'source_indices':indices, 'source_timestamps':timestamps[indices].tolist(),
                                      'video_pts':pts,'exported_utc':datetime.now(timezone.utc).isoformat()})
    print(scene, 'future' if future else 'observed', rgb.shape, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene',choices=SCENES)
    p.add_argument('--t0',type=int,required=True)
    p.add_argument('--future',action='store_true')
    p.add_argument('--attach-native',action='store_true')
    p.add_argument('--run-root',type=Path,default=RUN)
    args=p.parse_args()
    if args.attach_native:
        attach_native(args.scene,args.run_root)
    else:
        export(args.scene,args.t0,args.future,args.run_root)
