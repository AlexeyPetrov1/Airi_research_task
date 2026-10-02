"""Sealed observed-only Berkeley inference, three greedy P8 calls per scene at most."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import resource
import subprocess
import time
import traceback
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'
REVISION = '3f5e790a511ff2cdf21c8d2a14cb4d8409c94629'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def strict_parse(raw):
    match = re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*', raw)
    if match is None:
        raise ValueError('Expected exactly one complete tracks block')
    frames = match.group(1).split(';')
    if len(frames) != 30:
        raise ValueError(f'Expected 30 steps, found {len(frames)}')
    delta = np.empty((8, 30, 3), dtype=np.float32)
    for step, frame in enumerate(frames):
        tokens = frame.split()
        if len(tokens) != 33 or Decimal(tokens[0]) != Decimal(step + 3):
            raise ValueError(f'Incomplete or incorrect time ID at {step}')
        seen = set()
        for off in range(1, len(tokens), 4):
            record = tokens[off:off+4]
            if any(re.fullmatch(r'[+-]?\d+', s) is None for s in record):
                raise ValueError('Noninteger quantized coordinate')
            point, x, y, z = map(int, record)
            if point not in range(1, 9) or point in seen:
                raise ValueError('Incorrect or duplicate point ID')
            seen.add(point)
            delta[point-1, step] = np.array([x, y, z]) / 1000
        if len(seen) != 8:
            raise ValueError('Missing point ID')
    return delta


def freeze_scene_inputs(scene, meta, action):
    """Seal model inputs together with their semantic and temporal identity."""
    scene = Path(scene)
    paths = [scene/'metadata.json']
    paths += [path for top in [scene/'observed', scene/'geometry', scene/'groups']
              for path in top.rglob('*') if path.is_file()]
    frozen = {path.relative_to(scene).as_posix(): sha(path) for path in paths}
    invariants = {'action': action,
        't0_source_frame': meta.get('t0_source_frame_index', meta.get('t0_source_frame', meta.get('t0'))),
        'observed_source_indices': meta['observed_source_indices'],
        'history_source_indices': meta['history_source_indices'],
        'history_timestamps': meta['history_timestamps']}
    t0 = invariants['t0_source_frame']
    if invariants['history_source_indices'] != list(range(t0-2, t0+1)):
        raise ValueError('Frozen H3 source indices must end exactly at t0')
    if invariants['observed_source_indices'][-3:] != invariants['history_source_indices']:
        raise ValueError('Frozen observed prefix and H3 indices disagree')
    path = scene/'predictions/input_freeze.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if prior['sha256'] != frozen or prior.get('scene_invariants') != invariants or prior.get('action') != action:
            raise ValueError('Observed inputs or action/t0/history identity changed after freezing')
    else:
        write(path, {'sha256': frozen, 'action': action, 'scene_invariants': invariants,
                     'observed_only': True, 'created_utc': datetime.now(timezone.utc).isoformat()})
    return frozen


def recover_saved_group(out, group, prior):
    """Validate a completed saved generation without spending another model call.

    Only a post-generation failure with raw text, the parsed tensor, and the
    original processor inputs is recoverable. An OOM/incomplete generation is
    not silently retried. Its attempted call remains counted.
    """
    out, group = Path(out), Path(group)
    required = [out/'raw_model_output.txt', out/'future_3d.npy', out/'processor_inputs.pt']
    if prior.get('generation_started') is not True or not all(path.exists() for path in required):
        raise RuntimeError('Existing generation attempt has no complete recoverable saved output: '+str(out))
    raw = (out/'raw_model_output.txt').read_text(encoding='utf-8')
    delta = strict_parse(raw)
    future = np.load(out/'future_3d.npy', allow_pickle=False)
    batch = torch.load(out/'processor_inputs.pt', map_location='cpu', weights_only=False)
    anchor = batch['anchor_3d'].detach().cpu().float().numpy().reshape(-1)
    expected_anchor = np.load(group/'points_3d_history.npy', allow_pickle=False)[-1, 0]
    if anchor.shape != (3,) or not np.array_equal(anchor, expected_anchor):
        raise ValueError('Saved processor anchor disagrees with frozen group history')
    if (future.shape != (8, 30, 3) or not np.isfinite(future).all() or not np.any(future)
            or not np.allclose(future, delta+anchor, atol=1e-4)):
        raise ValueError('Saved output fails strict text/anchor reconstruction; cannot recover')
    digest = sha(out/'future_3d.npy')
    if prior.get('prediction_sha256') and prior['prediction_sha256'] != digest:
        raise ValueError('Saved prediction checksum changed; cannot recover')
    if prior.get('raw_output_sha256') and prior['raw_output_sha256'] != sha(out/'raw_model_output.txt'):
        raise ValueError('Saved raw model text checksum changed; cannot recover')
    recovered = dict(prior)
    recovered.update(success=True, parse_status='FULL_8x30x3', parsed_point_times=240,
        prediction_shape=list(future.shape), prediction_sha256=digest,
        raw_output_sha256=sha(out/'raw_model_output.txt'), recovered_from_saved_artifacts=True,
        recovery_utc=datetime.now(timezone.utc).isoformat(), recovery_additional_forward_calls=0,
        previous_failure=prior.get('error'))
    if 'prediction_seconds' not in recovered:
        recovered['prediction_seconds'] = None
        recovered['timing_limitation'] = 'Original process did not persist inference duration before interruption'
    np.savez_compressed(out/'prediction.npz', future_3d=future,
                        parsed_visibility=np.ones((8, 30), dtype=bool))
    write(out/'model_run.json', recovered)
    return recovered


def finalize_scene(scene, groups, statuses, checkpoint_info, elapsed_seconds):
    """Assemble validated group outputs without loading or calling a model."""
    dest = scene/'predictions'
    future = np.concatenate([np.load(dest/group.name/'future_3d.npy') for group in groups])
    selected_ids = np.load(scene/'observed/selected_point_ids.npy')
    group_ids = np.concatenate([np.load(group/'point_ids.npy') for group in groups])
    if not np.array_equal(group_ids, selected_ids) or len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError('Merged groups do not identify the frozen selected points')
    if future.shape != (len(selected_ids), 30, 3) or not np.isfinite(future).all():
        raise ValueError('Incomplete merged prediction')
    np.save(dest/'future_3d.npy', future)
    np.savez_compressed(dest/'prediction.npz', future_3d=future, selected_point_ids=selected_ids)
    durations = [group.get('prediction_seconds') for group in statuses]
    status = dict(checkpoint_info)
    status.update(success=True, successful_chunks=len(groups), attempted_chunks=len(groups),
        groups=statuses, future_horizon=30, prediction_shape=list(future.shape),
        prediction_seconds=sum(durations) if all(value is not None for value in durations) else None,
        elapsed_seconds=elapsed_seconds,
        peak_cuda_allocated_gib=max((group['peak_cuda_allocated_gib'] for group in statuses if group.get('peak_cuda_allocated_gib') is not None), default=None),
        peak_cuda_reserved_gib=max((group['peak_cuda_reserved_gib'] for group in statuses if group.get('peak_cuda_reserved_gib') is not None), default=None),
        completed_utc=datetime.now(timezone.utc).isoformat())
    write(dest/'model_run.json', status)
    print(scene.name, 'COMPLETE', future.shape, flush=True)


def recover_only(scene_dirs):
    """Finish complete saved generations on CPU, or fail without extra calls."""
    from berkeley_input_audit import validate_group_payloads
    for scene in scene_dirs:
        started = time.monotonic()
        validate_group_payloads(scene)
        meta = json.loads((scene/'metadata.json').read_text())
        action = meta.get('instruction', meta.get('task'))
        if not (scene/'predictions/input_freeze.json').exists():
            raise ValueError('Recovery requires the original pre-generation input freeze')
        frozen = freeze_scene_inputs(scene, meta, action)
        groups = sorted(path for path in (scene/'groups').glob('group_*') if path.is_dir())
        receipts = []
        for group in groups:
            out = scene/'predictions'/group.name
            prior = json.loads((out/'model_run.json').read_text())
            receipts.append(recover_saved_group(out, group, prior))
        keys = ('model_id', 'checkpoint_hf_revision', 'config_sha256', 'model_bytes',
                'git_commit', 'inference_script_sha256', 'torch_version', 'dtype',
                'seed', 'decoding', 'max_new_tokens', 'model_load_seconds')
        info = {key: receipts[0].get(key) for key in keys}
        info.update(action=action, recovery_only=True, recovery_additional_forward_calls=0)
        if not all(sha(scene/name) == digest for name, digest in frozen.items()):
            raise ValueError('Frozen inputs changed during CPU recovery')
        finalize_scene(scene, groups, receipts, info, time.monotonic()-started)


def run(scene_dirs):
    from berkeley_input_audit import validate_group_payloads
    # Pure CPU guards run before loading the model or consuming any call budget.
    for scene in scene_dirs:
        validate_group_payloads(scene)
        audit = json.loads((scene/'geometry/input_audit.json').read_text())
        if audit.get('success') is not True:
            raise ValueError('Observed input quality audit must pass before inference')
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    start = time.monotonic()
    torch.manual_seed(0)
    processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = MolmoMotion.from_pretrained(str(CHECKPOINT))
    finally:
        torch.set_default_dtype(previous_dtype)
    model._internal = model._internal.cuda().eval()
    torch.cuda.synchronize()
    load_seconds = time.monotonic() - start
    checkpoint_info = {
        'model_id': 'allenai/MolmoMotion-4B-H3-F30', 'checkpoint_hf_revision': REVISION,
        'config_sha256': sha(CHECKPOINT/'config.yaml'),
        'model_bytes': (CHECKPOINT/'model.pt').stat().st_size,
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'inference_script_sha256': sha(Path(__file__)),
        'torch_version': torch.__version__, 'dtype': 'bfloat16', 'seed': 0,
        'decoding': 'official default greedy', 'max_new_tokens': 4800,
        'model_load_seconds': load_seconds,
    }
    for scene in scene_dirs:
        meta = json.loads((scene/'metadata.json').read_text(encoding='utf-8'))
        action = meta.get('instruction', meta.get('task'))
        assert action in ['Pick up the blue cup and put it into the brown cup.', 'Put the ranch bottle into the pot.']
        observed = scene/'observed'
        rgb = np.load(observed/'history_rgb.npy')
        assert rgb.shape[0] == 3
        frames = [Image.fromarray(f).convert('RGB') for f in rgb]
        dest = scene/'predictions'
        dest.mkdir(exist_ok=True)
        groups = sorted((scene/'groups').glob('group_*'))
        assert 1 <= len(groups) <= 3
        frozen = freeze_scene_inputs(scene, meta, action)
        status = dict(checkpoint_info, action=action, successful_chunks=0, attempted_chunks=0,
                      success=False, groups=[], future_horizon=30)
        scene_start = time.monotonic()
        for group in groups:
            out = dest/group.name
            out.mkdir(exist_ok=True)
            status_path = out/'model_run.json'
            if status_path.exists():
                prior = json.loads(status_path.read_text())
                if prior.get('success'):
                    stored = np.load(out/'future_3d.npy')
                    assert stored.shape == (8,30,3) and np.isfinite(stored).all() and not np.all(stored == 0)
                    assert sha(out/'future_3d.npy') == prior['prediction_sha256'], 'Saved prediction changed'
                    parsed = strict_parse((out/'raw_model_output.txt').read_text(encoding='utf-8'))
                    anchor = np.load(group/'points_3d_history.npy')[-1, 0]
                    assert np.allclose(stored, parsed+anchor, atol=1e-4), 'Saved raw output and coordinates disagree'
                    if prior.get('raw_output_sha256'):
                        assert sha(out/'raw_model_output.txt') == prior['raw_output_sha256'], 'Saved raw text changed'
                    status['groups'].append(prior)
                    continue
                if prior.get('generation_started'):
                    recovered = recover_saved_group(out, group, prior)
                    status['groups'].append(recovered)
                    continue
            points2d = torch.from_numpy(np.load(group/'points_2d_at_t0.npy')).float()
            points3d = torch.from_numpy(np.load(group/'points_3d_history.npy')).float()
            assert points2d.shape == (8,2) and points3d.shape == (3,8,3)
            assert torch.isfinite(points3d).all() and (points3d[...,2] > 0).all()
            batch = processor(history_frames=frames, points_2d_at_t0=points2d,
                points_3d_history=points3d, action=action, future_horizon=30)
            torch.save(batch, out/'processor_inputs.pt')
            group_status = dict(group=group.name, success=False, generation_started=False,
                created_utc=datetime.now(timezone.utc).isoformat(), **checkpoint_info)
            write(status_path, group_status)
            try:
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                prediction_start = time.monotonic()
                group_status['generation_started'] = True
                write(status_path, group_status)
                print(scene.name, group.name, 'generating', flush=True)
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model.predict_trajectory(**batch)
                torch.cuda.synchronize()
                group_status['prediction_seconds'] = time.monotonic()-prediction_start
                (out/'raw_model_output.txt').write_text(output.future_text, encoding='utf-8')
                future = output.future_3d.detach().cpu().float().numpy()
                np.save(out/'future_3d.npy', future)
                delta = strict_parse(output.future_text)
                anchor = batch['anchor_3d'].detach().cpu().float().numpy().squeeze()
                success = (future.shape == (8,30,3) and np.isfinite(future).all()
                           and not np.all(future == 0) and np.allclose(future, delta+anchor, atol=1e-4))
                if not success:
                    raise ValueError('Invalid output or strict parser mismatch')
                np.savez_compressed(out/'prediction.npz', future_3d=future,
                                    parsed_visibility=np.ones((8,30), dtype=bool))
                group_status.update(success=True, parse_status='FULL_8x30x3',
                    parsed_point_times=240, prediction_shape=list(future.shape),
                    raw_output_sha256=sha(out/'raw_model_output.txt'),
                    prediction_sha256=sha(out/'future_3d.npy'))
            except Exception as exc:
                group_status.update(error=str(exc), traceback=traceback.format_exc(), parse_status='FAILED')
                raise
            finally:
                group_status.update(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                    finished_utc=datetime.now(timezone.utc).isoformat())
                write(status_path, group_status)
            status['groups'].append(group_status)
            del batch, output
            torch.cuda.empty_cache()
        assert all(sha(scene/name) == digest for name,digest in frozen.items()), 'Inputs mutated during inference'
        finalize_scene(scene, groups, status['groups'], dict(checkpoint_info, action=action), time.monotonic()-scene_start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--recover-only', action='store_true', help='CPU-only completion from saved actual outputs; never loads model or calls forward')
    args = parser.parse_args()
    (recover_only if args.recover_only else run)(args.scene_dir)
