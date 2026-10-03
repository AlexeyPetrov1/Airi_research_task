"""Read the seven immutable bundles; open future artifacts only for evaluation."""
from pathlib import Path

import numpy as np

from .io import Source
from .sample import ExperimentSample


def restore_layout(value, strides):
    """Preserve NumPy reduction order for original transposed/sliced arrays."""
    strides = tuple(strides)
    if value.strides == strides:
        return value
    low = sum(min(0,(size-1)*step) for size,step in zip(value.shape,strides))
    high = sum(max(0,(size-1)*step) for size,step in zip(value.shape,strides))
    buffer = np.empty(max(high-low+value.dtype.itemsize,1),dtype=np.uint8)
    restored = np.ndarray(value.shape,dtype=value.dtype,buffer=buffer,offset=-low,strides=strides)
    np.copyto(restored,value)
    return restored


def load_bundle(config, root, evaluation=True):
    root = Path(root)
    folder = config['bundle']
    source = Source(root)
    manifest = source.json(folder+'/manifest.json')
    if manifest['schema_version'] != 1 or manifest['dataset'] != config['dataset']:
        raise ValueError('Bundle schema/dataset does not match config')
    with source.array(folder+'/observed.npz') as packet:
        observed = dict(packet)
    if manifest.get('history_media'):
        with source.array(manifest['history_media']) as media:
            observed['history_frames'] = media['history_frames']
    observed = {key:restore_layout(value,manifest['observed_strides'][key]) for key,value in observed.items()}
    auxiliary = manifest.get('observed_aux_values', {}).copy()
    for key, fields in manifest.get('observed_aux_arrays', {}).items():
        auxiliary[key] = {field:observed.pop('aux/'+key+'/'+field) for field in fields}
    model_files = source.files.copy()
    processors = source.json(folder+'/processor_fingerprints.json') if manifest['expected_status'] == 'COMPLETE' else []
    meta = manifest['observed_metadata'].copy()
    gate = meta.get('geometry_gate')
    failed = [key for key, valid in gate['checks'].items() if not valid] if gate else []
    status = 'SKIPPED_GEOMETRY_GATE' if failed else 'COMPLETE'
    if status != manifest['expected_status']:
        raise ValueError('Geometry gate disagrees with frozen expected status')
    sample = ExperimentSample(
        dataset=manifest['dataset'], episode_id=manifest['episode_id'],
        action=manifest['action'], future_times=np.array(manifest['future_times'], dtype=manifest['future_times_dtype']),
        history_frames=observed.pop('history_frames'), history_timestamps=observed.pop('history_timestamps'),
        points_2d=observed.pop('points_2d'), points_3d_history=observed.pop('points_3d_history'),
        point_ids=observed.pop('point_ids'), **observed, metadata=meta,
        source_files=source.files, model_input_files=model_files, processor_fingerprints=processors,
        observed_aux=auxiliary,
        status=status, failure_reason=', '.join(failed) if failed else None)
    if failed:
        return sample
    sample.saved_prediction = source.array(folder+'/legacy_prediction.npy')
    if evaluation:
        description = source.json(folder+'/evaluation.json')
        with source.array(folder+'/evaluation.npz') as packet:
            arrays = {key:restore_layout(value,description['strides'][key]) for key,value in packet.items()}
        sample.metadata = description['metadata']
        sample.evaluation = description['values']
        for key in description['arrays']:
            sample.evaluation[key] = arrays[key]
        for key, fields in description['nested_arrays'].items():
            sample.evaluation[key] = {field:arrays[key+'/'+field] for field in fields}
        for key in ('camera_intrinsics', 'camera_poses'):
            if key in arrays:
                setattr(sample, key, arrays[key])
    return sample
