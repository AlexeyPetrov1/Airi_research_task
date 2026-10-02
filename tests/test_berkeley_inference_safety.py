"""CPU-only safety regressions using artificial temporary fixtures, not runs."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from berkeley_infer import freeze_scene_inputs, recover_only, recover_saved_group, strict_parse
from berkeley_input_audit import validate_group_payloads


def artificial_scene(path):
    obs, group = path/'observed', path/'groups/group_00'
    obs.mkdir()
    group.mkdir(parents=True)
    (path/'geometry').mkdir()
    ids = np.arange(8)
    history2d = np.arange(48, dtype=np.float32).reshape(3, 8, 2)
    history3d = np.arange(72, dtype=np.float32).reshape(3, 8, 3)/1000 + [0., 0., 1.]
    history3d = history3d.astype(np.float32)
    np.save(obs/'selected_point_ids.npy', ids)
    np.save(obs/'points_2d_history.npy', history2d)
    np.save(obs/'points_3d_history.npy', history3d)
    np.save(group/'point_ids.npy', ids)
    np.save(group/'points_2d_at_t0.npy', history2d[-1])
    np.save(group/'points_3d_history.npy', history3d)
    meta = {'instruction': 'Pick up the blue cup and put it into the brown cup.',
            't0': 12, 'observed_source_indices': list(range(5, 13)),
            'history_source_indices': [10, 11, 12], 'history_timestamps': [2., 2.2, 2.4]}
    (path/'metadata.json').write_text(json.dumps(meta))
    (path/'geometry/input_audit.json').write_text(json.dumps({'success': True}))
    return obs, group, meta


def artificial_saved_generation(scene, group):
    out = scene/'predictions/group_00'
    out.mkdir(parents=True, exist_ok=True)
    frames = [str(t) + ' ' + ' '.join(f'{p} {p} {2*p} {t}' for p in range(1, 9))
              for t in range(3, 33)]
    raw = '<tracks coords="'+';'.join(frames)+'">fixture</tracks>'
    anchor = np.load(group/'points_3d_history.npy')[-1, 0]
    future = strict_parse(raw)+anchor
    (out/'raw_model_output.txt').write_text(raw)
    np.save(out/'future_3d.npy', future)
    torch.save({'anchor_3d': torch.from_numpy(anchor[None].copy())}, out/'processor_inputs.pt')
    prior = {'success': False, 'generation_started': True, 'prediction_seconds': 7.,
             'error': 'artificial receipt write failure', 'peak_cuda_allocated_gib': 9.,
             'peak_cuda_reserved_gib': 10., 'group': 'group_00'}
    (out/'model_run.json').write_text(json.dumps(prior))
    return out, prior, future


class InferenceSafetyTests(unittest.TestCase):
    def test_stale_group_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            _, group, _ = artificial_scene(scene)
            self.assertEqual(validate_group_payloads(scene)['group_count'], 1)
            points = np.load(group/'points_3d_history.npy')
            points[1, 2, 0] += .1
            np.save(group/'points_3d_history.npy', points)
            with self.assertRaisesRegex(ValueError, 'Stale or reordered'):
                validate_group_payloads(scene)

    def test_instruction_and_t0_identity_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            _, _, meta = artificial_scene(scene)
            frozen = freeze_scene_inputs(scene, meta, meta['instruction'])
            self.assertIn('metadata.json', frozen)
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                freeze_scene_inputs(scene, meta, 'Put the ranch bottle into the pot.')
            meta['history_timestamps'][0] += .01
            (scene/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'identity changed'):
                freeze_scene_inputs(scene, meta, meta['instruction'])

    def test_cpu_recovery_completes_without_gpu_or_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            _, group, meta = artificial_scene(scene)
            freeze_scene_inputs(scene, meta, meta['instruction'])
            out, prior, expected = artificial_saved_generation(scene, group)
            with (patch('torch.cuda.is_available', side_effect=AssertionError('GPU queried')),
                  patch.dict(sys.modules, {'molmo_motion': None})):
                recover_only([scene])
            np.testing.assert_array_equal(np.load(scene/'predictions/future_3d.npy'), expected)
            receipt = json.loads((scene/'predictions/model_run.json').read_text())
            self.assertTrue(receipt['success'])
            self.assertEqual(receipt['attempted_chunks'], 1)
            self.assertEqual(receipt['recovery_additional_forward_calls'], 0)
            self.assertEqual(receipt['prediction_seconds'], 7.)
            # A complete-looking tensor with the wrong anchor cannot be reused.
            torch.save({'anchor_3d': torch.zeros(1, 3)}, out/'processor_inputs.pt')
            with self.assertRaisesRegex(ValueError, 'anchor disagrees'):
                recover_saved_group(out, group, prior)

    def test_incomplete_generation_cannot_be_silently_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            _, group, _ = artificial_scene(scene)
            out = scene/'predictions/group_00'
            out.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'no complete recoverable'):
                recover_saved_group(out, group, {'generation_started': True, 'success': False})


if __name__ == '__main__':
    unittest.main()
