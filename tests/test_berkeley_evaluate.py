"""Scientific validity checks; fixtures are artificial, never experiment outputs."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from berkeley_evaluate import (ALIGNMENT, TIMES, freeze_inputs, lift_native_depth,
                              metric_values, project, validate_timestamps, velocity)
from berkeley_infer import strict_parse


class BerkeleyEvaluationTests(unittest.TestCase):
    def test_canonical_future_export_arrays_must_match_metadata(self):
        metadata = {"t0": 47, "history_source_indices": [45, 46, 47],
                    "future_source_indices": list(range(48, 58)),
                    "history_timestamps": np.array([9., 9.2, 9.4], np.float32).tolist(),
                    "future_timestamps": np.array(9.4+TIMES, np.float32).tolist()}
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / "evaluation").mkdir()
            np.save(scene / "evaluation/timestamps.npy", np.array(metadata["future_timestamps"], np.float32))
            np.save(scene / "evaluation/source_indices.npy", np.array(metadata["future_source_indices"]))
            validate_timestamps(metadata, scene)
            np.save(scene / "evaluation/source_indices.npy", np.arange(49, 59))
            with self.assertRaisesRegex(ValueError, "source_indices.npy"):
                validate_timestamps(metadata, scene)
            np.save(scene / "evaluation/source_indices.npy", np.arange(48, 58))
            bad_times = np.asarray(metadata["future_timestamps"]) + .2
            np.save(scene / "evaluation/timestamps.npy", bad_times)
            with self.assertRaisesRegex(ValueError, "timestamps.npy"):
                validate_timestamps(metadata, scene)

    def test_physical_timestamp_alignment_without_interpolation(self):
        # A forecast with value equal to physical time must align exactly at
        # 0.2, 0.4, ..., 2.0 seconds, not at prediction steps 1,4,...,28.
        forecast = np.arange(1, 31, dtype=float) / 15
        np.testing.assert_allclose(forecast[ALIGNMENT], TIMES, atol=1e-14)
        metadata = {"t0": 47, "history_source_indices": [45, 46, 47],
                    "future_source_indices": list(range(48, 58)),
                    "history_timestamps": np.array([9., 9.2, 9.4], np.float32).tolist(),
                    "future_timestamps": np.array(9.4+TIMES, np.float32).tolist()}
        self.assertLess(validate_timestamps(metadata)["max_nominal_alignment_error_s"], 1e-5)
        metadata["future_source_indices"][0] = 47
        with self.assertRaisesRegex(ValueError, "next ten"):
            validate_timestamps(metadata)

    def test_common_visibility_mask_and_unavailable_final_error(self):
        truth = np.zeros((2, 10, 2))
        mask = np.zeros((2, 10), bool)
        mask[0, 0] = mask[1, 1] = True
        forecast = np.full_like(truth, 1e6)
        forecast[0, 0], forecast[1, 1] = [3, 4], [0, 10]
        values, _ = metric_values(forecast, truth, mask, "2D_px")
        self.assertEqual(values["ADE_2D_px"], 7.5)
        self.assertIsNone(values["FDE_2D_px"])
        self.assertEqual(values["valid_pairs"], 2)
        self.assertEqual(values["valid_point_coverage"], .1)
        baseline, _ = metric_values(np.zeros_like(truth), truth, mask, "2D_px")
        self.assertEqual(baseline["valid_pairs"], values["valid_pairs"])
        # A method cannot improve its metric by invalidating a bad forecast.
        forecast[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "common GT"):
            metric_values(forecast, truth, mask, "2D_px")

    def test_metric_projection_and_constant_velocity_units(self):
        k = np.array([[100., 0, 20], [0, 200, 30], [0, 0, 1]])
        xyz = np.array([[[.1, .2, 1.]], [[.3, .4, 1.]], [[.5, .6, 1.]]])
        speed = velocity(xyz, np.array([-.4, -.2, 0.]))
        np.testing.assert_allclose(speed, [[1, 1, 0]])
        np.testing.assert_allclose(project(xyz[-1]+speed*.2, k), [[90, 190]])
        depth = np.ones((10, 7, 7))
        depth[:, 3, 3] = 1000  # one outlier must not become physical depth
        uv = np.tile(np.array([3., 3.]), (1, 10, 1))
        recovered = lift_native_depth(depth, uv, k)
        np.testing.assert_allclose(recovered[..., 2], 1.)
        np.testing.assert_allclose(project(recovered, k), uv)

    def test_prediction_gate_rejects_zero_and_post_freeze_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / "predictions").mkdir()
            path = scene / "predictions/future_3d.npy"
            np.save(path, np.zeros((8, 30, 3)))
            (scene / "predictions/model_run.json").write_text(json.dumps(
                {"success": True, "successful_chunks": 1, "groups": [{"success": True}]}))
            with self.assertRaisesRegex(ValueError, "zero-filled"):
                freeze_inputs(scene)
            np.save(path, np.ones((8, 30, 3)))
            freeze_inputs(scene)
            self.assertTrue((scene / "evaluation/prediction_freeze.json").exists())
            np.save(path, np.full((8, 30, 3), 2.))
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_inputs(scene)

    def test_native_depth_rejects_saturation_and_requires_majority_support(self):
        depth = np.zeros((10, 7, 7))
        # Twelve positive readings do not support a full 25-pixel patch.
        depth[0, 1:6, 1:6] = np.r_[np.ones(12), np.zeros(13)].reshape(5, 5)
        depth[1, 1:6, 1:6] = np.r_[np.ones(13), np.zeros(12)].reshape(5, 5)
        # Sensor saturation cannot count as support or become measured Z.
        depth[2, 1:6, 1:6] = np.r_[np.ones(12), np.full(13, 65.535)].reshape(5, 5)
        depth[3, 1:6, 1:6] = np.r_[np.ones(13), np.full(12, 65.535)].reshape(5, 5)
        depth[4, 1:6, 1:6] = 10.
        depth[5, 1:6, 1:6] = np.inf
        uv = np.tile(np.array([3., 3.]), (1, 10, 1))
        result = lift_native_depth(depth, uv, np.eye(3))
        self.assertTrue(np.isnan(result[0, [0, 2, 4, 5, 6, 7, 8, 9]]).all())
        np.testing.assert_allclose(result[0, [1, 3], 2], 1.)

    def test_strict_parse_rejects_incomplete_duplicate_or_wrong_timestamp(self):
        frames = [str(t) + " " + " ".join(f"{p} {p*10} {-p} {t}" for p in range(1, 9))
                  for t in range(3, 33)]
        text = '<tracks coords="' + ";".join(frames) + '">motion</tracks>'
        parsed = strict_parse(text)
        self.assertEqual(parsed.shape, (8, 30, 3))
        np.testing.assert_allclose(parsed[7, -1], [.08, -.008, .032])
        for bad in (text.replace("1 10 -1", "2 10 -1", 1),
                    text.replace('coords="3 ', 'coords="2 ', 1),
                    '<tracks coords="' + ";".join(frames[:-1]) + '">motion</tracks>'):
            with self.assertRaises(ValueError):
                strict_parse(bad)


if __name__ == "__main__":
    unittest.main()
