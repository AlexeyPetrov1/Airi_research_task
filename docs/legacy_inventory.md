# Legacy inventory

All original scripts remain in place. No original run is overwritten.
The JSON companion records imports, functions, constants and file literals; literals are evidence of inputs/outputs, not an inferred execution trace.
The CSV companion inventories every saved research artifact by path, extension and size.

## `scripts/amend_dobbe_vipe_actions.py`

One pre-reconstruction correction: source-faithful action text.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: .
Imports/reused functions: `from dobbe_vipe_v1 import RUN, SCENES, read, write, sha`
Constants: 
Input/output file references: `protocol.json`; `protocol_action_correction.json`; `tracks.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/analyze_fmb_ablation_prompt.py`

Count changed millimetre-quantized history components in FMB ablations.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import numpy as np`; `from analyze_fmb_prompt_sensitivity import quantized`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, write_json`
Constants: 
Input/output file references: `comparison.csv`; `comparison.json`; `experiment_config.json`; `history_3d.npy`; `history_sensor_3d.npy`; `manifest.json`; `prediction_15hz.npy`; `runs/fmb_ablation_prompt_analysis_v1`; `src/molmo_motion/processor.py::_format_history_tracks, history minus final-frame point-0 anchor, components rounded to integer millimetres`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/analyze_fmb_pnp_annotation_noise.py`

Estimate current-window planar CAD/PnP sensitivity to 2 px RGB annotation noise.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `camera_points`, `distribution`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import os`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `RUN=Path(os.environ.get('FMB_QUANT_RUN', Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`; `NOISE_SIGMA_PX=2.0`; `DRAWS=300`
Input/output file references: `Estimate current-window planar CAD/PnP sensitivity to 2 px RGB annotation noise.`; `k_nominal.json`; `pnp_annotation_noise_sensitivity.json`; `pnp_correspondences.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/analyze_fmb_prompt_sensitivity.py`

Inspect how sub-mm K changes alter quantized MolmoMotion history tokens.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `quantized`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`
Input/output file references: `history_sensor_3d.npy`; `history_sensor_3d_K_focal_0925.npy`; `history_sensor_3d_K_simple.npy`; `prediction_15hz.npy`; `prompt_quantization_sensitivity.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `src/molmo_motion/processor.py::_format_history_tracks`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/analyze_fmb_temporal_forecast.py`

Detect temporally constant decoded 3D forecasts without changing the metric.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`
Input/output file references: `experiment_config.json`; `gt_2d.npy`; `points_2d_at_t0.npy`; `prediction_15hz.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `temporal_forecast_diagnostic.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/annotate_fmb_future_2d.py`

Annotate future peg surface coordinates from RGB only, blind to predictions.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `check_frozen`, `direct_registration`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import os`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import SOURCE as DEFAULT_SOURCE`; `from probe_fmb_dense_flow import follow as dense_follow`; `from probe_fmb_mask_ecc import follow as ecc_follow, segment`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`; `SOURCE=Path(os.environ.get('FMB_QUANT_SOURCE', DEFAULT_SOURCE))`; `T0=int(os.environ.get('FMB_QUANT_T0', '126'))`; `FUTURE=tuple(range(T0 + 1, T0 + 21))`; `REPEAT_FRAMES=tuple((T0 + offset for offset in (1, 6, 11, 16, 20)))`; `REPEAT_POINTS=(0, 7)`
Input/output file references: `/160`; `annotation_repeatability.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `gt_tracking_diagnostics.json`; `input_freeze.json`; `obs/side_1`; `points_2d_at_t0.npy`; `reason_mask.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/assess_fmb_second_depth.py`

Measure raw depth validity inside the yellow peg's visible interior.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_L_L_4_vertical_n_0.npy'`; `OUT=ROOT / 'runs/fmb_second_scene'`
Input/output file references: `candidate_depth_roi.json`; `data/fmb/single_object_manipulation_dataset/1_L_L_4_vertical_n_0.npy`; `obs/`; `runs/fmb_second_scene`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_dobbe_3651_recovered.py`

Audit the recovered HoNY source record against LeRobot episode 3651.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `video_info`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import pyarrow.parquet as pq`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data' / 'dobbe_oxe'`; `RAW=DATA / 'target_raw'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'dobbe_3651_recovered_audit.json'`; `FRAMES=(0, 40, 100, 200, 241)`
Input/output file references: `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30`; `Raw RGB/depth/label frame counts differ`; `compressed_video_h264.mp4`; `dobbe_3651_recovered_audit.json`; `labels.json`; `lerobot_episode_003651.mp4`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_dobbe_approx.py`

Audit the exploratory F2-NeRF transfer, frozen MolmoMotion run, and sparse GT.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import numpy as np`; `from validate_dobbe_geometry import CAMERA_TO_LABEL`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `STUDY=ROOT / 'runs/dobbe_rgbd_study'`; `RUN=STUDY / 'approx_history'`
Input/output file references: `audit.json`; `decision.json`; `f2nerf_second/cams_meta.npy`; `f2nerf_second/receipt.json`; `future_3d_gt_candidate.npy`; `future_evaluation.json`; `future_gt_valid.npy`; `future_projection_diagnostic.json`; `input_freeze.json`; `manifest.json`; `model_run.json`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `points_3d_history_world_candidate.npy`; `prediction_15hz.npy`; `protocol.json`; `runs/dobbe_rgbd_study`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_dobbe_colmap_clean.py`

Verify frozen controls against actual SQLite databases and sparse models.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import json`; `import sqlite3`; `import cv2`; `import numpy as np`; `import pycolmap`; `from dobbe_colmap_clean import OUT, ROOT, components, digest, plain, write`; `from summarize_dobbe_colmap_clean import RUNS, read`
Constants: 
Input/output file references: `.json`; `audit.json`; `causal/oracle split`; `causal_geometry_validation.json`; `clean_decision.json`; `clean_protocol.json`; `config.json`; `images/main/`; `mask_policy_frozen.json`; `masks/main/`; `match_pairs.csv`; `points_3d_history_candidate.npy`; `runs/dobbe_rgbd_study/approx_history/input_freeze.json`; `runs/dobbe_rgbd_study/approx_history/points_3d_history_candidate.npy`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_dobbe_rgbd_sample.py`

Audit one HoNY RGB-D delivery record; do not infer missing camera calibration.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs' / 'plex_dobbe_preflight'`; `SAMPLE=RUN / 'dobbe_rgbd_sample'`
Input/output file references: `Drawer_Closing/Home10/Env2/2022-12-22--00-09-08`; `compressed_video_h264.mp4`; `dobbe_rgbd_sample_audit.json`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_dobbe_rgbd_study.py`

Check saved evidence, causal indices, receipts and report links.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `load`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import json`; `import re`; `import sqlite3`; `from pathlib import Path`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_rgbd_study'`
Input/output file references: `/`; `_static_correspondences.json`; `_static_geometry.json`; `audit.json`; `colmap_*/summary.json`; `colmap_runs.csv`; `colmap_simple_pinhole_*_fixedpp_rectified/summary.json`; `correspondence_robustness.json`; `data/dobbe_oxe/second_raw`; `data/dobbe_oxe/target_raw`; `decision.json`; `future_evaluation.json`; `http://`; `https://`; `image_rotation_diagnostic.json`; `model_run.json`; `points_3d_history_candidate.npy`; `protocol.json`; `report/README.md`; `report/dobbe_approx_colmap_f2nerf_molmo.md`; `report/dobbe_rgbd_colmap_study.md`; `runs/dobbe_rgbd_study`; `runs/plex_dobbe_preflight/dobbe_3651_rgbd_receipt.json`; `second_download_receipt.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_dobbe_vipe_static.py`

Independent causal SIFT checks separate geometry from background-tracker errors.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `audit`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import cv2`; `import numpy as np`; `from dobbe_vipe_v1 import scene_dir, RUN, read, write`; `from dobbe_vipe_geometry import lift, project, stats`
Constants: 
Input/output file references: `.npz`; `SIFT mutual Lowe .7; fundamental RANSAC 1px; t0 background mask; depth/c2w not used for match selection`; `geometry_all.npz`; `independent_static_sift.json`; `protocol.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/audit_fmb_ablation_inputs.py`

Check the frozen input invariants for the paired FMB ablations.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from run_fmb_ablation_suite import RUNS, SUITE, VARIANTS, digest, verify_frozen, write_json`
Constants: 
Input/output file references: `experiment_config.json`; `history_3d.npy`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_audit.json`; `k_nominal.json`; `manifest.json`; `moge2_history_v1/manifest.json`; `points_2d_at_t0.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_fmb_episode_5201.py`

Audit the ShareRobot FMB episode against the selected public RLDS record.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `phash`, `ssim`, `camera_check`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import io`; `import json`; `import mmap`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image, ImageDraw`; `from probe_fmb_rlds_record import find_feature_values`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`; `OLD=ROOT / 'runs' / 'sharerobot_transfer_berkeley_ur5_episode_26'`; `REV='3266d92902b038ce40e7fb8aac5bfd9287eb3e45'`; `GCS_BASE='https://storage.googleapis.com/gresearch/robotics/fmb/0.0.1'`; `ZIP_URL='https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip'`; `SOURCE_NAME='1_M_L_3_vertical_n_2.npy'`
Input/output file references: `/`; `/affordance/affordance.json`; `/dataset_info.json`; `/features.json`; `/fmb-train.tfrecord-01228-of-02017`; `/planning/images`; `/planning/jsons`; `/trajectory/trajectory.json`; `0/240`; `1_M_L_3_vertical_n_2.npy`; `FMB collection environment defaults to 10 Hz; RealSense capture defaults to 15 FPS. Neither records actual timestamps in this NPY/RLDS episode.`; `FMB robot/camera code`; `JPEG decode only; no resize/crop/color transform`; `No eight persistent distinguishable physical points and independent 30-step future validity mask have been selected; 2D/3D metrics are unavailable.`; `Provide or extract one Planning RGB such as frame_28 for a third visual pair, plus side_1's recorded depth_units (meters/raw unit) and active aligned-color K/256x256 depth resize metadata.`; `camera_motion_check.json`; `depth_calibration_check.json`; `episode_metadata/file_path`; `floor(148 * frame_index / 30)`; `https://functional-manipulation-benchmark.github.io/dataset/index.html`; `https://functional-manipulation-benchmark.github.io/static/files/side_1`; `https://github.com/FlagOpen/ShareRobot/issues/4#issuecomment-3094378733`; `https://github.com/rail-berkeley/fmb/tree/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra`; `https://github.com/realsenseai/librealsense/wiki/Projection-in-RealSense-SDK-2.0`; `https://huggingface.co/datasets/BAAI/ShareRobot/blob/`; `https://huggingface.co/datasets/BAAI/ShareRobot/tree/`; `https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip`; `https://storage.googleapis.com/gresearch/robotics/fmb/0.0.1`; `insert_only_1_M_L_3_vertical_n_2.npy`; `lerobot/fmb episode_5201`; `metrics.json`; `obs/side_1`; `obs/side_1_depth`; `planning_rows.json`; `preflight.json`; `raw BGR and depth schemas, side_1 intrinsics; no depth scale/timestamps`; `share_to_source_mapping.csv`; `side_1 RGB/depth and episode_metadata/file_path`; `steps/observation/image_side_1`; `steps/observation/image_side_1_depth`; `temporal_alignment.json`; `x=20..234, y=83..244; upper object/gripper excluded`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_fmb_gripper_calibration.py`

Audit whether published FMB observations identify gripper-based camera K.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `from scipy.spatial.transform import Rotation`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT / 'runs' / 'sharerobot_fmb_episode_5201' / 'fmb_reference_code' / 'robot_infra' / 'franka_server.py'`; `DATA=Path('F:/AIRI_task/data/fmb/single_object_manipulation_dataset')`; `OUT=ROOT / 'runs' / 'fmb_effective_k_256_calibration' / 'gripper_calibration_audit.json'`
Input/output file references: `1_M_L_3_*.npy`; `F:/AIRI_task/data/fmb/single_object_manipulation_dataset`; `gripper_calibration_audit.json`; `obs/gripper_pose`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_fmb_quantitative_2d.py`

Verify frozen inputs, timeline, GT, and model artifacts for the FMB run.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import csv`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`; `SOURCE_OVERRIDE=os.environ.get('FMB_QUANT_SOURCE')`
Input/output file references: `annotation_repeatability.json`; `cad_dimension_check.json`; `cad_silhouette_prediction_comparison.json`; `calibration_sensitivity.csv`; `current_state.json`; `error_by_time.csv`; `evaluation_time_10hz_s.npy`; `experiment_config.json`; `geometry_prediction_comparison.json`; `gt_2d.npy`; `history_depth_probe.json`; `history_pnp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_freeze.json`; `k_nominal.json`; `k_sensitivity_variants.json`; `leakage_and_artifact_audit.json`; `metrics.csv`; `metrics.json`; `model_run.json`; `nominal_repeatability.json`; `parser_validation.json`; `pnp_annotation_noise_sensitivity.json`; `prediction_10hz.npy`; `prediction_15hz.npy`; `prediction_time_15hz_s.npy`; `preflight.json`; `reason_mask.npy`; `rigidity_check.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `sensor_vs_pnp.json`; `temporal_forecast_diagnostic.json`; `validity_mask.npy`; `visual_review.json`; `window_selection.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_fmb_second_scene_export.py`

Independently verify the complete FMB second-scene export.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data/fmb_second_scene'`; `RUN=ROOT / 'runs/fmb_second_scene'`; `REF=ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_2.npy'`; `CAMERAS=('side_1', 'side_2', 'wrist_1', 'wrist_2')`; `STATE_MAP={'obs/tcp_pose': 'tcp_pose', 'obs/tcp_vel': 'tcp_vel', 'obs/tcp_force': 'tcp_force', 'obs/tcp_torque': 'tcp_torque', 'obs/q': 'q', 'obs/dq': 'dq', 'obs/jacobian': 'jacobian', 'obs/gripper_pose': 'gripper_pose', 'actions': 'action', 'primitive': 'prim`
Input/output file references: `.mp4`; `.npy`; `/`; `Exact RGB/depth pixel registration is separate from export fidelity and remains unresolved`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_2.npy`; `data/fmb_second_scene`; `data_audit.json`; `metadata.json`; `obs/`; `obs/dq`; `obs/gripper_pose`; `obs/jacobian`; `obs/q`; `obs/tcp_force`; `obs/tcp_pose`; `obs/tcp_torque`; `obs/tcp_vel`; `raw/source_demo.npy`; `robot_state.npz`; `runs/fmb_second_scene`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_fmb_study.py`

Final integrity audit for the frozen two-episode FMB study.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `check_variant`, `audit_media`, `audit_report_links`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import re`; `import cv2`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import measure`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, digest, verify_frozen, write_json`
Constants: 
Input/output file references: `evaluation.json`; `forecast_comparison.mp4`; `forecast_vs_observed.mp4`; `geometry_bundle_v1/history_clouds_common_axes.png`; `geometry_bundle_v1/manifest.json`; `geometry_methods.mp4`; `gt_2d.npy`; `history_3d.npy`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `http://`; `https://`; `input_and_geometry.mp4`; `input_audit.json`; `manifest.json`; `media_manifest.json`; `model_run.json`; `models/moge-2-vitl/model.pt`; `moge2_history_v1/depth_comparison.png`; `moge2_history_v1/manifest.json`; `moge2_history_v1/moge2_history_maps.npz`; `parser_validation.json`; `points_2d_at_t0.npy`; `prediction_15hz.npy`; `prediction_15hz_model_order.npy`; `prediction_2d.npy`; `report/fmb_geometry_forecast_study.md`; `rigid_correction_v1/evaluation.json`; `runs/fmb_ablation_conclusions_v1/paired_effects.csv`; `runs/fmb_ablation_suite_v1_summary.csv`; `runs/fmb_depth_k_factorial_v1/comparison.csv`; `runs/fmb_geometry_forecast_comparison_v1/comparison.csv`; `runs/fmb_gt_sensitivity_v1/comparison.csv`; `runs/fmb_study_audit.json`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/audit_plex_cereal_states.py`

Recover Cereal free-joint poses directly from recorded PLEX qpos states.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `joints_in_xml_order`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import xml.etree.ElementTree as ET`; `from pathlib import Path`; `import h5py`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `HDF5=ROOT / 'data' / 'plex' / 'PickPlaceCereal_demo_act_norm.hdf5'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'plex_cereal_state_geometry.json'`; `JOINT_DIMS={'free': (7, 6), 'ball': (4, 3), 'hinge': (1, 1), 'slide': (1, 1)}`
Input/output file references: `plex_cereal_state_geometry.json`
Related saved experiments (dataset-level): 

## `scripts/audit_sharerobot_preflight.py`

Repeat the read-only ShareRobot transfer preflight from saved evidence frames.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `sha256`, `image_info`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `from urllib.parse import quote`; `import requests`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_transfer_berkeley_ur5_episode_26'`; `SHAREROBOT_REVISION='3266d92902b038ce40e7fb8aac5bfd9287eb3e45'`; `FMB_LEROBOT_REVISION='76c86f111d328229773288bf60d5b5b50e047e8d'`; `TRAJECTORY_URL=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{SHAREROBOT_REVISION}/trajectory/trajectory.json'`; `FMB_ZIP_URL='https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip'`; `FMB_LEROBOT_INFO=f'https://huggingface.co/datasets/lerobot/fmb/resolve/{FMB_LEROBOT_REVISION}/meta/info.json'`
Input/output file references: `/`; `/43_berkeley_autolab_ur5#episode_26/`; `/57_fmb#episode_5201/`; `/meta/info.json`; `/trajectory/images/`; `/trajectory/trajectory.json`; `BLOCKED/PARTIAL`; `https://functional-manipulation-benchmark.github.io/static/files/side_1`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `https://huggingface.co/datasets/lerobot/fmb/resolve/`; `https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip`; `preflight.json`
Related saved experiments (dataset-level): 

## `scripts/berkeley_arc_expansion.py`

Shared-scale and camera-up controls on frozen, genuine CASE-AUGE forecasts.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: expand.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha`, `snapshot`, `expand`, `scene_data`, `extent_diagnostics`, `annotated`, `render_families`, `grid_plot`, `gallery`, `generate`, `verify`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `from datetime import datetime, timezone`; `import hashlib`; `import json`; `from pathlib import Path`; `import subprocess`; `import cv2`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_temporal_diagnostics import BASE, OUT, load_scene, scale_displacement, write`; `from berkeley_baseline_comparison import controls, scores`; `from berkeley_evaluate import ALIGNMENT, TIMES, project, points_on, trails_on, label, save_rgb, write_video`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=OUT / 'CASE-AUGE'`; `DEST=ROOT / 'runs/berkeley_ur5_arc_expansion_v1'`; `VARIANTS={'reference_0333': (1 / 3, 0.0), 'scale_036': (0.36, 0.0), 'scale_038': (0.38, 0.0), 'scale_040': (0.4, 0.0), 'scale_042': (0.42, 0.0), 'lift_005': (1 / 3, 0.005), 'lift_010': (1 / 3, 0.01), 'scale_036_lift005': (0.36, 0.005), 'scale_038_lift005': (0`; `FAMILIES={'scale': ['reference_0333', 'scale_036', 'scale_040'], 'height': ['scale_036', 'scale_036_lift005', 'scale_036_lift010'], 'controls': ['reference_0333', 'scale_036_lift005', 'CV_3D']}`
Input/output file references: `*.mp4`; `.mp4`; `/future_3d.npy`; `/point_ids.npy`; `</`; `<\/`; `All 24 immutable point IDs
Green: future tracker reference
Blue/orange/pink: P8 groups
Same camera and parameters in both scenes
No future fitting of parameters
Full extent; no curve clipping`; `Cup root predictions/point_ids.npy has pilot-only P8 IDs; all 24 observed IDs independently validated against three group inputs and outputs`; `K.npy`; `aligned_3d.npz`; `comparison.json`; `full_30_steps.npz`; `generation_receipt.json`; `geometry/K_median.npy`; `groups/group_`; `metadata.json`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `p0.npy`; `p0_i + alpha * (raw_i(t) - p0_i) - [0, lift_m*smoothstep(t/2), 0]`; `phase_schedule/protocol.json`; `point_ids.npy`; `predictions/future_3d.npy`; `predictions/group_`; `predictions/model_run.json`; `projected_2d.npz`; `protocol.json`; `render_receipts.json`; `results.json`; `runs/berkeley_ur5_arc_expansion_v1`; `scripts/berkeley_arc_gallery.html`; `verification.json`; `visual_review.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_arc_phase.py`

Follow-up: expand late motion without amplifying the first second.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: temporal_expand.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `temporal_expand`, `main`.
Imports/reused functions: `from datetime import datetime, timezone`; `import json`; `from pathlib import Path`; `import numpy as np`; `from berkeley_arc_expansion import ROOT, DEST, SOURCE, BASE, OUT, scene_data, snapshot, sha, render_families, extent_diagnostics`; `from berkeley_temporal_diagnostics import write`; `from berkeley_evaluate import project, ALIGNMENT`; `from berkeley_baseline_comparison import scores`
Constants: `PHASE=DEST / 'phase_schedule'`; `VARIANTS={'ramp_036_lift005': (0.36, 0.005, 0.0), 'late_036_lift005': (0.36, 0.005, 1.0), 'late_038_lift005': (0.38, 0.005, 1.0), 'late_036_lift010': (0.36, 0.01, 1.0)}`
Input/output file references: `aligned_3d.npz`; `alpha(t)=1/3+(alpha_end-1/3)*smoothstep(clamp((t-onset_s)/(2-onset_s),0,1))`; `full_30_steps.npz`; `gallery_arrays.json`; `projected_2d.npz`; `protocol.json`; `results.json`; `verification.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_arc_report.py`

Save the completed visual decision, combined viewer and selected full arrays.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from datetime import datetime, timezone`; `import json`; `from pathlib import Path`; `import numpy as np`; `from berkeley_arc_expansion import ROOT, DEST, scene_data, annotated, sha`; `from berkeley_evaluate import save_rgb, project, ALIGNMENT`; `from berkeley_temporal_diagnostics import write`
Constants: `SELECTED='late_036_lift005'`
Input/output file references: `../../berkeley_ur5_improvement_v1/CASE-AUGE/`; `/predictions/future_3d.npy`; `</`; `<\/`; `All 24 IDs and all ten real future times for primary constant/late choices and CV; not every generated candidate movie watched in full`; `K.npy`; `Late placing bend to the right is already premature. Expansion raises/extends it but cannot restore the observed vertical lift.`; `Shared late expansion modestly improves global endpoint error, while average error remains slightly worse than x1/3.`; `Use the delayed small expansion to demonstrate the requested shape change; retain original x1/3 and CV as accuracy controls.`; `[Полный понятный разбор процесса и выводов](research_story.md) · [Визуальная оценка](visual_review.json) · [Выбранные массивы и параметры](selected/metadata.json).`; `[Предыдущее исследование и источник настоящих модельных прогнозов](../berkeley_ur5_improvement_v1/research_story.md).`; `alpha(t)=1/3+(0.36-1/3)*smoothstep(clamp(t-1,0,1)); Y-=0.005*same_smoothstep`; `both all24_candidate_grid.png; all 11 constant/lift candidates`; `bottle/controls_group_00..02`; `bottle/height_group_00_contact.png`; `cup/controls_group_00..02`; `cup/height_group_00_contact.png`; `direction and lift/place timing`; `full_30_steps.npz`; `phase_schedule/bottle/timing_group_00..02`; `phase_schedule/cup/timing_group_00..02`; `phase_schedule/results.json`; `point_ids.npy`; `postprocessed_future_3d.npy`; `projected_2d.npy`; `projected_2d.npz`; `python -m pytest tests/test_berkeley_arc_expansion.py -q`; `python scripts/berkeley_arc_expansion.py --stage generate --output-dir runs/berkeley_arc_repeat`; `python scripts/berkeley_arc_expansion.py --stage verify`; `python scripts/berkeley_arc_phase.py --parent-run runs/berkeley_arc_repeat`; `python scripts/berkeley_arc_verify.py`; `results.json`; `scripts/berkeley_arc_gallery.html`; `selected/metadata.json`; `visual_review.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_arc_verify.py`

Verify the full amplitude/phase experiment independently from its renderer.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from datetime import datetime, timezone`; `import json`; `import re`; `import cv2`; `import numpy as np`; `from berkeley_arc_expansion import ROOT, DEST, BASE, OUT, sha, snapshot, scene_data`; `from berkeley_evaluate import ALIGNMENT, TIMES, project`; `from berkeley_baseline_comparison import scores`; `from berkeley_temporal_diagnostics import write`
Constants: 
Input/output file references: `*.mp4`; `K.npy`; `Verify the full amplitude/phase experiment independently from its renderer.`; `final_verification.json`; `full_30_steps.npz`; `phase_schedule/executed_experiment.py`; `phase_schedule/protocol.json`; `phase_schedule/results.json`; `point_ids.npy`; `postprocessed_future_3d.npy`; `projected_2d.npy`; `projected_2d.npz`; `protocol.json`; `results.json`; `selected/metadata.json`; `visual_review.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_baseline_comparison.py`

Compare all saved Berkeley forecasts and causal baselines without ML calls.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `scores`, `controls`, `overview`, `main`.
Imports/reused functions: `import json`; `from pathlib import Path`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_temporal_diagnostics import BASE, OUT, load_scene, scale_displacement, fingerprint, render`; `from berkeley_evaluate import project, velocity, ALIGNMENT, TIMES`
Constants: `DEST=OUT / 'baseline_comparison'`; `THRESHOLDS=(5, 10, 20, 40)`
Input/output file references: `/`; `/all24_baselines.png).`; `/baselines_group_`; `: all 24 frozen IDs; green=reference; blue/orange/pink=P8 groups`; `baseline_comparison/`; `comparison.json`; `geometry/K_median.npy`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `predictions/future_3d.npy`; `results.json`; `| Сцена | Метод | ADE px ↓ | FDE px ↓ | PWT@10px ↑ | PWT@20px ↑ | Длина пути / reference |`; `Новые contact sheets и MP4 сопоставляют static / CV в той же геометрии / выбранную модель ×1/3 на настоящих RGB; зелёный — tracker reference. Все три группы сохранены.`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_contract_audit.py`

Source-grounded audit of ordinal model time and prepared-case provenance.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import hashlib, json`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`
Input/output file references: `Cadence mismatch remains a hypothesis. Fixed 1/3 scaling, timewarp, oracle scale and H1 are separate controls; none alone establishes the cause.`; `data/checkpoints/MolmoMotion-4B-H3-F30/config.yaml`; `geometry/case_provenance.json`; `predictions/input_freeze.json`; `runs/berkeley_ur5_improvement_v1`; `sources/model_time_contract.json`; `src/molmo_motion/data/trajectory_3d_dataset.py`; `src/molmo_motion/processor.py`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_audit.py`

Data-only native RGB-D and selection audit, independent of model outputs.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import json, hashlib`; `import numpy as np`; `from PIL import Image`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import requests`; `from berkeley_data_extract_native import ROOT, SCENES`
Constants: `RUN=Path(__file__).resolve().parents[1] / 'runs/berkeley_ur5_molmomotion'`
Input/output file references: `.npy`; `candidates_all.json`; `https://github.com/yunliangchen/ur5bc/blob/a4285610da52cb30215951c7ac37454fcfae01c8/ur5/robot_env.py`; `observed/mask.png`; `runs/berkeley_ur5_molmomotion`; `selection_and_depth_audit.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_capture_source.py`

Save original Berkeley capture code establishing depth alignment and units.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: .
Imports/reused functions: `from pathlib import Path`; `import requests, json, hashlib`
Constants: `ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes/capture_source')`
Input/output file references: `/`; `/commits/main`; `/git/trees/`; `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes/capture_source`; `https://api.github.com/repos/yunliangchen/ur5bc`; `https://github.com/yunliangchen/ur5bc`; `https://raw.githubusercontent.com/yunliangchen/ur5bc/`; `receipt.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_discover.py`

Bounded, reproducible Berkeley UR5 candidate retrieval (no depth-video use).

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `get`, `main`.
Imports/reused functions: `from pathlib import Path`; `import json, hashlib`; `from concurrent.futures import ThreadPoolExecutor`; `import requests`
Constants: `DEST=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')`; `REV='6306aa91c00b9b0de28831b7fceaa42b8096bb4f'`; `BASE=f'https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/resolve/{REV}/'`
Input/output file references: `.mp4`; `/`; `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`; `candidates_all.json`; `data/chunk-000/episode_`; `https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/resolve/`; `lerobot/berkeley_autolab_ur5`; `meta/episodes.jsonl`; `meta/info.json`; `meta/tasks.jsonl`; `retrieval_receipt.json`; `videos/chunk-000/observation.images.image/episode_`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_export_run.py`

Export future native depth after completed predictions, without overwriting artifacts.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, json, hashlib, subprocess, sys`; `import numpy as np`; `from berkeley_data_extract_native import ROOT, SOURCE, SHARD, SCENES`
Constants: `REPO=Path(__file__).resolve().parents[1]`
Input/output file references: `.npy`; `/`; `All episode robot pose/gripper states and translation actions match LeRobot exactly; observed RGB RMSE 2.95-4.12/255 due to AV1 compression.`; `native_depth.npy`; `native_depth_future.npy`; `native_depth_metadata.json`; `predictions/model_run.json`; `runs/berkeley_ur5_molmomotion`; `scripts/berkeley_data_extract_native.py`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_extract_native.py`

Extract verified Berkeley native metric depth; future decoding is a separate command.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, hashlib, io, json, struct`; `import numpy as np`; `import pandas as pd`; `import av`; `from PIL import Image`; `from probe_fmb_rlds_record import find_feature_values`
Constants: `ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')`; `SHARD='berkeley_autolab_ur5-train.tfrecord-00004-of-00412'`; `SOURCE='https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/d2590280290484e2a7eb53a91bba32ac2ff669a0/' + SHARD`; `SCENES=[('bottle', 9, 47, 40, 0), ('cup', 10, 63, 56, 1)]`
Input/output file references: `.mp4`; `.npy`; `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`; `_receipt.json`; `data/chunk-000/episode_`; `https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/features/image_feature.py`; `https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/d2590280290484e2a7eb53a91bba32ac2ff669a0/`; `predictions/model_run.json`; `receipt.json`; `runs/berkeley_ur5_molmomotion`; `source_frame_indices.npy`; `source_timestamps.npy`; `steps/action/world_vector`; `steps/observation/image`; `steps/observation/image_with_depth`; `steps/observation/natural_language_instruction`; `steps/observation/robot_state`; `videos/chunk-000/observation.images.image/episode_`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_inspect.py`

berkeley_data_inspect.py

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import json`; `import numpy as np`; `import pandas as pd`; `import av`; `from PIL import Image, ImageDraw`; `import requests`
Constants: `ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')`
Input/output file references: `.mp4`; `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`; `candidate_gripper_events.json`; `data/chunk-000`; `dataset_info.json`; `features.json`; `https://huggingface.co/api/datasets/lerobot-raw/berkeley_autolab_ur5_raw/tree/main`; `https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/main/`; `raw_tree.json`; `videos/chunk-000/observation.images.image`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_native.py`

Bounded TFRecord retrieval and exact RGB/state identity audit.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sheets`, `main`.
Imports/reused functions: `from pathlib import Path`; `import json, struct, hashlib, time, io, os`; `import requests`; `import numpy as np`; `import pandas as pd`; `import av`; `from PIL import Image, ImageDraw`
Constants: `ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')`
Input/output file references: `.mp4`; `/image`; `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`; `Bounded TFRecord retrieval and exact RGB/state identity audit.`; `_frames/frame_`; `_summary.json`; `https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/d2590280290484e2a7eb53a91bba32ac2ff669a0/`; `raw_tree.json`; `videos/chunk-000/observation.images.image`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_data_states.py`

berkeley_data_states.py

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: .
Imports/reused functions: `from pathlib import Path`; `import json`; `import pandas as pd`; `import numpy as np`
Constants: `ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')`
Input/output file references: `/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes`; `data/chunk-000/episode_`; `detailed_states.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_evaluate.py`

Evaluate frozen Berkeley predictions with independent author AllTracker tracks.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: project, velocity, metric_values.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `read_json`, `write_json`, `sha256`, `find_file`, `load_npy`, `freeze_inputs`, `scene_inputs`, `track_future`, `project`, `velocity`, `validate_timestamps`, `metric_values`, `lift_native_depth`, `label`, `points_on`, `trails_on`, `save_rgb`, `write_video`, `visualize`, `evaluate`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import csv`; `import hashlib`; `import json`; `import sys`; `import time`; `from datetime import datetime, timezone`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `ALIGNMENT=np.arange(2, 30, 3, dtype=np.int64)`; `TIMES=np.arange(1, 11, dtype=np.float64) / 5`
Input/output file references: `.cache/torch/hub/checkpoints/alltracker.pth`; `.json`; `.npy`; `.npz`; `.pt`; `/`; `5x5 median of finite native metric depth with 0<Z<10 m; >=max(3,ceil(patch_area/2)) valid samples (13/25 for full patch)`; `Berkeley history uses real 5 FPS frames; MolmoMotion training/inference forecast is 15 FPS.`; `Exact future export timestamps/indices are required: `; `Expected [8/16/24,30,3], received `; `Frozen predictions/causal inputs changed since evaluation began`; `GT tracks may drift during gripper/object/container occlusion; coverage and video require joint interpretation.`; `Incomplete model point/time identifiers`; `RGB/prediction dimensions disagree`; `Source metadata and saved timestamps/indices disagree: `; `data_generation/third_party/alltracker`; `evaluation/alltracker_execution.json`; `evaluation/error_by_horizon.csv`; `evaluation/evaluation_results.npz`; `evaluation/evaluation_tracks_2d.npz`; `evaluation/future_rgb.npy`; `evaluation/future_source_indices.npy`; `evaluation/future_timestamps.npy`; `evaluation/ground_truth_3d_est.npz`; `evaluation/native_depth_future.npy`; `evaluation/native_depth_metadata.json`; `evaluation/prediction_freeze.json`; `evaluation/rgb.npy`; `evaluation/source_indices.npy`; `evaluation/timestamps.npy`; `geometry/K_median.npy`; `geometry/K_per_frame.npy`; `geometry/camera_motion_audit.json`; `geometry/depth_observed.npy`; `geometry/depth_source.json`; `geometry/filter_metadata.json`; `geometry/intrinsics_stability.json`; `intrinsics_stability.json`; `metadata.json`; `metrics.json`; `observed/grounding_metadata.json`; `observed/history_rgb.npy`; `observed/history_source_indices.npy`; `observed/history_timestamps.npy`; `observed/mask.png`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/query_points_100.npy`; `observed/selected_point_ids.npy`; `pred_vs_gt_2d.mp4`; `predictions/future_3d.npy`; `predictions/inference.json`; `predictions/model_run.json`; `predictions/prediction_15hz.npy`; `s | GT green / prediction pink X`; `side_by_side.mp4`; `visualization_manifest.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_finalize_report.py`

Build final readable indexes and secondary tables from saved real evidence.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `read`, `write`, `main`.
Imports/reused functions: `import json`; `from berkeley_temporal_diagnostics import OUT`
Constants: 
Input/output file references: `/`; `/comparison.json`; `/geometry_comparison.mp4), [PNG](`; `/geometry_contact_sheet.png) | [MP4](`; `/geometry_group_01_comparison.mp4), [PNG](`; `/geometry_group_01_contact_sheet.png) | [MP4](`; `/geometry_group_02_comparison.mp4), [PNG](`; `/geometry_group_02_contact_sheet.png) |`; `BASELINE-U/`; `CASE-H1/cup/comparison.json`; `all24_visual_manifest.json`; `final_verification.json`; `fixed_summary.json`; `oracle_summary.json`; `sources/sharerobot_archive_headers.json`; `sources/sharerobot_pixel_mapping.json`; `summary.json`; `tests_receipt.json`; `visual_review_final.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_flagged_point_review.py`

Display the retained strict-geometry outlier without changing frozen inputs.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `import json`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_temporal_diagnostics import OUT, load_scene, scale_displacement, fingerprint`; `from berkeley_evaluate import project, ALIGNMENT, TIMES`
Constants: 
Input/output file references: `CASE-NOK-STRICT/bottle`; `bottle/strict_retained_id30.json`; `bottle/strict_retained_id30.png`; `geometry/K_median.npy`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `observed/points_3d_history.npy`; `own-point x1/3`; `predictions/future_3d.npy`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_geometry_cases.py`

Observed-only controlled geometry cases with frozen Berkeley point identities.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha`, `write`, `prepare`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `from pathlib import Path`; `import shutil`; `import subprocess`; `import sys`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `BASE=ROOT / 'runs/berkeley_ur5_molmomotion'`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`
Input/output file references: `../../../../berkeley_ur5_molmomotion`; `Independent MoGe rays; all raw XYZ and author trust/smoothing recomputed from native depth and unchanged 2D tracks. Fixed IDs retained for comparability.`; `Measured/smoothed sensor Z changed`; `f=H/(2*tan(fovy/2)), cx=W/2, cy=H/2`; `geometry/K_median.npy`; `geometry/K_per_frame.npy`; `geometry/case_provenance.json`; `geometry/depth_source.json`; `geometry/filter_diagnostics.npz`; `geometry/filter_metadata.json`; `geometry/moge_observed_outputs.npz`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `geometry/strict_geometry_audit.json`; `groups/group_`; `metadata.json`; `models/moge-2-vitl/model.pt`; `observed/native_depth.npy`; `observed/observed_tracks_2d.npz`; `observed/points_3d_history.npy`; `observed/rgb.npy`; `observed/selected_point_ids.npy`; `points_3d_filtered.npy`; `points_3d_history.npy`; `points_3d_history.pt`; `points_3d_raw.npy`; `runs/berkeley_ur5_improvement_v1`; `runs/berkeley_ur5_molmomotion`; `sources/auge_config.py`; `third_party/MoGe`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_geometry_compare.py`

Visual-first comparison of actual geometry-case predictions and frozen baseline.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `import argparse`; `import json`; `import numpy as np`; `from berkeley_temporal_diagnostics import BASE, OUT, load_scene, render, evaluate, scale_displacement, write, fingerprint, velocity, TIMES`
Constants: 
Input/output file references: `/`; `AugE/learned K are calibration hypotheses, not measured ground truth`; `comparison.json`; `geometry/K_median.npy`; `observed/points_3d_history.npy`; `predictions/future_3d.npy`; `predictions/model_run.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_geometry_infer.py`

Actual author MolmoMotion inference for controlled geometry pilots/full cases.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `parse_tracks`, `recover_selected`, `finalize_selected`, `main`.
Imports/reused functions: `from pathlib import Path`; `import hashlib`; `import argparse`; `import json`; `import resource`; `import subprocess`; `import time`; `import traceback`; `import re`; `from decimal import Decimal`; `import numpy as np`; `import torch`; `from PIL import Image`; `from berkeley_infer import ROOT, CHECKPOINT, REVISION, sha, write, strict_parse, freeze_scene_inputs, recover_saved_group, finalize_scene`; `from berkeley_input_audit import validate_group_payloads`
Constants: `EXECUTED_SCRIPT_BYTES=Path(__file__).read_bytes()`; `EXECUTED_SCRIPT_SHA256=hashlib.sha256(EXECUTED_SCRIPT_BYTES).hexdigest()`
Input/output file references: `<tracks coords="([^"]+)">[^<]*</tracks>\s*`; `Incomplete output or anchor/text disagreement`; `allenai/MolmoMotion-4B-H`; `data/checkpoints/MolmoMotion-4B-H1-F32`; `future_3d.npy`; `geometry/K_median.npy`; `geometry/case_provenance.json`; `geometry/input_audit.json`; `groups/group_`; `groups/group_00/point_ids.npy`; `metadata.json`; `model.pt`; `model_run.json`; `observed/history_rgb.npy`; `points_2d_at_t0.npy`; `points_3d_history.npy`; `prediction.npz`; `predictions/future_3d.npy`; `predictions/group_00/future_3d.npy`; `predictions/model_run.json`; `predictions/point_ids.npy`; `processor_inputs.pt`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_gpu_queue.py`

Queue genuine Berkeley grounding behind existing GPU work; never stop other jobs.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `command_line`, `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, json, subprocess, sys, time`
Constants: `ROOT=Path(__file__).resolve().parents[1]`
Input/output file references: `/cmdline`; `/proc/`; `gpu_queue.json`; `scripts/berkeley_ground_molmopoint.py`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_ground_molmopoint.py`

Genuine official MolmoPoint pointing using only each Berkeley scene's t0.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha256`, `write_json`, `checkpoint_info`, `numpy_metadata`, `prepare_scene`, `select_point`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import gc`; `import hashlib`; `import json`; `import os`; `import sys`; `import time`; `from datetime import datetime, timezone`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `WORKSPACE=ROOT.parent`; `CHECKPOINT=WORKSPACE / 'models/MolmoPoint-Vid-4B'`; `REVISION='b331ed6c6352e6db967325493c6dae515541a919'`; `AUTHOR=ROOT / 'data_generation/third_party/sam3/molmo2_pointing.py'`
Input/output file references: `.cache/huggingface`; `.cache/huggingface/download/config.json.metadata`; `allenai/MolmoPoint-Vid-4B`; `config.json`; `data_generation/third_party/sam3/molmo2_pointing.py`; `model.safetensors.index.json`; `models/MolmoPoint-Vid-4B`; `observed/molmopoint_generated_token_ids.npy`; `observed/molmopoint_grounding.json`; `observed/molmopoint_input_ids.npy`; `observed/molmopoint_overlay.png`; `observed/molmopoint_preparation.json`; `observed/molmopoint_raw_output.txt`; `observed/molmopoint_t0.png`; `observed/molmopoint_t0_single_frame.mp4`; `observed/rgb.npy`; `predictions/model_run.json`; `processor_config.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_grounding_setup.py`

Inspect official grounding access and obtain user-approved official SAM2.1.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: .
Imports/reused functions: `from pathlib import Path`; `import json`; `import time`; `import requests`; `from huggingface_hub import HfApi, get_token, hf_hub_download`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_molmomotion'`
Input/output file references: `allenai/MolmoPoint-Vid-4B`; `facebook/sam2.1-hiera-large`; `facebook/sam3`; `grounding_availability.json`; `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`; `https://huggingface.co/facebook/sam3/resolve/main/sam3.pt`; `models/sam2.1_hiera_large.pt`; `runs/berkeley_ur5_molmomotion`; `sam2.1_hiera_large.pt`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_improvement_continue.py`

Continue already-authorized full cases after the live H1 pilot finishes.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `write`, `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import json`; `import subprocess`; `import sys`; `import time`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`
Input/output file references: `/cmdline`; `/proc/`; `CASE-H1/cup/predictions/model_run.json`; `continuation.json`; `runs/berkeley_ur5_improvement_v1`; `scripts/berkeley_geometry_compare.py`; `scripts/berkeley_geometry_infer.py`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_improvement_queue.py`

Run one authorized case pilot after an identified live workflow completes.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, json, subprocess, sys, time`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`
Input/output file references: `/cmdline`; `/proc/`; `queue.json`; `runs/berkeley_ur5_improvement_v1`; `scripts/berkeley_geometry_compare.py`; `scripts/berkeley_geometry_infer.py`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_improvement_sources.py`

Fetch pinned official calibration and DELTA sources; inspect ShareRobot mapping.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `write`, `fetch`, `main`.
Imports/reused functions: `from pathlib import Path`; `import hashlib`; `import json`; `import sys`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`
Input/output file references: `.json`; `.tools/ijson`; `/`; `/config.py`; `delta_tree.json`; `densetrack3d/models/predictor/predictor.py`; `https://api.github.com/repos/snap-research/DELTA_densetrack3d/commits/main`; `https://api.github.com/repos/snap-research/DELTA_densetrack3d/git/trees/`; `https://huggingface.co/api/datasets/BAAI/ShareRobot`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `https://raw.githubusercontent.com/BerkeleyAutomation/AugE-Toolkit/`; `https://raw.githubusercontent.com/snap-research/DELTA_densetrack3d/`; `official_sources.json`; `planning/jsons/`; `runs/berkeley_ur5_improvement_v1/sources`; `sharerobot_manifest_search.json`; `sharerobot_metadata.json`; `sharerobot_search_scope.json`; `trajectory/`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_improvement_verify.py`

Independent CPU audit of local evidence; Git publication is checked separately.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha`, `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, hashlib, json, re`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`; `BASE=ROOT / 'runs/berkeley_ur5_molmomotion'`
Input/output file references: `*comparison.mp4`; `/`; `/future_3d.npy`; `/points_3d_history.npy`; `<tracks coords="([^"]+)">[^<]*</tracks>\s*`; `CASE-*/**/predictions/group_*/model_run.json`; `Missing/duplicate point`; `displacement_one_third_full_3d.npy`; `final_verification.json`; `fixed_comparison.mp4`; `future_3d.npy`; `geometry/case_provenance.json`; `geometry/strict_geometry_audit.json`; `groups/`; `observed/mask.png`; `observed/native_depth.npy`; `observed/observed_tracks_2d.npz`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `predictions/future_3d.npy`; `predictions/group_`; `predictions/input_freeze.json`; `predictions/model_run.json`; `progress_verification.json`; `protocol.json`; `runs/berkeley_ur5_improvement_v1`; `runs/berkeley_ur5_molmomotion`; `share_extract_*.json`; `sources/sharerobot_pixel_mapping.json`; `visual_review_final.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_infer.py`

Sealed observed-only Berkeley inference, three greedy P8 calls per scene at most.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `write`, `sha`, `strict_parse`, `freeze_scene_inputs`, `recover_saved_group`, `finalize_scene`, `recover_only`, `run`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import re`; `import resource`; `import subprocess`; `import time`; `import traceback`; `from datetime import datetime, timezone`; `from decimal import Decimal`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `REVISION='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629'`
Input/output file references: `<tracks coords="([^"]+)">[^<]*</tracks>\s*`; `Observed inputs or action/t0/history identity changed after freezing`; `Saved output fails strict text/anchor reconstruction; cannot recover`; `allenai/MolmoMotion-4B-H3-F30`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `future_3d.npy`; `geometry/input_audit.json`; `history_rgb.npy`; `metadata.json`; `model.pt`; `model_run.json`; `observed/selected_point_ids.npy`; `point_ids.npy`; `points_2d_at_t0.npy`; `points_3d_history.npy`; `prediction.npz`; `predictions/input_freeze.json`; `processor_inputs.pt`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_input_audit.py`

Pre-inference quality/fence audit, using observed data only.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `validate_group_payloads`, `audit`.
Imports/reused functions: `from pathlib import Path`; `import argparse, json`; `import cv2`; `import numpy as np`
Constants: 
Input/output file references: `/`; `Pre-inference quality/fence audit, using observed data only.`; `Reject stale/reordered model payloads before any expensive forward call.`; `Selected IDs must be 8/16/24 unique integer candidate IDs`; `geometry/K_median.npy`; `geometry/camera_motion_audit.json`; `geometry/depth_observed.npy`; `geometry/input_audit.json`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `grounding_metadata.json`; `history_rgb.npy`; `history_source_indices.npy`; `metadata.json`; `native_depth.npy`; `observed_tracks_2d.npz`; `point_ids.npy`; `points_2d_at_t0.npy`; `points_2d_history.npy`; `points_3d_history.npy`; `predictions/input_freeze.json`; `rgb.npy`; `selected_point_ids.npy`; `source_indices.npy`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_preprocess.py`

Observed-only Berkeley external-camera preparation for MolmoMotion.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `write_json`, `read_json`, `sha256`, `load_observed`, `save_rgb`, `draw_points`, `official_kmeans`, `grounding`, `estimate_depth`, `alltracker_tracks`, `track_observed`, `camera_motion_audit`, `robust_lift`, `load_author_filter`, `spread_select`, `filter_and_export`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import ast`; `import gc`; `import hashlib`; `import importlib.util`; `import json`; `import os`; `from pathlib import Path`; `import sys`; `import time`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `WORKSPACE=ROOT.parent`; `ALLTRACKER=ROOT / 'data_generation/third_party/alltracker'`; `FILTER_SOURCE=ROOT / 'data_generation/third_party/vipe/track-filter-smooth.py'`; `KMEANS_SOURCE=ROOT / 'data_generation/third_party/sam3/querypoints_from_video.py'`
Input/output file references: `.cache/huggingface/download/model.safetensors.metadata`; `.cache/torch/hub/checkpoints/alltracker.pth`; `Berkeley real H3 at 5 Hz (-0.4,-0.2,0s), versus MolmoMotion training 15 Hz; no frame interpolation/duplication`; `Invalid target mask shape/area: `; `Observed-only sparse background optical flow with forward/backward check.`; `This run requires actual MolmoPoint --pointing-json and official SAM2.1; no manual points/masks`; `Validate temporal fence using only files in observed/ and metadata.`; `configs/sam2.1/sam2.1_hiera_l.yaml`; `data_generation/third_party/alltracker`; `data_generation/third_party/sam3/querypoints_from_video.py`; `data_generation/third_party/vipe/track-filter-smooth.py`; `geometry/K_median.npy`; `geometry/K_per_frame.npy`; `geometry/camera_motion_audit.json`; `geometry/camera_poses.npy`; `geometry/depth_observed.npy`; `geometry/depth_source.json`; `geometry/filter_diagnostics.npz`; `geometry/filter_metadata.json`; `geometry/intrinsics_stability.json`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `geometry/unidepth_confidence_observed.npy`; `geometry/unidepth_depth_observed.npy`; `groups/group_`; `https://github.com/facebookresearch/sam2`; `metadata.json`; `models/unidepth-v2-vits14`; `observed-only background Shi-Tomasi + pyramidal LK; forward/backward <1px; object mask excluded`; `observed/alltracker_execution.json`; `observed/frame_t`; `observed/grounding_metadata.json`; `observed/history_rgb.npy`; `observed/mask.png`; `observed/mask_overlay.png`; `observed/native_depth.npy`; `observed/native_depth_metadata.json`; `observed/observed_tracks_2d.npz`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/query_in_mask.npy`; `observed/query_points_100.npy`; `observed/rgb.npy`; `observed/sam_candidates.npz`; `observed/selected_point_ids.npy`; `observed/source_indices.npy`; `point_ids.npy`; `points_2d_at_t0.npy`; `points_2d_at_t0.pt`; `points_3d_history.npy`; `points_3d_history.pt`; `third_party/UniDepth`; `third_party/sam2`; `viz/depth_t0.png`; `viz/history_contact_sheet.png`; `viz/intrinsics_stability.png`; `viz/mask_and_100_queries.png`; `viz/selected_24_points.png`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_report.py`

Build the final Berkeley report strictly from completed experimental artifacts.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `read`, `digest`, `number`, `relative`, `link`, `optional_link`, `scene_bundle`, `build`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `from datetime import datetime, timezone`; `import hashlib`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`
Constants: `REPO=Path(__file__).resolve().parents[1]`; `DATA=REPO.parent / 'data/berkeley_ur5_two_scenes'`; `DEFAULT_RUN=REPO / 'runs/berkeley_ur5_molmomotion'`; `PNG_NAMES=['history_contact_sheet', 'mask_and_100_queries', 'selected_24_points', 'depth_t0', 'intrinsics_stability', 'trajectory_3d', 'final_overlay_t0', 'error_by_time', 'visibility_coverage', 'trajectory_2d_full_extent']`; `VIDEO_NAMES=['pred_vs_gt_2d', 'side_by_side']`
Input/output file references: `    --pointing-json "$RUN/$SCENE/observed/molmopoint_grounding.json" \`; `    --sam-checkpoint ../models/sam2.1_hiera_large.pt`; `  $PRE scripts/berkeley_evaluate.py "$RUN/$SCENE" --stage track --max-side 512`; `  $PRE scripts/berkeley_preprocess.py filter --scene-dir "$RUN/$SCENE"`; `  $PRE scripts/berkeley_preprocess.py ground --scene-dir "$RUN/$SCENE" \`; `  $PRE scripts/berkeley_preprocess.py track --scene-dir "$RUN/$SCENE" --max-side 512`; `  $PY scripts/berkeley_evaluate.py "$RUN/$SCENE" --stage evaluate`; `  $PY scripts/berkeley_preprocess.py depth --scene-dir "$RUN/$SCENE"`; `  $PY scripts/berkeley_scene.py "$SCENE" --t0 "$T0" --attach-native --run-root "$RUN"`; ` "$RUN/analysis_protocol.json"`; ` "$RUN/grounding_availability.json"`; ` / `; ` bottle candidates. Визуально проверены cup 4/7/10/12 и bottle 2/5/6/9. Полный список и причины ранжирования: `; ` bytes**. Record 0 соответствует bottle9, record 1 — cup10. Все pose/gripper states и translation actions каждого эпизода совпадают с LeRobot точно. Lossless native RGB сравнивался с AV1 RGB на наблюдаемых кадрах; отличия обусловлены видеокодеком. `; `$PY scripts/berkeley_data_discover.py`; `$PY scripts/berkeley_data_export_run.py --future --run-root "$RUN"`; `$PY scripts/berkeley_data_extract_native.py`; `$PY scripts/berkeley_data_inspect.py`; `$PY scripts/berkeley_data_native.py`; `$PY scripts/berkeley_ground_molmopoint.py --scene-dir "$RUN/cup" "$RUN/bottle"`; `$PY scripts/berkeley_infer.py --scene-dir "$RUN/cup" "$RUN/bottle"`; `$PY scripts/berkeley_input_audit.py --scene-dir "$RUN/cup" "$RUN/bottle"`; `$PY scripts/berkeley_report.py --run-root "$RUN"`; `$PY scripts/berkeley_scene.py bottle --t0 47 --future --run-root "$RUN"`; `$PY scripts/berkeley_scene.py bottle --t0 47 --run-root "$RUN"`; `$PY scripts/berkeley_scene.py cup --t0 63 --future --run-root "$RUN"`; `$PY scripts/berkeley_scene.py cup --t0 63 --run-root "$RUN"`; `$PY scripts/berkeley_verify.py --run-root "$RUN"`; `$PY scripts/berkeley_video_audit.py --scene-dir "$RUN/cup" "$RUN/bottle"`; `- SAM3 weights вернули HTTP 403 / GatedRepo. По явному выбору пользователя применён официальный Meta SAM2.1 Hiera-Large, prompted настоящим MolmoPoint. `; `.mp4`; `/`; `Build the final Berkeley report strictly from completed experimental artifacts.

This command never modifies observed/, geometry/, groups/, or predictions/.
It refuses missing final metrics, uncompleted inference, and existing reports.
`; `Focal range / median: **`; `K/metrics mismatch: `; `Observed lift/filter/smooth`; `PRE=/mnt/f/AIRI_task/.venv-vipe/bin/python`; `PY=/mnt/f/AIRI_task/.venv/bin/python`; `Prediction/metrics shape mismatch: `; `RUN=$(mktemp -d runs/berkeley_ur5_reproduce_XXXXXX)`; `Required final artifact absent/empty: `; `UniDepthV2 / K`; `analysis_protocol.json`; `candidates_all.json`; `capture_source/receipt.json`; `cd /mnt/f/AIRI_task/molmo-motion`; `data/berkeley_ur5_two_scenes`; `evaluation/alltracker_execution.json`; `evaluation/evaluation_results.npz`; `evaluation/evaluation_tracks_2d.npz`; `evaluation/ground_truth_3d_est.npz`; `evaluation/video_audit.png`; `geometry/K_median.npy`; `geometry/camera_motion_audit.json`; `geometry/depth_source.json`; `geometry/filter_metadata.json`; `geometry/intrinsics_stability.json`; `grounding_availability.json`; `metadata.json`; `metrics.json`; `native_observed_receipt.json`; `observed/alltracker_execution.json`; `observed/grounding_metadata.json`; `observed/mask.png`; `observed/mask_overlay.png`; `observed/molmopoint_grounding.json`; `observed/molmopoint_overlay.png`; `observed/observed_tracks_2d.npz`; `observed/query_points_100.npy`; `points_2d_at_t0.npy`; `points_3d_history.npy`; `prediction.npz`; `predictions/future_3d.npy`; `predictions/model_run.json`; `predictions/prediction.npz`; `runs/berkeley_ur5_molmomotion`; `selection_and_depth_audit.json`; `summary.json`; `verification.json`; `visual_review.json`; `| Group | Parse status | Shape | Prediction, s | Peak allocated GPU, GiB | Raw output / arrays |`; `| Сцена | Episode | t0, кадр / с | Points / chunks | Prediction | ADE 2D, px | FDE 2D, px | Coverage 2D | ADE 3D_est, м | FDE 3D_est, м |`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_scene.py`

Export Berkeley observations before inference; future export is prediction-gated.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `write`, `attach_native`, `export`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import hashlib`; `import json`; `from datetime import datetime, timezone`; `import av`; `import numpy as np`; `import pandas as pd`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT.parent / 'data/berkeley_ur5_two_scenes'`; `RUN=ROOT / 'runs/berkeley_ur5_molmomotion'`; `SCENES={'cup': {'episode': 10, 'instruction': 'Pick up the blue cup and put it into the brown cup.', 'object': 'blue cup', 'reason': 'Printed motif gives visible texture for persistent points; exposed cup separated from brown cup; ep4 initial overlap, ep7/1`
Input/output file references: `.mp4`; `.npy`; `Data quality only; episode/t0 screening precedes predictions. Future screening used only to select episode/t0, never to form model inputs.`; `Printed motif gives visible texture for persistent points; exposed cup separated from brown cup; ep4 initial overlap, ep7/12 plain sides.`; `White bottle body exposed through lift; ep2/5/6 have substantially stronger gripper occlusion of body.`; `data/berkeley_ur5_two_scenes`; `data/chunk-000`; `future_rgb.npy`; `history_rgb.npy`; `history_source_indices.npy`; `history_timestamps.npy`; `lerobot/berkeley_autolab_ur5`; `metadata.json`; `native_observed_receipt.json`; `observed/native_depth.npy`; `observed/native_depth_metadata.json`; `observed/source_indices.npy`; `predictions/input_freeze.json`; `predictions/model_run.json`; `rgb.npy`; `runs/berkeley_ur5_molmomotion`; `source_frames.json`; `source_indices.npy`; `timestamps.npy`; `videos/chunk-000/observation.images.image`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_archive_headers.py`

Check whether ShareRobot's split archive parts are separate gzip streams.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import json, time`; `from urllib.parse import quote`; `from concurrent.futures import ThreadPoolExecutor`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`
Input/output file references: `/`; `https://api.github.com/repos/FlagOpen/ShareRobot/issues/4`; `https://api.github.com/repos/FlagOpen/ShareRobot/issues/4/comments`; `https://github.com/FlagOpen/ShareRobot/issues/4`; `https://huggingface.co/api/datasets`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `planning/images/rt_frames_success.tar.gz.part.`; `runs/berkeley_ur5_improvement_v1/sources`; `sharerobot_archive_headers.json`; `sharerobot_metadata.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_extract_range.py`

Extract only exact Berkeley targets from a bounded, pinned gzip range.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `check_png`, `main`.
Imports/reused functions: `from pathlib import Path`; `import argparse, hashlib, io, json, re, struct, tarfile, time, zlib`; `from urllib.parse import quote`; `import numpy as np`; `from PIL import Image`; `import requests`; `from berkeley_share_range_index import recover, valid_start, MIB, OUT`
Constants: 
Input/output file references: `.json`; `/`; `/43_berkeley_autolab_ur5#episode_(9|10)/frame_(\d+)\.png$`; `Truncated/trailing PNG`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `planning/images/rt_frames_success.tar.gz.part.`; `sharerobot_metadata.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_mapping_probe.py`

Evidence for ShareRobot sequence mapping; bounded image/archive access probe.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `from urllib.parse import quote`; `import io`; `import json`; `import tarfile`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`
Input/output file references: `/`; `/frame_0.png`; `/planning/images/`; `Evidence for ShareRobot sequence mapping; bounded image/archive access probe.`; `Matching ShareRobot IDs/instructions do not prove RGB or temporal mapping. Pixel correspondence remains to be established.`; `Retrieve ShareRobot frames for these exact sequences and match to native/LeRobot episode frames by pixels.`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `planning/images/rt_frames_success.tar.gz.part.aa`; `rt_frames_success/rtx_frames_success_20/43_berkeley_autolab_ur5#episode_`; `runs/berkeley_ur5_improvement_v1/sources`; `share_planning_task.json`; `sharerobot_mapping_probe.json`; `sharerobot_metadata.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_mirror_audit.py`

Find bounded access to the exact ShareRobot planning images in Hub mirrors.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import json`; `from concurrent.futures import ThreadPoolExecutor`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`
Input/output file references: `#episode_10/`; `#episode_9/`; `https://huggingface.co/api/datasets/`; `runs/berkeley_ur5_improvement_v1/sources`; `sharerobot_archive_headers.json`; `sharerobot_mirror_audit.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_pixel_mapping.py`

Prove ShareRobot target sequence identity against the original RLDS RGB.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `digest`, `main`.
Imports/reused functions: `from pathlib import Path`; `import hashlib, io, json, struct`; `import numpy as np`; `from PIL import Image, ImageDraw`; `from probe_fmb_rlds_record import find_feature_values`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `BASE=ROOT / 'runs/berkeley_ur5_molmomotion'`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`; `DATA=ROOT.parent / 'data/berkeley_ur5_two_scenes'`
Input/output file references: `../../berkeley_ur5_molmomotion/sources/native_observed_receipt.json`; `ShareRobot planning retains 30 sampled frames over the whole episode, not the dense 5 FPS model history or future. Model inputs remain the original verified native/LeRobot stream; mapping does not substitute sparse ShareRobot frames into H3.`; `data/berkeley_ur5_two_scenes`; `https://github.com/FlagOpen/ShareRobot/issues/4#issuecomment-3094378733`; `rtx_frames_success_20/43_berkeley_autolab_ur5#episode_`; `runs/berkeley_ur5_improvement_v1/sources`; `runs/berkeley_ur5_molmomotion`; `sharerobot_archive_headers.json`; `sharerobot_metadata.json`; `sharerobot_pixel_mapping.json`; `sources/native_observed_receipt.json`; `steps/observation/image`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_range_index.py`

Bounded exploratory tar-name recovery from PNG-heavy split gzip byte ranges.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `recover`, `valid_start`, `list_members`, `main`.
Imports/reused functions: `from pathlib import Path`; `from concurrent.futures import ThreadPoolExecutor`; `from urllib.parse import quote`; `import argparse, hashlib, io, json, re, tarfile, time, zlib`; `import requests`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`; `MIB=1024 ** 2`
Input/output file references: `.json`; `/`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `planning/images/rt_frames_success.tar.gz.part.`; `runs/berkeley_ur5_improvement_v1/sources`; `sharerobot_metadata.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_share_viewer_probe.py`

Bounded Dataset Viewer checks for images embedded in a ShareRobot mirror.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `get`, `main`.
Imports/reused functions: `from pathlib import Path`; `import json`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1/sources'`; `SERVER='https://datasets-server.huggingface.co'`
Input/output file references: `/first-rows`; `/search`; `/splits`; `FedorX8/ShareRobot-test`; `IffYuan/sharerobot_trajectory`; `advaitgupta/sharerobot-bench`; `berkeley_autolab_ur5#episode_10/`; `berkeley_autolab_ur5#episode_9/`; `https://datasets-server.huggingface.co`; `runs/berkeley_ur5_improvement_v1/sources`; `sharerobot_viewer_probe.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_temporal_diagnostics.py`

CPU-only cadence diagnostics on frozen real Berkeley predictions.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: scale_displacement.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha`, `write`, `scale_displacement`, `oracle_alpha`, `load_scene`, `evaluate`, `render`, `fingerprint`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `from datetime import datetime, timezone`; `import hashlib`; `import json`; `from pathlib import Path`; `import subprocess`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_evaluate import ALIGNMENT, TIMES, project, metric_values, velocity, points_on, trails_on, label, save_rgb, write_video`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `BASE=ROOT / 'runs/berkeley_ur5_molmomotion'`; `OUT=ROOT / 'runs/berkeley_ur5_improvement_v1'`
Input/output file references: `Displacement x 1/3`; `Fixed 1/3 diagnostics must precede oracle fitting`; `H3 speed (m/s)`; `Raw/smoothed direction cosine`; `_comparison.mp4`; `_immutability_check.json`; `_render_receipt.json`; `displacement_one_third_full_3d.npy`; `evaluation/evaluation_results.npz`; `evaluation/future_rgb.npy`; `evaluation/ground_truth_3d_est.npz`; `fixed_arrays.npz`; `fixed_summary.json`; `geometry/K_median.npy`; `geometry/points_3d_raw.npy`; `metrics.json`; `observed/history_timestamps.npy`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `oracle_arrays.npz`; `oracle_summary.json`; `predictions/future_3d.npy`; `protocol.json`; `runs/berkeley_ur5_improvement_v1`; `runs/berkeley_ur5_molmomotion`; `smoothing_velocity.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_verify.py`

Independent artifact/content audit of the completed two-scene experiment.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `sha`, `verify`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import json`; `import hashlib`; `import numpy as np`; `import cv2`; `from berkeley_infer import strict_parse`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/berkeley_ur5_molmomotion'`
Input/output file references: `.mp4`; `Independent artifact/content audit of the completed two-scene experiment.`; `allenai/MolmoPoint-Vid-4B`; `evaluation/evaluation_results.npz`; `future_3d.npy`; `geometry/depth_observed.npy`; `metadata.json`; `metrics.json`; `observed/grounding_metadata.json`; `observed/molmopoint_generated_token_ids.npy`; `observed/molmopoint_grounding.json`; `observed/molmopoint_raw_output.txt`; `observed/native_depth.npy`; `observed/native_depth_metadata.json`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `point_ids.npy`; `predictions/future_3d.npy`; `predictions/input_freeze.json`; `predictions/model_run.json`; `runs/berkeley_ur5_molmomotion`; `sam2.1_hiera_large.pt`; `verification.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_video_audit.py`

Decode real encoded comparison videos into reproducible audit contact sheets.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `audit`.
Imports/reused functions: `import argparse`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: 
Input/output file references: `Wrong frame/time geometry: `; `evaluation/video_audit.json`; `evaluation/video_audit.png`; `viz/side_by_side.mp4`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/berkeley_visual_overview.py`

Show all 24 frozen points at common full plot bounds, colored by P8 group.

Dataset: **berkeley**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/berkeley_bottle.json configs/berkeley_cup.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import json`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_temporal_diagnostics import BASE, OUT, load_scene, scale_displacement, fingerprint`; `from berkeley_evaluate import project`
Constants: `COLORS=['#4aa3ff', '#ff9518', '#f240b8']`
Input/output file references: ` | own-point displacement x1/3`; `: all 24 IDs, all 30 model steps; blue/orange/pink = P8 groups; green = tracked future`; `all24_visual_manifest.json`; `geometry/K_median.npy`; `observed/points_3d_history.npy`; `predictions/future_3d.npy`; `predictions/model_run.json`
Related saved experiments (dataset-level): berkeley_ur5_arc_expansion_v1, berkeley_ur5_improvement_v1, berkeley_ur5_molmomotion, fmb_v2_berkeley_matched, sharerobot_transfer_berkeley_ur5_episode_26

## `scripts/bootstrap_fmb_board_k.py`

Resample board hole features to assess RGB-only K stability.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import numpy as np`; `from calibrate_fmb_board_k import HOLDOUT_FEATURES, TRAIN, cad_features, extract`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from calibrate_fmb_two_boards_k import fit_multi`; `from probe_fmb_tcp_tip_k import OUT`
Constants: 
Input/output file references: `.npy`; `Resample the 10 training hole-center features with replacement; retain six outer/side anchors and all five board placements.`; `bootstrap_K.csv`; `bootstrap_K_summary.json`; `multi_board_bundle_calibration.json`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/build_motion_visualizations.py`

Render two distinct evidence-backed motion figures and their MP4 companions.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `font`, `text`, `legend`, `current_valid`, `mean_path`, `draw_path`, `rgb_overlay`, `fmb_plane`, `history_panels`, `history_intro`, `attach_history_to_summary`, `save_video`, `fmb_data`, `render_fmb`, `project`, `track_observed`, `dobbe_data`, `inset_3d`, `render_dobbe`, `package_data`, `audit`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import csv`; `import hashlib`; `import json`; `import shutil`; `from pathlib import Path`; `import cv2`; `import imageio.v2 as imageio`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from PIL import Image, ImageDraw, ImageFont`; `from scipy.spatial.transform import Rotation`; `from dobbe_two_colmap import RUN as DOBBE, RAW, verify_freeze, read, write, sha`; `from validate_dobbe_geometry import CAMERA_TO_LABEL`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'visualizations'`; `FMB=ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'`; `BG='#101827'`; `CARD='#172236'`; `TEXT='#F3F6FC'`; `MUTED='#B9C5D8'`; `GREEN='#43E2AB'`; `ORANGE='#FFA45B'`; `PURPLE='#BCA5FF'`; `CYAN='#64CFFF'`; `FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')`; `BOLD=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')`
Input/output file references: `.mp4`; `/3  ·  frame `; `/3 · frame `; `/60`; `/8`; `/8 outside RGB`; `/96`; `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf`; `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`; `A/B shared non-geometry parameters identical`; `Dobb·E  /  Same MolmoMotion, two COLMAP intrinsics`; `Dobb·E / episode_3651 / Home15`; `Dots: 8 positions · Trails: means · RGB-proxy GT: 160/160 future observations · Physical F30 timing assumes 15 Hz`; `FMB  /  Prediction vs observed motion`; `FMB / episode_5201 / fixed side_1`; `K A / K B → two histories → two forecasts`; `K.npy`; `MODEL INPUT  /  three real RGB frames with the same eight selected point IDs`; `Nearest real frames to 0.67 / 1.33 s are 0.70 / 1.30 s; no synthetic RGB frames.`; `Same RGB, point IDs, measured HoNY depth, Dobb·E poses and action in A / B.`; `Selected points / IDs 0–7`; `baseline_constant_velocity_2d.npy`; `both raw text independently parsed 240/240`; `comparison_by_step.csv`; `comparison_metrics.json`; `compressed_video_h264.mp4`; `data/artifact_audit.json`; `data/dobbe`; `data/dobbe_provenance.json`; `data/fmb`; `data/fmb_provenance.json`; `dobbe_two_colmap_molmo_comparison.mp4`; `episode_3651 · Home15 · same RGB / 8 IDs / HoNY depth / Dobb·E poses / action / checkpoint / seed`; `existing image-derived ECC RGB proxy, 160/160`; `experiment_config.json`; `fmb_prediction_vs_reality.mp4`; `geometry_audit.json`; `gt_2d.npy`; `history.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `history_world.npy`; `input_freeze.json`; `labels.json`; `metrics.json`; `model_run.json`; `n/a (behind camera)`; `obs/side_1`; `observed_rgb_uv.npy`; `observed_rgb_valid.npy`; `official/database_audit.json`; `official/execution.json`; `official_database_audit.json`; `official_execution.json`; `parser_validation.json`; `points_2d_at_t0.npy`; `prediction.npy`; `prediction_15hz.npy`; `prediction_2d.npy`; `projected_forecast_30hz.npy`; `protocol.json`; `reconstruction.json`; `rgb_tracking.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `shared/history_uv.npy`; `shared/points_2d_at_t0.npy`; `shared/poses_c2w.npy`; `shared/rgb_0098.png`; `t0 unproject/project numerical round trip`; `validity_mask.npy`
Related saved experiments (dataset-level): 

## `scripts/build_track_keys_cache.py`

Pre-scan molmo-motion-1m tracks NPZs to record which object keys each
video actually exposes. The upstream `*_split.json` for some entries
references keys (e.g. `obj1`) that don't exist in the corresponding
`_3d.npz` (upstream re-clustered the tracks but didn't update the split).
We use this cache to filter at Trajectory3DDataset init time so training
never hits a KeyError mid-step.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `_scan_one`, `build_cache_for_dataset`, `main`.
Imports/reused functions: `import argparse`; `import json`; `import os`; `from concurrent.futures import ProcessPoolExecutor, as_completed`; `from pathlib import Path`; `import numpy as np`; `from tqdm import tqdm`
Constants: `MOLMO_MOTION_1M_ROOT=os.environ.get('MOLMO_MOTION_1M_ROOT', 'data/molmo-motion-1m')`; `DATASETS={'egodex': ('egodex/annotations/egodex_split.json', 'egodex/tracks/object', '_3d.npz', True), 'hepic': ('hdepic/annotations/hdepic_split.json', 'hdepic/tracks', '_3d.npz', True), 'ytvis': ('ytvis/annotations/ytvis_split.json', 'ytvis/tracks', '_3d.np`
Input/output file references: ` missing/corrupted NPZs`; `.json`; `_3d.npz`; `data/molmo-motion-1m`; `egodex/annotations/egodex_split.json`; `egodex/tracks/object`; `hdepic/annotations/hdepic_split.json`; `hdepic/tracks`; `molmospaces/annotations/molmospaces_split.json`; `molmospaces/tracks`; `stereo4d/annotations/stereo4d_split.json`; `stereo4d/tracks`; `ytvis/annotations/ytvis_split.json`; `ytvis/tracks`
Related saved experiments (dataset-level): 

## `scripts/calibrate_fmb_board_k.py`

Estimate FMB side_1 effective K from the stationary metric assembly board.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `cad_features`, `ordered_quad`, `extract`, `unpack`, `project`, `score`, `fit`, `depth_crosscheck`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares, linear_sum_assignment`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT, SOURCE`
Constants: `BOARD=OUT / 'peg_board_top_wires.json'`; `TRAIN=(0, 16, 32, 48)`; `HOLDOUT=(8, 24, 40, 56)`; `HOLDOUT_FEATURES=(8, 12, 16)`
Input/output file references: `Medium board: projected hole-center layout matches RGB to mean 0.51 px versus 3.84 and 1.95 px for large/small solid.`; `board_calibration_probe.json`; `board_rgb_observations.json`; `obs/side_1`; `obs/side_1_depth`; `peg_board_top_wires.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/calibrate_fmb_board_shared_plane_k.py`

RGB-only K assuming the repositioned boards share one physical table plane.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `camera_pose`, `fit`, `multi_score`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, depth_crosscheck, extract, project, score, unpack`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT`
Constants: 
Input/output file references: `.npy`; `multi_board_bundle_calibration.json`; `obs/side_1`; `obs/side_1_depth`; `shared_plane_board_calibration.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/calibrate_fmb_multi_boards_k.py`

Joint RGB-only side_1 K from several placements of the official FMB board.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `background_alignment`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, depth_crosscheck, extract`; `from calibrate_fmb_two_boards_k import fit_multi, score_multi`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT, SOURCE`
Constants: `DATA=SOURCE.parents[3] / 'data/fmb/single_object_manipulation_dataset'`; `EPISODES=(2, 3, 4, 5, 6)`
Input/output file references: `.npy`; `board_cad_rgb_correspondences.json`; `board_calibration_frames.json`; `data/fmb/single_object_manipulation_dataset`; `multi_board_bundle_calibration.json`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/calibrate_fmb_multi_horizontal_contact_k.py`

Shared RGB-only K from three different tabletop poses of the same FMB peg.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `k_from_x`, `seed_for_k`, `fit_bundle`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from calibrate_fmb_board_k import HOLDOUT_FEATURES, cad_features, depth_crosscheck, extract, fit as fit_board, project`; `from check_fmb_cross_orientation_depth import check as peg_depth_check`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_cad_silhouette_k import project_model, support`; `from probe_fmb_cross_orientation_silhouette_k import observed_hull`; `from probe_fmb_board_peg_contact_bundle import fit_joint as fit_one_episode`; `from probe_fmb_table_contact_k import peg_pose_on_table, solve_contact`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `DATA=ROOT / 'data/fmb/single_object_manipulation_dataset'`; `EPISODES=(0, 4, 8)`; `CONTACT_AXIS={0: 'x', 4: 'y', 8: 'y'}`
Input/output file references: `.npy`; `board_peg_table_contact_K_probe.json`; `data/fmb/single_object_manipulation_dataset`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `multi_board_bundle_calibration.json`; `multi_horizontal_contact_K_probe.json`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/calibrate_fmb_two_boards_k.py`

Shared side_1 K from two positions of the same official FMB medium board.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `fit_multi`, `score_multi`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, depth_crosscheck, extract, project, score, unpack`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT, SOURCE`
Constants: `SOURCES=(SOURCE, SOURCE.parents[3] / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy')`
Input/output file references: `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `obs/side_1`; `obs/side_1_depth`; `two_board_bundle_calibration.json`; `two_board_rgb_observations.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_dobbe_3651_cup_alignment.py`

Measure red-cup RGB side-edge agreement with depth discontinuities.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `red_cup_mask`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data' / 'dobbe_oxe' / 'target_raw'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight'`; `FRAMES=(0, 20, 40, 60, 80, 100, 180, 220)`
Input/output file references: `compressed_video_h264.mp4`; `dobbe_3651_cup_rgb_depth_alignment.json`; `u_depth=u_rgb, v_depth=(v_rgb+.5)*192/256-.5`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/check_dobbe_colmap_official_math.py`

Numerical checks for alignment and SIMPLE_RADIAL convention conversion.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: .
Imports/reused functions: `import numpy as np`; `import cv2`; `import pycolmap`; `from scipy.spatial.transform import Rotation`; `from dobbe_colmap_official import CAMERA_TO_LABEL, camera_candidate, sim3`
Constants: 
Input/output file references: `PASS: Sim(3) scale, rotation, translation and orientation; SIMPLE_RADIAL/OpenCV projection`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/check_dobbe_rgb_depth_alignment.py`

Test the published 256x256 RGB ↔ 256x192 depth mapping on past frames.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read_frames`, `score`, `inspect_scene`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import liblzfse`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_rgbd_study'`; `SCENES={'main': ('target_raw', [0, 20, 40, 60, 80]), 'second': ('second_raw', [0, 30, 60, 90, 120])}`
Input/output file references: `.tools/lzfse`; `compressed_video_h264.mp4`; `data/dobbe_oxe`; `rgb_depth_alignment.json`; `runs/dobbe_rgbd_study`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/check_fmb_board_annotation_stability.py`

Stress the automatic board landmark extraction against mask thresholds.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from calibrate_fmb_board_k import cad_features, extract`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from probe_fmb_tcp_tip_k import OUT`
Constants: `FRAMES=(0, 16, 32)`
Input/output file references: `.npy`; `board_annotation_stability.json`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_board_distortion.py`

Compare zero-distortion and k1/k2 RGB board calibration on held-out data.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `project_distorted`, `evaluate`, `fit`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, extract, project`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from probe_fmb_tcp_tip_k import OUT`
Constants: 
Input/output file references: `.npy`; `Compare zero-distortion and k1/k2 RGB board calibration on held-out data.`; `distortion_model_comparison.json`; `multi_board_bundle_calibration.json`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_cad_scale.py`

Compare the known 150 mm FMB peg with its RGB size and sensor depth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `STEPS=[0, 5, 10, 20, 37, 74, 110, 120, 125, 130, 135, 140, 145]`; `FY_256_CANDIDATE=380.209 * 256 / 480`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `cad_scale_check.json`; `obs/side_1`; `obs/side_1_depth`; `runs/sharerobot_fmb_episode_5201`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_cross_orientation_depth.py`

Independent Z16 check of the RGB-only horizontal-peg silhouette fit.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `surface_points`, `check`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import D, H, K as K_NOMINAL, W`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `SOURCE=ROOT / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy'`; `RADIUS=0.00288`
Input/output file references: `Metric comparison is independent of RGB fit, but RGB/depth registration and Z16 scale were not recorded in the episode.`; `cross_orientation_sensor_depth_check.json`; `cross_orientation_silhouette_k_probe.json`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_horizontal_peg_segmentation.py`

Measure RGB silhouette sensitivity for a newly sampled horizontal FMB peg.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `hull`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `SOURCE=ROOT / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy'`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `STEPS=tuple(range(0, 50, 5))`; `HUE_MAX=(13, 20, 25, 30, 35)`; `ANGLES=np.arange(0, 2 * np.pi, np.pi / 8)`; `DIRS=np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))`
Input/output file references: `data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy`; `horizontal_peg_segmentation_stability.json`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_nominal_repeat.py`

Check whether a second run with exactly the same frozen inputs is identical.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`
Constants: `RUN=Path(os.environ.get('FMB_QUANT_RUN', Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`
Input/output file references: `model_run.json`; `nominal_repeatability.json`; `prediction_15hz.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `variants/nominal_repeat`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_rgb_depth_alignment.py`

Check RGB/depth boundary agreement for the red peg in episode_5201.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `median_finite`, `crossing`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image, ImageDraw`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`; `STEPS=(0, 37, 74, 110, 120, 130, 140)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `obs/side_1`; `obs/side_1_depth`; `rgb_depth_alignment_check.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_share_sampling.py`

Match locally available ShareRobot PNGs to all four FMB RGB streams.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `from PIL import Image`; `from audit_fmb_episode_5201 import phash, ssim`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`; `SHARE=ROOT / 'runs' / 'sharerobot_transfer_berkeley_ur5_episode_26'`; `CAMERAS=('side_1', 'side_2', 'wrist_1', 'wrist_2')`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `four_camera_sampling_check.json`; `obs/`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_fmb_silhouette_history_uncertainty.py`

Propagate RGB-silhouette K sensitivity into the MolmoMotion 3D history.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `one_history`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import OUT, SOURCE`; `from probe_fmb_cad_silhouette_k import initial_pose, project_model, red_hull, support`
Constants: `HISTORY=(130, 135, 140)`
Input/output file references: `cad_history_candidate/manifest.json`; `effective_k_rounded_cad_history_candidates.npz`; `effective_k_rounded_cad_history_uncertainty.json`; `effective_k_rounded_cad_silhouette.json`; `obs/side_1`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/check_setup.py`

Check the isolated WSL environment without running model inference.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import platform`; `import resource`; `import sys`; `from importlib.metadata import version`; `from pathlib import Path`; `import av`; `import cv2`; `import decord`; `import imageio`; `import numpy`; `import torch`; `from PIL import Image`; `from torchcodec.decoders import VideoDecoder`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`
Constants: 
Input/output file references: `data_generation/third_party/vipe/assets/examples/dog-example.mp4`
Related saved experiments (dataset-level): 

## `scripts/check_two_colmap_geometry.py`

Checks real calibration provenance and geometry without touching inference.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from dobbe_two_colmap import ROOT, RAW, RUN, A_RUN, read, write, sha, verify_freeze`; `from validate_dobbe_geometry import CAMERA_TO_LABEL, evaluate`
Constants: 
Input/output file references: `22 automatic feature correspondences, depth-Z/ray ambiguity and uncertain labels; do not treat a smaller diagnostic error as certification`; `Geometry audit PASS; external depth/labels diagnostics saved`; `K.npy`; `geometry_audit.json`; `history.npy`; `images/main/`; `labels.json`; `official/images/`; `protocol.json`; `runs/dobbe_colmap_clean_rerun_v1/causal_validation_correspondences.json`; `shared/history_depth_m.npy`; `shared/history_uv.npy`; `shared/poses_c2w.npy`
Related saved experiments (dataset-level): 

## `scripts/collect_dobbe_vipe_results.py`

Collect the small reviewable experiment artifacts, without raw data or weights.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `collect`.
Imports/reused functions: `import shutil`; `from dobbe_vipe_v1 import ROOT, RUN, read, write, sha`
Constants: 
Input/output file references: `/`; `ablation_input_audit.json`; `ablation_input_audit_initial.json`; `alltracker_execution.json`; `c2w_t0.npy`; `causal_tracks.mp4`; `control_protocol.json`; `data_generation/third_party/alltracker/nets/alltracker.py`; `data_generation/third_party/alltracker/nets/blocks.py`; `data_generation/third_party/vipe/track-filter-smooth.py`; `evaluation/alltracker_execution.json`; `evaluation/conditional_error_over_time.png`; `evaluation/conditional_projection.npz`; `evaluation/future_access_receipt.json`; `evaluation/future_comparison.png`; `evaluation/future_overlay.mp4`; `evaluation/metrics.json`; `evaluation/observed_future_tracks.npz`; `frame_map.json`; `frames/`; `geometry_gate.json`; `hybrid_alignment_review.json`; `implementation_checks.json`; `independent_static_sift.json`; `input_freeze.json`; `manifest.json`; `mask_annotation.json`; `model_run.json`; `paired_forecast_summary.json`; `paired_selection_audit.json`; `paired_selection_protocol.json`; `points_2d_t0.npy`; `points_3d_camera_t0_diagnostic.npy`; `points_3d_world.npy`; `prediction_15hz.npy`; `prediction_parsed_visibility.npy`; `processor_equivalence.json`; `protocol.json`; `protocol_action_correction.json`; `query_points.npz`; `rejected_sensor_queries/*/*`; `report/dobbe_vipe_results`; `scripts/*dobbe_vipe*`; `src/molmo_motion/modeling.py`; `src/molmo_motion/processor.py`; `static_sift_*.npz`; `tests/test_dobbe_vipe_v1.py`; `verification.json`; `vipe/intrinsics/causal_15hz.npz`; `vipe/intrinsics/causal_15hz_camera.txt`; `vipe/pose/causal_15hz.npz`; `vipe_default_execution.json`; `vipe_no_vda_execution.json`; `vipe_no_vda_gdown6_failure_execution.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/collect_fmb_second_assets.py`

Collect the official camera and object assets without altering their bytes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data/fmb_second_scene'`; `BASE='https://functional-manipulation-benchmark.github.io/static/'`; `ASSETS={'calibration/raw/side_1': 'files/side_1', 'calibration/raw/side_2': 'files/side_2', 'calibration/raw/wrist_1': 'files/wrist_1', 'calibration/raw/wrist_2': 'files/wrist_2', 'object/peg_official.step': 'files/peg.step', 'object/shape_color_reference.p`
Input/output file references: `asset_provenance.json`; `calibration/raw`; `calibration/raw/side_1`; `calibration/raw/side_2`; `calibration/raw/wrist_1`; `calibration/raw/wrist_2`; `camera/rs_capture.py`; `data/fmb_second_scene`; `doc/FMB%20Shape%20and%20Color%20Number%20Reference%20Sheet%20-%20Google%20Docs.pdf`; `envs/franka_fmb_env.py`; `files/peg.step`; `files/side_1`; `files/side_2`; `files/wrist_1`; `files/wrist_2`; `https://functional-manipulation-benchmark.github.io/static/`; `https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/`; `https://raw.githubusercontent.com/rail-berkeley/fmb/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/`; `object/peg_official.step`; `object/shape_color_reference.pdf`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_depth_k_factorial.py`

Fixed-depth 2x2 comparison: sensor/MoGe-2 depth and nominal/0.9 focal K.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `metric`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json`
Constants: 
Input/output file references: `Fixed-depth 2x2 comparison: sensor/MoGe-2 depth and nominal/0.9 focal K.`; `comparison.csv`; `comparison.json`; `evaluation.json`; `metrics.json`; `runs/fmb_depth_k_factorial_v1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_effective_k_candidates.py`

Compare camera-K hypotheses using held-out RGB board and sensor geometry.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `history_rigidity`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import numpy as np`; `from calibrate_fmb_board_k import HOLDOUT, HOLDOUT_FEATURES, TRAIN, cad_features, depth_crosscheck, extract`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from calibrate_fmb_two_boards_k import fit_multi, score_multi`; `from probe_fmb_tcp_tip_k import OUT`
Constants: `FIRST_RUN=OUT.parent / 'sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'`
Input/output file references: `.npy`; `K_comparison.csv`; `K_comparison.json`; `board_peg_table_contact_K_probe.json`; `cross_orientation_silhouette_k_probe.json`; `history_points_2d.npy`; `k_sensitivity_variants.json`; `multi_board_bundle_calibration.json`; `multi_episode_tcp_tip_K_probe.json`; `multi_horizontal_contact_K_probe.json`; `not evaluated on RGB/TCP tip holdout`; `not independent: this K was fitted to RGB/TCP tip data`; `obs/side_1`; `obs/side_1_depth`; `shared_tcp_offset_K_probe.json`; `sharerobot_fmb_episode_5201/effective_k_rounded_cad_silhouette.json`; `sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_geometry_forecasts.py`

Diagnostic comparison of sensor-depth and alternative CAD-geometry runs.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `compare_variant`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import os`; `from pathlib import Path`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import GT_TIMES, interpolate, measure, project`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`
Input/output file references: `Planar RGB tracked correspondences are weak: frame 124 PnP implies large rotation/motion inconsistent with robot TCP; diagnostic stress case.`; `_comparison.json`; `_disagreement_by_time.csv`; `experiment_config.json`; `gt_2d.npy`; `history_sensor_3d.npy`; `k_nominal.json`; `model_run.json`; `prediction_10hz.npy`; `prediction_15hz.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `sensor_vs_cad_silhouette_tcp.json`; `sensor_vs_pnp.json`; `validity_mask.npy`; `variants/cad_silhouette/prediction_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_gt_methods.py`

Re-score the same frozen forecasts against both RGB-derived 2D trackers.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `predictions`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import interpolate, measure, project`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json`
Constants: 
Input/output file references: `baseline_constant_velocity_2d.npy`; `baseline_stationary_2d.npy`; `comparison.csv`; `comparison.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `k_nominal.json`; `prediction_2d.npy`; `runs/fmb_gt_sensitivity_v1`; `validity_mask.npy`; `variants/cad_silhouette/prediction_15hz.npy`; `variants/pnp/prediction_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_metric_geometry.py`

Compare monocular predictions with tentative FMB CAD PnP and sensor depth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `peg_mask`, `robust`, `temporal_comparison`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from itertools import combinations`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `K_RGB=np.asarray([[152.0836, 0, 124.2908], [0, 202.7781333333, 129.2973333333], [0, 0, 1]], dtype=np.float64)`
Input/output file references: `*_metric_predictions.npz`; `1_M_L_3_vertical_n_2.npy`; `assumed 0.0001 m/count and candidate RGB K; interior 5x5 valid-depth median`; `cad_history_candidate/manifest.json: projected visible interior CAD locations`; `cad_pnp_probe.json`; `manifest.json`; `metric_geometry_comparison.json`; `obs/side_1`; `obs/side_1_depth`; `points_3d_history_candidate.npy`; `runs/sharerobot_fmb_episode_5201`; `sensor_history_candidate.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_fmb_sensor_to_cad.py`

Deproject interior sensor depth at the same eight CAD/PnP RGB landmarks.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `from itertools import combinations`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `DEST=OUT / 'cad_history_candidate'`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `Deproject interior sensor depth at the same eight CAD/PnP RGB landmarks.`; `manifest.json`; `obs/side_1_depth`; `points_3d_history_candidate.npy`; `runs/sharerobot_fmb_episode_5201`; `sensor_cad_comparison.json`; `sensor_history_candidate.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/compare_two_fmb_trials.py`

Descriptive side-by-side audit of two frozen FMB 2D experiments.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `one`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `from pathlib import Path`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `FIRST=ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'`; `SECOND_BASE=ROOT / 'runs/fmb_second_example_1_M_L_3_vertical_n_3'`; `SECOND=SECOND_BASE / 'quantitative_2d_sensor_t130'`
Input/output file references: `ShareRobot episode_5201 / FMB trial 2`; `Trial 2 / ShareRobot 5201`; `Trial 3 / raw FMB`; `Trials differ in source episode, t0, object motion, point locations, and unverified effective camera/depth preprocessing. The additional raw FMB file has no verified ShareRobot episode mapping.`; `Two local FMB trials, same prediction/evaluation protocol`; `experiment_config.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `gt_tracking_diagnostics.json`; `history_depth_probe.json`; `history_sensor_3d.npy`; `k_nominal.json`; `metrics.json`; `model_run.json`; `points_2d_at_t0.npy`; `prediction_15hz.npy`; `prediction_2d.npy`; `preflight.json`; `projection_check.json`; `runs/fmb_second_example_1_M_L_3_vertical_n_3`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `sensor_vs_cad_silhouette_tcp.json`; `two_trial_comparison.csv`; `two_trial_comparison.json`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/convert_dobbe_control_f2nerf.py`

Run the supplied F2-NeRF COLMAP converter on the RGB-only control scene.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import subprocess`; `import sys`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `WORKSPACE=ROOT.parent`; `STUDY=ROOT / 'runs/dobbe_rgbd_study'`; `COLMAP=STUDY / 'second/colmap_pinhole_all_fixedpp_init62_107'`; `STAGE=STUDY / 'f2nerf_second'`; `CONVERTER=WORKSPACE / 'third_party/f2-nerf/scripts/colmap2poses.py'`
Input/output file references: `cams_meta.npy`; `receipt.json`; `runs/dobbe_rgbd_study`; `second/colmap_pinhole_all_fixedpp_init62_107`; `second/rgb_prefix`; `summary.json`; `third_party/f2-nerf/scripts/colmap2poses.py`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_colmap_clean.py`

Clean RGB-only COLMAP controls; oracle and causal outputs stay separate.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `plain`, `write`, `digest`, `mask_rgb`, `prepare`, `components`, `audit_database`, `summarize`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import csv`; `import hashlib`; `import json`; `import time`; `import sqlite3`; `from collections import Counter`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import pycolmap`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_colmap_clean_rerun_v1'`; `SOURCE=ROOT / 'data/dobbe_oxe/target_raw/compressed_video_h264.mp4'`; `PAIR_BASE=2147483647`; `POLICY={'version': 1, 'design_rgb_frames_causal_only': [0, 32, 64, 95], 'camera_border_px': [10, 10, 35, 21], 'apparatus_corridors_xy': [[[0, 256], [0, 209], [46, 146], [74, 148], [40, 256]], [[44, 146], [76, 146], [134, 256], [87, 256]], [[151, 256], [161,`
Input/output file references: `Conservative apparatus/hand corridors and RGB red cup; imperfect foreground mask, not learned segmentation`; `Oracle intrinsics, poses and graph cannot initialize/select causal models`; `clean_protocol.json`; `config.json`; `data/dobbe_oxe/target_raw/compressed_video_h264.mp4`; `images/main`; `images/main/`; `mask_policy_frozen.json`; `masks/main`; `masks/main/`; `match_graph.json`; `match_images.csv`; `match_pairs.csv`; `runs/dobbe_colmap_clean_rerun_v1`; `state.json`; `summary.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_colmap_official.py`

Prepare and inspect official automatic_reconstructor outputs, without fitting K.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `write`, `sha`, `csv_write`, `prepare`, `prepare_subset`, `database_audit`, `load_measurements`, `sim3`, `camera_candidate`, `track_diagnostics`, `analyze`, `audit`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import csv`; `import hashlib`; `import json`; `import shutil`; `import sqlite3`; `import subprocess`; `from pathlib import Path`; `import cv2`; `import imageio_ffmpeg`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `import pycolmap`; `from scipy.spatial.transform import Rotation`; `from dobbe_colmap_clean import components, summarize`; `from inspect_hony_scenes import liblzfse`; `from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, evaluate`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_colmap_official_v1'`; `SOURCE=ROOT / 'data/dobbe_oxe/target_raw'`; `PAIR_BASE=2147483647`; `RUNS={'full': list(range(243)), 'causal': list(range(96)), 'causal_stride5': list(range(0, 96, 5))}`
Input/output file references: `22 automatic RGB matches, not manually certified static landmarks; Z/ray unresolved`; `Not accepted: largest causal model reprojection on fixed 22 measured-depth/labels correspondences is worse than diagnostic baseline`; `audit.json`; `compressed_video_h264.mp4`; `data/dobbe_oxe/target_raw`; `decision.json`; `execution.json`; `experiment_results.csv`; `label_motion_audit.json`; `labels.json`; `match_graph.json`; `match_images.csv`; `match_pairs.csv`; `protocol.json`; `runs/dobbe_colmap_clean_rerun_v1/causal_validation_correspondences.json`; `runs/dobbe_colmap_clean_rerun_v1/mask_policy_frozen.json`; `runs/dobbe_colmap_clean_rerun_v1/masks/main`; `runs/dobbe_colmap_clean_rerun_v1/masks/main/`; `runs/dobbe_colmap_official_v1`; `summary.json`; `trajectory.csv`; `u_d=u_cv; v_d=(v_cv+0.5)*192/256-0.5`; `validation_sources.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_two_colmap.py`

A sealed K-only MolmoMotion comparison, using two RGB-only COLMAP pipelines.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `write`, `sha`, `prepare`, `reconstruction_summary`, `build_inputs`, `verify_freeze`, `infer`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import resource`; `import shutil`; `import sqlite3`; `import time`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import pycolmap`; `from scipy.spatial.transform import Rotation`; `from dobbe_colmap_clean import summarize`; `from inspect_hony_scenes import liblzfse`; `from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/dobbe_two_colmap_v1'`; `RAW=ROOT / 'data/dobbe_oxe/target_raw'`; `OLD=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `A_RUN=ROOT / 'runs/dobbe_colmap_clean_rerun_v1/main_causal_pinhole_masked_stride1_all_dsp_affine_guided'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `HISTORY=[96, 97, 98]`
Input/output file references: `A/B inference complete; all non-geometry inputs identical`; `Freeze method selection before seeing the new B reconstruction/forecasts.`; `K.npy`; `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30`; `compressed_video_h264.mp4`; `config.json`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `data/dobbe_oxe/target_raw`; `history.npy`; `history_depth_m.npy`; `history_uv.npy`; `history_world.npy`; `images/main/`; `input_freeze.json`; `labels.json`; `manifest.json`; `model.pt`; `model_run.json`; `official COLMAP 4.2.1 automatic_reconstructor; video/high, PINHOLE, single camera, no mask, no manual K/initial pair/mapper thresholds`; `official/database.db`; `official/database_audit.json`; `official/execution.json`; `official/images`; `points_2d_at_t0.npy`; `poses_c2w.npy`; `prediction.npy`; `protocol.json`; `reconstruction.json`; `runs/dobbe_colmap_clean_rerun_v1/main_causal_pinhole_masked_stride1_all_dsp_affine_guided`; `runs/dobbe_rgbd_study/approx_history`; `runs/dobbe_two_colmap_v1`; `s, parsed 240/240`; `shared/points_2d_at_t0.npy`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_vipe_geometry.py`

Geometry I/O, diagnostics and selection; all arrays use causal ViPE indices.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: project, stats.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `restore_intrinsics`, `load_vipe`, `sample_depth`, `lift`, `project`, `stats`, `independent_sift_geometry`, `rigid_error`, `select_spread`, `geometry`, `render_geometry`, `render_causal_tracks`.
Imports/reused functions: `from __future__ import annotations`; `import importlib.util`; `import io`; `import json`; `import zipfile`; `import cv2`; `import numpy as np`; `from dobbe_vipe_v1 import RUN, VIPE, GATE, read, write, sha, scene_dir`
Constants: 
Input/output file references: `.npz`; `Geometry I/O, diagnostics and selection; all arrays use causal ViPE indices.`; `Incomplete/misaligned ViPE `; `Incomplete/misaligned ViPE depth IDs`; `Tracking/ViPE frame identity mismatch`; `c2w_t0.npy`; `causal_tracks.mp4`; `estimated camera trajectory X/Z`; `geometry_all.npz`; `geometry_gate.json`; `measured_depth_prefix.npy`; `points_2d_t0.npy`; `points_3d_world.npy`; `protocol.json`; `static_sift_*.npz`; `tracks.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_vipe_inference.py`

Sealed inference, processor equivalence and measured-depth ablation.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `export_variants`, `save_variant`, `inputs`, `infer`, `render_forecast`.
Imports/reused functions: `from __future__ import annotations`; `import time`; `import resource`; `import traceback`; `import cv2`; `import numpy as np`; `from dobbe_vipe_v1 import ROOT, RUN, CHECKPOINT, GATE, read, write, sha, freeze, scene_dir`; `from dobbe_vipe_geometry import lift, rigid_error, project, stats, independent_sift_geometry`
Constants: 
Input/output file references: `c2w_t0.npy`; `geometry_all.npz`; `geometry_gate.json`; `hybrid_alignment_review.json`; `input_freeze.json`; `measured_depth_prefix.npy`; `model.pt`; `model_run.json`; `paired_selection_protocol.json`; `points_2d_t0.npy`; `points_3d_camera_t0_diagnostic.npy`; `points_3d_world.npy`; `prediction_15hz.npy`; `prediction_parsed_visibility.npy`; `processor_equivalence.json`; `protocol.json`; `runs/dobbe_rgbd_study/approx_history/points_2d_at_t0_candidate.npy`; `tracks.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/dobbe_vipe_v1.py`

Causal 15 Hz Dobb·E -> ViPE -> AllTracker -> MolmoMotion experiment.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `write`, `sha`, `freeze`, `scene_dir`, `prepare_branch`, `load_video`, `encode`, `prepare`, `masks`, `run_receipt`, `vipe`, `track`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import os`; `import subprocess`; `import sys`; `import time`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/dobbe_vipe_v1'`; `VIPE=ROOT / 'data_generation/third_party/vipe'`; `ALLTRACKER=ROOT / 'data_generation/third_party/alltracker'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `SCENES={'A': {'raw': 'data/dobbe_oxe/target_raw', 't0': 98, 'recording': 'Pick_and_Place/Home15/Env1/2023-04-25--02-05-30', 'action': 'Pick up the red cup.'}, 'B': {'raw': 'data/dobbe_oxe/second_raw', 't0': 123, 'recording': 'Pick_and_Place/Home7/Env1/2023-`; `GATE={'static_median_px_max': 4.0, 'static_p90_px_max': 10.0, 'static_pairs_min': 3, 'static_observations_min': 30, 'rigid_relative_median_max': 0.15, 'history_visible_points_min': 8, 'depth_ratio_median_min': 0.5, 'depth_ratio_median_max': 2.0, 'focal_re`
Input/output file references: `.cache/huggingface`; `.cache/torch`; `.cache/torch/hub`; `.cache/torch/hub/checkpoints/alltracker.pth`; `.tools/lzfse`; `/usr/local/cuda-12.8`; `Causal RGB lossless encoding changed frame pixels/order`; `Drawer_Closing/Home10/Env2/2022-12-22--00-09-08`; `No RGB/depth/pose after t0 enters prediction, mask, tracking, filtering or geometry gate.`; `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30`; `Pick_and_Place/Home7/Env1/2023-04-27--10-47-40`; `Successful cached execution has different inputs/command`; `_execution.json`; `_failure.json`; `alltracker_execution.json`; `causal_15hz.mp4`; `compressed_video_h264.mp4`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `data/dobbe_oxe/second_raw`; `data/dobbe_oxe/target_raw`; `data_generation/third_party/alltracker`; `data_generation/third_party/vipe`; `frame_map.json`; `https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth`; `labels.json`; `mask_annotation.json`; `measured_depth_prefix.npy`; `protocol.json`; `query_points.npz`; `runs/dobbe_rgbd_study/approx_history/points_2d_at_t0_candidate.npy`; `runs/dobbe_vipe_v1`; `runs/plex_dobbe_preflight/dobbe_rgbd_sample`; `sift_audited supports variants/processor/infer only; preserve the original geometry gate`; `tracks.npz`; `u_model=u_vipe; v_model=(v_vipe+0.5)*256/192-0.5`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/document_fmb_action_traceability.py`

Record exact source primitive and ShareRobot action wording for frozen run.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `from probe_fmb_cad_pnp import SOURCE`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `BASE=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `RUN=BASE / 'quantitative_2d_sensor_t126'`
Input/output file references: `The frozen English action paraphrases the FMB primitive insert and ShareRobot's rectangular-object/square-slot task; color, peg, and blue board are from object metadata and RGB scene.`; `action_traceability.json`; `experiment_config.json`; `planning_rows.json`; `runs/sharerobot_fmb_episode_5201`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/download_dobbe_3651_lerobot.py`

Download only the LeRobot RGB video and states for Dobb-E episode 3651.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `checksum`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DEST=ROOT / 'data' / 'dobbe_oxe'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight'`; `BASE='https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot/resolve/main/'`; `FILES={'lerobot_episode_003651.mp4': ('videos/chunk-003/observation.images.wrist_image/episode_003651.mp4', '495dbe8c0a5d6472fd515eb5d1325a66116721a476393a80a22b8f3ad940b51d'), 'lerobot_episode_003651.parquet': ('data/chunk-003/episode_003651.parquet', '54`
Input/output file references: `data/chunk-003/episode_003651.parquet`; `dobbe_3651_lerobot_receipt.json`; `https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot/resolve/main/`; `lerobot_episode_003651.mp4`; `videos/chunk-003/observation.images.wrist_image/episode_003651.mp4`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/download_dobbe_oxe_shard.py`

Download only the OXE/TFDS Dobb-e shard containing a candidate ordinal.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'data' / 'dobbe_oxe'`; `REPO='lerobot-raw/dobbe_raw'`; `BASE=f'https://huggingface.co/datasets/{REPO}/resolve/main'`; `CANDIDATE_ORDINAL=3651`
Input/output file references: `/`; `/dataset_info.json`; `/resolve/main`; `Download only the OXE/TFDS Dobb-e shard containing a candidate ordinal.`; `dataset_info.json`; `download_receipt.json`; `https://huggingface.co/datasets/`; `lerobot-raw/dobbe_raw`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/download_fmb_raw_candidate.py`

Download one original FMB NPY, verifying its Hugging Face SHA-256.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `digest`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import os`; `import sys`; `from pathlib import Path`; `import requests`
Constants: `REVISION='f99fd55c072eea5573523c96aa527aed3c665690'`; `BASE=f'https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/resolve/{REVISION}/single_object_manipulation_dataset/'`; `DEST=Path(__file__).resolve().parents[2] / 'data/fmb/single_object_manipulation_dataset'`
Input/output file references: `.npy`; `/`; `/single_object_manipulation_dataset/`; `Download one original FMB NPY, verifying its Hugging Face SHA-256.

Usage: python scripts/download_fmb_raw_candidate.py FILENAME.npy
`; `data/fmb/single_object_manipulation_dataset`; `https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/resolve/`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/download_plex_cereal_member.py`

Download only the PickPlaceCereal HDF5 member from Microsoft's PLEX ZIP.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import shutil`; `import zipfile`; `from pathlib import Path`; `from inspect_remote_zip import HTTPRangeFile`
Constants: `URL='https://download.microsoft.com/download/e/5/1/e5106eb4-0f53-4d65-afd1-58c03d60bf98/robosuite_demo_data%20%281%29.zip'`; `MEMBER='robosuite_demo_data/PickPlaceCereal/Panda/raw/PickPlaceCereal_demo_act_norm.hdf5'`; `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'data' / 'plex' / Path(MEMBER).name`
Input/output file references: `download_receipt.json`; `https://download.microsoft.com/download/e/5/1/e5106eb4-0f53-4d65-afd1-58c03d60bf98/robosuite_demo_data%20%281%29.zip`; `robosuite_demo_data/PickPlaceCereal/Panda/raw/PickPlaceCereal_demo_act_norm.hdf5`
Related saved experiments (dataset-level): 

## `scripts/evaluate_author_davis.py`

Evaluate and visualize the saved single-episode F30 prediction without GPU.

Dataset: **davis**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: metrics, parse_raw_forecast, project.
Frozen selected experiment: `molmo-motion-experiment --config configs/author_davis.json`.

Functions: `check_original_hashes`, `parse_raw_forecast`, `verify_saved_data`, `metrics`, `self_test`, `project`, `plot_error`, `plot_tracks`, `draw_frame`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import json`; `import re`; `from decimal import Decimal`; `from pathlib import Path`; `import imageio.v2 as imageio`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from PIL import Image, ImageDraw, ImageFont`; `from matplotlib.lines import Line2D`; `import torch`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/author_davis_bmx_trees_f30'`; `SAMPLE=ROOT / 'examples/data/davis_bmx_trees'`; `SOURCE=ROOT / 'data/pointmotionbench/davis'`; `METHODS=['MolmoMotion', 'Static', 'Constant velocity']`; `COLORS={'MolmoMotion': '#e63946', 'Static': '#3267d6', 'Constant velocity': '#dd8b17'}`; `POINT_COLORS=['#ff595e', '#ffca3a', '#8ac926', '#1982c4', '#6a4c93', '#ff924c', '#40c9a2', '#f72585']`; `LABEL_OFFSETS=[(-62, -48), (20, -57), (35, -8), (-70, 8), (40, 30), (-65, 48), (-72, -17), (40, 55)]`
Input/output file references: `/32`; `<tracks coords="([^"]+)">[^<]*</tracks>\s*`; `Dequantized answer contains NaN/Inf`; `Saved 3D coordinates are consistent with the first sequence frame's camera/world anchor, not camera t0=frame 2. A single K without per-frame extrinsics cannot project common-frame 3D coordinates onto later moving RGB frames.`; `bmx-trees_2d.npz`; `bmx-trees_3d.npz`; `correction_baseline_hashes.json`; `correction_validation.json`; `data/pointmotionbench/davis`; `examples/data/davis_bmx_trees`; `frames/`; `ground_truth.npz`; `intrinsics_K.pt`; `meta.json`; `metrics.csv`; `metrics.json`; `points_2d_at_t0.pt`; `points_3d_history.pt`; `prediction has NaN/Inf; cannot silently mask model omissions`; `prediction.npz`; `processor_inputs.pt`; `projection_check.json`; `real_continuation_gt.mp4`; `runs/author_davis_bmx_trees_f30`
Related saved experiments (dataset-level): author_davis_bmx_trees_f30, author_davis_coordinate_audit

## `scripts/evaluate_dobbe_approx_future.py`

Track sealed cup points in future HoNY RGB-D after prediction is frozen.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `sha256`, `red_surface_mask`, `metric`, `resample_model`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from inspect_hony_scenes import ROOT, liblzfse`; `from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, unproject`
Constants: `RUN=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `RAW=ROOT / 'data/dobbe_oxe/target_raw'`
Input/output file references: `/8 depth GT`; `compressed_video_h264.mp4`; `data/dobbe_oxe/target_raw`; `future_3d_gt_candidate.npy`; `future_evaluation.json`; `future_gt_valid.npy`; `future_rgb_uv.npy`; `input_freeze.json`; `labels.json`; `manifest.json`; `model_run.json`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `prediction_15hz.npy`; `runs/dobbe_rgbd_study/approx_history`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/evaluate_dobbe_vipe.py`

Future-only evaluation after a successful sealed forecast.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `relative_label_poses`, `evaluate`, `render`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from dobbe_vipe_v1 import ROOT, ALLTRACKER, scene_dir, read, write, sha, freeze, load_video, encode`; `from dobbe_vipe_geometry import project, sample_depth, stats`
Constants: 
Input/output file references: `.cache/torch/hub`; `.cache/torch/hub/checkpoints/alltracker.pth`; `/8`; `Pixel comparison is conditional on the published pose/basis and estimated K; future 2D tracking is model-derived, not manual GT.`; `alltracker_execution.json`; `compressed_video_h264.mp4`; `conditional_projection.npz`; `future_15hz.mp4`; `future_access_receipt.json`; `future_overlay.mp4`; `geometry_all.npz`; `geometry_gate.json`; `input_freeze.json`; `labels.json`; `measured_depth_prefix.npy`; `metrics.json`; `model_run.json`; `observed_future_tracks.npz`; `points_2d_t0.npy`; `points_3d_camera_t0_diagnostic.npy`; `prediction_15hz.npy`; `protocol.json`; `tracks.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/evaluate_fmb_quantitative_2d.py`

Evaluate frozen MolmoMotion predictions on native FMB 10 Hz RGB GT.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `project`, `interpolate`, `measure`, `write_csv`, `draw_points`, `visuals`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import os`; `import shutil`; `from pathlib import Path`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from probe_fmb_cad_pnp import SOURCE as DEFAULT_SOURCE`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`; `BASE=RUN.parent`; `SOURCE=Path(os.environ.get('FMB_QUANT_SOURCE', DEFAULT_SOURCE))`; `PRED_TIMES=np.arange(1, 31, dtype=float) / 15`; `GT_TIMES=np.arange(1, 21, dtype=float) / 10`; `DIAGONAL=float(np.hypot(256, 256))`
Input/output file references: ` (valid / 25)`; `/160 pairs; FDE: `; `/8 points`; `annotation_repeatability.json`; `baseline_constant_velocity_2d.npy`; `baseline_stationary_2d.npy`; `calibration_sensitivity.csv`; `error_by_time.csv`; `evaluation_time_10hz_s.npy`; `experiment_config.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `history_depth_probe.json`; `history_pnp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `k_sensitivity_variants.json`; `metrics.csv`; `metrics.json`; `model_run.json`; `obs/side_1`; `per_point_metrics.csv`; `prediction_10hz.npy`; `prediction_15hz.npy`; `prediction_2d.npy`; `prediction_time_15hz_s.npy`; `projection_check.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/evaluate_fmb_rigid_correction.py`

Postprocess frozen FMB forecasts with a per-frame proper rigid fit.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `rigid_fit`, `pairwise_distances`, `evaluate_one`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `from pathlib import Path`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import GT_TIMES, interpolate, measure, project`; `from run_fmb_ablation_suite import RUNS, write_json`
Constants: 
Input/output file references: `corrected_prediction_10hz.npy`; `corrected_prediction_15hz.npy`; `corrected_prediction_2d.npy`; `evaluation.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `history_sensor_3d.npy`; `k_nominal.json`; `prediction_15hz.npy`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/export_fmb_geometry_bundle.py`

Export comparable, explicitly sparse/dense FMB history geometry records.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, digest, verify_frozen, write_json`
Constants: `METHODS={'sensor': 'history_sensor_3d.npy', 'planar_pnp': 'history_pnp_3d.npy', 'cad_silhouette_tcp': 'history_cad_silhouette_tcp_3d.npy', 'moge2_raw': 'history_moge2_raw_3d.npy', 'moge2_history_scaled': 'history_moge2_history_scaled_3d.npy'}`
Input/output file references: `.npz`; `Approximate planar RGB/CAD correspondences, no dense depth map`; `Export comparable, explicitly sparse/dense FMB history geometry records.`; `RGB/CAD and TCP derived point positions, no dense depth map`; `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `history_cad_silhouette_tcp_3d.npy`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_pnp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `k_nominal.json`; `manifest.json`; `moge2_history_v1/manifest.json`; `moge2_history_v1/moge2_history_maps.npz`; `obs/side_1_depth`; `sensor Z16, 0.0001 m/raw`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/export_fmb_second_scene.py`

Export a complete original FMB demonstration and visual preflight.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `write_json`, `sha256`, `ranges`, `labeled_frame`, `export_rgb_and_video`, `export_depth`, `export_state`, `make_contact`, `make_synced`, `depth_visual`, `make_rgb_depth`, `make_plots`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import shutil`; `from pathlib import Path`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from PIL import Image, ImageDraw`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `NAME='1_L_L_4_vertical_n_0.npy'`; `SOURCE=ROOT.parent / 'data/fmb/single_object_manipulation_dataset' / NAME`; `DATA=ROOT / 'data/fmb_second_scene'`; `RUN=ROOT / 'runs/fmb_second_scene'`; `CAMERAS=('side_1', 'side_2', 'wrist_1', 'wrist_2')`; `STATE_KEYS=('obs/tcp_pose', 'obs/tcp_vel', 'obs/tcp_force', 'obs/tcp_torque', 'obs/q', 'obs/dq', 'obs/jacobian', 'obs/gripper_pose', 'actions', 'primitive')`; `SOURCE_URL='https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/blob/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset/' + NAME`
Input/output file references: `.mp4`; `.npy`; `1_L_L_4_vertical_n_0.npy`; `data/fmb/single_object_manipulation_dataset`; `data/fmb_second_scene`; `depth_statistics.json`; `https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/blob/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset/`; `https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip`; `metadata.json`; `obs/`; `obs/dq`; `obs/gripper_pose`; `obs/jacobian`; `obs/q`; `obs/side_1`; `obs/tcp_force`; `obs/tcp_pose`; `obs/tcp_torque`; `obs/tcp_vel`; `raw/source_demo.npy`; `robot_state.npz`; `runs/fmb_second_scene`; `visuals/gripper_timeline.png`; `visuals/object_configuration.png`; `visuals/primitive_timeline.png`; `visuals/synchronized_four_cameras.png`; `visuals/tcp_orientation_change.png`; `visuals/tcp_xyz.png`; `visuals/valid_depth_fraction.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/extend_fmb_2d_metrics.py`

Additional visible-motion metrics from the frozen FMB 2D annotations.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `angle_degrees`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import interpolate, project`; `from inspect_fmb_2d_window import red_component`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: 
Input/output file references: `annotation_repeatability.json`; `baseline_constant_velocity_2d.npy`; `baseline_stationary_2d.npy`; `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `gt_2d.npy`; `k_nominal.json`; `metrics.csv`; `metrics.json`; `obs/side_1`; `points_2d_at_t0.npy`; `prediction_15hz.npy`; `runs/fmb_extended_2d_metrics_v1`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/extract_dobbe_3651_rgbd.py`

Fetch the gripper-matched HoNY RGB-D capture from one split ZIP part.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read_range`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import struct`; `import sys`; `import zlib`; `from pathlib import Path`; `import requests`; `from index_dobbe_depth_zip import BASE, OUT, confirmed_params, range_get`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DEST=ROOT / 'data' / 'dobbe_oxe' / 'target_raw'`; `INDEX=OUT / 'dobbe_depth_nested_index.json'`; `GDOWN_TARGET=ROOT.parent / '.tools' / 'gdown'`; `FOLDER_ID='1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO'`; `SOURCE='iphone_data_depth/Pick_and_Place/Home15/Env1/2023-04-25--02-05-30/'`; `PART_SIZE=1073741824`; `INNER_START_GLOBAL=105`
Input/output file references: `/`; `Inner ZIP capture directory, including iphone_data_depth/`; `ZIP central/local header mismatch`; `compressed_video_h264.mp4`; `dobbe_3651_rgbd_receipt.json`; `dobbe_depth_nested_index.json`; `iphone_data_depth/Pick_and_Place/Home15/Env1/2023-04-25--02-05-30/`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/extract_dobbe_depth_sample.py`

Fetch one HoNY RGB-D record from the last split ZIP with HTTP ranges.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `fetch_range`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import struct`; `import zlib`; `from pathlib import Path`; `import requests`; `from index_dobbe_depth_zip import BASE, OUT, confirmed_params, range_get`
Constants: 
Input/output file references: `compressed_video_h264.mp4`; `dobbe_depth_nested_index.json`; `dobbe_rgbd_sample_receipt.json`; `iphone_data_depth/Drawer_Closing/Home10/Env2/2022-12-22--00-09-08/`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/extract_sharerobot_candidate_rows.py`

Stream one ShareRobot planning JSON and retain only two target episode rows.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'sharerobot_candidate_rows.json'`; `IJSON_TARGET=ROOT.parent / '.tools' / 'ijson'`; `SOURCE='https://huggingface.co/datasets/BAAI/ShareRobot/resolve/main/planning/jsons/planning_task.json'`; `TARGETS=('plex_robosuite#episode_46', 'dobbe#episode_3651')`
Input/output file references: `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/main/planning/jsons/planning_task.json`; `sharerobot_candidate_rows.json`
Related saved experiments (dataset-level): 

## `scripts/fetch_plex_scene_assets.py`

Fetch only asset files referenced by a recorded PLEX Robosuite scene.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `relative_asset`, `fetch`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import concurrent.futures`; `import hashlib`; `import json`; `import posixpath`; `import urllib.error`; `import urllib.request`; `import xml.etree.ElementTree as ET`; `from pathlib import Path`; `import h5py`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `HDF5=ROOT / 'data' / 'plex' / 'PickPlaceCereal_demo_act_norm.hdf5'`; `ASSETS=ROOT / 'data' / 'plex' / 'robosuite_assets'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'plex_asset_receipt.json'`; `COMMIT='c2fca799bb8ebd42bad9c4c5fa4157fea4251f98'`; `OLD_PREFIX='/mnt/d/code/robosuite/robosuite/models/assets/'`; `BASE=f'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/{COMMIT}/robosuite/models/assets/'`
Input/output file references: `../`; `.//*[@file]`; `/`; `/mnt/d/code/robosuite/robosuite/models/assets/`; `/robosuite/models/assets/`; `https://raw.githubusercontent.com/ARISE-Initiative/robosuite/`; `plex_asset_receipt.json`
Related saved experiments (dataset-level): 

## `scripts/finalize_fmb_geometry_preflight.py`

Reconcile recovered FMB calibration evidence into episode_5201 preflight.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `save_json`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import numpy as np`; `from audit_fmb_episode_5201 import camera_check`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`; `INTRINSICS_URL='https://functional-manipulation-benchmark.github.io/static/files/side_1'`; `FMB_CODE='https://github.com/rail-berkeley/fmb/tree/d4da6ce044a9806f41e58bf7423b5a3c05289925'`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `Official capture code has no transformation taking aligned 640x480 depth to the published 256x256 depth; the JSON has no active color profile/stream metadata. Exact depth-pixel K cannot be reproduced from published code.`; `Scale inference uses the target episode's later TCP trajectory and is diagnostic; an unbiased future test should confirm/freeze calibration independently.`; `camera_motion_check_extended.json`; `depth_calibration_check.json`; `geometry_probe.json`; `https://functional-manipulation-benchmark.github.io/static/files/side_1`; `https://github.com/rail-berkeley/fmb/tree/d4da6ce044a9806f41e58bf7423b5a3c05289925`; `intrinsics_decode.json`; `obs/side_1`; `obs/side_1_depth`; `preflight.json`; `set_gripper waits up to 1 s between commands and another 1.2 s on close / 0.6 s on open; do not assume globally uniform measured time.`; `temporal_alignment.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/find_second_fmb_episode.py`

Inspect a few nearby raw FMB episode files without downloading the dataset.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from huggingface_hub import HfApi`
Constants: `REPO='charlesxu0124/functional-manipulation-benchmark'`; `PREFIX='single_object_manipulation_dataset/1_M_L_3_vertical_n_'`
Input/output file references: `.npy`; `charlesxu0124/functional-manipulation-benchmark`; `single_object_manipulation_dataset/1_M_L_3_vertical_n_`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fit_dobbe_3651_effective_k.py`

Exploratory shared K/registration fit from HoNY RGB-D and recorded poses.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `pair_residual`, `summarize`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import numpy as np`; `from scipy.ndimage import map_coordinates`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_dobbe_3651_camera_geometry import DATA, OUT, PAIRS, load_frames, make_matches`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `TRAIN={(0, 20), (20, 40), (40, 60), (180, 200)}`; `HOLDOUT={(60, 80), (200, 220)}`; `C=np.asarray([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)`
Input/output file references: `Exploratory shared K/registration fit from HoNY RGB-D and recorded poses.

This is a diagnostic with held-out frame pairs. It cannot validate the sensor
extrinsic, Record3D depth semantics, or ShareRobot pixel identity by itself.
`; `dobbe_3651_effective_k_fit.json`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/fmb_colmap_official.py`

Official RGB-only COLMAP controls on two FMB episodes and two separate views.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `camera_candidate`, `digest`, `prepare`, `board_check`, `depth_check`, `motion_check`, `pair_motion`, `analyze`, `audit`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `from pathlib import Path`; `import shutil`; `import sqlite3`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `import pycolmap`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from dobbe_colmap_official import camera_candidate as radial_candidate, csv_write, database_audit, read, write`; `from dobbe_colmap_clean import summarize`; `from calibrate_fmb_board_k import cad_features, extract, TRAIN, HOLDOUT, HOLDOUT_FEATURES`; `from probe_fmb_cad_pnp import K as K_NOMINAL`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/fmb_colmap_official_v1'`; `EPISODES={'episode2': {'source': ROOT / 'runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy', 't0': 126}, 'episode3': {'source': ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy', 't0': 130}}`; `VIEWS=('side_1', 'wrist_1')`; `SCOPES=('full', 'causal')`; `MODEL='SIMPLE_RADIAL'`
Input/output file references: `Approximate segmented CAD outline/hole features; depth registration, scale and color intrinsics not certified; K is frozen, only board pose fits RGB`; `Diagnostics only, neither source depth/TCP nor nominal K is passed to COLMAP`; `Fixed side camera has no camera-translation parallax; wrist and side K are camera specific; depth registration/scale and active color calibration are not independently certified`; `Image displacement is not triangulation angle; planar/static matches can pass verification`; `Scale fits these same samples; moving robot/object points included; unverified RGB-depth registration and units; not K certification`; `Sensor depth, CAD and TCP are post-reconstruction diagnostics only; depth scale 0.0001 m/count remains conditional`; `_nominal_board_check.json`; `audit.json`; `camera_diagnostics.csv`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `decision.json`; `execution.json`; `experiment_results.csv`; `obs/`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `protocol.json`; `runs/fmb_colmap_official_pinhole_v1`; `runs/fmb_colmap_official_v1`; `runs/fmb_effective_k_256_calibration/peg_board_top_wires.json`; `runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy`; `scripts/calibrate_fmb_board_k.py`; `summary.json`; `trajectory.csv`; `validation_sources.json`; `verified_pair_displacements.csv`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_checkpoint_verify.py`

Verify native model bytes against the pinned Hub LFS SHA-256 receipts.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from fmb_v2_scene import ROOT, RUN`; `from berkeley_preprocess import sha256, write_json`
Constants: 
Input/output file references: `.cache/huggingface/download/model.pt.metadata`; `checkpoint_byte_audit.json`; `data/checkpoints`; `model.pt`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_continue.py`

Continue this local experiment after H3, preserving the prediction/evaluation fence.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `import os`; `from pathlib import Path`; `import subprocess`; `import sys`; `import time`; `from fmb_v2_scene import ROOT, RUN, SPECS`; `from berkeley_preprocess import write_json`
Constants: 
Input/output file references: `Continue this local experiment after H3, preserving the prediction/evaluation fence.`; `continuation_status.json`; `evaluation/alltracker_future.npz`; `evaluation/future_rgb.npy`; `group_*/model_run.json`; `predictions/model_run.json`; `predictions_h1/model_run.json`; `scripts/fmb_v2_evaluate.py`; `scripts/fmb_v2_infer.py`; `scripts/fmb_v2_scene.py`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_environment.py`

Read-only environment/checkpoint probe; no secrets are printed.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json`; `from pathlib import Path`; `import torch, transformers, requests`
Constants: 
Input/output file references: `Read-only environment/checkpoint probe; no secrets are printed.`; `h1_checkpoint_hub.json`; `https://huggingface.co/api/models/allenai/MolmoMotion-4B-H1-F32`; `runs/fmb_v2_berkeley_matched`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_evaluate.py`

Prediction-gated independent AllTracker reference, nominal-time metrics and media.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `interpolate_prediction`, `check_freeze`, `track`, `metric`, `uncertainty`, `visualize`, `evaluate`, `json_summary`, `audit_reference`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import numpy as np`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from berkeley_preprocess import alltracker_tracks, write_json, read_json, sha256, save_rgb`; `from berkeley_evaluate import project, velocity, points_on, trails_on, label, write_video`; `from fmb_v2_preprocess import lift`
Constants: `TIMES=np.arange(1, 21) / 10`
Input/output file references: `All 24 points: green reference, full 30/32-step model trajectories`; `Linear interpolation of predicted XYZ only; source frames/reference not interpolated`; `alltracker_future.npz`; `evaluation/alltracker_execution.json`; `evaluation/alltracker_future.npz`; `evaluation/ecc_uncertainty.npz`; `evaluation/future_rgb.npy`; `evaluation/reference_uncertainty.json`; `evaluation/references_and_methods.npz`; `future_3d.npy`; `future_rgb.npy`; `geometry/K_median.npy`; `metadata.json`; `metrics.json`; `native_depth_future.npy`; `observed/history_timestamps.npy`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/rgb.npy`; `observed/selected_point_ids.npy`; `prediction_vs_real.mp4`; `predictions/future_3d.npy`; `predictions/input_freeze.json`; `predictions/model_run.json`; `predictions_h1/future_3d.npy`; `predictions_h1/model_run.json`; `references_and_methods.npz`; `side_by_side.mp4`; `timestamps.npy`; `video_receipts.json`; `viz/manual_reference_audit.png`; `viz/reference_uncertainty.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_ground.py`

Use the Berkeley MolmoPoint runner with a FMB observed-only prompt adapter.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `prepare_scene`.
Imports/reused functions: `import sys`; `from datetime import datetime, timezone`; `import cv2`; `import numpy as np`; `import berkeley_ground_molmopoint as original`; `from berkeley_preprocess import load_observed`
Constants: 
Input/output file references: `observed/molmopoint_grounding.json`; `observed/molmopoint_input_ids.npy`; `observed/molmopoint_preparation.json`; `observed/molmopoint_t0.png`; `observed/rgb.npy`; `predictions/input_freeze.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_infer.py`

Sealed BF16 greedy FMB H3/H1 inference with complete text/parser auditing.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `strict_parse`, `run`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `from decimal import Decimal`; `import json`; `import re`; `import resource`; `import time`; `import traceback`; `import numpy as np`; `import torch`; `from PIL import Image`; `from berkeley_preprocess import sha256, write_json, read_json`; `from berkeley_infer import freeze_scene_inputs`; `from berkeley_input_audit import validate_group_payloads`
Constants: `ROOT=Path(__file__).resolve().parents[1]`
Input/output file references: `/`; `<tracks coords="([^"]+)">[^<]*</tracks>\s*`; `Duplicate/out-of-range point ID`; `Invalid time ID/record count at `; `Sealed BF16 greedy FMB H3/H1 inference with complete text/parser auditing.`; `allenai/MolmoMotion-4B-H`; `anchor.npy`; `data/checkpoints`; `future_3d.npy`; `geometry/input_audit.json`; `h1_checkpoint_hub.json`; `input_hashes.json`; `metadata.json`; `model.pt`; `model_run.json`; `observed/history_rgb.npy`; `observed/selected_point_ids.npy`; `points_2d_at_t0.npy`; `points_3d_history.npy`; `predictions/input_freeze.json`; `predictions/model_run.json`; `processor_inputs.pt`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_preprocess.py`

Berkeley methods applied to FMB, with FMB depth/registration quality checks.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `depth`, `registration`, `lift`, `author_filter`, `observed_masks`, `filter_scene`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import time`; `import numpy as np`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `.npy`; `.npz`; `Berkeley methods applied to FMB, with FMB depth/registration quality checks.`; `Exactly the Berkeley 5x5 finite positive median, >=13/25 on a full patch.`; `Measured FMB Z16 with supported, unconfirmed 0.0001 m/raw scale`; `Nearest-edge distances are descriptive and include texture/internal board edges; do not certify pointwise registration.`; `configs/sam2.1/sam2.1_hiera_l.yaml`; `geometry/K_median.npy`; `geometry/K_per_frame.npy`; `geometry/camera_motion_audit.json`; `geometry/camera_poses.npy`; `geometry/depth_observed.npy`; `geometry/depth_source.json`; `geometry/filter_diagnostics.npz`; `geometry/input_audit.json`; `geometry/intrinsics_stability.json`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `geometry/registration_audit.json`; `geometry/selection_gates.json`; `metadata.json`; `models/sam2.1_hiera_large.pt`; `observed/historical_masks_h3.npy`; `observed/historical_masks_metadata.json`; `observed/mask.png`; `observed/molmopoint_grounding.json`; `observed/native_depth.npy`; `observed/observed_tracks_2d.npz`; `observed/query_in_mask.npy`; `observed/query_points_100.npy`; `predictions/input_freeze.json`; `third_party/sam2`; `viz/fixed_camera_check.png`; `viz/historical_tracks.png`; `viz/history_xyz_clouds.png`; `viz/rgb_depth_registration_audit.png`; `viz/selected_24_points.png`; `viz/sensor_depth.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_report.py`

Build the independent FMB v2 report and verify completion evidence.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `verify`, `report`.
Imports/reused functions: `from pathlib import Path`; `import json`; `import re`; `import numpy as np`; `import cv2`; `from fmb_v2_scene import ROOT, RUN, SPECS`; `from berkeley_preprocess import read_json, write_json, sha256`; `from fmb_v2_infer import strict_parse`; `from fmb_v2_evaluate import check_freeze`
Constants: 
Input/output file references: `
[Видео рядом](../runs/fmb_v2_berkeley_matched/`; ` mm. [Метрики](../runs/fmb_v2_berkeley_matched/`; `/`; `/480 в 2D, `; `/480 в 3D_est. Максимальная поправка H3 ray smoothing: `; `/evaluation/metrics.json), [input audit](../runs/fmb_v2_berkeley_matched/`; `/geometry/input_audit.json).
`; `/viz/`; `/viz/prediction_vs_real.mp4)
`; `/viz/side_by_side.mp4) · [Видео наложения](../runs/fmb_v2_berkeley_matched/`; `RGB/depth registration audit`; `](../runs/fmb_v2_berkeley_matched/`; `anchor.npy`; `checkpoint_byte_audit.json`; `completion_audit.json`; `episode_5201/evaluation/metrics.json`; `episode_5201/evaluation/reference_uncertainty.json`; `evaluation/metrics.json`; `evaluation/references_and_methods.npz`; `fmb_control_n3/evaluation/reference_uncertainty.json`; `future_3d.npy`; `geometry/K_median.npy`; `geometry/camera_motion_audit.json`; `geometry/input_audit.json`; `geometry/registration_audit.json`; `metadata.json`; `model_run.json`; `observed/grounding_metadata.json`; `observed/molmopoint_grounding.json`; `observed/points_3d_history.npy`; `observed/rgb.npy`; `prediction_vs_real.mp4`; `preserved_v1.json`; `report/fmb_v2_berkeley_matched.md`; `side_by_side.mp4`; `visual_review.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_scene.py`

Independent FMB v2 export. Future export requires complete H3 and H1 receipts.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `preserved_v1`, `export`.
Imports/reused functions: `from pathlib import Path`; `import argparse`; `import json`; `import shutil`; `import sys`; `import numpy as np`; `import cv2`; `from berkeley_preprocess import sha256, write_json`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/fmb_v2_berkeley_matched'`; `SPECS={'episode_5201': {'source': ROOT / 'runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy', 't0': 126, 'share_robot_id': '57_fmb#episode_5201'}, 'fmb_control_n3': {'source': ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_verti`; `ACTION='Insert the red rectangular peg into the matching hole on the blue board.'`
Input/output file references: `FMB obs/side_1_depth`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `export_receipt.json`; `future_rgb.npy`; `history_rgb.npy`; `history_source_indices.npy`; `history_timestamps.npy`; `metadata.json`; `model_run.json`; `native_depth.npy`; `native_depth_future.npy`; `native_depth_metadata.json`; `obs/side_1`; `obs/side_1_depth`; `preserved_v1.json`; `provenance_v1_source_audit.json`; `rgb.npy`; `runs/fmb_second_example_1_M_L_3_vertical_n_3`; `runs/fmb_v2_berkeley_matched`; `runs/sharerobot_fmb_episode_5201`; `runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy`; `runs/sharerobot_fmb_episode_5201/preflight.json`; `sensor_z16.npy`; `source_indices.npy`; `timestamps.npy`; `viz/observed_8_frames.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_v2_search_share.py`

Bounded provenance search for control n3 in the pinned ShareRobot trajectory subset.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from concurrent.futures import ThreadPoolExecutor`; `import json`; `import numpy as np`; `import cv2`; `from fmb_v2_scene import ROOT, RUN, SPECS`; `from search_fmb_second_sharerobot import fetch`; `from berkeley_preprocess import write_json, sha256`
Constants: 
Input/output file references: `obs/`; `runs/fmb_second_scene/sharerobot_trajectory_manifest.json`; `second_share_search.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_abort_resource.py`

Archive this run's unfinished call before a documented resource-only retry.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import os, signal, time`; `from pathlib import Path`; `from datetime import datetime, timezone`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`
Constants: 
Input/output file references: `/proc`; `group_00/model_run.json`; `interruption_receipt.json`; `model_run.json`; `wrist_2/branches/A_vipe_full/predictions`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_after_vipe.py`

Continue observed-only preprocessing after the specific live ViPE worker.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import argparse, os, subprocess, time`; `from pathlib import Path`; `from fmb_wrist_prepare import ROOT, RUN, write`
Constants: 
Input/output file references: `.cache/huggingface`; `.cache/torch`; `.venv/bin/python`; `/proc`; `observed/molmopoint_grounding.json`; `preprocess_status.json`; `preprocess_wait.json`; `scripts/fmb_wrist_ground.py`; `scripts/fmb_wrist_preprocess.py`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_blinded_media.py`

Hide method labels for the assistant's first qualitative forecast assessment.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json`; `import cv2, numpy as np`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_math import from_anchor, project`; `from fmb_wrist_evaluate import prediction_gate, align_prediction, TIMES`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `/8`; `/CV`; `/Static`; `evaluation/blinding_map.json`; `evaluation/future_reference.npz`; `evaluation/future_rgb.npy`; `geometry/branch_selection.json`; `observed/points_3d_history.npy`; `predictions/future_3d.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_completion_audit.py`

Verify requested artifacts, actual native calls, hashes, coordinates and evaluation.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `audit`.
Imports/reused functions: `import json, re, subprocess`; `from datetime import datetime, timezone`; `from pathlib import Path`; `import cv2, numpy as np, torch`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`; `from fmb_wrist_math import project, from_anchor`; `from fmb_v2_infer import strict_parse`; `from berkeley_input_audit import validate_group_payloads`
Constants: 
Input/output file references: `/`; `/CV`; `/Static`; `24 same deterministic points across A/B/C`; `ADE/FDE/time/perpoint/baselines/uncertainty`; `First blinded judgments preserved before map/metric disclosure`; `Four meaningful coordinate/time tests passed`; `Report figures/videos/evidence links resolve`; `Requested controlled experiment completed, including failed geometry/cross-view validations; no calibrated physical GT or successful four-view fusion claimed.`; `Source-matched FMB / ShareRobot5201`; `all selected H3 points supported by masks/visibility/depth/author filters`; `anchor.npy`; `builtin_blinded_review.json`; `builtin_visual_review_*.json`; `builtin_visual_review_future.json`; `builtin_visual_review_observed.json`; `completion_audit.json`; `coordinate_test_receipt.json`; `crossview_observed_audit.json`; `crossview_observed_audit.json; side_future_crosscheck.json`; `data/checkpoints/MolmoMotion-4B-H3-F30/model.pt`; `evaluation/export_receipt.json`; `evaluation/export_receipt.json; future_alltracker.npz`; `evaluation/future_alltracker.npz`; `evaluation/future_reference.npz`; `evaluation/hand_eye_reference_sensitivity.json`; `evaluation/metrics.json`; `evaluation/metrics.json and hand_eye_reference_sensitivity.json`; `evaluation/metrics.json; future_reference.npz`; `filter_diagnostics.npz`; `forecast_pipeline_status.json`; `future_3d.npy`; `geometry/ hand-eye and control audits`; `geometry/K.npy`; `geometry/branch_selection.json`; `geometry/camera_poses.npy`; `geometry/depth_comparison.json`; `geometry/unidepth_K.npy`; `geometry_forecasts_side_by_side.mp4`; `group_00/raw_model_output.txt`; `hand_eye.json`; `hand_eye_bootstrap.json`; `independent_robot_rgbd_bootstrap.json`; `independent_robot_rgbd_hand_eye.json`; `index.html; video_gallery_receipt.json`; `input_hashes.json`; `interruption_receipt.json`; `metadata.json`; `metadata.json; observed/source_indices.npy`; `model_run.json`; `native_square_control.json`; `observed grounding/SAM/AllTracker and UniDepth`; `observed/historical_masks_h3.npy`; `observed/mask.png`; `observed/molmopoint_grounding.json`; `observed/observed_tracks_2d.npz`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `observed/selected_point_ids.npy; geometry/branch_selection.json`; `observed/source_indices.npy`; `points_3d_history.npy`; `predictions/future_3d.npy`; `predictions/input_freeze.json`; `predictions/interrupted_resource_attempt_00/`; `predictions/model_run.json`; `preserved_prior_runs.json`; `primary_prediction_vs_real.mp4`; `processor_inputs.pt`; `report/fmb_v2_berkeley_matched.md`; `report/fmb_wrist_v3.md`; `report/fmb_wrist_v3.md local links`; `requirements_progress.json`; `runs/fmb_v2_berkeley_matched/completion_audit.json`; `runs/fmb_v2_berkeley_matched/preserved_v1.json`; `selection/candidates.json`; `selection/candidates.json; geometry_ranking.json; target_depth_support.json`; `selection/geometry_ranking.json`; `selection/target_depth_support.json`; `side_future_crosscheck.json`; `source_episode_5201_provenance.json`; `sources/source_receipts.json`; `video_gallery_receipt.json`; `vipe_*_receipt.json`; `vipe_default_receipt.json`; `vipe_rectified_default_receipt.json`; `viz/*.png`; `viz/real_future.mp4`; `wrist_2/branches/A_vipe_full/predictions/interrupted_resource_attempt_00`; `wrist_2/metadata.json`; `wrist_future_world_motion.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_crossview.py`

Observed-only wrist/side geometry checks with explicit correspondence limits.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `rgb_masks`, `scene_points`, `align_rigid`, `run`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, json`; `from pathlib import Path`; `import cv2, numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from scipy.spatial import cKDTree`; `from fmb_wrist_prepare import ROOT, RUN, CAMERAS, write`; `from fmb_wrist_math import *`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `Different surfaces/partial FOV, ambiguous factory RGB K and independently estimated X; nearest-surface agreement is not shared-point accuracy`; `Observed-only wrist/side geometry checks with explicit correspondence limits.`; `crossview_observed_audit.json`; `geometry/estimated_side_c2base.npy`; `geometry/hand_eye_X.npy`; `geometry/hand_eye_Y.npy`; `geometry/observed_target_color_centroids.npy`; `geometry/selected_hand_eye_X.npy`; `geometry/vipe_camera_poses.npy`; `observed/rgb.npy`; `observed/sensor_z16.npy`; `observed/tcp_pose_xyzw.npy`; `sources/official_K_candidates.json`; `viz/crossview_board_matches.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_evaluate.py`

Prediction-gated, moving-camera future reference, common-mask metrics, media.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: scores.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `prediction_gate`, `export_future`, `track_future`, `align_prediction`, `scores`, `evaluate`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, json, subprocess`; `from datetime import datetime, timezone`; `from pathlib import Path`; `import cv2, numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import imageio.v2 as imageio`; `from fmb_wrist_prepare import ROOT, RUN, write, sha, CAMERAS`; `from fmb_wrist_math import *`; `import berkeley_preprocess as b`
Constants: `TIMES=np.arange(1, 21) / 10`
Input/output file references: `/8`; `/CV`; `/Static`; `K.npy`; `One frozen official K sensor/TCP control for every geometry; no branch-specific future reference`; `Reference inherits observed hand-eye/K/scale uncertainty and can drift on low-texture target. All rejected geometries are explicitly diagnostic.`; `alltracker_receipt.json`; `builtin_visual_review_observed.json`; `export_receipt.json`; `future_alltracker.npz`; `future_reference.npz`; `future_rgb.npy`; `future_sensor_depth.npy`; `future_tcp_pose_xyzw.npy`; `geometry/branch_selection.json`; `geometry/selected_hand_eye_X.npy`; `geometry_forecasts_side_by_side.mp4`; `metrics.json`; `obs/`; `obs/tcp_pose`; `observed/observed_tracks_2d.npz`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `predictions/future_3d.npy`; `predictions/model_run.json`; `primary_prediction_vs_real.mp4`; `source_indices.npy`; `wrist_2/branches`; `wrist_2/metadata.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_finish_observed.py`

Run remaining observed-only steps and keep a failure receipt.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import subprocess`; `from fmb_wrist_prepare import ROOT, RUN, write`
Constants: 
Input/output file references: `.venv-vipe/bin/python`; `geometry_status.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_future_crossview.py`

Descriptive world-frame motion replication, with distinct visible point sets.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_math import tcp_matrices, transform, backproject, sample_z`; `from fmb_wrist_evaluate import prediction_gate, TIMES`
Constants: 
Input/output file references: `Estimated base-frame point displacement, not calibrated motion GT. Different surfaces/point sets, rotation, frozen uncertain X/K/scale and asynchronous capture confound replication.`; `evaluation/estimated_world_point_displacements.npy`; `evaluation/future_alltracker.npz`; `evaluation/future_reference.npz`; `evaluation/future_sensor_depth.npy`; `evaluation/future_tcp_pose_xyzw.npy`; `geometry/selected_hand_eye_X.npy`; `wrist_future_world_motion.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_geometry.py`

Compare observed geometry, calibrate hand-eye, and export identical H3 IDs.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `load_vipe`, `plot_depth`, `diagnose`, `export`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, io, json, shutil, zipfile`; `from pathlib import Path`; `import cv2, numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`; `from fmb_wrist_math import *`; `import berkeley_preprocess as b`; `from fmb_v2_preprocess import author_filter`
Constants: 
Input/output file references: ` common valid points across A/B/C, cannot silently reduce scope`; `.npz`; `Computational payload/coordinate integrity only; physical validation gate separately recorded`; `K.npy`; `Observed RGB color target/board and disclosed coarse robot/background polygons; target t0 uses `; `Observed RGB-region medians; different pixels/surfaces across moving-camera frames`; `c2w.npy`; `depth.npy`; `filter_diagnostics.npz`; `geometry/K.npy`; `geometry/branch_selection.json`; `geometry/camera_poses.npy`; `geometry/depth_comparison.json`; `geometry/geometry_diagnostics.json`; `geometry/hand_eye.json`; `geometry/hand_eye_X.npy`; `geometry/hand_eye_Y.npy`; `geometry/hand_eye_bootstrap.json`; `geometry/independent_robot_rgbd_X.npy`; `geometry/independent_robot_rgbd_hand_eye.json`; `geometry/input_audit.json`; `geometry/official_K.npy`; `geometry/official_profile_validation.json`; `geometry/points_3d_filtered.npy`; `geometry/points_3d_raw.npy`; `geometry/rejected_canonical_vipe_hand_eye_static.json`; `geometry/selected_hand_eye.json`; `geometry/selected_hand_eye_X.npy`; `geometry/selected_hand_eye_Y.npy`; `geometry/static_correspondences.json`; `geometry/tcp_camera_poses_in_vipe_world.npy`; `geometry/unidepth_K.npy`; `geometry/vipe_K.npy`; `geometry/vipe_camera_poses.npy`; `geometry/vipe_depth.npy`; `geometry/vipe_source_selection.json`; `metadata.json`; `observed/historical_masks_h3.npy`; `observed/mask.png`; `observed/native_depth.npy`; `observed/observed_tracks_2d.npz`; `observed/points_2d_history.npy`; `observed/points_3d_history.npy`; `observed/rgb.npy`; `observed/selected_point_ids.npy`; `observed/sensor_z16.npy`; `observed/source_indices.npy`; `observed/tcp_pose_xyzw.npy`; `point_ids.npy`; `points_2d_at_t0.npy`; `points_3d_filtered.npy`; `points_3d_history.npy`; `points_3d_raw.npy`; `sources/official_K_candidates.json`; `viz/camera_paths_3d.png`; `viz/depth_comparison.png`; `viz/pose_K_comparison.png`; `viz/selected_24_points.png`; `viz/selected_history_tracks.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_ground.py`

Native observed-image MolmoPoint grounding with wrist camera provenance.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `prepare_scene`.
Imports/reused functions: `import fmb_v2_ground as adapted`; `import berkeley_ground_molmopoint as original`; `from berkeley_preprocess import read_json`
Constants: 
Input/output file references: `metadata.json`; `observed/molmopoint_preparation.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_inspect_vipe.py`

Inspect actual ViPE IDs and pose calibration before downstream model work.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json, zipfile`; `import numpy as np`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_geometry import load_vipe`; `from fmb_wrist_math import tcp_matrices, hand_eye, static_pairs, static_reprojection`
Constants: 
Input/output file references: `/observed_10hz.npz`; `observed/tcp_pose_xyzw.npy`; `vipe_default/`; `vipe_default/pose/observed_10hz.npz`; `vipe_early_hand_eye_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_math.py`

Explicit c2w moving-camera geometry and observed-only AX=XB calibration.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: transform, project, from_anchor.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `tcp_matrices`, `transform`, `backproject`, `project`, `to_anchor`, `from_anchor`, `sample_z`, `lift`, `mean_transform`, `hand_eye`, `static_pairs`, `static_reprojection`.
Imports/reused functions: `from __future__ import annotations`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`
Constants: 
Input/output file references: `G_i X = Y C_i, with G and C both camera/EE-to-world transforms.`; `Observed LK correspondences on blue board/gray background, excluding camera-fixed gripper.`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_native_control.py`

Retain quantitative unmodified-square ViPE control without changing primary inputs.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json`; `import numpy as np`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_geometry import load_vipe`; `from fmb_wrist_math import tcp_matrices, hand_eye, static_reprojection, static_pairs`
Constants: 
Input/output file references: `geometry/native_square_control.json`; `geometry/official_K.npy`; `observed/rgb.npy`; `observed/sensor_z16.npy`; `observed/tcp_pose_xyzw.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_observed_audit.py`

Preservation receipts and image-only observed inspection material.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2, numpy as np`; `import matplotlib`; `from fmb_wrist_prepare import ROOT, RUN, CAMERAS, write, sha`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `observed/rgb.npy`; `observed/sensor_z16.npy`; `observed/source_indices.npy`; `preserved_prior_runs.json`; `report/fmb_v2_berkeley_matched.md`; `runs/fmb_v2_berkeley_matched/completion_audit.json`; `runs/fmb_v2_berkeley_matched/preserved_v1.json`; `runs/sharerobot_fmb_episode_5201/preflight.json`; `source_episode_5201_provenance.json`; `source_fmb_control_provenance.json`; `viz/observed_prefix_review.png`; `viz/registration_review.png`; `wrist_2/metadata.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_prepare.py`

Observed-only FMB wrist selection, source receipts, and causal exports.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `write`, `sha`, `sources`, `inspect`, `export`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, hashlib, json, os, urllib.request`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_WRIST_RUN', str(ROOT / 'runs/fmb_wrist_v3')))`; `DATA=ROOT.parent / 'data/fmb/single_object_manipulation_dataset'`; `CAMERAS=['side_1', 'side_2', 'wrist_1', 'wrist_2']`
Input/output file references: `*.npy`; `/`; `/commits/main`; `/envs/`; `/git/trees/`; `1_M_L_3_vertical_n_2.npy`; `CANDIDATE: requires RGB profile and actual crop/resize validation`; `calibration/static/files`; `camera/`; `candidates.json`; `data/fmb/single_object_manipulation_dataset`; `functional-manipulation-benchmark/functional-manipulation-benchmark.github.io`; `history_rgb.npy`; `history_source_indices.npy`; `https://api.github.com/repos/`; `https://raw.githubusercontent.com/`; `metadata.json`; `obs/`; `obs/tcp_pose`; `observed_10hz.mp4`; `official_K_candidates.json`; `rail-berkeley/fmb`; `rgb.npy`; `runs/fmb_wrist_v3`; `sensor_z16.npy`; `source_indices.npy`; `source_receipts.json`; `static/files/`; `tcp_pose_xyzw.npy`; `timestamps.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_prepare_control.py`

Observed-selected better-excited control; preserve primary 5201 diagnostics.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json, os, shutil`; `from pathlib import Path`; `import fmb_wrist_prepare as prepare`
Constants: 
Input/output file references: `../fmb_wrist_v3/wrist_2/geometry/independent_robot_rgbd_bootstrap.json`; `1_M_L_3_horizontal_n_0.npy`; `control_selection.json`; `runs/fmb_wrist_v3_control_horizontal_n0`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_preprocess.py`

Frozen author models on the observed wrist prefix; no future file loading.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sam`, `track`, `unidepth`, `h3_masks`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, gc, json, sys`; `from pathlib import Path`; `import cv2, numpy as np`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `configs/sam2.1/sam2.1_hiera_l.yaml`; `lpiccinelli/unidepth-v2-vits14`; `models/sam2.1_hiera_large.pt`; `models/unidepth-v2-vits14`; `models/unidepth-v2-vits14/config.json`; `observed/historical_masks_h3.npy`; `observed/mask.png`; `observed/molmopoint_grounding.json`; `observed/observed_tracks_2d.npz`; `observed/query_points_100.npy`; `observed/rgb.npy`; `observed/source_indices.npy`; `third_party/UniDepth`; `third_party/sam2`; `unidepth_K.npy`; `unidepth_depth.npy`; `unidepth_local_ids.npy`; `unidepth_receipt.json`; `viz/candidate_history_tracks.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_rank_candidates.py`

Rank observed candidates by independently held-out RGBD/TCP calibration.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `from pathlib import Path`; `import numpy as np`; `import fmb_wrist_robot_rgbd_calibration as calibration`; `from fmb_wrist_prepare import ROOT, RUN, DATA, write`
Constants: 
Input/output file references: `Rank observed candidates by independently held-out RGBD/TCP calibration.`; `geometry/independent_robot_rgbd_bootstrap.json`; `geometry/independent_robot_rgbd_hand_eye.json`; `obs/`; `obs/tcp_pose`; `observed/rgb.npy`; `observed/sensor_z16.npy`; `observed/tcp_pose_xyzw.npy`; `selection/candidates.json`; `selection/geometry_candidates`; `selection/geometry_ranking.json`; `sources/official_K_candidates.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_record_blinded_review.py`

Record actual first built-in future inspection before revealing method identities.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `seen`.
Imports/reused functions: `from datetime import datetime, timezone`; `import numpy as np`; `from fmb_wrist_prepare import RUN, write, sha`
Constants: 
Input/output file references: `/viz/reference_all24_review.png`; `All five forecasts are still in frame. Slots2/3 appear close to reference, with modest point offsets. Slots1/4/5 already shift head points upward. No physical3D ranking inferred.`; `First qualitative judgments saved before method-map/metric disclosure`; `Only sampled RGB/projection images; cannot establish calibrated3D GT or exact identities on a low-texture object. Prior reconstruction hypotheses were known, so blinding hides labels but is not fully independent.`; `Red surface near gripper/boundary; sampled membership appears correct, but contact/occlusion risk remains`; `builtin_blinded_review.json`; `observed/selected_point_ids.npy`; `wrist_1/viz/blinded_review_20.png`; `wrist_2/viz/blinded_review_01.png`; `wrist_2/viz/blinded_review_10.png`; `wrist_2/viz/blinded_review_20.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_record_future_review.py`

Serialize the assistant's completed built-in image review, preserving first judgments.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `seen`.
Imports/reused functions: `import json`; `from datetime import datetime, timezone`; `from fmb_wrist_prepare import RUN, write, sha`
Constants: 
Input/output file references: `A has a large persistent absolute3D offset while looking much better in2D. B/C overlap Static closely; C CV has smaller2D error but larger3D_est error. B CV projections grow extremely large. Ranking differs between2D and3D_est.`; `Model3D_est error grows to about343mm at2s, well above Static/CV throughout the horizon. The2D curve has a large projection spike above7000px around1.6s; these off-image projections remain in numerical metrics.`; `Visible red peg moves downward toward insertion in the fixed view. The colored mask follows the visible surface in sampled frames126/127/136/146, with top clipping at126 and changing visible contour. This is an approximate centroid control.`; `builtin_blinded_review.json`; `builtin_visual_review_future.json`; `evaluation/blinding_map.json`; `side_1/viz/side_future_color_control.png`; `side_2/viz/side_future_color_control.png`; `wrist_1/viz/forecast_errors.png`; `wrist_1/viz/full_30step_xyz.png`; `wrist_2/viz/forecast_errors.png`; `wrist_2/viz/full_30step_xyz.png`; `wrist_2/viz/reference_hand_eye_sensitivity.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_record_observed_review.py`

Serialize the assistant's completed built-in image review; never run a HF VLM.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `reviewed`.
Imports/reused functions: `import json`; `from datetime import datetime, timezone`; `import numpy as np`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`
Constants: 
Input/output file references: `/viz/selected_24_points.png`; `Controlled estimated-geometry comparison; A/B rejected as diagnostics, C provisional within-view primary`; `Qualitative review, single previously studied episode, concentrated wrist2 point coverage, depth scale/registration and hand-eye uncertain; no certified 3D GT.`; `ViPE camera path shows depth-direction spikes and >30 degree suffix rotation residual. ViPE focal lengths are much larger than official/UniDepth. Selected independent TCP path disagrees with ViPE; do not claim matching calibrated poses.`; `builtin_visual_review_observed.json`; `geometry/K.npy`; `geometry/branch_selection.json`; `geometry/camera_poses.npy`; `geometry/official_K.npy`; `geometry/selected_hand_eye.json`; `geometry/selected_hand_eye_X.npy`; `geometry/vipe_source_selection.json`; `metadata.json`; `observed/history_rgb.npy`; `observed/points_3d_history.npy`; `observed/selected_point_ids.npy`; `side_1/viz/crossview_board_matches.png`; `wrist_1/viz/intrinsics_observed_frames.png`; `wrist_1/viz/pose_K_comparison.png`; `wrist_1/viz/selected_history_tracks.png`; `wrist_2/viz/depth_comparison.png`; `wrist_2/viz/history_3d_clouds.png`; `wrist_2/viz/pose_K_comparison.png`; `wrist_2/viz/selected_history_tracks.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_rectified_vipe.py`

Restore recorded 4:3 aspect before default ViPE; keep source 10Hz and raw model RGB.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, json, os, resource, subprocess, time`; `from pathlib import Path`; `import cv2, numpy as np, imageio.v2 as imageio`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`
Constants: 
Input/output file references: `.cache/huggingface`; `.cache/torch`; `.venv-vipe/bin/python`; `/proc`; `data_generation/third_party/vipe`; `fx_model=fx; fy_model=fy*256/192; cx_model=cx; cy_model=(cy+.5)*256/192-.5`; `geometry/vipe_source_selection.json`; `observed/aspect_rectified`; `observed/rgb.npy`; `observed_10hz.mp4`; `preprocess_status.json`; `vipe_rectified_default_receipt.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_reference_uncertainty.py`

Propagate frozen observed calibration bootstrap into evaluation-only references.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_math import tcp_matrices, backproject, sample_z, transform`; `from fmb_wrist_evaluate import prediction_gate, align_prediction, scores`
Constants: 
Input/output file references: `/`; `/CV`; `/Static`; `future_alltracker.npz`; `future_reference.npz`; `future_sensor_depth.npy`; `future_tcp_pose_xyzw.npy`; `geometry/independent_robot_rgbd_bootstrap.json`; `hand_eye_reference_sensitivity.json`; `observed/points_3d_history.npy`; `predictions/future_3d.npy`; `viz/reference_hand_eye_sensitivity.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_report.py`

Generate the final research report from actual outputs and built-in reviews.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `fmt`, `build`.
Imports/reused functions: `import json`; `import numpy as np`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`
Constants: 
Input/output file references: ` partial contour control](../runs/fmb_wrist_v3/`; `![Observed wrist clouds](../runs/fmb_wrist_v3/wrist_crossview_base_clouds.png)`; `![Side/wrist motion magnitude](../runs/fmb_wrist_v3/side_future_motion_crosscheck.png)`; `![Wrist world motion replication](../runs/fmb_wrist_v3/wrist_future_world_motion.png)`; `![Кандидаты, четыре камеры](../runs/fmb_wrist_v3/selection/all_candidates.png)`; `**. Global physical certification: **False**. A/B — диагностические controls; wrist_1 повторяет C.`; `/`; `/480 2D samples. Off-image/behind-camera forecasts сохраняются в numerical errors. Static/CV используют тот же H3 и геометрию своей ветки.`; `/480 3D и `; `/CV`; `/Static`; `/evaluation/metrics.json) · [Side-by-side видео](../runs/fmb_wrist_v3/`; `/geometry/depth_comparison.json) · [Bootstrap X](../runs/fmb_wrist_v3/`; `/geometry/independent_robot_rgbd_bootstrap.json) · [Square ViPE control](../runs/fmb_wrist_v3/`; `/geometry/native_square_control.json)`; `/observed/`; `/viz/`; `/viz/geometry_forecasts_side_by_side.mp4) · [Primary vs real](../runs/fmb_wrist_v3/`; `/viz/primary_prediction_vs_real.mp4)`; `/viz/side_future_color_control.png)`; `Absolute ADE3D_est остаётся основной метрикой и включает исходный geometry offset. Displacement-only диагностика вычитает собственныйt0 каждого forecast и reference, чтобы отдельно показать ошибку движения. Она не исправляет scale/pose uncertainty.`; `Observed wrist board nearest-surface median/p90 disagreement `; `Robot/background regions обозначены грубо по RGB. Temporal std региональных медиан включает смену поверхности и camera motion; это описательная вариация, не чистый sensor noise.`; `[Future review и hashes](../runs/fmb_wrist_v3/builtin_visual_review_future.json)`; `[Observed four-view audit](../runs/fmb_wrist_v3/crossview_observed_audit.json)`; `[Observed visual review](../runs/fmb_wrist_v3/builtin_visual_review_observed.json) · [Первая blinded future оценка](../runs/fmb_wrist_v3/builtin_blinded_review.json)`; `[Все depth regions и temporal std](../runs/fmb_wrist_v3/`; `[Геометрическое ранжирование](../runs/fmb_wrist_v3/selection/geometry_ranking.json) · [Depth support](../runs/fmb_wrist_v3/selection/target_depth_support.json) · [Provenance5201](../runs/fmb_wrist_v3/source_episode_5201_provenance.json)`; `[Метрики](../runs/fmb_wrist_v3/`; `[Новые видео: галерея](../runs/fmb_wrist_v3/index.html) · [Протокол](../runs/fmb_wrist_v3/PROTOCOL.md) · [Воспроизведение](../runs/fmb_wrist_v3/REPRODUCE.md) · [Pinned sources](../runs/fmb_wrist_v3/sources/source_receipts.json)`; `[Полный completion audit](../runs/fmb_wrist_v3/completion_audit.json)`; `](../runs/fmb_wrist_v3/`; `builtin_visual_review_future.json`; `completion_audit.json`; `crossview_observed_audit.json`; `evaluation/hand_eye_reference_sensitivity.json`; `evaluation/metrics.json`; `geometry/branch_selection.json`; `geometry/depth_comparison.json`; `geometry/geometry_diagnostics.json`; `geometry/hand_eye.json`; `geometry/independent_robot_rgbd_bootstrap.json`; `geometry/independent_robot_rgbd_hand_eye.json`; `geometry/native_square_control.json`; `report/fmb_wrist_v3.md`; `requirements_progress.json`; `side_future_crosscheck.json`; `wrist_1/evaluation/metrics.json`; `wrist_2/evaluation/metrics.json`; `wrist_future_world_motion.json`; `| AX=XB heldout median/p90 rotation | `; `| AX=XB heldout median/p90 translation | `; `| Geometry forecast | Initial reference offset mm | Displacement-only ADE/FDE mm |`; `| Geometry | Static median/p90 px | H3 rigidity variation | Local gate |`; `| Independent X heldout median/p90 reprojection | `; `| Method | ADE/FDE 3D_est mm | ADE/FDE 2D px | Outside | Amplitude ratio |`; `| Region | Valid px | median/p90 absΔZ m | ViPE/sensor |`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_resample_candidates.py`

Observed-only depth-supported author KMeans retry; preserve the initial 100.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json, shutil`; `import cv2, numpy as np`; `from fmb_wrist_prepare import RUN, write, sha`; `from fmb_wrist_math import sample_z`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `*/predictions/model_run.json`; `SAM intersect observed t0 sensor/ViPE local depth support`; `alltracker_execution.json`; `candidate_resampling.json`; `geometry/branch_selection.json`; `geometry/initial_100_candidates`; `geometry/vipe_depth.npy`; `observed_tracks_2d.npz`; `query_points_100.npy`; `rgb.npy`; `sensor_z16.npy`; `source_indices.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_resume_geometry.py`

Wait for verified observed preprocessing, then run causal geometry checks.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import argparse, json, subprocess, time`; `from pathlib import Path`; `from fmb_wrist_prepare import ROOT, RUN, write`
Constants: 
Input/output file references: `.venv-vipe/bin/python`; `/proc`; `geometry_status.json`; `preprocess_status.json`; `scripts/fmb_wrist_crossview.py`; `scripts/fmb_wrist_geometry.py`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_robot_rgbd_calibration.py`

Independent observed RGBD/TCP hand-eye control, without ViPE poses/depth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, json`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_math import tcp_matrices, static_pairs, sample_z, backproject, project, transform`
Constants: 
Input/output file references: `Independent observed RGBD/TCP hand-eye control, without ViPE poses/depth.`; `Observed conservative board/gray-background LK FB and homography filters`; `Strong RGB/depth registration and static-match assumptions. Limited rotation diversity; factory RGB K and units remain uncertain. This estimated X is an independent observed control, not measured hand-eye.`; `geometry/independent_robot_rgbd_X.npy`; `geometry/independent_robot_rgbd_bootstrap.json`; `geometry/independent_robot_rgbd_hand_eye.json`; `independent RGBD/TCP X`; `observed/rgb.npy`; `observed/sensor_z16.npy`; `observed/tcp_pose_xyzw.npy`; `sources/official_K_candidates.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_run_forecasts.py`

Serialize this experiment with other live GPU work, then gate future evaluation.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `busy`.
Imports/reused functions: `import os, subprocess, time`; `from pathlib import Path`; `from fmb_wrist_prepare import ROOT, RUN, write`
Constants: 
Input/output file references: `.venv/bin/python`; `/proc`; `Model/evaluation outputs ready; built-in blinded image review pending`; `[0-9]*/cmdline`; `forecast_pipeline_status.json`; `wrist_1/branches/C_sensor_tcp_official`; `wrist_2/branches`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_selection_probe.py`

Explain common observed eligibility without changing gates.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import cv2, json`; `import numpy as np`; `from fmb_wrist_prepare import RUN, write`
Constants: 
Input/output file references: `filter_diagnostics.npz`; `geometry/initial_100_eligibility.json`; `observed/historical_masks_h3.npy`; `observed/mask.png`; `observed/observed_tracks_2d.npz`; `points_3d_filtered.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_side_future.py`

Evaluation-only side controls: partial-contour caveats and motion magnitude.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json`; `import cv2, numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import RUN, write`; `from fmb_wrist_crossview import rgb_masks, scene_points`; `from fmb_wrist_evaluate import prediction_gate, TIMES`; `from fmb_wrist_math import transform`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `evaluation/estimated_base_target_centroids.npy`; `evaluation/future_alltracker.npz`; `evaluation/future_reference.npz`; `evaluation/future_rgb.npy`; `evaluation/future_sensor_depth.npy`; `evaluation/source_indices.npy`; `evaluation/target_color_centroids.npy`; `geometry/estimated_side_c2base.npy`; `side_future_crosscheck.json`; `sources/official_K_candidates.json`; `viz/side_future_color_control.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_target_probe.py`

Check target sensor support before choosing a wrist hybrid scene.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json`; `import cv2, numpy as np`; `from fmb_wrist_prepare import ROOT, RUN, DATA, write`; `from fmb_wrist_math import sample_z`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `/`; `obs/`; `selection/candidates.json`; `selection/target_depth_support.json`; `selection/target_depth_support.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_audit.py`

Prove completion against actual outputs and the unchanged earlier experiment.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `run`.
Imports/reused functions: `import json, re`; `from datetime import datetime, timezone`; `from pathlib import Path`; `import cv2, numpy as np, torch`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_math import tcp_matrices`; `from fmb_v2_infer import strict_parse`; `from fmb_wrist_v4_diagnose import BASE, OUT`; `from fmb_wrist_v4_prepare import forecast_tcp, MODEL_TIMES`; `from fmb_wrist_v4_refine import predict`; `from fmb_wrist_v4_evaluate import align`
Constants: 
Input/output file references: `/`; `/group`; `24 frozen IDs and finite H1/H3`; `C_sensor_tcp_official/CV`; `C_sensor_tcp_official/Static`; `Exact original reference SHA/masks`; `Final report and all evidence/video links resolve`; `Test/code hashes`; `anchor.npy`; `builtin_visual_review.json`; `calibration_ranking_sensitivity.json`; `code_snapshots/fmb_wrist_v4_infer_initial.py`; `complete30/32 step plots exist`; `completion_audit.json`; `coordinate_tests.json`; `data/checkpoints`; `evaluation/future_reference.npz`; `evaluation/metrics.json`; `future_3d.npy`; `geometry/official_K.npy`; `geometry/selected_hand_eye_X.npy`; `history_rgb.npy`; `history_xyz.npy`; `hybrid_correction.npy`; `inference_status.json`; `input_spec.json`; `media_receipt.json`; `metrics.json`; `model.pt`; `model_run.json`; `motion_policy_amendment_v2.json`; `observed/history_rgb.npy`; `observed/selected_point_ids.npy`; `observed/source_indices.npy`; `observed/tcp_pose_xyzw.npy`; `observed_diagnosis.json`; `observed_reference.npz`; `point_ids.npy`; `predictions_and_projections.npz`; `preregistration.json`; `preserved_v3.json`; `processor_inputs.pt`; `report/fmb_wrist_v3.md`; `report/fmb_wrist_v4_improvement.md`; `robot_observed_model_selection.json`; `runtime_interruption_00/receipt.json`; `same X/K/24 point IDs`; `scripts/fmb_wrist_v4_infer.py`; `selected_motion_policy_v2_15hz.npy`; `viz/forecast_comparison.mp4`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_diagnose.py`

Review v3 and preregister improvement experiments using observed inputs only.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json`; `from datetime import datetime, timezone`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_math import tcp_matrices, transform, sample_z, backproject`
Constants: `BASE=ROOT / 'runs/fmb_wrist_v3'`; `OUT=ROOT / 'runs/fmb_wrist_v4_improvement'`
Input/output file references: `branches/C_sensor_tcp_official/observed/points_3d_history.npy`; `denoised_rigid_H3.npy`; `geometry/official_K.npy`; `geometry/selected_hand_eye_X.npy`; `observed/observed_tracks_2d.npz`; `observed/selected_point_ids.npy`; `observed/sensor_z16.npy`; `observed/tcp_pose_xyzw.npy`; `observed_diagnosis.json`; `observed_reference.npz`; `preserved_v3.json`; `report/fmb_wrist_v3.md`; `runs/fmb_wrist_v3`; `runs/fmb_wrist_v4_improvement`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_evaluate.py`

Evaluate all frozen improvement variants on the unchanged v3 masks/reference.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: align.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `align`, `run`.
Imports/reused functions: `import argparse, json`; `from datetime import datetime, timezone`; `import numpy as np`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_math import from_anchor, project`; `from fmb_wrist_evaluate import scores`; `from fmb_wrist_v4_diagnose import BASE, OUT`
Constants: `TIMES=np.arange(1, 21) / 10`
Input/output file references: `C_sensor_tcp_official/CV`; `C_sensor_tcp_official/Static`; `Evaluate all frozen improvement variants on the unchanged v3 masks/reference.`; `branches/C_sensor_tcp_official/observed/points_3d_history.npy`; `branches/C_sensor_tcp_official/predictions/future_3d.npy`; `evaluation/future_reference.npz`; `future_3d.npy`; `history_xyz.npy`; `hybrid_correction.npy`; `inference_status.json`; `input_spec.json`; `metrics.json`; `motion_policy_amendment_v2.json`; `observed_diagnosis.json`; `observed_reference.npz`; `predictions_and_projections.npz`; `preregistration.json`; `robot_attached_forecast_15hz.npy`; `selected_motion_policy_v2_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_infer.py`

Run preregistered native neural ablations, one GPU workload at a time.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `wait_gpu`, `run`.
Imports/reused functions: `import argparse, json, os, resource, subprocess, time, traceback`; `from datetime import datetime, timezone`; `import numpy as np`; `import torch`; `from PIL import Image`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_v4_diagnose import OUT`; `from fmb_v2_infer import strict_parse`
Constants: 
Input/output file references: `.cache/huggingface`; `.cache/torch`; `Inspect/archive incomplete attempts explicitly before resumption`; `anchor.npy`; `data/checkpoints`; `future_3d.npy`; `history_rgb.npy`; `history_xyz.npy`; `inference_status.json`; `input_ids.npy`; `input_spec.json`; `model_run.json`; `preregistration.json`; `processor_inputs.pt`; `query_uv.npy`; `scripts/fmb_wrist_v4_infer.py`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_interrupt_runtime.py`

Preserve this run's slow partial call before an explicit CPU-thread correction.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json, os, signal, time, shutil`; `from datetime import datetime, timezone`; `from pathlib import Path`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_v4_diagnose import OUT`
Constants: 
Input/output file references: `/proc`; `First native call took692s; explicit resumption with4 CPU threads, matching the earlier continuation configuration. Frozen scientific inputs/checkpoint/greedy decoding unchanged.`; `code_snapshots/fmb_wrist_v4_infer_initial.py`; `inference_status.json`; `model_run.json`; `receipt.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_media.py`

Render comparable original/new forecasts and a local video gallery.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import argparse, json`; `import cv2, imageio.v2 as imageio, numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_v4_diagnose import BASE, OUT`
Constants: 
Input/output file references: `"></video><p><a href="`; `/8`; `/viz/`; `</h2><video controls src="`; `</html>`; `Render comparable original/new forecasts and a local video gallery.`; `comparison_20.png">Кадр через2s</a></p></article>`; `evaluation/future_reference.npz`; `evaluation/future_rgb.npy`; `forecast_comparison.mp4`; `future_3d.npy`; `media_receipt.json`; `metrics.json`; `motion_policy_amendment_v2.json`; `predictions_and_projections.npz`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_prepare.py`

Freeze neural ablations and observed-selected robot-aware forecasts before scoring.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: forecast_tcp.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `forecast_tcp`, `run`.
Imports/reused functions: `import json`; `from datetime import datetime, timezone`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_math import transform`; `from fmb_wrist_v4_diagnose import BASE, OUT`
Constants: `MODEL_TIMES=np.arange(1, 31) / 15`
Input/output file references: `*.npy`; `Reuse exactly frozen v3 independent AllTracker/sensor/TCP reference and common masks; no calibration or reference changes`; `branches/C_sensor_tcp_official/observed/points_3d_history.npy`; `camera_t0; estimated meters; unchanged official K/frozen observed hand-eye X`; `denoised_rigid_H3.npy`; `history_rgb.npy`; `history_xyz.npy`; `input_spec.json`; `metadata.json`; `observed/history_rgb.npy`; `observed/points_2d_history.npy`; `observed_reference.npz`; `point_ids.npy`; `preregistration.json`; `query_uv.npy`; `robot_attached_forecast_15hz.npy`; `robot_observed_model_selection.json`; `wrist_2/observed_reference.npz`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_progress.py`

Report only fully completed neural variants without changing selection.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json, numpy as np`; `from fmb_wrist_prepare import write`; `from fmb_wrist_v4_diagnose import BASE, OUT`; `from fmb_wrist_v4_evaluate import align`; `from fmb_wrist_evaluate import scores`
Constants: 
Input/output file references: `evaluation/future_reference.npz`; `future_3d.npy`; `history_xyz.npy`; `model_run.json`; `neural_progress.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_record_review.py`

Record the assistant's actual built-in image inspection from this session.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `from datetime import datetime, timezone`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_v4_diagnose import OUT`
Constants: 
Input/output file references: `At1s robot and hybrid displayed forecasts are all in image, mostly near the cap with visible offsets. Original H3, cleaned H3 and H1 each show7/8 off-image, unlike the actual reference points.`; `At1s robot/hybrid stay inside the frame and nearer the red face, though several forecasts lie on gripper/board. Original H3 has5/8, cleaned H3 has6/8 and H1 has3/8 displayed off-image points.`; `Cleaned H3 groups behave inconsistently: some nearly flat, another drifts strongly inY/Z with Z crossing zero. H1 is mostly stationary after minor early changes. Full30/32-step extents preserve these failures.`; `Final six-panel comparison retains the same robot endpoint limitation. Robot and bounded hybrid each have7/8 displayed off-image points; cleaned H3 and H1 have8/8. No visually accurate2s endpoint is demonstrated.`; `H1 removes the enormous original/cleaned H3 projection spikes, while staying close to Static in3D_est. Robot/hybrid errors are substantially lower, but still rise at the end. Full axes retain all large errors.`; `New H3 remains worse than original in both dimensions; H1 almost follows original/Static. Robot/hybrid3D_est curves are lower and nearly overlap. CV retains better final2D error than robot despite worse overall2D ADE.`; `One already studied episode; comparisons show8/24 points, numerical errors and native plots cover all24. Green low-texture correspondences are estimates, not certified material point identities.`; `Original H3 has vertically stretched, largely off-image trajectories. All eight shown robot forecasts remain inside the image but shift right onto gripper/board instead of the red target.`; `Robot and hybrid stay in image but drift right away from target; hybrid is closer. Native H1 is less numerically catastrophic than original H3 but7/8 displayed points are off-image at2s. Cleaned H3 has8/8 off-image.`; `builtin_visual_review.json`; `inference_status.json`; `wrist_1/viz/comparison_10.png`; `wrist_1/viz/comparison_20.png`; `wrist_1/viz/errors.png`; `wrist_1/viz/full_neural_outputs.png`; `wrist_1/viz/provisional_robot_comparison_20.png`; `wrist_2/viz/comparison_10.png`; `wrist_2/viz/comparison_20.png`; `wrist_2/viz/errors.png`; `wrist_2/viz/full_neural_outputs.png`; `wrist_2/viz/provisional_robot_comparison_20.png`; `wrist_2/viz/provisional_robot_errors.png`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_refine.py`

Compare rigid-centroid translation with full TCP-pose extrapolation on observed data.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: Selected functions borrowed unchanged: centroid_forecast, predict.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `centroid_forecast`, `predict`, `run`.
Imports/reused functions: `import json`; `from datetime import datetime, timezone`; `import numpy as np`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_math import transform`; `from fmb_wrist_v4_diagnose import OUT`; `from fmb_wrist_v4_prepare import forecast_tcp, MODEL_TIMES`
Constants: 
Input/output file references: `First built-in provisional comparison still shows endpoint drift despite lower ADE. Compare a simpler rigid-centroid motion forecast against full TCP rotation/lever-arm extrapolation.`; `Only observed77..126 target points and TCP, fixed original K/X; no127..146 files opened by this selector`; `motion_policy_amendment_v2.json`; `observed_reference.npz`; `selected_motion_policy_v2_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_report.py`

Write the evidence-grounded improvement report, keeping v3 untouched.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `read`, `run`.
Imports/reused functions: `import json`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_v4_diagnose import OUT`
Constants: 
Input/output file references: `![Все24 native trajectories, полный30/32-step range](../runs/fmb_wrist_v4_improvement/`; `![Ошибки во времени](../runs/fmb_wrist_v4_improvement/`; `![Сравнение через2s](../runs/fmb_wrist_v4_improvement/`; `/`; `/480 2D samples. Off-image прогнозы остаются в numerical errors. Future camera poses используются только для проекции и reference.`; `/480 3D samples и`; `/7 вариантах, ниже Static в `; `/7 и ниже CV в `; `/7. Меняется только reference; прогноз остаётся при исходном X. Это описательная проверка чувствительности, а не доверительный интервал либо полная совместная неопределённость.`; `/calibration_ranking_sensitivity.json)`; `/metrics.json) · [Видео](../runs/fmb_wrist_v4_improvement/`; `/viz/comparison_20.png)`; `/viz/errors.png)`; `/viz/forecast_comparison.mp4)`; `/viz/full_neural_outputs.png)`; `[Observed TCP selection](../runs/fmb_wrist_v4_improvement/robot_observed_model_selection.json) · [Исправленная target-motion selection](../runs/fmb_wrist_v4_improvement/motion_policy_amendment_v2.json)`; `[Observed diagnosis](../runs/fmb_wrist_v4_improvement/observed_diagnosis.json)`; `[Pinned primary-source receipts](../runs/fmb_wrist_v4_improvement/sources/source_receipts.json)`; `[Reference sensitivity](../runs/fmb_wrist_v4_improvement/`; `[Метрики](../runs/fmb_wrist_v4_improvement/`; `[Новые видео](../runs/fmb_wrist_v4_improvement/index.html) · [Аудит](../runs/fmb_wrist_v4_improvement/completion_audit.json) · [Команды воспроизведения](../runs/fmb_wrist_v4_improvement/REPRODUCE.md)`; `[Оценка и image hashes](../runs/fmb_wrist_v4_improvement/builtin_visual_review.json)`; `[Первичная preregistration](../runs/fmb_wrist_v4_improvement/preregistration.json)`; `builtin_visual_review.json`; `calibration_ranking_sensitivity.json`; `metrics.json`; `motion_policy_amendment_v2.json`; `report/fmb_wrist_v4_improvement.md`; `| Native H1 | Официальный H1-F32, прежние t0 RGB/XYZ и action | Нет |`; `| Robot pose forecast | Экстраполяция прошлой TCP translation/rotation, объект жёстко связан с камерой | Нет |`; `| Метод | ADE/FDE3D_est mm | ADE/FDE2D px | Вне изображения |`; `| Поправка MolmoMotion | Полный нейронный выход оценивается непосредственно | В hybrid общий median displacement ограничен observed p90 drift: 2,41/3,23 мм; форма детали сохраняется |`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_sensitivity.py`

Assess fixed-forecast rankings under seven frozen observed calibration fits.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: .
Imports/reused functions: `import json, numpy as np`; `from fmb_wrist_prepare import write, sha`; `from fmb_wrist_math import tcp_matrices, backproject, sample_z, transform`; `from fmb_wrist_evaluate import scores`; `from fmb_wrist_v4_diagnose import BASE, OUT`
Constants: 
Input/output file references: `Descriptive reference-only sensitivity, not a confidence interval or full joint calibration/prediction uncertainty. Robot forecasts retain their originally frozen X; no new neural inference or calibration fitting.`; `calibration_ranking_sensitivity.json`; `evaluation/future_alltracker.npz`; `evaluation/future_reference.npz`; `evaluation/future_sensor_depth.npy`; `evaluation/future_tcp_pose_xyzw.npy`; `geometry/independent_robot_rgbd_bootstrap.json`; `predictions_and_projections.npz`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_v4_sources.py`

Save current primary-source receipts for the improvement review.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import json, urllib.request`; `from datetime import datetime, timezone`; `from fmb_wrist_prepare import ROOT, write, sha`; `from fmb_wrist_v4_diagnose import OUT`
Constants: 
Input/output file references: `/`; `data_generation/README.md`; `hf_model_inventory.json`; `https://api.github.com/repos/allenai/molmo-motion/commits/main`; `https://huggingface.co/api/models?author=allenai&search=MolmoMotion&sort=lastModified&direction=-1&limit=10`; `https://raw.githubusercontent.com/allenai/molmo-motion/`; `source_receipts.json`; `src/molmo_motion/data/trajectory_3d_dataset.py`; `src/molmo_motion/processor.py`; `upstream_commit.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_video_gallery.py`

Create a local reviewable HTML artifact and real-future camera clips.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `run`.
Imports/reused functions: `import html, json`; `import imageio.v2 as imageio`; `import numpy as np`; `from fmb_wrist_prepare import RUN, write, sha`
Constants: 
Input/output file references: `/viz/geometry_forecasts_side_by_side.mp4">Открыть сравнение вариантов геометрии</a></p></article>`; `/viz/primary_prediction_vs_real.mp4"></video><p><a href="`; `/viz/real_future.mp4"></video></article>`; `: MolmoMotion и reference</h2><video controls preload="metadata" src="`; `: запись FMB, t0 и2s future</h2><video controls preload="metadata" src="`; `</section></html>`; `evaluation/future_rgb.npy`; `video_gallery_receipt.json`; `viz/real_future.mp4`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_vipe.py`

Run official ViPE after currently live independent GPU work finishes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `busy_processes`, `run`.
Imports/reused functions: `from __future__ import annotations`; `import argparse, json, os, resource, subprocess, sys, time`; `from pathlib import Path`; `from fmb_wrist_prepare import ROOT, RUN, write, sha`
Constants: 
Input/output file references: `.cache/huggingface`; `.cache/torch`; `/`; `/proc`; `/scripts/`; `/usr/local/cuda-12.8`; `_receipt.json`; `data_generation/third_party/vipe`; `gpu_wait.json`; `observed/observed_10hz.mp4`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/fmb_wrist_visuals.py`

Observed-only readable plots for the frozen 24 IDs and geometry controls.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `observed`.
Imports/reused functions: `import json`; `import cv2`; `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from fmb_wrist_prepare import RUN, write`; `import berkeley_preprocess as b`
Constants: 
Input/output file references: `geometry/branch_selection.json`; `geometry/intrinsics_variation.json`; `geometry/official_K.npy`; `geometry/unidepth_K.npy`; `geometry/vipe_K.npy`; `metadata.json`; `observed/observed_tracks_2d.npz`; `observed/points_3d_history.npy`; `observed/rgb.npy`; `observed/selected_point_ids.npy`; `observed/source_indices.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/freeze_dobbe_approx_input.py`

Seal the exploratory MolmoMotion input before any future RGB-D is used.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `STUDY=ROOT / 'runs/dobbe_rgbd_study'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `MODEL_PT_SHA256_PREVIOUSLY_VERIFIED='506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205'`
Input/output file references: `allenai/MolmoMotion-4B-H3-F30`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `f2nerf_second/receipt.json`; `input_freeze.json`; `manifest.json`; `model.pt`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `points_3d_history_world_candidate.npy`; `runs/dobbe_rgbd_study`; `runs/dobbe_rgbd_study/approx_history`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/index_dobbe_depth_zip.py`

Index the inner HoNY RGB-D ZIP from only the tail of its final split part.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `confirmed_params`, `range_get`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import collections`; `import json`; `import re`; `import struct`; `import sys`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight'`; `GDOWN_TARGET=ROOT.parent / '.tools' / 'gdown'`; `FOLDER_ID='1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO'`; `BASE='https://drive.usercontent.google.com/download'`
Input/output file references: `/`; `Expected nested ZIP / ZIP64 footer not found`; `Index the inner HoNY RGB-D ZIP from only the tail of its final split part.

The outer split ZIP stores one uncompressed nested ZIP. This script reads its
central directory by HTTP Range and downloads no RGB/depth payload.
`; `dobbe_depth_nested_index.json`; `https://drive.usercontent.google.com/download`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/index_dobbe_pick_place_labels.py`

Index only Pick_and_Place HoNY gripper traces using HTTP ZIP ranges.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import zipfile`; `from pathlib import Path`; `from inspect_remote_zip import HTTPRangeFile`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'data' / 'dobbe_oxe' / 'pick_place_label_index.jsonl'`; `URL='https://dl.dobb-e.com/datasets/homes_of_new_york.zip'`
Input/output file references: `/`; `/labels.json`; `https://dl.dobb-e.com/datasets/homes_of_new_york.zip`; `iphone_data/Pick_and_Place/`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/infer_fmb_metric_depth.py`

Run one pretrained monocular metric model on selected FMB side_1 frames.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `load_model`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import torch`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `EXTERNAL=ROOT.parent / 'third_party'`; `DATA=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `STEPS=(0, 37, 130, 135, 140, 145)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `_metric_predictions.json`; `_metric_predictions.npz`; `models/moge-2-vitl/model.pt`; `models/moge-3-vitl/model.pt`; `models/unidepth-v2-vits14`; `obs/side_1`; `runs/sharerobot_fmb_episode_5201`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/inspect_dobbe_colmap_db.py`

Show strongest wide-baseline verified pairs in an RGB-only COLMAP DB.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import sqlite3`; `from pathlib import Path`
Constants: `P=2147483647`
Input/output file references: 
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/inspect_dobbe_history.py`

Save only the frozen episode_3651 history RGB frames for point inspection.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from inspect_hony_scenes import ROOT, liblzfse`
Constants: 
Input/output file references: `data/dobbe_oxe/target_raw/compressed_np_depth_float32.bin`; `data/dobbe_oxe/target_raw/compressed_video_h264.mp4`; `runs/dobbe_rgbd_study/approx_history`; `runs/dobbe_rgbd_study/protocol.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/inspect_fmb_2d_window.py`

Print red-object visibility and depth statistics for the quantitative FMB window.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `red_component`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import cv2`; `import numpy as np`; `from probe_fmb_cad_pnp import SOURCE`
Constants: 
Input/output file references: `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/inspect_fmb_board_step.py`

List dimensions and CAD vertices of the official FMB assembly boards.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `from OCP.BRep import BRep_Tool`; `from OCP.BRepBndLib import BRepBndLib`; `from OCP.Bnd import Bnd_Box`; `from OCP.STEPControl import STEPControl_Reader`; `from OCP.TopAbs import TopAbs_SOLID, TopAbs_VERTEX`; `from OCP.TopExp import TopExp_Explorer`; `from OCP.TopoDS import TopoDS`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `STEP=ROOT / 'runs/fmb_effective_k_256_calibration/peg_board_official.step'`; `OUT=STEP.with_name('peg_board_official_solids.json')`
Input/output file references: `peg_board_official_solids.json`; `runs/fmb_effective_k_256_calibration/peg_board_official.step`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/inspect_fmb_board_wires.py`

Export top-plane perimeter and hole wires of the official FMB boards.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `bounds`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from OCP.BRep import BRep_Tool`; `from OCP.BRepBndLib import BRepBndLib`; `from OCP.Bnd import Bnd_Box`; `from OCP.STEPControl import STEPControl_Reader`; `from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_WIRE`; `from OCP.TopExp import TopExp_Explorer`; `from OCP.TopoDS import TopoDS`; `from inspect_fmb_board_step import OUT, STEP`
Constants: 
Input/output file references: `peg_board_top_wires.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/inspect_fmb_peg_step.py`

Inspect the 54 solids in the official FMB peg STEP assembly.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `from OCP.BRepBndLib import BRepBndLib`; `from OCP.Bnd import Bnd_Box`; `from OCP.STEPControl import STEPControl_Reader`; `from OCP.BRep import BRep_Tool`; `from OCP.TopAbs import TopAbs_SOLID, TopAbs_VERTEX`; `from OCP.TopExp import TopExp_Explorer`; `from OCP.TopoDS import TopoDS`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `STEP=ROOT / 'runs/sharerobot_fmb_episode_5201/peg_official.step'`; `OUT=STEP.with_name('peg_official_solids.json')`
Input/output file references: `peg_official_solids.json`; `runs/sharerobot_fmb_episode_5201/peg_official.step`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/inspect_hony_scenes.py`

Inspect both selectively extracted HoNY captures before fixing a protocol.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `inspect`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `import liblzfse`
Constants: `ROOT=Path(__file__).resolve().parents[1]`
Input/output file references: `.tools/lzfse`; `compressed_video_h264.mp4`; `data/dobbe_oxe`; `labels.json`; `runs/dobbe_rgbd_study`; `scene_inspection.json`
Related saved experiments (dataset-level): 

## `scripts/inspect_plex_cereal_hdf5.py`

Inspect the small, selectively fetched official PLEX PickPlaceCereal HDF5.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import xml.etree.ElementTree as ET`; `from pathlib import Path`; `import h5py`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT / 'data' / 'plex' / 'PickPlaceCereal_demo_act_norm.hdf5'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'plex_cereal_hdf5_audit.json'`
Input/output file references: `.//*[@file]`; `.//body[@name="Cereal_main"]`; `.//camera[@name="agentview"]`; `plex_cereal_hdf5_audit.json`
Related saved experiments (dataset-level): 

## `scripts/inspect_remote_zip.py`

List a remote ZIP's member names using byte ranges, without fetching its payload.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `import io`; `import sys`; `import zipfile`; `import requests`
Constants: 
Input/output file references: `/`; `List a remote ZIP's member names using byte ranges, without fetching its payload.

Usage: python scripts/inspect_remote_zip.py URL [substring]
`; `bytes 0-0/`
Related saved experiments (dataset-level): 

## `scripts/inspect_second_fmb_window.py`

Inspect candidate 2 s windows in a second raw FMB episode before inference.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy'`; `OUT=ROOT / 'runs/fmb_second_example_1_M_L_3_vertical_n_3'`
Input/output file references: `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `runs/fmb_second_example_1_M_L_3_vertical_n_3`; `window_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/match_dobbe_lerobot_to_hony.py`

Match a LeRobot episode to a HoNY capture by exact gripper time series.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `import pyarrow.parquet as pq`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs' / 'plex_dobbe_preflight'`; `DATA=ROOT / 'data' / 'dobbe_oxe'`
Input/output file references: `IPEC-COMMUNITY/dobbe_lerobot episode_003651`; `dobbe_3651_source_match.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/normalize_fmb_second_assets.py`

Describe official FMB calibration and CAD evidence with explicit provenance.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `save`, `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data/fmb_second_scene'`; `CAMERAS=('side_1', 'side_2', 'wrist_1', 'wrist_2')`; `SERIALS={'side_1': '128422270679', 'side_2': '127122270146', 'wrist_1': '127122270350', 'wrist_2': '128422271851'}`; `PROJECT='https://functional-manipulation-benchmark.github.io/'`; `CODE='https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/'`
Input/output file references: `UNRESOLVED; STEP bodies have no per-body shape/size IDs`; `calibration/calibration.json`; `calibration/raw`; `camera/rs_capture.py`; `data/fmb_second_scene`; `data/fmb_second_scene/object/peg_official.step`; `data/fmb_second_scene/object/shape_color_reference.pdf`; `dataset/index.html`; `envs/franka_fmb_env.py#L113-L116`; `envs/franka_fmb_env.py#L225-L232`; `files/index.html`; `files/index.html#camera-mounts`; `https://functional-manipulation-benchmark.github.io/`; `https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/fmb_dataset_builder/fmb_single_object_dataset/fmb_single_object_dataset_dataset_builder.py`; `https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/`; `metadata.json`; `object/object_metadata.json`; `object/peg_official.step`; `static/doc/FMB%20Shape%20and%20Color%20Number%20Reference%20Sheet%20-%20Google%20Docs.pdf`; `static/files/`; `static/files/peg.step`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_dobbe_rgbd_study.py`

Make the compact evidence figures for the causal HoNY geometry study.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read_json`, `k_stability`, `metric_comparison`, `draw_matches`, `reprojection_overlay`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from validate_dobbe_geometry import OUT, ROOT, CAMERA_TO_LABEL, get_intrinsics, load_scene, unproject`
Constants: 
Input/output file references: ` observed green / projected red`; `/`; `0/`; `COLMAP RGB-only calibration: registered frames / input frames`; `_static_correspondences.json`; `_static_geometry.json`; `summary.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/plot_dobbe_vipe_ablation.py`

Standalone paired-depth figure from sealed observed inputs and predictions.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `plot`.
Imports/reused functions: `import numpy as np`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from dobbe_vipe_v1 import RUN, read, write`
Constants: 
Input/output file references: `A/sift_audited`; `Input/forecast sensitivity only; absolute 3D differences are not prediction error against GT.`; `Same RGB queries / K / c2w; both unsmoothed`; `c2w_t0.npy`; `paired_forecast_summary.json`; `paired_selection_audit.json`; `points_2d_t0.npy`; `points_3d_camera_t0_diagnostic.npy`; `prediction_15hz.npy`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/plot_fmb_ablation_trajectories.py`

Plot actual saved point trajectories for prescribed FMB input perturbations.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from matplotlib.patches import Rectangle`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE`
Constants: `SCENARIOS=('nominal', 'scale_090', 'scale_110', 'perm_keep_anchor', 'perm_new_anchor', 'focal_090', 'focal_110')`; `COLORS={'nominal': '#e15759', 'scale_090': '#f28e2b', 'scale_110': '#b07aa1', 'perm_keep_anchor': '#59a14f', 'perm_new_anchor': '#76b7b2', 'focal_090': '#4e79a7', 'focal_110': '#9c755f', 'GT': 'black'}`
Input/output file references: `gt_2d.npy`; `prediction_2d.npy`; `runs/fmb_ablation_trajectory_scenarios.png`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_annotation_repeatability.py`

Visualize existing blind algorithmic repeat checks without changing the GT.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS`
Constants: 
Input/output file references: `: 5/20 future frames`; `annotation_repeatability.json`; `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_board_calibration.py`

Visual summary of RGB-only board K fit and independent depth check.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT`
Constants: 
Input/output file references: `multi_board_bundle_calibration.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_effective_k_probe.py`

Plot diagnostic K estimates and independent depth residuals.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `params`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import OUT, SOURCE`
Constants: 
Input/output file references: `CAD/PnP Z minus sensor Z (mm)`; `effective_k_rounded_cad_silhouette.json`; `tcp_bottom_k_probe.json`; `tcp_cad_k_probe.json`; `tcp_tip_k_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_geometry_history.py`

Compare the actual frozen H3 point clouds on common metric axes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from run_fmb_ablation_suite import RUNS, write_json`
Constants: `SOURCES={'Sensor': 'history_sensor_3d.npy', 'Planar PnP': 'history_pnp_3d.npy', 'CAD silhouette + TCP': 'history_cad_silhouette_tcp_3d.npy', 'MoGe-2 raw': 'history_moge2_raw_3d.npy', 'MoGe-2 history scaled': 'history_moge2_history_scaled_3d.npy'}`; `FRAME_COLORS=('#4e79a7', '#f28e2b', '#e15759')`
Input/output file references: `experiment_config.json`; `history_cad_silhouette_tcp_3d.npy`; `history_clouds_common_axes.json`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_pnp_3d.npy`; `history_sensor_3d.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_K_sensitivity_combined.py`

Compare all frozen full-inference FMB K hypotheses on both episodes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from matplotlib.patches import Patch`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json`
Constants: `NAMES=('Nominal', 'Principal point', 'Prior focal 0.925*', 'Focal 0.9', 'Focal 1.1')`
Input/output file references: `/`; `calibration_sensitivity.csv`; `comparison.csv`; `comparison.json`; `evaluation.json`; `metrics.json`; `runs/fmb_K_sensitivity_combined_v1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_metric_depth.py`

Create a fixed-scale visual comparison for the FMB depth experiment.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from pathlib import Path`; `import cv2`; `import matplotlib.pyplot as plt`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `moge2_aspect_fov_metric_predictions.npz`; `moge3_aspect_fov_metric_predictions.npz`; `obs/side_1`; `obs/side_1_depth`; `runs/sharerobot_fmb_episode_5201`; `unidepthv2_metric_predictions.npz`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_moge2_history.py`

Show frozen historical FMB sensor and MoGe-2 depth with common scales.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: 
Input/output file references: ` / points`; `MoGe-2 depth / m`; `Sensor depth / m`; `data/fmb/single_object_manipulation_dataset`; `depth_comparison.json`; `experiment_config.json`; `history_points_2d.npy`; `moge2_history_v1/moge2_history_maps.npz`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_moge2_temporal_geometry.py`

Plot H3 static-background depth stability and object shape consistency.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `pair_changes`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: 
Input/output file references: `Same-pixel sensor/RGB registration unverified; ROI visually static only`; `Sensor vs RGB-derived MoGe-2 history; common rigidity scale, candidate FOV/K`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_sensor_3d.npy`; `moge2_history_v1/manifest.json`; `runs/fmb_moge2_temporal_geometry_v1`; `temporal_consistency.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/plot_fmb_tracking_uncertainty.py`

Show disagreement between two independent RGB tracking algorithms, not 3D truth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: `SELECTED_OFFSETS=(0, 5, 10, 15, 19)`
Input/output file references: `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `obs/side_1`; `tracker_disagreement.json`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/preflight_fmb_second_scene.py`

Write method readiness, candidate t0 points, and ShareRobot search status.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `save`, `center_and_area`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data/fmb_second_scene'`; `RUN=ROOT / 'runs/fmb_second_scene'`
Input/output file references: `/`; `1_L_L_4_vertical_n_0.npy`; `active 256x256 RGB/depth intrinsics`; `calibration/calibration.json`; `candidate_depth_roi.json`; `data/fmb_second_scene`; `data/fmb_second_scene/calibration/calibration.json`; `data/fmb_second_scene/object/object_metadata.json`; `evaluation/README.md`; `https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/fmb_dataset_builder/fmb_dataset/fmb_dataset_dataset_builder.py`; `https://huggingface.co/datasets/BAAI/ShareRobot`; `https://huggingface.co/datasets/BAAI/ShareRobot/tree/main/planning/images`; `method_readiness.json`; `object/object_metadata.json`; `obs/side_1`; `obs/tcp_pose`; `observed/README.md`; `raw/source_demo.npy`; `runs/fmb_second_scene`; `runs/fmb_second_scene/candidate_depth_roi.json`; `runs/fmb_second_scene/sharerobot_trajectory_search.json`; `runs/fmb_second_scene/visuals/sharerobot_search_best_pair.png`; `runs/sharerobot_fmb_episode_5201/preflight.json`; `shape/size/length metadata`; `sharerobot_mapping.json`; `sharerobot_trajectory_search.json`; `t0_candidates.json`; `visible broad yellow face at t0; top edge/gripper coverage grows late`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/prepare_author_davis.py`

Prepare the exact bmx-trees GT and original frames for the bundled example.

Dataset: **davis**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/author_davis.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import zipfile`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SAMPLE=ROOT / 'examples/data/davis_bmx_trees'`; `SOURCE=ROOT / 'data/pointmotionbench/davis'`; `RUN=ROOT / 'runs/author_davis_bmx_trees_f30'`
Input/output file references: `/JPEGImages/480p/bmx-trees/`; `DAVIS-2017-trainval-480p.zip/JPEGImages/480p/bmx-trees`; `allenai/PointMotionBench`; `bmx-trees_2d.npz`; `bmx-trees_3d.npz`; `data/pointmotionbench/davis`; `dataset_info.json`; `examples/data/davis_bmx_trees`; `ground_truth.npz`; `intrinsics_K.pt`; `meta.json`; `points_2d_at_t0.pt`; `points_3d_history.pt`; `runs/author_davis_bmx_trees_f30`
Related saved experiments (dataset-level): author_davis_bmx_trees_f30, author_davis_coordinate_audit

## `scripts/prepare_dobbe_approx_history.py`

Freeze an exploratory [3,8,3] red-cup history using transferred RGB-only K.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `digest`, `cup_mask`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from inspect_hony_scenes import ROOT, liblzfse`; `from validate_dobbe_geometry import CAMERA_TO_LABEL, depth_at, unproject`
Constants: `OUT=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `STUDY=ROOT / 'runs/dobbe_rgbd_study'`; `RAW=ROOT / 'data/dobbe_oxe/target_raw'`
Input/output file references: `data/dobbe_oxe/target_raw`; `f2nerf_second/receipt.json`; `labels.json`; `manifest.json`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `points_3d_history_world_candidate.npy`; `protocol.json`; `runs/dobbe_rgbd_study`; `runs/dobbe_rgbd_study/approx_history`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/prepare_dobbe_vipe_depth_pair.py`

Freeze sensor-supported common queries before depth-only A inference.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `prepare_pair`.
Imports/reused functions: `import importlib.util`; `import shutil`; `import cv2`; `import numpy as np`; `from dobbe_vipe_v1 import RUN, VIPE, GATE, read, write, freeze, sha`; `from dobbe_vipe_geometry import lift, sample_depth, rigid_error, select_spread, independent_sift_geometry`; `from dobbe_vipe_inference import save_variant`
Constants: 
Input/output file references: `/`; `A/sift_audited`; `Original selected cup rim queries include background/mixed measured depth`; `Two original queries sampled background/mixed sensor depth near the cup rim; choose one common supported pool for both depths before either paired prediction.`; `ablation_input_audit.json`; `ablation_input_audit_initial.json`; `c2w_t0.npy`; `geometry_all.npz`; `geometry_gate.json`; `hybrid/geometry_gate.json`; `hybrid/input_freeze.json`; `hybrid_alignment_review.json`; `input_freeze.json`; `measured_depth_prefix.npy`; `paired_selection_audit.json`; `paired_selection_protocol.json`; `points_2d_t0.npy`; `points_3d_world.npy`; `protocol.json`; `rejected_sensor_queries/hybrid/geometry_gate.json`; `tracks.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/prepare_dobbe_vipe_sift_control.py`

Separate post-hoc RGB correspondence control; main AllTracker gate is preserved.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `prepare`.
Imports/reused functions: `import shutil`; `import numpy as np`; `from dobbe_vipe_v1 import RUN, GATE, read, write, freeze, sha`; `from dobbe_vipe_geometry import independent_sift_geometry`; `from dobbe_vipe_inference import export_variants`
Constants: 
Input/output file references: `*/prediction_15hz.npy`; `A/rectified`; `A/rectified/geometry_gate.json`; `A/sift_audited`; `control_protocol.json`; `geometry_all.npz`; `geometry_gate.json`; `hybrid/geometry_gate.json`; `late observed prefix/H3 only; no reliable SIFT matches for early frames`; `model_run.json`; `protocol.json`; `static_sift_*.npz`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/prepare_fmb_cad_history.py`

Build a transparent, candidate-only H3/8-point FMB input from joint CAD PnP.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `DEST=OUT / 'cad_history_candidate'`; `STEPS=[130, 135, 140]`; `K=np.asarray([[152.0836, 0, 124.2908], [0, 202.7781333333, 129.2973333333], [0, 0, 1]], dtype=float)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `Build a transparent, candidate-only H3/8-point FMB input from joint CAD PnP.`; `cad_pnp_probe.json`; `manifest.json`; `obs/side_1`; `obs/side_1_depth`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `runs/sharerobot_fmb_episode_5201`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/prepare_fmb_moge2_history.py`

RGB-derived MoGe-2 history for the two frozen FMB forecast windows.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `extract_depth`, `local_depth`, `unproject`, `internal_consistency`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, digest, verify_frozen, write_json`
Constants: `MODEL_FILE=ROOT.parent / 'models/moge-2-vitl/model.pt'`; `MOGE_SOURCE=ROOT.parent / 'third_party/MoGe'`; `EXPECTED_MODEL_SHA256='3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5'`
Input/output file references: `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `k_nominal.json`; `manifest.json`; `models/moge-2-vitl/model.pt`; `moge2_history_maps.npz`; `obs/side_1`; `obs/side_1_depth`; `source BGR to RGB; inverse 256x256->256x192 aspect stretch; candidate horizontal FOV from K_nominal fx; outputs depth/points resized to 256x256`; `third_party/MoGe`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/prepare_fmb_quantitative_2d.py`

Freeze a sensor-depth MolmoMotion input before viewing any new prediction.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `write_json`, `sha256`, `deproject`, `project`, `k_values`, `history_uv_and_depth`, `cad_poses_and_local_points`, `pnp_history`, `compare_xyz`, `rigidity`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import D, H, K, SOURCE, W`; `from probe_fmb_future_tracking import initial_points, track`; `from probe_fmb_history_geometry import kabsch`; `from probe_fmb_history_silhouette_pose import fit_pose, fit_shared_orientation`; `from probe_fmb_effective_k import landmarks, pose_for_k`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `BASE=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `RUN=BASE / 'quantitative_2d_sensor_t126'`; `T0=126`; `STEPS=(124, 125, 126)`; `ACTION='Insert the red rectangular peg into the matching hole on the blue board.'`; `DEPTH_SCALE=0.0001`; `SOURCE_K=np.asarray([[380.209, 0, 311.477], [0, 380.209, 242.87], [0, 0, 1]])`
Input/output file references: `.npy`; `English description of source primitive insert, object_info rectangle/Jeans Red, and visible blue board; not a verbatim FMB metadata field`; `Freeze a sensor-depth MolmoMotion input before viewing any new prediction.

All geometry here uses FMB source steps 124--126 only. Future frames are used
only for the predeclared window motion/coverage check, never for XYZ or K.
`; `Only an upper interior patch of the uniform red face is observed as eight points; no full CAD width/depth/height endpoint pair is independently identified.`; `cad_dimension_check.json`; `cad_history_candidate/molmo_candidate_validation.json`; `cad_pnp_focal_depth_sweep.json`; `cad_pnp_local_sensitivity.json`; `cad_pose_fit.json`; `current_state.json`; `effective_k_rounded_cad_silhouette.json`; `experiment_config.json`; `fixed RGB mask row/fraction grid at t0; optical flow backward to history; selected by history local depth only`; `four_camera_sampling_check.json`; `geometry_probe.json`; `history_cad_silhouette_tcp_3d.npy`; `history_depth_probe.json`; `history_pnp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_freeze.json`; `k_nominal.json`; `k_sensitivity_variants.json`; `metric_geometry_comparison.json`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `planar CAD PnP plus RGB rounded-CAD silhouette/TCP diagnostic`; `pnp_correspondences.json`; `points_2d_at_t0.npy`; `preflight.json`; `published depth resize/registration unverified`; `rgb_depth_alignment_check.json`; `rigidity_check.json`; `runs/sharerobot_fmb_episode_5201`; `sensor Z16 5x5 positive median, 0.0001 m/raw, K_nominal`; `sensor_vs_cad_silhouette_tcp.json`; `sensor_vs_pnp.json`; `window_selection.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/prepare_second_fmb_quantitative_2d.py`

Freeze independent sensor-depth input for a second raw FMB trial.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `save_json`, `selected_history`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `import prepare_fmb_quantitative_2d as previous`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy'`; `BASE=ROOT / 'runs/fmb_second_example_1_M_L_3_vertical_n_3'`; `RUN=BASE / 'quantitative_2d_sensor_t130'`; `T0=130`; `STEPS=(128, 129, 130)`; `FUTURE=list(range(131, 151))`; `ACTION='Insert the red rectangular peg into the matching hole on the blue board.'`; `SELECTED=((28, 0.3), (28, 0.7), (44, 0.5), (44, 0.8), (60, 0.2), (60, 0.7), (76, 0.3), (76, 0.8))`
Input/output file references: `../window_point_search.json`; `.npy`; `8 fixed row/fraction inner red-face positions from pre-inference depth-valid candidate grid`; `cad_dimension_check.json`; `cad_pose_fit.json`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `experiment_config.json`; `history_cad_silhouette_tcp_3d.npy`; `history_depth_probe.json`; `history_pnp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_freeze.json`; `k_nominal.json`; `k_sensitivity_variants.json`; `obs/gripper_pose`; `obs/side_1`; `obs/side_1_depth`; `pnp_correspondences.json`; `points_2d_at_t0.npy`; `preflight.json`; `preflight_probe_failed.json`; `rigidity_check.json`; `runs/fmb_second_example_1_M_L_3_vertical_n_3`; `sensor_vs_cad_silhouette_tcp.json`; `sensor_vs_pnp.json`; `window_point_search.json`; `window_probe.json`; `window_selection.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_dobbe_3651_camera_geometry.py`

Probe RGB/depth/pose consistency for the matched HoNY red-cup capture.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `load_frames`, `make_matches`, `rotations`, `reprojection`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import itertools`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT / 'data' / 'dobbe_oxe' / 'target_raw'`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight'`; `PAIRS=[(0, 20), (20, 40), (40, 60), (60, 80), (180, 200), (200, 220)]`
Input/output file references: `Probe RGB/depth/pose consistency for the matched HoNY red-cup capture.

This is a diagnostic. It does not silently accept a guessed camera matrix.
`; `compressed_video_h264.mp4`; `dobbe_3651_camera_pose_probe.json`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_dobbe_3651_rgbd_self_calibration.py`

Exploratory effective-K fit using RGB-D rigidity, independent of saved poses.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `sample_z`, `xyz`, `rigid_align`, `pair_residual`, `stats`, `prepare`, `fit`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.ndimage import map_coordinates`; `from scipy.optimize import least_squares`; `from probe_dobbe_3651_camera_geometry import DATA, OUT, load_frames`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `PAIRS=[(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (100, 110), (110, 120), (180, 190), (190, 200), (200, 210), (210, 220)]`; `HOLDOUT={(30, 40), (110, 120), (200, 210)}`
Input/output file references: `dobbe_3651_rgbd_self_calibration.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_dobbe_depth_delivery.py`

Probe byte-range access to HoNY's final split ZIP member, no bulk download.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import re`; `import struct`; `import sys`; `from pathlib import Path`; `import requests`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight' / 'dobbe_depth_delivery_probe.json'`; `GDOWN_TARGET=ROOT.parent / '.tools' / 'gdown'`; `FOLDER_ID='1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO'`
Input/output file references: `dobbe_depth_delivery_probe.json`; `https://drive.usercontent.google.com/download`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_dobbe_f2nerf_transfer.py`

Test RGB-only F2-NeRF control-scene intrinsics on main measured RGB-D.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `import validate_dobbe_geometry as geometry`
Constants: 
Input/output file references: `Test RGB-only F2-NeRF control-scene intrinsics on main measured RGB-D.

The transfer is an explicit hypothesis: the two recordings may have used
different phones/camera settings. No future frames or depth fit the candidate K.
`; `data/dobbe_oxe/target_raw`; `f2nerf_intrinsics_transfer.json`; `f2nerf_second/receipt.json`; `labels.json`; `main_static_correspondences.json`; `second_static_geometry.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_dobbe_pose_branch.py`

Measure the effect of the required portrait RGB rotation on static pairs.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `import json`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `import validate_dobbe_geometry as geometry`
Constants: 
Input/output file references: `_static_correspondences.json`; `data/dobbe_oxe`; `image_rotation_diagnostic.json`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_dobbe_rgb_archive.py`

Inventory the official HoNY RGB ZIP over HTTP ranges without downloading it.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import collections`; `import json`; `import sys`; `import zipfile`; `from pathlib import Path`; `from inspect_remote_zip import HTTPRangeFile`
Constants: `URL='https://dl.dobb-e.com/datasets/homes_of_new_york.zip'`; `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'plex_dobbe_preflight'`
Input/output file references: `/`; `/cup`; `compressed_video_h264.mp4`; `dobbe_rgb_archive_inventory.json`; `https://dl.dobb-e.com/datasets/homes_of_new_york.zip`; `iphone_data/`; `iphone_data/r3d_files.txt`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/probe_fmb_board_peg_contact_bundle.py`

RGB-only joint K from board CAD and an unoccluded peg lying on its table.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `unpack_k`, `fit_joint`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from calibrate_fmb_board_k import HOLDOUT_FEATURES, cad_features, depth_crosscheck, extract, project`; `from check_fmb_cross_orientation_depth import check as peg_depth_check`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_cad_silhouette_k import project_model, support`; `from probe_fmb_cross_orientation_silhouette_k import observed_hull`; `from probe_fmb_table_contact_k import board_pose, peg_pose_on_table, solve_contact`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `SOURCE=ROOT / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy'`
Input/output file references: `board_peg_table_contact_K_probe.json`; `cross_orientation_silhouette_k_probe.json`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `multi_board_bundle_calibration.json`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_cad_pnp.py`

Explicitly labeled planar CAD PnP probe for the FMB medium long rectangle.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `solve_one`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/sharerobot_fmb_episode_5201'`; `SOURCE=OUT / '1_M_L_3_vertical_n_2.npy'`; `ANNOTATIONS={130: [[190, 19], [229, 21], [197, 149], [177, 149]], 135: [[180, 44], [214, 53], [190, 162], [167, 162]], 140: [[177, 63], [211, 67], [186, 168], [166, 168]]}`; `FRONT=np.asarray([[-W / 2, -D / 2, H], [W / 2, -D / 2, H], [W / 2, -D / 2, 0], [-W / 2, -D / 2, 0]], dtype=np.float64)`; `EIGHT=np.asarray([[-W / 2, -D / 2, H], [W / 2, -D / 2, H], [W / 2, -D / 2, 0], [-W / 2, -D / 2, 0], [-W / 2, D / 2, H], [W / 2, D / 2, H], [W / 2, D / 2, 0], [-W / 2, D / 2, 0]], dtype=np.float64)`; `K=np.asarray([[152.0836, 0, 124.2908], [0, 202.7781333333, 129.2973333333], [0, 0, 1]], dtype=np.float64)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `cad_pnp_history_candidate.npy`; `cad_pnp_history_joint_candidate.npy`; `cad_pnp_probe.json`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `runs/sharerobot_fmb_episode_5201`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_cad_silhouette_k.py`

Experimental RGB-only effective-K fit using the exact rounded CAD silhouette.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `rounded_cuboid_points`, `red_hull`, `support`, `project_model`, `k_from_x`, `initial_pose`, `fit`, `validate`, `motion_check`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import D, EIGHT, FRONT, H, K as K0, OUT, SOURCE, W`; `from probe_fmb_effective_k import INTERIOR, landmarks, pose_for_k`
Constants: `RADIUS=0.00288`; `TRAIN=(0, 128, 134, 140, 146)`; `HOLDOUT=(130, 138, 144)`; `ANGLES=np.arange(0, 2 * np.pi, np.pi / 8)`; `DIRS=np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))`; `MODEL=rounded_cuboid_points()`
Input/output file references: `effective_k_rounded_cad_silhouette.json`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_cad_silhouette_rigid_k.py`

RGB-only rounded-CAD K probe with one late object orientation.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `fit_shared`, `heldout_check`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import K as K0, OUT, SOURCE`; `from probe_fmb_cad_silhouette_k import DIRS, HOLDOUT, TRAIN, initial_pose, k_from_x, project_model, red_hull, support`; `from probe_fmb_effective_k import INTERIOR`
Constants: 
Input/output file references: `effective_k_rounded_cad_rigid_probe.json`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_cross_orientation_silhouette_k.py`

Diagnostic RGB-only K fit adding a horizontal peg view to vertical views.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `observed_hull`, `k_from_x`, `fit_horizontal_pose`, `fit_joint`, `heldout_error`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import H, K as K_NOMINAL, SOURCE as VERTICAL_SOURCE`; `from probe_fmb_cad_silhouette_k import DIRS, MODEL, TRAIN as VERTICAL_TRAIN, HOLDOUT as VERTICAL_HOLDOUT, project_model, support`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `HORIZONTAL_SOURCE=ROOT / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy'`; `HORIZONTAL_TRAIN=(0,)`; `HORIZONTAL_HOLDOUT=(5, 10)`
Input/output file references: `cross_orientation_silhouette_k_probe.json`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `molmo-motion/runs/sharerobot_fmb_episode_5201/effective_k_rounded_cad_silhouette.json`; `obs/side_1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_dense_flow.py`

Compare dense image flow with sparse KLT on the textureless red peg.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `bilinear`, `follow`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import SOURCE`; `from probe_fmb_future_tracking import initial_points`
Constants: `OUT=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`
Input/output file references: `dense_flow_probe.json`; `obs/side_1`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_effective_k.py`

Test whether published FMB RGB frames identify an effective 256px K.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `landmarks`, `project`, `pose_for_k`, `unpack_k`, `fit_bundle`, `validate`, `motion_rigidity`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import D, EIGHT, FRONT, K as K_CANDIDATE, OUT, SOURCE, W, H`
Constants: `TRAIN=(128, 132, 136, 140, 144)`; `HOLDOUT=(130, 138, 146)`; `STEPS=TRAIN + HOLDOUT`; `OBJECT=np.vstack([FRONT, [W / 2, D / 2, H]]).astype(np.float64)`; `INTERIOR=np.asarray([[(a - 0.5) * W, -D / 2, (1 - b) * H] for a in (0.2, 0.5, 0.8) for b in (0.25, 0.5, 0.75)])`
Input/output file references: `Could not identify top/bottom silhouette vertices: `; `Sensor depth is a diagnostic only and retains registration/scale uncertainty.`; `effective_k_bundle_probe.json`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_future_tracking.py`

Exploratory point and visibility probe before fixing the quantitative run.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `initial_points`, `candidate_grid`, `track`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import SOURCE`
Constants: `OUT=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`; `T0=126`
Input/output file references: `history_point_candidates.json`; `obs/side_1`; `obs/side_1_depth`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`; `tracking_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_geometry.py`

Independent physical checks for the FMB side_1 raw depth units.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `similarity_fit`, `fixed_scale_residual`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from PIL import Image, ImageDraw`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`; `SOURCE=OUT / '1_M_L_3_vertical_n_2.npy'`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `geometry_probe.json`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_history_geometry.py`

Pre-inference sensor-history consistency for candidate FMB point sets.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `kabsch`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `from probe_fmb_cad_pnp import K`
Constants: `RUN=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`; `SELECTED=((35, 0.35), (35, 0.55), (45, 0.25), (45, 0.45), (55, 0.35), (55, 0.75), (65, 0.25), (65, 0.55))`
Input/output file references: `history_point_candidates.json`; `history_sensor_candidate_3d.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_history_pnp.py`

Planar CAD PnP on eight RGB-tracked interior peg points, without depth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import K, SOURCE`
Constants: `OUT=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`
Input/output file references: `history_pnp_probe.json`; `history_silhouette_pose_probe.json`; `obs/tcp_pose`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`; `tracking_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_history_silhouette_pose.py`

RGB-only approximate rounded-CAD poses for the history 124--126.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `polygon_row_span`, `model_polygon`, `observations`, `fit_pose`, `fit_shared_orientation`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import D, K, SOURCE`; `from probe_fmb_cad_silhouette_k import MODEL`; `from probe_fmb_effective_k import landmarks, pose_for_k`
Constants: `OUT=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`
Input/output file references: `history_point_candidates.json`; `history_silhouette_pose_probe.json`; `obs/side_1`; `obs/tcp_pose`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_mask_ecc.py`

Probe affine registration of object masks as a second future 2D tracker.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `segment`, `follow`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_cad_pnp import SOURCE`; `from probe_fmb_future_tracking import initial_points`
Constants: `OUT=Path(__file__).resolve().parents[1] / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001'`
Input/output file references: `mask_ecc_probe.json`; `obs/side_1`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_multi_episode_tcp_tip_k.py`

Diagnostic effective-K fit from RGB peg tips and metric TCP trajectories.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `k_from_x`, `project`, `fit`, `score`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import red_end`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `DATA=ROOT / 'data/fmb/single_object_manipulation_dataset'`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `EPISODES=(0, 4, 8)`; `INTERVALS={0: (210, 290), 4: (193, 235), 8: (222, 251)}`
Input/output file references: `.npy`; `A rigid TCP-to-tip offset and exact RGB/TCP synchronization are assumed during insertion.`; `board_peg_table_contact_K_probe.json`; `cross_orientation_silhouette_k_probe.json`; `data/fmb/single_object_manipulation_dataset`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `multi_board_bundle_calibration.json`; `multi_episode_tcp_tip_K_probe.json`; `multi_horizontal_contact_K_probe.json`; `obs/side_1`; `obs/tcp_pose`; `sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126/k_sensitivity_variants.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_rgb_depth_affine.py`

Cross-validate a restricted RGB-to-depth affine from physical silhouettes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `fit`, `error`, `red_mask`, `vertical_edges`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'fmb_effective_k_256_calibration'`; `EP=ROOT / 'runs' / 'sharerobot_fmb_episode_5201'`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `obs/side_1`; `obs/side_1_depth`; `rgb_depth_affine_probe.json`; `rgb_depth_alignment_check.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_rlds_record.py`

Inspect one saved FMB TFDS TFRecord example without loading TensorFlow.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `varint`, `fields`, `find_feature_values`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import mmap`; `from pathlib import Path`
Constants: 
Input/output file references: `/`; `episode_metadata/`; `episode_metadata/episode_language_instruction`; `episode_metadata/file_path`; `rlds_record_summary.json`; `steps/observation/image_`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_shared_tcp_offset_k.py`

RGB/TCP calibration probe with one common tip offset across three episodes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import numpy as np`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_tcp_tip_k import fit, score`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `EPISODES=(0, 4, 8)`
Input/output file references: `Temporal RGB/TCP synchronization is assumed.`; `board_peg_table_contact_K_probe.json`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `multi_episode_tcp_tip_K_probe.json`; `shared_tcp_offset_K_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_stereo_rgb.py`

Test whether the two published FMB RGB views support stereo matching.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `red_properties`, `sift_stereo`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs' / 'fmb_effective_k_256_calibration'`; `EP=ROOT / 'runs' / 'sharerobot_fmb_episode_5201' / '1_M_L_3_vertical_n_2.npy'`; `STEPS=(0, 37, 74, 110, 120, 128, 136, 144)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `SIFT/F/H only checks tentative stereo correspondences. There are no verified nonplanar 3D matches or recorded per-camera effective COLOR K/extrinsics, so metric stereo triangulation is not validated.`; `obs/side_1`; `obs/side_2`; `stereo_rgb_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_table_contact_k.py`

RGB-only peg/table-contact diagnostic using FMB's metric board CAD.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `board_pose`, `peg_pose_on_table`, `solve_contact`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from calibrate_fmb_board_k import cad_features, extract, fit as fit_board`; `from check_fmb_cross_orientation_depth import check as depth_check`; `from probe_fmb_cad_pnp import D, H, K as K_NOMINAL, W`; `from probe_fmb_cad_silhouette_k import project_model, support`; `from probe_fmb_cross_orientation_silhouette_k import observed_hull`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `SOURCE=ROOT / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy'`; `BOARD_BOTTOM_Z_M=0.0485`
Input/output file references: `cross_orientation_silhouette_k_probe.json`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_horizontal_n_4.npy`; `horizontal_table_contact_probe.json`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `obs/side_1`; `obs/side_1_depth`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_tcp_bottom_k.py`

Calibrate from the peg's visible bottom cross-section across TCP motion.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `observed`, `predict`, `score`, `fit`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_silhouette_k import MODEL`; `from probe_fmb_tcp_cad_k import depth_crosscheck, initial_vector`; `from probe_fmb_tcp_tip_k import OUT, SOURCE, k_from_x, red_end`
Constants: `BOTTOM=MODEL[np.isclose(MODEL[:, 2], 0.0)]`; `TRAIN=tuple(range(72, 89, 2)) + tuple(range(100, 121, 2))`; `HOLDOUT=tuple(range(73, 89, 2)) + tuple(range(101, 121, 2))`; `LATE=(129, 131, 133, 135, 137, 139, 141, 143, 145)`
Input/output file references: `Rigid grasp and RGB/TCP synchronization are assumed.`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `tcp_bottom_k_probe.json`; `tcp_cad_k_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_tcp_cad_k.py`

Probe effective K from RGB, a STEP-derived rounded envelope and TCP motion.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `projected_support`, `summary`, `depth_crosscheck`, `fit`, `initial_vector`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import K as K_NOMINAL`; `from probe_fmb_cad_silhouette_k import MODEL, red_hull`; `from probe_fmb_tcp_tip_k import OUT, SOURCE, k_from_x`
Constants: `ANGLES=np.arange(0, 2 * np.pi, np.pi / 6)`; `DIRS=np.column_stack((np.cos(ANGLES), np.sin(ANGLES)))`; `TRAIN=(64, 66, 68, 70, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146)`; `HOLDOUT=(65, 67, 69, 129, 131, 133, 135, 137, 139, 141, 143, 145)`
Input/output file references: `1_M_L_3_vertical_n_2.npy`; `Directional silhouette supports in tcp_cad_k_probe.json`; `RGB/TCP diagnostic: peg red-mask silhouette visible`; `cad_rgb_correspondences.json`; `calibration_frames.json`; `effective_k_rounded_cad_silhouette.json`; `obs/gripper_pose`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `tcp_cad_k_probe.json`; `tcp_tip_k_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/probe_fmb_tcp_tip_k.py`

Diagnostic RGB/TCP camera calibration from the visible end of a grasped peg.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `red_end`, `k_from_x`, `project`, `fit`, `score`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import K as K_NOMINAL`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT / 'runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy'`; `OUT=ROOT / 'runs/fmb_effective_k_256_calibration'`; `TRAIN=tuple(range(64, 89, 2)) + tuple(range(100, 121, 2))`; `HOLDOUT=tuple(range(65, 89, 2)) + tuple(range(101, 121, 2)) + tuple(range(121, 126))`
Input/output file references: `obs/side_1`; `obs/tcp_pose`; `runs/fmb_effective_k_256_calibration`; `runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy`; `tcp_tip_k_probe.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/publish_remaining_assets.py`

Publish oversized local research files as resumable GitHub Release assets.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `import argparse`; `import concurrent.futures`; `import hashlib`; `import http.client`; `import json`; `from pathlib import Path`; `import subprocess`; `import threading`; `import time`; `import urllib.parse`; `import urllib.request`; `import urllib.error`
Constants: `PART_BYTES=1024 ** 3`; `LOCK=threading.Lock()`
Input/output file references: `.uploaded.json`; `/assets?per_page=100`; `/assets?per_page=100&page=`; `/releases`; `/releases/`; `/releases/assets/`; `/releases/tags/`; `AlexeyPetrov1/Airi_research_task`; `application/octet-stream`; `application/vnd.github+json`; `https://api.github.com/repos/`
Related saved experiments (dataset-level): 

## `scripts/query_fmb_episode_files.py`

Print metadata for a few exact FMB raw files without downloading the dataset.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from huggingface_hub import HfApi`
Constants: `REPO='charlesxu0124/functional-manipulation-benchmark'`; `FILES=[f'single_object_manipulation_dataset/1_M_L_3_vertical_n_{i}.npy' for i in range(4, 10)]`
Input/output file references: `.npy`; `charlesxu0124/functional-manipulation-benchmark`; `single_object_manipulation_dataset/1_M_L_3_vertical_n_`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/render_dobbe_approx_prediction_overlay.py`

Project exploratory future 3D predictions onto real future cup RGB frames.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `project_world_from_h2`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from inspect_hony_scenes import ROOT`; `from validate_dobbe_geometry import CAMERA_TO_LABEL`
Constants: `RUN=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `RAW=ROOT / 'data/dobbe_oxe/target_raw'`
Input/output file references: `/8`; `/8; pred on cup `; `compressed_video_h264.mp4`; `data/dobbe_oxe/target_raw`; `future_evaluation.json`; `future_gt_valid.npy`; `future_projection_diagnostic.json`; `future_rgb_uv.npy`; `labels.json`; `manifest.json`; `model_run.json`; `points_3d_history_candidate.npy`; `prediction_15hz.npy`; `runs/dobbe_rgbd_study/approx_history`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/render_fmb_moge2_comparison.py`

Render actual future RGB with saved sensor and MoGe-2 forecasts.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from render_fmb_study_media import BLUE, ORANGE, GREEN, header, point, video`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, write_json`
Constants: `COLORS={'sensor': ORANGE, 'moge2_raw': GREEN, 'moge2_scaled': (205, 75, 175)}`; `LABELS={'sensor': 'FMB sensor Z', 'moge2_raw': 'MoGe-2 RGB depth', 'moge2_scaled': 'MoGe-2 / H3 scale'}`
Input/output file references: `/8`; `MoGe-2 / H3 scale`; `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `forecast_comparison.mp4`; `forecast_media_manifest.json`; `gt_2d.npy`; `obs/side_1`; `prediction_2d.npy`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/render_fmb_study_media.py`

Render FMB figures and videos only from saved frames, tracks and forecasts.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `header`, `point`, `make_real_panel`, `make_pred_panel`, `make_error_panel`, `make_chart_panel`, `video`, `geometry_cloud_panel`, `render_one`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import interpolate, project`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: `BLUE=(255, 100, 25)`; `ORANGE=(0, 145, 255)`; `GRAY=(150, 150, 150)`; `GREEN=(50, 190, 60)`; `WHITE=(245, 245, 245)`; `BLACK=(15, 15, 15)`; `COLOR={'sensor': ORANGE, 'pnp': GREEN, 'cad': (210, 90, 190)}`
Input/output file references: `/8`; `XY and XZ views, both at 1 px/mm with fixed camera-coordinate axes.`; `baseline_constant_velocity_2d.npy`; `baseline_stationary_2d.npy`; `blue sensor; green CAD; 1 px/mm`; `data/fmb/single_object_manipulation_dataset`; `experiment_config.json`; `forecast_vs_observed.mp4`; `geometry_methods.mp4`; `gt_2d.npy`; `history_cad_silhouette_tcp_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_and_geometry.mp4`; `k_nominal.json`; `media_manifest.json`; `obs/side_1`; `obs/side_1_depth`; `prediction_15hz.npy`; `validity_mask.npy`; `variants/cad_silhouette/prediction_15hz.npy`; `variants/pnp/prediction_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/restore_remaining_assets.py`

Restore large research inputs from the verified GitHub Release manifest.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `digest`, `main`.
Imports/reused functions: `import argparse`; `import hashlib`; `import json`; `from pathlib import Path`; `import urllib.request`
Constants: 
Input/output file references: `report/remaining_assets_manifest.json`
Related saved experiments (dataset-level): 

## `scripts/run_author_davis.py`

Run the released H3/F30 model once on the bundled DAVIS bmx-trees episode.

Dataset: **davis**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/author_davis.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import resource`; `import shutil`; `import subprocess`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`; `from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SAMPLE=ROOT / 'examples/data/davis_bmx_trees'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `RUN=ROOT / 'runs/author_davis_bmx_trees_f30'`; `HORIZON=30`
Input/output file references: `Model text lacks at least one of the 8 x 30 point/timestamp entries`; `Run the released H3/F30 model once on the bundled DAVIS bmx-trees episode.`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `davis_bmx_trees/bike_rider/t0=2/point_indices=[9,12,17,18,27,29,32,34]`; `examples/data/davis_bmx_trees`; `model.pt`; `points_2d_at_t0.pt`; `points_3d_history.pt`; `prediction.npz`; `processor_inputs.pt`; `run_status.json`; `runs/author_davis_bmx_trees_f30`
Related saved experiments (dataset-level): author_davis_bmx_trees_f30, author_davis_coordinate_audit

## `scripts/run_dobbe_approx_molmo.py`

Run one exploratory MolmoMotion forecast from the sealed HoNY cup history.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `sha256`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import json`; `import resource`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/dobbe_rgbd_study/approx_history'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`
Input/output file references: `data/checkpoints/MolmoMotion-4B-H3-F30`; `input_freeze.json`; `model_run.json`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `prediction_15hz.npy`; `runs/dobbe_rgbd_study/approx_history`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/run_dobbe_rgb_colmap.py`

RGB-only COLMAP controls with a clean default and an explicit legacy profile.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `database_complete`, `remove_db_files`, `save_rgb_prefix`, `legacy_main`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `import sqlite3`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `import pycolmap`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `STUDY=ROOT / 'runs/dobbe_rgbd_study'`; `SCENES={'main': (ROOT / 'data/dobbe_oxe/target_raw', 96), 'second': (ROOT / 'data/dobbe_oxe/second_raw', 121)}`
Input/output file references: `*/database.db`; `compressed_video_h264.mp4`; `data/dobbe_oxe/second_raw`; `data/dobbe_oxe/target_raw`; `runs/dobbe_rgbd_study`; `summary.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/run_dobbe_vipe_sift_inference.py`

Run all predeclared A controls before opening any A future frames.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: .
Imports/reused functions: `from dobbe_vipe_inference import infer, export_variants`; `from prepare_dobbe_vipe_depth_pair import prepare_pair`
Constants: 
Input/output file references: 
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/run_fmb_ablation_suite.py`

Frozen, paired FMB ablations. Run in the Linux GPU environment.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `digest`, `write_json`, `verify_frozen`, `make_variant`, `prepare`, `infer`, `evaluate`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import csv`; `import hashlib`; `import json`; `import os`; `import resource`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUNS={'first': ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126', 'second': ROOT / 'runs/fmb_second_example_1_M_L_3_vertical_n_3/quantitative_2d_sensor_t130'}`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `MODEL_REVISION='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629'`; `SUITE='ablation_suite_v1'`; `CORE_VARIANTS=('text_paraphrase', 'text_generic', 'text_counterfactual', 'no_history', 'scale_090', 'scale_110', 'perm_keep_anchor', 'perm_new_anchor', 'focal_090', 'focal_110')`; `MOGE_VARIANTS=('moge2_raw', 'moge2_scaled', 'moge2_focal_090')`; `VARIANTS=CORE_VARIANTS + MOGE_VARIANTS`; `PERM_KEEP=np.array([0, 2, 4, 6, 1, 3, 5, 7])`; `PERM_NEW=np.array([4, 0, 6, 2, 7, 1, 5, 3])`; `TEXT={'text_paraphrase': 'Place the red rectangular piece into the matching opening in the blue board.', 'text_generic': 'Move the object to complete the task.', 'text_counterfactual': 'Move the red rectangular piece away from the blue board.'}`
Input/output file references: `F30 j/15 s interpolated to native FMB j/10 s`; `History / geometry`; `MoGe-2 / focal 0.9`; `Reorder / new anchor`; `Reorder / same anchor`; `allenai/MolmoMotion-4B-H3-F30`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `evaluation.json`; `experiment_config.json`; `fmb_ablation_suite_v1_summary.csv`; `gt_2d.npy`; `gt_dense_flow_alternative.npy`; `history_3d.npy`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_points_2d.npy`; `history_sensor_3d.npy`; `input_freeze.json`; `k_nominal.json`; `manifest.json`; `metrics.json`; `model_run.json`; `moge2_history_maps.npz`; `parser_validation.json`; `points_2d_at_t0.npy`; `prediction_10hz.npy`; `prediction_15hz.npy`; `prediction_15hz_model_order.npy`; `prediction_2d.npy`; `preflight.json`; `runs/fmb_ablation_suite_v1_summary.png`; `runs/fmb_second_example_1_M_L_3_vertical_n_3/quantitative_2d_sensor_t130`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`; `validity_mask.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/run_fmb_cad_candidate.py`

Exploratory MolmoMotion H3/F30 run using the explicitly unverified CAD PnP input.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import resource`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/sharerobot_fmb_episode_5201/cad_history_candidate'`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `ACTION='Insert the red rectangular peg into the matching hole on the blue board.'`
Input/output file references: `Exploratory MolmoMotion H3/F30 run using the explicitly unverified CAD PnP input.`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `manifest.json`; `molmo_candidate_future.npy`; `molmo_candidate_status.json`; `points_2d_at_t0_candidate.npy`; `points_3d_history_candidate.npy`; `runs/sharerobot_fmb_episode_5201/cad_history_candidate`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/run_fmb_quantitative_2d.py`

Run the frozen FMB t0=126 H3/F30 input with MolmoMotion in WSL GPU env.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `verify_freeze`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import hashlib`; `import json`; `import os`; `import resource`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`; `CHECKPOINT=ROOT / 'data/checkpoints/MolmoMotion-4B-H3-F30'`; `MODEL_REVISION='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629'`; `MODEL_PT_SHA256_PREVIOUSLY_VERIFIED='506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205'`
Input/output file references: `Run the frozen FMB t0=126 H3/F30 input with MolmoMotion in WSL GPU env.`; `Three real history observations are 0.1 s apart; model typical training/prediction timing is about 15 Hz. Processor has no explicit timestamp input.`; `allenai/MolmoMotion-4B-H3-F30`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `experiment_config.json`; `history_cad_silhouette_tcp_3d.npy`; `history_pnp_3d.npy`; `history_sensor_3d.npy`; `history_sensor_3d_K_focal_0925.npy`; `history_sensor_3d_K_simple.npy`; `input_freeze.json`; `model_run.json`; `points_2d_at_t0.npy`; `prediction_15hz.npy`; `preflight.json`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/search_fmb_second_sharerobot.py`

Check the individually available ShareRobot FMB trajectory RGBs by pixels.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `fetch`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import hashlib`; `import io`; `import json`; `from concurrent.futures import ThreadPoolExecutor, as_completed`; `from pathlib import Path`; `from urllib.parse import quote`; `import cv2`; `import numpy as np`; `import requests`; `from PIL import Image`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/fmb_second_scene'`; `DATA=ROOT / 'data/fmb_second_scene/raw/source_demo.npy'`; `MANIFEST=RUN / 'sharerobot_trajectory_manifest.json'`; `REV='3266d92902b038ce40e7fb8aac5bfd9287eb3e45'`; `BASE=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{REV}/trajectory/images/'`; `CAMERAS=('side_1', 'side_2', 'wrist_1', 'wrist_2')`
Input/output file references: `/`; `/trajectory/images/`; `/trajectory/trajectory.json`; `1_L_L_4_vertical_n_0.npy`; `data/fmb_second_scene/raw/source_demo.npy`; `https://huggingface.co/datasets/BAAI/ShareRobot/blob/`; `https://huggingface.co/datasets/BAAI/ShareRobot/resolve/`; `obs/`; `runs/fmb_second_scene`; `sharerobot_trajectory_manifest.json`; `sharerobot_trajectory_search.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/select_dobbe_second_scene.py`

List complete HoNY RGB-D captures from the existing split-ZIP index.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from collections import defaultdict`; `from pathlib import Path`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `INDEX=ROOT / 'runs/plex_dobbe_preflight/dobbe_depth_nested_index.json'`; `NEEDED={'compressed_video_h264.mp4', 'compressed_np_depth_float32.bin', 'labels.json'}`
Input/output file references: `/Home15/`; `/Pick_and_Place/`; `Home15/Env1/2023-04-25--02-05-30`; `compressed_video_h264.mp4`; `labels.json`; `runs/plex_dobbe_preflight/dobbe_depth_nested_index.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/select_fmb_second_scene.py`

Record the source-backed FMB reference and visual candidate decision.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sha256`, `write_json`, `red_peg_quality`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/fmb_second_scene'`; `SOURCE_DIR=ROOT.parent / 'data/fmb/single_object_manipulation_dataset'`; `REFERENCE='1_M_L_3_vertical_n_2.npy'`; `SELECTED='1_L_L_4_vertical_n_0.npy'`; `HF='https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/tree/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset'`; `NAMING='https://functional-manipulation-benchmark.github.io/dataset/index.html'`
Input/output file references: `/`; `1_L_L_4_vertical_n_0.npy`; `1_M_L_3_vertical_n_2.npy`; `2_M_L_7_vertical_n_0.npy`; `candidate_selection.csv`; `candidate_survey.json`; `data/fmb/single_object_manipulation_dataset`; `https://functional-manipulation-benchmark.github.io/dataset/index.html`; `https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/tree/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset`; `https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip`; `obs/`; `original raw RGB/depth`; `reference_5201.json`; `runs/fmb_second_scene`; `runs/fmb_second_scene/candidate_depth_roi.json`; `runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_1.png`; `runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_2.png`; `runs/sharerobot_fmb_episode_5201/preflight.json`; `selection.json`; `single_object_manipulation_dataset/`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/select_second_fmb_window.py`

Search pre-inference candidate windows/inner points for second FMB trial.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `candidates`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from inspect_fmb_2d_window import red_component`; `from probe_fmb_future_tracking import track`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `SOURCE=ROOT.parent / 'data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy'`; `OUT=ROOT / 'runs/fmb_second_example_1_M_L_3_vertical_n_3'`
Input/output file references: `Search pre-inference candidate windows/inner points for second FMB trial.`; `data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`; `obs/side_1`; `obs/side_1_depth`; `obs/tcp_pose`; `runs/fmb_second_example_1_M_L_3_vertical_n_3`; `window_point_search.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/smoke_inference.py`

Load the released checkpoint and optionally forecast one future frame.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `import resource`; `import time`; `import traceback`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`; `from molmo_motion import MolmoMotion, MolmoMotionProcessor`
Constants: 
Input/output file references: `.json`; `.npz`; `data/checkpoints/MolmoMotion-4B-H3-F30`; `examples/data/davis_bmx_trees`; `model.pt`; `points_2d_at_t0.pt`; `points_3d_history.pt`; `runs/smoke_real_f1`
Related saved experiments (dataset-level): 

## `scripts/stress_dobbe_3651_joint_calibration.py`

Stress-test a joint K, image warp and camera-offset hypothesis.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `residual`, `metrics`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import sys`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.ndimage import map_coordinates`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from fit_dobbe_3651_effective_k import C, DATA, HOLDOUT, OUT, PAIRS, TRAIN, load_frames, make_matches`
Constants: `ROOT=Path(__file__).resolve().parents[1]`
Input/output file references: `dobbe_3651_joint_calibration_stress.json`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/stress_dobbe_correspondence_quality.py`

Audit duplicate and depth-edge sensitivity of saved causal RGB matches.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `poses_from_labels`, `filtered_group`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `import validate_dobbe_geometry as geometry`
Constants: 
Input/output file references: `_static_correspondences.json`; `correspondence_robustness.json`; `data/dobbe_oxe`; `labels.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/stress_fmb_board_shared_bias.py`

Stress board K against shared RGB feature-localization bias across episodes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import numpy as np`; `from calibrate_fmb_board_k import HOLDOUT_FEATURES, TRAIN, cad_features, extract`; `from calibrate_fmb_multi_boards_k import DATA, EPISODES`; `from calibrate_fmb_two_boards_k import fit_multi`; `from probe_fmb_tcp_tip_k import OUT`
Constants: 
Input/output file references: `.npy`; `multi_board_bundle_calibration.json`; `obs/side_1`; `shared_annotation_bias_stress.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/stress_fmb_cad_pnp.py`

Probe local sensitivity of the candidate three-frame planar PnP fit.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `quantiles`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import ANNOTATIONS, EIGHT, FRONT, K, OUT, SOURCE`
Constants: 
Input/output file references: `cad_pnp_local_sensitivity.json`; `cad_pnp_probe.json`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/summarize_dobbe_colmap_clean.py`

Summarize the four frozen COLMAP controls and independently score causal K.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `validate_causal`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import copy`; `import csv`; `import json`; `import cv2`; `import matplotlib`; `import matplotlib.pyplot as plt`; `import numpy as np`; `from matplotlib.colors import LogNorm`; `from dobbe_colmap_clean import OUT, ROOT, digest, write`; `from validate_dobbe_geometry import evaluate, load_scene`
Constants: `RUNS=['main_full-oracle_pinhole_masked_stride1_all', 'main_causal_pinhole_masked_stride1_all', 'main_causal_pinhole_masked_stride5_all', 'main_causal_pinhole_masked_stride1_all_dsp_affine_guided']`; `LABELS=['Full 243 (oracle)', 'Causal 96', 'Causal stride 5', 'Causal DSP/affine/guided']`
Input/output file references: `../dobbe_rgbd_study/approx_history/manifest.json`; `/summary.json`; `Causal DSP/affine/guided`; `Legacy 9/96 used guessed focal prior, free principal point and relaxed thresholds; it did not establish general COLMAP failure`; `Summarize the four frozen COLMAP controls and independently score causal K.

No intrinsics are fitted to depth/labels. Historical RGB correspondences are
filtered by the frozen foreground masks before applying every candidate K.
`; `causal_geometry_validation.json`; `causal_validation_correspondences.json`; `clean_decision.json`; `experiment_results.csv`; `masks/main/`; `match_graph.json`; `match_pairs.csv`; `runs/dobbe_rgbd_study/approx_history/points_3d_history_candidate.npy`; `runs/dobbe_rgbd_study/main_static_correspondences.json`; `summary.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/summarize_dobbe_rgbd_study.py`

Collect reproducible tables and record the geometry gate outcome.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `read`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `from pathlib import Path`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_rgbd_study'`
Input/output file references: `../dobbe_colmap_clean_rerun_v1/clean_decision.json`; `_static_geometry.json`; `approx_history/manifest.json`; `colmap_*/summary.json`; `colmap_runs.csv`; `decision.json`; `dobbe_colmap_clean_rerun_v1/clean_decision.json`; `future_evaluation.json`; `geometry_metrics.csv`; `main_static_geometry.json`; `model_run.json`; `points_3d_history_candidate.npy`; `protocol.json`; `runs/dobbe_rgbd_study`; `second_static_geometry.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/summarize_fmb_ablation_results.py`

Assemble paired, descriptive FMB ablation effects from saved evaluations.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `import numpy as np`; `from run_fmb_ablation_suite import ROOT, RUNS, SUITE, VARIANTS, write_json`
Constants: 
Input/output file references: `evaluation.json`; `history_points_2d.npy`; `metrics.json`; `paired_effects.csv`; `paired_effects.json`; `runs/fmb_ablation_conclusions_v1`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/summarize_fmb_geometry_forecast.py`

Relate frozen historical geometry diagnostics to observed 2D forecast error.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `pair_distance`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import json`; `from pathlib import Path`; `import matplotlib`; `import matplotlib.pyplot as plt`; `from matplotlib.lines import Line2D`; `import numpy as np`; `from evaluate_fmb_quantitative_2d import interpolate, measure, project`; `from run_fmb_ablation_suite import ROOT, RUNS, write_json`
Constants: `METHODS={'sensor': ('history_sensor_3d.npy', 'prediction_15hz.npy'), 'planar PnP': ('history_pnp_3d.npy', 'variants/pnp/prediction_15hz.npy'), 'CAD silhouette + TCP': ('history_cad_silhouette_tcp_3d.npy', 'variants/cad_silhouette/prediction_15hz.npy'), 'MoGe`
Input/output file references: `/`; `Sensor depth is an imperfect reference, not independent 3D ground truth. CAD/PnP points are approximate correspondences on a smooth face. Two episodes and few methods do not support a population correlation estimate.`; `ablation_suite_v1/moge2_raw/prediction_15hz.npy`; `ablation_suite_v1/moge2_scaled/prediction_15hz.npy`; `comparison.csv`; `comparison.json`; `gt_2d.npy`; `history_cad_silhouette_tcp_3d.npy`; `history_moge2_history_scaled_3d.npy`; `history_moge2_raw_3d.npy`; `history_pnp_3d.npy`; `history_sensor_3d.npy`; `k_nominal.json`; `prediction_15hz.npy`; `runs/fmb_geometry_forecast_comparison_v1`; `validity_mask.npy`; `variants/cad_silhouette/prediction_15hz.npy`; `variants/pnp/prediction_15hz.npy`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/survey_fmb_peg_pose_diversity.py`

Inspect whether additional same-object FMB episodes provide diverse peg views.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `peg_mask`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from calibrate_fmb_multi_boards_k import background_alignment`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `DATA=ROOT / 'data/fmb/single_object_manipulation_dataset'`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `EPISODES=(('vertical', 2), ('vertical', 3), ('vertical', 4), ('vertical', 5), ('vertical', 6), ('horizontal', 0), ('horizontal', 4), ('horizontal', 8))`; `FRACTIONS=(0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1)`
Input/output file references: `.npy`; `data/fmb/single_object_manipulation_dataset`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `obs/side_1`; `peg_pose_diversity_survey.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/survey_fmb_second_candidates.py`

Make compact, unmodified-source previews for second-scene selection.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `sheet`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import csv`; `import hashlib`; `import json`; `from pathlib import Path`; `import numpy as np`; `from PIL import Image, ImageDraw, ImageFont`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `DATA=ROOT.parent / 'data/fmb/single_object_manipulation_dataset'`; `OUT=ROOT / 'runs/fmb_second_scene'`; `FIELDS=['shape', 'size', 'length', 'color', 'angle', 'distractor']`
Input/output file references: `*.npy`; `candidate_survey.json`; `data/fmb/single_object_manipulation_dataset`; `obs/`; `obs/tcp_pose`; `runs/fmb_second_scene`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/survey_fmb_tcp_tip_observability.py`

Survey usable RGB peg-end observations during FMB horizontal episodes.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `ranges`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_tcp_tip_k import red_end`
Constants: `ROOT=Path(__file__).resolve().parents[2]`; `DATA=ROOT / 'data/fmb/single_object_manipulation_dataset'`; `OUT=ROOT / 'molmo-motion/runs/fmb_effective_k_256_calibration'`; `EPISODES=(0, 4, 8)`
Input/output file references: `.npy`; `data/fmb/single_object_manipulation_dataset`; `molmo-motion/runs/fmb_effective_k_256_calibration`; `obs/side_1`; `obs/tcp_pose`; `tcp_tip_observability_horizontal.json`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/sweep_fmb_pnp_focal_depth.py`

Conditional focal sweep for the FMB CAD/RGB fit against interior depth.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `import cv2`; `import numpy as np`; `from scipy.optimize import least_squares`; `from scipy.spatial.transform import Rotation`; `from probe_fmb_cad_pnp import ANNOTATIONS, D, FRONT, H, K, OUT, SOURCE, W`
Constants: 
Input/output file references: `Conditional focal sweep for the FMB CAD/RGB fit against interior depth.

This cannot calibrate color intrinsics: it assumes D405 default depth units,
published RGB/depth interior registration, and a uniform focal adjustment.
`; `Published 256x256 RGB/depth registration is not exact or documented.`; `cad_pnp_focal_depth_sweep.json`; `cad_pnp_probe.json`; `obs/side_1_depth`; `obs/tcp_pose`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/validate_dobbe_geometry.py`

Validate candidate RGB-only K against HoNY measured depth and labels.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `load_scene`, `depth_at`, `choose_correspondences`, `k_candidates`, `get_intrinsics`, `unproject`, `evaluate`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `from pathlib import Path`; `import cv2`; `import numpy as np`; `from scipy.spatial.transform import Rotation`; `from inspect_hony_scenes import ROOT, liblzfse`
Constants: `OUT=ROOT / 'runs/dobbe_rgbd_study'`; `P_OLD=np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64)`; `P_NEW=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)`; `RGB_CW_TO_RAW_OPENGL=np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)`; `CV_TO_OPENGL=np.diag([1.0, -1.0, -1.0])`; `CAMERA_TO_LABEL=P_OLD @ RGB_CW_TO_RAW_OPENGL @ CV_TO_OPENGL`; `PAIRS={'main': [(0, 20), (20, 40), (40, 60), (60, 80), (80, 95)], 'second': [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 120)]}`
Input/output file references: `_static_correspondences.json`; `_static_geometry.json`; `compressed_video_h264.mp4`; `data/dobbe_oxe`; `labels.json`; `runs/dobbe_rgbd_study`; `summary.json`; `u_d=u_r; v_d=(v_r+0.5)*192/256-0.5`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/verify_dobbe_vipe_results.py`

Verify sealed causal inputs and completed/explicitly rejected executions.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `verify`.
Imports/reused functions: `import numpy as np`; `from dobbe_vipe_v1 import RUN, ROOT, GATE, read, sha, write`
Constants: 
Input/output file references: `A/sift_audited`; `A/sift_audited/`; `C/pure_vipe`; `Verify sealed causal inputs and completed/explicitly rejected executions.`; `causal_15hz.mp4`; `evaluation/future_access_receipt.json`; `frame_map.json`; `geometry_gate.json`; `hybrid/model_run.json`; `input_freeze.json`; `model_run.json`; `paired_hybrid/c2w_t0.npy`; `paired_hybrid/points_2d_t0.npy`; `paired_selection_audit.json`; `paired_vipe/c2w_t0.npy`; `paired_vipe/points_2d_t0.npy`; `prediction_15hz.npy`; `prediction_parsed_visibility.npy`; `processor_equivalence.json`; `protocol.json`; `report/dobbe_vipe_results/manifest.json`; `tracks.npz`; `verification.json`; `vipe_no_vda_execution.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `scripts/verify_fmb_molmo_candidate.py`

Offline consistency check for the candidate MolmoMotion output.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `main`.
Imports/reused functions: `import json`; `from pathlib import Path`; `import numpy as np`; `from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=ROOT / 'runs/sharerobot_fmb_episode_5201/cad_history_candidate'`
Input/output file references: `Candidate output failed text/tensor consistency`; `molmo_candidate_future.npy`; `molmo_candidate_validation.json`; `points_3d_history_candidate.npy`; `runs/sharerobot_fmb_episode_5201/cad_history_candidate`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/verify_fmb_quantitative_2d.py`

Independently parse raw MolmoMotion tracks for the frozen FMB 2D experiment.

Dataset: **fmb**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/fmb_wrist_1.json configs/fmb_wrist_2.json`.

Functions: `verify_one`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `import os`; `from pathlib import Path`; `import numpy as np`; `from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `RUN=Path(os.environ.get('FMB_QUANT_RUN', ROOT / 'runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126'))`
Input/output file references: `history_cad_silhouette_tcp_3d.npy`; `history_pnp_3d.npy`; `history_sensor_3d.npy`; `history_sensor_3d_K_focal_0925.npy`; `history_sensor_3d_K_simple.npy`; `parser_validation.json`; `prediction_15hz.npy`; `runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126`
Related saved experiments (dataset-level): fmb_ablation_conclusions_v1, fmb_ablation_prompt_analysis_v1, fmb_colmap_official_pinhole_v1, fmb_colmap_official_v1, fmb_depth_k_factorial_v1, fmb_effective_k_256_calibration, fmb_extended_2d_metrics_v1, fmb_geometry_forecast_comparison_v1, fmb_gt_sensitivity_v1, fmb_K_sensitivity_combined_v1, fmb_moge2_temporal_geometry_v1, fmb_second_example_1_M_L_3_vertical_n_3, fmb_second_scene, fmb_v2_berkeley_matched, fmb_wrist_publication_20261003, fmb_wrist_v3, fmb_wrist_v3_control_horizontal_n0, fmb_wrist_v4_improvement, sharerobot_fmb_episode_5201

## `scripts/visualize_trajectory.py`

Visualize a predicted 3D trajectory from MolmoMotion.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `project_camera_xyz_to_pixel`, `overlay_mode`, `threed_mode`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image, ImageDraw`
Constants: 
Input/output file references: 
Related saved experiments (dataset-level): 

## `scripts/write_dobbe_protocol.py`

Freeze frame indices before geometry estimation and prediction.

Dataset: **dobbe**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Frozen selected experiment: `molmo-motion-experiment --config configs/dobbe.json configs/dobbe_blocked.json`.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import json`; `from pathlib import Path`
Constants: `ROOT=Path(__file__).resolve().parents[1]`; `OUT=ROOT / 'runs/dobbe_rgbd_study/protocol.json'`
Input/output file references: `Only calibration_past and history may enter camera/pose/mapping/point selection; future_evaluation_only is sealed until prediction is frozen.`; `Pick_and_Place/Home15/Env1/2023-04-25--02-05-30`; `Pick_and_Place/Home7/Env1/2023-04-27--10-47-40`; `runs/dobbe_rgbd_study/protocol.json`
Related saved experiments (dataset-level): dobbe_colmap_clean_rerun_v1, dobbe_colmap_official_v1, dobbe_rgbd_study, dobbe_two_colmap_v1, dobbe_vipe_v1, plex_dobbe_preflight

## `examples/01_quickstart.py`

MolmoMotion quickstart: predict a 3D point trajectory and render it as an MP4.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `project_camera_xyz_to_pixel`, `render_trajectory_mp4`, `load_example`, `prediction_from_jsonl`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `from pathlib import Path`; `import numpy as np`; `import torch`; `from PIL import Image`
Constants: `EXAMPLES_DIR=Path(__file__).parent / 'data'`
Input/output file references: `Load the released-model ``(P, F, 3)`` prediction for ``video`` from the
    eval JSONL (``pred_raw_combined``) -- identical in shape/units to
    ``out.future_3d``.`; `Sub-directory of examples/data/ to run on.`; `_2d.mp4`; `allenai/MolmoMotion-4B-H3-F30`; `intrinsics_K.pt`; `meta.json`; `points_2d_at_t0.pt`; `points_3d_history.pt`
Related saved experiments (dataset-level): 

## `launch_scripts/eval.py`

Evaluation entry point for MolmoMotion.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import sys`; `from molmo_motion.eval.model_evaluator import EvalConfig`; `from molmo_motion.exceptions import OLMoCliError`; `from molmo_motion.util import clean_opt, prepare_torchrun_environment`
Constants: 
Input/output file references: ` configs/eval_h3.yaml --load_path=outputs/step20000`
Related saved experiments (dataset-level): 

## `launch_scripts/eval_pointmotionbench.py`

PointMotionBench all-point evaluation driver.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `_compute_metrics`, `main`.
Imports/reused functions: `from __future__ import annotations`; `import argparse`; `import json`; `import os`; `import sys`; `from pathlib import Path`; `import numpy as np`
Constants: `_BENCH_TO_SUFFIX={'hot3d': 'hot3d_bench', 'worldtrack': 'worldtrack_bench', 'davis': 'davis_bench'}`; `_PWT_THRESHOLDS=(0.01, 0.02, 0.05, 0.1, 0.2)`
Input/output file references: `metrics.json`; `summary.json`
Related saved experiments (dataset-level): 

## `launch_scripts/sft.py`

Recipe training entry point for MolmoMotion.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `get_model`, `get_training_mixture`, `main`.
Imports/reused functions: `import argparse`; `import dataclasses`; `import os`; `from os.path import join`; `from typing import List`; `from omegaconf import OmegaConf, omegaconf`; `from molmo_motion.data.data_loader import WeightedDataset, KwargsMixture, DataLoaderConfig`; `from molmo_motion.data.dynamic_packer import PackingConfig`; `from molmo_motion.models.molmo.molmo import MolmoConfig`; `from molmo_motion.models.molmo2.molmo2 import Molmo2Config`; `from molmo_motion.models.molmo2.molmo2_preprocessor import Molmo2PreprocessorConfig`; `from molmo_motion.preprocessing.multicrop_preprocessor import MultiCropConfig`; `from molmo_motion.preprocessing.video_preprocessor import VideoPreprocessorConfig`; `from molmo_motion.torch_util import get_world_size`; `from molmo_motion.train.optim import OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType`; `from molmo_motion.train.run_trainer import run_trainer`; `from molmo_motion.train.trainer_config import FSDPConfig, BatchDivisor, SpeedMonitorConfig, TrainConfig, WandbConfig, CompilerConfig`; `from molmo_motion.util import prepare_torchrun_environment, select_checkpoint, clean_opt`
Constants: 
Input/output file references: ` is not part of the public release. Only 'trajectory_3d_*' mixtures are supported — see the dataset-name grammar in molmo_motion/data/get_dataset.py.`
Related saved experiments (dataset-level): 

## `launch_scripts/train.py`

YAML-config training entry point for MolmoMotion.

Dataset: **shared**. Retained in place; historical preparation and diagnostics remain available.
Replacement: No full replacement; archival script retained.
Shared/author script; see the final configurations in README.

Functions: `main`.
Imports/reused functions: `from __future__ import annotations`; `import sys`; `from molmo_motion.exceptions import OLMoCliError`; `from molmo_motion.train.run_trainer import run_trainer`; `from molmo_motion.train.trainer_config import TrainConfig`; `from molmo_motion.util import clean_opt, prepare_torchrun_environment`
Constants: 
Input/output file references: ` configs/pretrain_h3.yaml --save_folder=outputs/run1`
Related saved experiments (dataset-level): 

