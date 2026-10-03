# FMB wrist v3 reproduction

Use WSL Ubuntu from `F:/AIRI_task/molmo-motion`. MolmoMotion/AllTracker/SAM/UniDepth use `/mnt/f/AIRI_task/.venv/bin/python`; ViPE and EXR geometry use `/mnt/f/AIRI_task/.venv-vipe/bin/python`. The official raw FMB NPY, pinned MolmoMotion checkpoint, SAM2.1, AllTracker, UniDepth and native ViPE priors are cached locally. Pinned capture/calibration sources and hashes are in `sources/source_receipts.json`.

Start a fresh run directory rather than replacing this experiment. In a WSL shell:

```bash
cd /mnt/f/AIRI_task/molmo-motion
export FMB_WRIST_RUN=/mnt/f/AIRI_task/molmo-motion/runs/fmb_wrist_reproduction
export HF_HOME=/mnt/f/AIRI_task/.cache/huggingface
export TORCH_HOME=/mnt/f/AIRI_task/.cache/torch
```

Preparation and observed-only selection:

```bash
mkdir -p "$FMB_WRIST_RUN"
cp -a runs/fmb_wrist_v3/sources "$FMB_WRIST_RUN/sources"
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_prepare.py inspect
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_target_probe.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_prepare.py export --source 1_M_L_3_vertical_n_2.npy --t0 126
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_observed_audit.py
```

Observed author models, using 50 genuine frames at nominal10Hz:

```bash
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_vipe.py --cameras wrist_2 wrist_1
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_ground.py --scene-dir "$FMB_WRIST_RUN/wrist_2" "$FMB_WRIST_RUN/wrist_1"
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py sam
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py track
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py unidepth
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py h3-masks
```

Calibration/aspect controls and final same24-ID export:

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_robot_rgbd_calibration.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_rank_candidates.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_rectified_vipe.py
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_geometry.py diagnose
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_resample_candidates.py
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_geometry.py export
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py track
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_visuals.py
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_crossview.py
/mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_native_control.py
```

The resampling control preserves the original100 candidates, selects100 author KMeans queries on the observedt0 SAM/depth-supported interior, and reruns observed AllTracker. All eligibility thresholds and final24-point/group ordering remain fixed. Export refuses to replace an existing frozen geometry selection.

Before inference, inspect historical RGB, SAM, selected-point tracks, pose/K/depth plots and cross-view images with the assistant's built-in `view_image`. Save honest judgments, source frame IDs and image hashes to `builtin_visual_review_observed.json`, and freeze geometry/selection/X/K hashes. The companion `fmb_wrist_record_observed_review.py` records judgments from the original session; reproducing a new run requires actual image inspection and appropriately updated judgments. It also writes the provisional physical-certification fields in branch selection before freezing them. It is not an automatic vision model. The copied source snapshots retain the original pinned commits; calling `prepare.py sources` instead would resolve the then-current remote main branches.

Nine successful P8 calls on wrist_2 plus three C replication calls on wrist_1:

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_v2_infer.py --scene-dir \
  "$FMB_WRIST_RUN/wrist_2/branches/A_vipe_full" \
  "$FMB_WRIST_RUN/wrist_2/branches/B_sensor_vipe" \
  "$FMB_WRIST_RUN/wrist_2/branches/C_sensor_tcp_official" \
  "$FMB_WRIST_RUN/wrist_1/branches/C_sensor_tcp_official"
```

Keep GPU work serialized. In the original session a first incomplete A/group0 call was explicitly terminated because another DAS workload occupied the same GPU. Its real processor payload/status and interruption receipt were archived under `predictions/interrupted_resource_attempt_00`. Twelve successful outputs plus one interrupted attempt are disclosed. Incomplete generations must be inspected and preserved before any explicit retry; the runner does not silently retry them.

Future is opened only after successful frozen predictions:

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py export
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py track
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py evaluate
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_blinded_media.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_reference_uncertainty.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_future_crossview.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_side_future.py
```

Inspect blinded forecast sheets before reading `blinding_map.json` or metrics, and inspect all24 future reference IDs separately. Save the first actual qualitative judgments to `builtin_blinded_review.json` **before** revealing method names or reading numerical scores. Then unblind, inspect the full30-step XYZ plots, error curves, reference sensitivity and side/wrist controls, and save the final interpretation to `builtin_visual_review_future.json`, preserving the first receipt and its hash. The serializers `fmb_wrist_record_blinded_review.py` and `fmb_wrist_record_future_review.py` contain judgments from this session; a new run needs new actual image inspection and appropriately updated judgments. No external Hugging Face VLM is used.

The common reference uses sensor local5×5 median, official K prior, frozen observed X and evaluation-only future TCP. All H3/model/reference XYZ live in camera_t0; future projection compensates camera motion. Source10Hz and model15Hz are aligned by interpolating XYZ at nominal times. Off-image predictions remain in errors. Bootstrap calibration sensitivity is descriptive, not a confidence interval or certified physical GT.

Final report and audit:

```bash
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_report.py
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_video_gallery.py
/mnt/f/AIRI_task/.venv/bin/python -m pytest tests/test_fmb_wrist_math.py -q
/mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_completion_audit.py
```

Before running the audit on a new directory, record the actual pytest exit code/output and current SHA256 of `tests/test_fmb_wrist_math.py`, `scripts/fmb_wrist_math.py`, and `scripts/fmb_wrist_evaluate.py` in `coordinate_test_receipt.json`. Independently verify the source-matching provenance and preserve/hash prior runs; do not copy success claims into a new receipt. Build the gallery before opening the report's video-gallery link. In this session a pending `completion_audit.json` existed before the final report generation, so the report includes its link; create a pending false-success receipt before building a new final report if that link is desired.

The completion audit checks real native outputs/parser/payloads, hashes, videos, requirements and preservation of914 FMBv1 files and FMBv2 report/audit. It explicitly retains failed cross-view/physical-geometry validation. Its original-session interrupted-attempt check and forecast-controller status receipt should be adapted for an independently reproduced attempt ledger; never fabricate that interruption in a new run. The original audit is a verification of the recorded experiment, not an assertion that a new run will reproduce identical predictions or calibration estimates.
