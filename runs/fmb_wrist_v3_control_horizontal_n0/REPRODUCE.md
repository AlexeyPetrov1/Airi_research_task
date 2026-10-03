# FMB wrist v3 reproduction

Run from `F:/AIRI_task/molmo-motion`, using WSL Ubuntu. The working MolmoMotion environment is `/mnt/f/AIRI_task/.venv`; the separate installed ViPE environment is `/mnt/f/AIRI_task/.venv-vipe`. Existing raw FMB NPY, checkpoints, SAM2.1, AllTracker, and ViPE priors are required. Official sources are pinned in `sources/source_receipts.json`. Source NPY is trusted official data with pickle dictionaries.

Preparation (CPU; selection never opens future pixels):

```powershell
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_prepare.py sources
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_prepare.py inspect
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_target_probe.py
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_prepare.py export --source 1_M_L_3_vertical_n_2.npy --t0 126
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_observed_audit.py
```

Native default ViPE, sequential moving views, at nominal source 10 Hz:

```powershell
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_vipe.py --cameras wrist_2 wrist_1
```

Observed grounding/tracking/depth comparison:

```powershell
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_ground.py --scene-dir runs/fmb_wrist_v3/wrist_2 runs/fmb_wrist_v3/wrist_1
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py sam
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py track
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py unidepth
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_preprocess.py h3-masks
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_geometry.py diagnose
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_geometry.py export
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv-vipe/bin/python scripts/fmb_wrist_crossview.py
```

Export refuses to replace frozen geometry selections. Inspect the saved observed images using the assistant's built-in `view_image` capability and save `builtin_visual_review_observed.json` with image hashes, frames, explicit judgments and uncertainty before inference. No external Hugging Face VLM is used. Diagnostic branches must retain their historical gate failures and cannot be called validated primary geometry.

H3 inference: use `scripts/fmb_v2_infer.py --scene-dir` on the three directories under `runs/fmb_wrist_v3/wrist_2/branches/`, and on the accepted primary replication directory under wrist_1. The runner seals observed inputs, action, IDs, timestamp metadata and actual processor payload. It audits all 8×30×3 outputs and anchors. Incomplete generation attempts cannot be silently retried.

Only after successful primary scene calls and observed visual review:

```powershell
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py export
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py track
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python scripts/fmb_wrist_evaluate.py evaluate
```

Future camera poses are robot TCP × the frozen observed-only hand-eye transform. The common reference also uses sensor depth under the explicit scale hypothesis, and K chosen from historical validation. Future data is strictly evaluation-only. 3D errors are estimated geometry errors, not calibrated physical ground truth. Source 10 Hz and model 15 Hz are aligned by interpolating XYZ at time coordinates, not by frame number.

Inspect saved reference and forecast review sheets with built-in image inspection and save the final review receipt. A visual diagnosis is not an independently reproducible external VLM score.

Meaningful coordinate tests:

```powershell
wsl -d Ubuntu --exec /mnt/f/AIRI_task/.venv/bin/python -m pytest tests/test_fmb_wrist_math.py -q
```

All prior FMB files checked in the preservation receipt remain byte-identical. The process wrappers resume only when their specific live process has ended and required completion receipts are successful; they do not infer completion from elapsed time.
