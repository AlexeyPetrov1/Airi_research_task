# FMB v2 reproduction

Run from `molmo-motion` inside Linux/WSL using the existing CUDA environment. The original FMB datasets and checkpoints are external payloads. This run uses an RTX 4070, 12 GiB VRAM; do not run another GPU workload concurrently.

Existing v2 outputs are deliberately protected. For a fresh repetition, use a separate repository checkout and empty `runs/fmb_v2_berkeley_matched`; do not delete or overwrite FMB v1. Scripts export only the selected observed indices before inference, despite the official pickle dictionary containing the full source recording.

Required local sources:

- `runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy`, SHA-256 in the saved provenance receipt.
- `../data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_3.npy`.
- Source provenance files and pinned ShareRobot trajectory manifest from the existing repository.
- `../models/MolmoPoint-Vid-4B`, revision `b331ed6c6352e6db967325493c6dae515541a919`.
- `../models/sam2.1_hiera_large.pt` and `../third_party/sam2`.
- `../models/unidepth-v2-vits14`, revision `038c238f06c87b6c2f5b3749fd51fbf442b1f218`, and `../third_party/UniDepth`.
- `../.cache/torch/hub/checkpoints/alltracker.pth`; vendored AllTracker and author track-filter-smooth code.
- H3 and H1 checkpoints below. Use the native `model.pt` loader, not a different quantization.

The environment used PyTorch `2.9.1+cu128`, Transformers `4.57.6`. Missing preparation dependencies were installed as `hydra-core==1.3.2`, `iopath==0.1.10`, `portalocker==3.2.0`, `scikit-learn==1.7.2`, `joblib==1.5.2`, `threadpoolctl==3.6.0`. Other dependencies are those in the established Berkeley environment and repository setup. No model/runtime upgrade was performed.

```bash
PY=../.venv/bin/python
HF=../.venv/bin/hf
MAIN=runs/fmb_v2_berkeley_matched/episode_5201
CONTROL=runs/fmb_v2_berkeley_matched/fmb_control_n3

# Selective download of two source recordings from the pinned public mirror.
# The downloader checks length and SHA-256 against the Hub, and verifies existing files.
$PY scripts/download_fmb_raw_candidate.py 1_M_L_3_vertical_n_2.npy
$PY scripts/download_fmb_raw_candidate.py 1_M_L_3_vertical_n_3.npy
# The main source-audit folder is part of the existing v1 provenance evidence.
cp -n ../data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_2.npy \
  runs/sharerobot_fmb_episode_5201/1_M_L_3_vertical_n_2.npy

$HF download allenai/MolmoMotion-4B-H3-F30 model.pt config.yaml \
  --revision 3f5e790a511ff2cdf21c8d2a14cb4d8409c94629 \
  --local-dir data/checkpoints/MolmoMotion-4B-H3-F30
HF_HUB_DISABLE_XET=1 $HF download allenai/MolmoMotion-4B-H1-F32 model.pt config.yaml \
  --revision 14e69b0d5dd55b2f09c885030b8756cd053ea6a4 \
  --local-dir data/checkpoints/MolmoMotion-4B-H1-F32

$PY scripts/fmb_v2_environment.py
$PY scripts/fmb_v2_checkpoint_verify.py
$PY scripts/fmb_v2_scene.py all
$PY scripts/fmb_v2_search_share.py
$PY scripts/fmb_v2_ground.py --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_preprocess.py depth --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_preprocess.py ground --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_preprocess.py track --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_preprocess.py observed-masks --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_preprocess.py filter --scene-dir "$MAIN" "$CONTROL"
$PY -m pytest tests/test_fmb_v2_protocol.py -q

# Inspect mask overlays, registration overlays, camera-flow checks and selected points.
# These are data-quality reviews before either model inference.
$PY scripts/fmb_v2_infer.py --mode h3 --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_infer.py --mode h1 --scene-dir "$MAIN"
$PY scripts/fmb_v2_scene.py all --future
$PY scripts/fmb_v2_evaluate.py track --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_evaluate.py evaluate --scene-dir "$MAIN" "$CONTROL"
$PY scripts/fmb_v2_evaluate.py audit-reference --scene-dir "$CONTROL"

# Inspect future reference/video samples and write a truthful visual_review.json.
# fmb_v2_report.py requires that review receipt and checks all completion evidence.
$PY scripts/fmb_v2_report.py
```

The author trust filter has an upstream logging bug when all dropped tracks have no visible samples: `dropped_mw.min()` receives an empty array. The FMB adapter guards only this diagnostic print, leaving every trust/filter computation and the vendored file unchanged. A regression test covers the case.

`predictions/input_freeze.json` seals metadata, observations, geometry and groups before H3. H1 uses the same frozen t0 coordinates and final XYZ. Each model call saves its actual CPU processor input, anchor, hashes, raw text, strict parser result, timing and peak CUDA memory. Inference can recover an already saved full output without generating a new one; an incomplete generation requires inspection rather than an automatic retry.

`completion_audit.json` verifies six complete H3 calls and three complete H1 calls, same-ID references, unchanged inputs and FMB v1 artifacts, common masks, 20-frame 10-FPS videos and visual review. All time/metric depth units remain subject to the nominal frequency and depth-scale qualifications in the report.
