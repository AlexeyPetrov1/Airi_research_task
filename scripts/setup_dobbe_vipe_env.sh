#!/usr/bin/env bash
# Separate from the working MolmoMotion environment; no grounding-model stack.
set -euo pipefail
TASK_BASE=/mnt/f/AIRI_task
TASK_UV="$TASK_BASE/.tools/uv_linux/bin/uv"
TASK_ENV="$TASK_BASE/.venv-vipe"
mkdir -p "$TASK_BASE/molmo-motion/runs/dobbe_vipe_v1"
# Keep the new build/download cache on ext4: DrvFS rename races interrupt uv.
export UV_CACHE_DIR=/home/lesha/.cache/airi-vipe-uv
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
export USE_SYSTEM_EIGEN=1
export CPLUS_INCLUDE_PATH=/usr/include/eigen3
if ! test -x "$TASK_ENV/bin/python"; then
  "$TASK_UV" venv --python "$TASK_BASE/.venv/bin/python" "$TASK_ENV"
fi
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" setuptools wheel ninja numpy scipy scikit-learn matplotlib pillow opencv-python imageio imageio-ffmpeg av OpenEXR einops omegaconf hydra-core tqdm rerun-sdk python-pycg timm transformers==4.57.1 gdown==5.2.0 kornia viser click prettytable tensorboard huggingface-hub accelerate
# PyTorch CUDA wheels provide the CUDA library headers (cusparse/cublas/etc.).
for task_include in "$TASK_ENV"/lib/python*/site-packages/nvidia/*/include; do
  export CPATH="${CPATH:+$CPATH:}$task_include"
done
cd "$TASK_BASE/molmo-motion/data_generation/third_party/vipe"
"$TASK_UV" pip install --python "$TASK_ENV/bin/python" --no-build-isolation --no-deps -e .
"$TASK_UV" pip freeze --python "$TASK_ENV/bin/python" > "$TASK_BASE/molmo-motion/runs/dobbe_vipe_v1/environment.freeze.txt"
"$TASK_ENV/bin/python" -m vipe.cli.main --help
