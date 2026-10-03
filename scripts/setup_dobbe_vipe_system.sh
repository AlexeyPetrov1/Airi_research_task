#!/usr/bin/env bash
# Run as WSL root. Installs only build tools and CUDA compiler/runtime headers.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends build-essential libeigen3-dev libgl1 libglib2.0-0 curl ca-certificates
if ! test -x /usr/local/cuda-12.8/bin/nvcc; then
  curl -fL --retry 3 https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/airi-cuda-keyring.deb
  dpkg -i /tmp/airi-cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-cudart-dev-12-8
fi
/usr/local/cuda-12.8/bin/nvcc --version
