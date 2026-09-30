"""Check the isolated WSL environment without running model inference."""

from __future__ import annotations

import json
import platform
import resource
import sys
from importlib.metadata import version
from pathlib import Path

import av
import cv2
import decord
import imageio
import numpy
import torch
from PIL import Image
from torchcodec.decoders import VideoDecoder

from molmo_motion import MolmoMotion, MolmoMotionProcessor


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    sample = repo / "data_generation/third_party/vipe/assets/examples/dog-example.mp4"
    assert sample.is_file(), f"Missing bundled decoder sample: {sample}"
    assert torch.cuda.is_available(), "PyTorch cannot access the NVIDIA GPU"

    value = torch.tensor([2.0, 3.0], device="cuda")
    assert torch.equal((value * value).cpu(), torch.tensor([4.0, 9.0]))
    torch.cuda.synchronize()

    cap = cv2.VideoCapture(str(sample))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, "OpenCV failed to decode the bundled video"

    reader = decord.VideoReader(str(sample), ctx=decord.cpu(0), num_threads=1)
    assert len(reader) > 0 and reader[0].shape[2] == 3
    with av.open(str(sample)) as container:
        decoded = next(container.decode(video=0))
        assert decoded.width > 0 and decoded.height > 0
    torchcodec_reader = VideoDecoder(str(sample), device="cpu")
    assert len(torchcodec_reader) > 0 and torchcodec_reader[0].numel() > 0

    print(json.dumps({
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "peak_cuda_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 2),
        "peak_process_ram_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "opencv": cv2.__version__,
        "pyav": av.__version__,
        "decord": decord.__version__,
        "torchcodec": version("torchcodec"),
        "torchcodec_frames": len(torchcodec_reader),
        "imageio": imageio.__version__,
        "numpy": numpy.__version__,
        "pillow": Image.__version__,
        "molmo_motion_imports": [MolmoMotion.__name__, MolmoMotionProcessor.__name__],
        "video_frames": len(reader),
        "video_first_frame_shape": list(frame.shape),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
