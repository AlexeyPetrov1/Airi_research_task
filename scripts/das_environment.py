"""Inspect local packages or record pinned checkpoint metadata without loading GPU models."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', action='store_true')
p.add_argument('--revision',default='a116c52b08182f7a5916eae7a47fcd61a9467a04')
p.add_argument('--output', type=Path)
a = p.parse_args()
if a.checkpoint:
    from huggingface_hub import HfApi
    info = HfApi().model_info('alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control', revision=a.revision, files_metadata=True)
    result = {'repo_id': info.id, 'revision': info.sha,
              'files': [{'path': f.rfilename, 'bytes': f.size} for f in info.siblings]}
else:
    result = {}
    for name in ['torch', 'torchvision', 'numpy', 'diffusers', 'transformers', 'accelerate',
                 'huggingface-hub', 'omegaconf', 'einops', 'ftfy', 'sentencepiece',
                 'imageio-ffmpeg', 'safetensors', 'psutil', 'opencv-python', 'matplotlib', 'scipy']:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
if a.output:
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
