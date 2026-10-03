"""Resume official Wan VAE decode from saved diffusion latents, without denoising."""
import argparse
import json
from pathlib import Path
import time
import torch
from accelerate import cpu_offload
from omegaconf import OmegaConf
from das_wanfun_runtime import setup
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--artifact-dir',type=Path,required=True)
p.add_argument('--das-root',type=Path,required=True);p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
a=p.parse_args();out=a.artifact_dir
target=out/'generated_molmomotion_seed42.mp4'
if target.exists():raise FileExistsError('Preserve existing video')
setup(a.das_root)
from videox_fun.models import AutoencoderKLWan
from videox_fun.utils.utils import save_videos_grid
torch.set_num_threads(4)
if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
start=time.monotonic()
saved=torch.load(out/'generated_latents.pt',map_location='cpu',weights_only=True)
config=OmegaConf.load(a.das_root/'config/wan2.1/wan_civitai.yaml')
vae=AutoencoderKLWan.from_pretrained(saved['vae_checkpoint'],additional_kwargs=OmegaConf.to_container(config['vae_kwargs'])).to(torch.bfloat16)
vae.eval()
if a.device=='cuda':cpu_offload(vae,execution_device=torch.device('cuda:0'),offload_buffers=True)
print('Decode saved diffusion latents on',a.device,flush=True)
with torch.inference_mode():
    decoded=vae.decode(saved['latents'].to(a.device,torch.bfloat16)).sample
    video=(decoded/2+.5).clamp(0,1).cpu().float()
save_videos_grid(video,str(target),fps=8)
if a.device=='cuda':torch.cuda.synchronize()
receipt={'success':True,'method':'Original official Wan VAE decode and save_videos_grid from frozen denoising latents; no denoising rerun',
         'device':a.device,'dtype':'bfloat16','seconds':time.monotonic()-start,
         'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if a.device=='cuda' else 0,
         'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if a.device=='cuda' else 0,'latent_sha256':sha256(out/'generated_latents.pt'),
         'video_sha256':sha256(target),'vae_checkpoint':saved['vae_checkpoint'],'das_commit':saved['das_commit']}
write_json(out/'decode_recovery.json',receipt)
print(json.dumps(receipt,indent=2))
