"""CLI around official DaS Wanfun inference with stage/resource evidence."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import types
import numpy as np
import psutil
from PIL import Image
from das_prepare_control import sha256, write_json
from das_wanfun_runtime import load_official_infer, setup


def research_branch(root):
    """Read HEAD even when Windows created the worktree and WSL runs Python."""
    gitdir=root/'.git'
    if gitdir.is_file():
        pointer=gitdir.read_text().strip().removeprefix('gitdir: ')
        if len(pointer)>2 and pointer[1]==':':
            pointer='/mnt/'+pointer[0].lower()+'/'+pointer[3:].replace('\\','/')
        gitdir=Path(pointer)
        if not gitdir.is_absolute():gitdir=root/gitdir
    head=(gitdir/'HEAD').read_text().strip()
    return head.removeprefix('ref: refs/heads/')


def main():
    p = argparse.ArgumentParser()
    for flag in ['image','checkpoint_path','output_path','das_root']:
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--tracking_path', type=Path)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_inference_steps', type=int, default=25)
    p.add_argument('--import-only', action='store_true')
    p.add_argument('--min-free-vram-gib',type=float,default=9)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--reference-mode',choices=['original','initial-only','clean-background'],default='original')
    p.add_argument('--reference-image',type=Path)
    p.add_argument('--vacancy-strength',type=float,default=0.)
    p.add_argument('--vacancy-context-px',type=int,default=0)
    p.add_argument('--background-bundle',type=Path)
    p.add_argument('--reference-alpha',type=Path)
    a = p.parse_args()
    if a.reference_mode=='clean-background' and a.reference_image is None:p.error('clean-background requires --reference-image')
    if not 0<=a.vacancy_strength<=1:p.error('vacancy-strength must be between zero and one')
    if not 0<=a.vacancy_context_px<=64:p.error('vacancy-context-px must be between zero and 64')
    if a.vacancy_strength and (a.reference_mode!='clean-background' or not a.background_bundle or not a.reference_alpha):
        p.error('vacancy prior requires clean-background reference, background-bundle, and reference-alpha')
    out = a.output_path.parent
    out.mkdir(parents=True, exist_ok=True)
    if a.import_only:
        setup(a.das_root)
        from videox_fun.pipeline import WanFunControlPipeline
        print('Import PASS', WanFunControlPipeline)
        return
    is_main = a.tracking_path is not None
    stem = '' if is_main else '_no_control'
    log = open(out/f'generation{stem}.log','a',encoding='utf8',buffering=1)
    class Tee:
        def __init__(self, original): self.original=original
        def write(self,text): self.original.write(text); log.write(text)
        def flush(self): self.original.flush(); log.flush()
        def isatty(self): return False
    sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
    import torch
    torch.set_num_threads(4)
    if a.output_path.exists():
        raise FileExistsError(f'Refusing to overwrite {a.output_path}')
    state = {'stage':'waiting_for_gpu','success':False,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             'peak_process_rss_bytes':0,'peak_system_ram_used_bytes':0,
             'peak_cuda_allocated_bytes':0,'peak_cuda_reserved_bytes':0,'peak_device_used_bytes':0}
    started = time.monotonic()
    finished = threading.Event()
    def monitor():
        while not finished.is_set():
            process = psutil.Process()
            rss = process.memory_info().rss
            ram = psutil.virtual_memory()
            state['peak_process_rss_bytes'] = max(state['peak_process_rss_bytes'],rss)
            state['peak_system_ram_used_bytes'] = max(state['peak_system_ram_used_bytes'],ram.total-ram.available)
            if torch.cuda.is_initialized():
                free,total = torch.cuda.mem_get_info()
                state['peak_device_used_bytes'] = max(state['peak_device_used_bytes'],total-free)
                state['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
                state['peak_cuda_reserved_bytes'] = torch.cuda.max_memory_reserved()
            state['wall_seconds'] = time.monotonic()-started
            write_json(out/f'resource_usage{stem}.json',state)
            finished.wait(1)
    sampler = threading.Thread(target=monitor,daemon=True)
    sampler.start()
    def stage(name):
        state['stage']=name
        print(f'STAGE {name}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    try:
        if a.wait_pid:
            while psutil.pid_exists(a.wait_pid):
                print(f'Waiting for existing GPU experiment PID {a.wait_pid}',flush=True)
                time.sleep(15)
        free,total = torch.cuda.mem_get_info()
        while free < a.min_free_vram_gib*2**30:
            print(f'GPU busy: {free/2**30:.2f} GiB free; waiting 15 s',flush=True)
            time.sleep(15)
            free,total = torch.cuda.mem_get_info()
        state['queue_wait_seconds']=time.monotonic()-started
        # Keep a small real CUDA allocation during CPU model loading so other
        # queue implementations can see this process in nvidia-smi.
        gpu_presence_guard=torch.ones(262144,device='cuda',dtype=torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if is_main:
            validation = json.loads((out/'control_validation.json').read_text(encoding='utf-8-sig'))
            if not validation.get('generation_ready') or validation['control_sha256']!=sha256(a.tracking_path):
                raise RuntimeError('Prepared control requires recorded visual review and matching hash')
        elif not (out/'generated_molmomotion_seed42.mp4').exists() or not json.loads((out/'resource_usage.json').read_text()).get('success'):
            raise RuntimeError('Run and validate the controlled generation before the optional ablation')
        config = {'model':'alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control','checkpoint':str(a.checkpoint_path),
            'checkpoint_revision':json.loads((out/'checkpoint_receipt.json').read_text())['revision'],
            'git_commit':subprocess.check_output(['git','-C',str(a.das_root),'rev-parse','HEAD'],text=True).strip(),
            'research_branch':research_branch(Path(__file__).resolve().parents[1]),
            'dtype':'bfloat16','seed':a.seed,'num_inference_steps':a.num_inference_steps,
            'num_frames':49,'resolution':[720,480],'fps':8,'GPU':torch.cuda.get_device_name(),
            'VRAM_bytes':total,'RAM_bytes':psutil.virtual_memory().total,'RAM_scope':'WSL Linux memory limit; physical Windows host has 32 GiB',
            'offload':'official enable_sequential_cpu_offload(device=cuda:0)',
            'prompt':a.prompt,'control_video':str(a.tracking_path) if is_main else None,
            'no_control_method':None if is_main else 'Official control_video=None branch; zero latent condition; no black video VAE encoding',
            'command':sys.argv,'image_sha256':sha256(a.image),
            'control_sha256':sha256(a.tracking_path) if is_main else None,
            'control_variant':validation.get('variant','original_sparse_measured_background') if is_main else None,
            'reference_mode':a.reference_mode,
            'persistent_reference_image_sha256':sha256(a.reference_image) if a.reference_image else sha256(a.image) if a.reference_mode=='original' else None,
            'vacancy_strength':a.vacancy_strength,
            'vacancy_context_px':a.vacancy_context_px,
            'vacancy_background_bundle_sha256':sha256(a.background_bundle) if a.background_bundle else None,
            'vacancy_reference_alpha_sha256':sha256(a.reference_alpha) if a.reference_alpha else None,
            'runtime_adapter':'AST extraction removes only unused local imports; missing distributed module shim for one GPU',
            'guidance_scale':6.0,'scheduler':'FlowMatchEulerDiscreteScheduler','teacache_threshold':0.10}
        write_json(out/f'config{stem}.json',config)
        package_names = {dist.metadata['Name'] for dist in importlib.metadata.distributions()}
        packages = sorted((name,importlib.metadata.version(name)) for name in package_names)
        (out/f'environment{stem}.txt').write_text(platform.platform()+'\n'+sys.version+'\n'+'\n'.join(f'{n}=={v}' for n,v in packages)+'\n',encoding='utf8')
        (out/f'gpu_info{stem}.txt').write_text(subprocess.check_output(['nvidia-smi'],text=True),encoding='utf8')
        stage('imports')
        infer = load_official_infer(a.das_root)
        from videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel
        from videox_fun.pipeline import WanFunControlPipeline
        for model_cls, name in [(WanTransformer3DModel,'load_transformer'),(AutoencoderKLWan,'load_vae'),
                                (WanT5EncoderModel,'load_text_encoder'),(CLIPModel,'load_clip_image_encoder')]:
            original = model_cls.from_pretrained
            def wrapped(cls,*args,_original=original,_name=name,**kwargs):
                stage(_name)
                return _original(*args,**kwargs)
            model_cls.from_pretrained = classmethod(wrapped)
        original_call = WanFunControlPipeline.__call__
        def measured_call(self,*args,**kwargs):
            # Official no-control route; the unused tracking tensor is not encoded.
            if not is_main: kwargs['control_video']=None
            # Native reference ablation: keep original start_image and CLIP,
            # remove only the full_ref spatial image tokens repeated at each step.
            if a.reference_mode=='initial-only':kwargs['ref_image']=None
            elif a.reference_mode=='clean-background':
                empty=np.array(Image.open(a.reference_image).convert('RGB'))
                assert empty.shape==(480,720,3)
                kwargs['ref_image']=torch.from_numpy(empty.copy()).permute(2,0,1)[None,:,None].float()/255
            stage('pipeline_encode_and_denoise')
            original_encode, original_decode = self.vae.encode, self.vae.decode
            image_latents=[]
            def encode(*args,**kwargs):
                stage('vae_encode'); value=original_encode(*args,**kwargs)
                if a.vacancy_strength and args[0].shape[2]==1:
                    image_latents.append(value.latent_dist.mode().detach())
                stage('pipeline'); return value
            def decode(*args,**kwargs):
                stage('save_generated_latents')
                torch.save({'latents':args[0].detach().cpu(),'vae_checkpoint':str(a.checkpoint_path/'Wan2.1_VAE.pth'),
                            'dtype':'bfloat16','das_commit':config['git_commit']},out/f'generated_latents{stem}.pt')
                stage('vae_decode'); return original_decode(*args,**kwargs)
            self.vae.encode,self.vae.decode=encode,decode
            if a.vacancy_strength:
                from das_vacancy_prior import make_latent_masks,VacancyPrior
                bundle=np.load(a.background_bundle)
                alpha=np.array(Image.open(a.reference_alpha).convert('L'))/255.
                masks=make_latent_masks(alpha,bundle['moving_mask'],a.vacancy_context_px)
                prior=VacancyPrior(masks,a.vacancy_strength,a.seed+9001,image_latents)
                kwargs['callback_on_step_end']=prior
                kwargs['callback_on_step_end_tensor_inputs']=['latents']
            self.transformer.register_forward_pre_hook(lambda *unused: stage('denoise'))
            result=original_call(self,*args,**kwargs)
            if a.vacancy_strength:
                np.save(out/'vacancy_latent_masks.npy',masks)
                write_json(out/'vacancy_prior_receipt.json',{'applied_steps':prior.applied_steps,
                    'strength':a.vacancy_strength,'context_pixels':a.vacancy_context_px,'noise_seed':a.seed+9001,'protected_start_latent':bool(not masks[0].any()),
                    'latent_mask_shape':list(masks.shape),'latent_mask_mean':float(masks.mean()),
                    'method':'After each FlowMatch Euler step, blend only vacated-site latents with correctly noised clean-reference latents at sigma[index+1]. Foreground-touching cells protected.',
                    'native_DaS':False,'postprocessed_RGB':False})
            return result
        WanFunControlPipeline.__call__=measured_call
        image = np.array(Image.open(a.image).convert('RGB'))
        assert image.shape==(480,720,3)
        if is_main:
            import imageio.v2 as imageio
            with imageio.get_reader(str(a.tracking_path)) as reader:
                frames=np.stack([frame for frame in reader])
            assert frames.shape==(49,480,720,3)
            tracking=torch.from_numpy(frames.copy()).permute(0,3,1,2).float()/255
        else:
            # Used only for the official adapter's shape preparation. The pipeline
            # receives None, so this placeholder is neither encoded nor conditioned.
            tracking=torch.zeros((49,3,480,720))
        owner=types.SimpleNamespace(device='cuda:0',dtype=torch.bfloat16)
        os.chdir(a.das_root)
        stage('official_inference')
        infer(owner,prompt=a.prompt,model_path=str(a.checkpoint_path),tracking_tensor=tracking,
              image_tensor=torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
              output_path=str(a.output_path),num_inference_steps=a.num_inference_steps,
              dtype=torch.bfloat16,fps=8,seed=a.seed)
        torch.cuda.synchronize()
        state['success']=True
        stage('complete')
        state['output_sha256']=sha256(a.output_path)
    except BaseException as exc:
        state['error_type']=type(exc).__name__
        state['error']=str(exc)
        state['stack_trace']=traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        state['wall_seconds']=time.monotonic()-started
        state['active_seconds']=state['wall_seconds']-state.get('queue_wait_seconds',0.)
        if torch.cuda.is_initialized():
            state['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated()
            state['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
        finished.set(); sampler.join()
        write_json(out/f'resource_usage{stem}.json',state)
        log.flush()


if __name__=='__main__': main()
