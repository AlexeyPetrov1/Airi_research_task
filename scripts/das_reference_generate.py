"""Actual DaS Wanfun diffusion with an explicit photographic latent guide.

No generated RGB is composited, replaced, or interpolated after VAE decoding.
The guide is rendered from t0 + one authorized future endpoint, not a real clip.
This is an inference extension and NOT a causal prediction benchmark.
"""
import argparse, importlib.metadata, json, os, platform, subprocess, sys
import threading, time, traceback, types
from pathlib import Path
import numpy as np
import psutil
from PIL import Image
from das_prepare_control import sha256, write_json
from das_wanfun_runtime import load_official_infer
from das_generate import research_branch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_molmomotion/cup/das_reference_repair'
DAS=Path('/mnt/f/AIRI_task/third_party/DiffusionAsShader-Wanfun')
MODEL=Path('/mnt/f/AIRI_task/models/Wan2.1-Fun-V1.1-1.3B-Control')

class PhotographicPrior:
    def __init__(self,guide,strength,seed):
        self.guide=guide;self.strength=strength;self.seed=seed;self.noise=None;self.steps=0;self.sigmas=[]
    def __call__(self,pipeline,index,timestep,kwargs):
        import torch
        x=kwargs['latents'];guide=self.guide.to(x.device,x.dtype)
        assert x.shape==guide.shape
        if self.noise is None:
            generator=torch.Generator(device=x.device).manual_seed(self.seed)
            self.noise=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)
        sigma=pipeline.scheduler.sigmas[index+1].to(x.device,x.dtype)
        assert 0<=float(sigma)<=1
        noised_guide=(1-sigma)*guide+sigma*self.noise
        weights=torch.full((1,1,x.shape[2],1,1),self.strength,device=x.device,dtype=x.dtype)
        weights[:,:,0]=1. # exact initial VAE latent, correctly noised at each step
        kwargs['latents']=x*(1-weights)+noised_guide*weights
        assert torch.isfinite(kwargs['latents']).all()
        self.steps+=1;self.sigmas.append(float(sigma))
        return kwargs

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='guided_strength045')
    p.add_argument('--strength',type=float,default=.45);p.add_argument('--steps',type=int,default=30)
    p.add_argument('--preparation-subdir',default='')
    p.add_argument('--no-prior',action='store_true');p.add_argument('--no-control',action='store_true');a=p.parse_args()
    if not 0<a.strength<.9:p.error('Use a partial guide strength in (0, .9)')
    if a.no_control and not a.no_prior:p.error('Uncontrolled ablation must also disable the photographic motion guide')
    out=(OUT/a.name).resolve();assert out.is_relative_to(OUT.resolve()),'Output must remain inside this experiment'
    out.mkdir(exist_ok=True);output=out/'generated_seed42.mp4'
    if output.exists():raise FileExistsError('Refusing to overwrite actual output')
    prep_root=(OUT/a.preparation_subdir).resolve();assert prep_root.is_relative_to(OUT.resolve())
    prep=json.loads((prep_root/'preparation.json').read_text())
    review=json.loads((prep_root/'preparation_visual_review.json').read_text(encoding='utf-8-sig'))
    assert review['accepted'] and review['guide_sha256']==sha256(prep_root/'guide_720x480.npz')==prep['guide_sha256']
    assert review['control_sha256']==sha256(prep_root/'control_endpoint_720x480.mp4')==prep['control_sha256']
    log=open(out/'generation.log','a',encoding='utf8',buffering=1)
    class Tee:
        def __init__(self,original):self.original=original
        def write(self,value):self.original.write(value);log.write(value)
        def flush(self):self.original.flush();log.flush()
    sys.stdout,sys.stderr=Tee(sys.stdout),Tee(sys.stderr)
    import torch
    torch.set_num_threads(4)
    started=time.monotonic();finished=threading.Event()
    state={'success':False,'stage':'queue','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'peak_process_rss_bytes':0,'peak_system_ram_used_bytes':0,'peak_cuda_allocated_bytes':0,'peak_cuda_reserved_bytes':0}
    def monitor():
        while not finished.is_set():
            state['wall_seconds']=time.monotonic()-started
            state['peak_process_rss_bytes']=max(state['peak_process_rss_bytes'],psutil.Process().memory_info().rss)
            ram=psutil.virtual_memory();state['peak_system_ram_used_bytes']=max(state['peak_system_ram_used_bytes'],ram.total-ram.available)
            if torch.cuda.is_initialized():
                state['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated();state['peak_cuda_reserved_bytes']=torch.cuda.max_memory_reserved()
            write_json(out/'resource_usage.json',state);finished.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def stage(name):
        if state['stage']!=name:print(f'STAGE {name} {time.monotonic()-started:.1f}s',flush=True)
        state['stage']=name
    try:
        free,total=torch.cuda.mem_get_info()
        while free<9*2**30:
            print('Waiting for free GPU memory',flush=True);time.sleep(15);free,total=torch.cuda.mem_get_info()
        guard=torch.ones(262144,device='cuda');torch.cuda.reset_peak_memory_stats()
        prompt=('A realistic fixed-camera laboratory video. A UR5 robot gripper firmly holds exactly one light-blue paper cup with a white cartoon face. '
            'The robot lifts this same cup smoothly upwards a short distance on the left side of the wooden table. '
            'The gripper and cup move together and retain their solid shapes. The brown cup stays on the table. '
            'The vacated original cup location shows empty wood. Detailed natural surfaces, consistent lighting, stationary background.')
        config={'model':'alibaba-pai/Wan2.1-Fun-V1.1-1.3B-Control','checkpoint_revision':json.loads((prep_root/'checkpoint_receipt.json').read_text())['revision'],
            'das_commit':subprocess.check_output(['git','-C',str(DAS),'rev-parse','HEAD'],text=True).strip(),
            'research_branch':research_branch(ROOT),'dtype':'bfloat16','seed':42,'num_inference_steps':a.steps,
            'num_frames':49,'resolution':[720,480],'fps':8,'GPU':torch.cuda.get_device_name(),'VRAM_bytes':total,'RAM_bytes':psutil.virtual_memory().total,
            'RAM_scope':'WSL limit; host has 32 GiB','offload':'official sequential CPU offload','guidance_scale':5.,'teacache':False,
            'persistent_reference':'initial-only; ref_image=None','photographic_guide_strength':0 if a.no_prior else a.strength,
            'initial_latent_strength':0 if a.no_prior else 1,'guide_noise_seed':42,'prompt':prompt,'future_used':True,
            'future_information':('Only the shared descriptive prompt; no future RGB/control/guide tensor is passed to the model' if a.no_control else
                'RGB frame 73 (2 s endpoint), SAM masks and dense 2D correspondence; no intermediate future RGB in generation'),
            'future_pixels_or_pose_condition_passed_to_model':not a.no_control,
            'causal_MolmoMotion_benchmark':False,'native_DaS_unmodified_inference':False,'postprocessed_RGB':False,
            'preparation_root':str(prep_root),'background_mode':prep.get('background_mode','initial'),
            'preparation_sha256':sha256(prep_root/'preparation.json'),'guide_sha256':None if a.no_control else prep['guide_sha256'],
            'control_sha256':None if a.no_control else prep['control_sha256'],'trajectory_control':not a.no_control,
            'command':sys.argv}
        write_json(out/'config.json',config)
        packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())
        (out/'environment.txt').write_text(platform.platform()+'\n'+sys.version+'\n'+'\n'.join(f'{n}=={v}' for n,v in packages)+'\n')
        (out/'gpu_info.txt').write_text(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi'],text=True))
        stage('imports');infer=load_official_infer(DAS)
        from videox_fun.pipeline import WanFunControlPipeline
        original_call=WanFunControlPipeline.__call__
        def guided_call(self,*args,**kwargs):
            self.transformer.disable_teacache()
            kwargs['ref_image']=None;kwargs['guidance_scale']=5.
            if a.no_control:kwargs['control_video']=None
            kwargs['negative_prompt']='Duplicate cups, two blue cups, stationary ghost cup, transparent cup, broken gripper, distorted robot, deformed object, flicker, low quality, grainy texture, watermark.'
            stage('encode_photographic_guide')
            if not a.no_prior:
                guide=np.load(prep_root/'guide_720x480.npz')['frames']
                tensor=torch.from_numpy(guide.copy()).permute(3,0,1,2)[None].to('cuda',torch.bfloat16)/127.5-1
                guide_latents=self.vae.encode(tensor)[0].mode().detach()
                del tensor,guide
                assert guide_latents.shape==(1,16,13,60,90)
                torch.save({'latents':guide_latents.cpu(),'guide_sha256':prep['guide_sha256']},out/'photographic_guide_latents.pt')
                prior=PhotographicPrior(guide_latents,a.strength,42)
                kwargs['callback_on_step_end']=prior;kwargs['callback_on_step_end_tensor_inputs']=['latents']
            original_decode=self.vae.decode
            def decode(latents,*args,**kwargs):
                stage('save_generated_latents')
                torch.save({'latents':latents.detach().cpu(),'model':config['model'],'das_commit':config['das_commit']},out/'generated_latents.pt')
                stage('decode');return original_decode(latents,*args,**kwargs)
            self.vae.decode=decode
            self.transformer.register_forward_pre_hook(lambda *unused:stage('denoise'))
            stage('pipeline');result=original_call(self,*args,**kwargs)
            if not a.no_prior:
                assert prior.steps==a.steps and prior.sigmas[-1]==0
                write_json(out/'photographic_prior_receipt.json',{'applied_steps':prior.steps,'sigma_after_each_step':prior.sigmas,
                    'strength':a.strength,'initial_strength':1.,'noise_seed':42,'latent_shape':list(guide_latents.shape),
                    'formula':'After Euler step: x <- (1-w)x + w[(1-sigma_next)*guide + sigma_next*fixed_noise]',
                    'postprocessed_RGB':False,'native_DaS':False,'future_used':True})
            return result
        WanFunControlPipeline.__call__=guided_call
        import imageio.v2 as imageio
        with imageio.get_reader(str(prep_root/'control_endpoint_720x480.mp4')) as reader:control=np.stack([f for f in reader])
        image=np.array(Image.open(prep_root/'image_t0_720x480.png').convert('RGB'))
        tracking=torch.from_numpy(control.copy()).permute(0,3,1,2).float()/255
        owner=types.SimpleNamespace(device='cuda:0',dtype=torch.bfloat16);os.chdir(DAS)
        stage('official_model_loading')
        infer(owner,prompt=prompt,model_path=str(MODEL),tracking_tensor=tracking,
            image_tensor=torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
            output_path=str(output),num_inference_steps=a.steps,dtype=torch.bfloat16,fps=8,seed=42)
        torch.cuda.synchronize();state['success']=True;state['output_sha256']=sha256(output);stage('complete')
    except BaseException as e:
        state.update(error_type=type(e).__name__,error=str(e),stack_trace=traceback.format_exc());traceback.print_exc();raise
    finally:
        state['wall_seconds']=time.monotonic()-started
        finished.set();thread.join();write_json(out/'resource_usage.json',state);log.flush()

if __name__=='__main__':main()
