"""Run preregistered native neural ablations, one GPU workload at a time."""
import argparse,json,os,resource,subprocess,time,traceback
from datetime import datetime,timezone
import numpy as np
import torch
from PIL import Image
from fmb_wrist_prepare import ROOT,write,sha
from fmb_wrist_v4_diagnose import OUT
from fmb_v2_infer import strict_parse

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def wait_gpu():
    quiet=0
    while quiet<3:
        r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        active=[line for line in r.stdout.splitlines() if line.strip() and line.split(',')[0].strip()!=str(os.getpid())]
        if active:
            quiet=0;write(OUT/'inference_status.json',{'status':'WAITING_FOR_OTHER_GPU_PROCESS','active_processes':active,'updated_utc':datetime.now(timezone.utc).isoformat()})
            print('GPU occupied; waiting without interrupting other work',active,flush=True);time.sleep(10)
        else:quiet+=1;time.sleep(3)

def run(resume=False):
    assert (OUT/'preregistration.json').exists()
    os.environ.setdefault('HF_HOME',str(ROOT.parent/'.cache/huggingface'))
    os.environ.setdefault('TORCH_HOME',str(ROOT.parent/'.cache/torch'))
    torch.set_num_threads(4)
    wait_gpu()
    from molmo_motion import MolmoMotion,MolmoMotionProcessor
    variants=[('H3_rigid_observed',3,30,'3f5e790a511ff2cdf21c8d2a14cb4d8409c94629'),
              ('H1_native',1,32,'14e69b0d5dd55b2f09c885030b8756cd053ea6a4')]
    for name,h,f,revision in variants:
        checkpoint=ROOT/'data/checkpoints'/f'MolmoMotion-4B-H{h}-F{f}'
        torch.manual_seed(0);start=time.monotonic()
        processor=MolmoMotionProcessor.from_pretrained(str(checkpoint))
        previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
        try:model=MolmoMotion.from_pretrained(str(checkpoint))
        finally:torch.set_default_dtype(previous)
        model._internal=model._internal.cuda().eval();torch.cuda.synchronize()
        load_seconds=time.monotonic()-start
        assert processor.config.history_size==h
        assert not getattr(model._internal.config,'use_2d_point_features',False),'Audit point-feature normalization before changing checkpoint configuration'
        for camera in ['wrist_2','wrist_1']:
            folder=OUT/camera/'variants'/name;spec=read(folder/'input_spec.json')
            assert all(sha(folder/p)==digest for p,digest in spec['sha256'].items())
            hist=np.load(folder/'history_xyz.npy');xy=np.load(folder/'query_uv.npy')
            frames=[Image.fromarray(im) for im in np.load(folder/'history_rgb.npy')]
            combined=[];calls=[]
            for group in range(3):
                dest=folder/f'group_{group:02d}';dest.mkdir(exist_ok=True);status=dest/'model_run.json'
                xyz=hist[:,group*8:group*8+8]
                if status.exists():
                    prior=read(status)
                    assert resume and prior['success'],'Inspect/archive incomplete attempts explicitly before resumption'
                    assert prior['input_spec_sha256']==sha(folder/'input_spec.json')
                    raw=(dest/'raw_model_output.txt').read_text(encoding='utf-8');stored=np.load(dest/'future_3d.npy');anchor=np.load(dest/'anchor.npy')
                    assert prior['raw_sha256']==sha(dest/'raw_model_output.txt') and prior['prediction_sha256']==sha(dest/'future_3d.npy')
                    np.testing.assert_allclose(stored,strict_parse(raw,h,f)+anchor,atol=1e-4)
                    assert np.array_equal(anchor,xyz[-1,0])
                    combined.append(stored);calls.append(prior);print(camera,name,group,'verified completed output retained',flush=True);continue
                batch=processor(history_frames=frames,points_2d_at_t0=torch.from_numpy(xy[group*8:group*8+8]).float(),
                                points_3d_history=torch.from_numpy(xyz).float(),action=spec['action'],future_horizon=f)
                torch.save(batch,dest/'processor_inputs.pt')
                tokens=batch['input_ids'][0].cpu().numpy();np.save(dest/'input_ids.npy',tokens)
                (dest/'decoded_prompt.txt').write_text(processor.tokenizer.decode(tokens[tokens>=0]),encoding='utf-8')
                anchor=batch['anchor_3d'].cpu().float().numpy().reshape(3)
                assert np.array_equal(anchor,xyz[-1,0]);np.save(dest/'anchor.npy',anchor)
                info={'model_id':checkpoint.name,'revision':revision,'history':h,'horizon':f,'dtype':'bfloat16','seed':0,
                      'decoding':'Native greedy, best-of-1','point_features_enabled':False,'internal_model_class':type(model._internal).__name__,'camera':camera,'variant':name,'group':group,
                      'input_spec_sha256':sha(folder/'input_spec.json'),'processor_inputs_sha256':sha(dest/'processor_inputs.pt'),
                      'script_sha256':sha(ROOT/'scripts/fmb_wrist_v4_infer.py'),'config_sha256':sha(checkpoint/'config.yaml'),
                      'model_load_seconds':load_seconds,'future_input_used':False,'generation_started':True,'success':False}
                info['torch_CPU_threads']=torch.get_num_threads()
                write(status,info);write(OUT/'inference_status.json',{'status':'GENERATING','camera':camera,'variant':name,'group':group,'pid':os.getpid()})
                print(camera,name,group,'generation started',flush=True)
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                try:
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=model.predict_trajectory(**batch)
                    torch.cuda.synchronize();info['prediction_seconds']=time.monotonic()-start
                    (dest/'raw_model_output.txt').write_text(result.future_text,encoding='utf-8')
                    pred=result.future_3d.cpu().float().numpy();np.save(dest/'future_3d.npy',pred)
                    parsed=strict_parse(result.future_text,h,f)
                    assert pred.shape==(8,f,3) and np.isfinite(pred).all()
                    np.testing.assert_allclose(pred,parsed+anchor,atol=1e-4)
                    info.update(success=True,parse_status=f'FULL_8x{f}x3',raw_sha256=sha(dest/'raw_model_output.txt'),prediction_sha256=sha(dest/'future_3d.npy'))
                    combined.append(pred)
                except Exception as exc:
                    info.update(error=str(exc),traceback=traceback.format_exc());raise
                finally:
                    info['peak_cuda_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
                    info['peak_cuda_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
                    info['peak_ram_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
                    write(status,info)
                calls.append(info);del batch,result;torch.cuda.empty_cache()
                print(camera,name,group,'complete',info['prediction_seconds'],flush=True)
            all_points=np.concatenate(combined);np.save(folder/'future_3d.npy',all_points)
            write(folder/'model_run.json',{'success':True,'calls':calls,'shape':list(all_points.shape),'future_input_used':False})
            assert all(sha(folder/p)==digest for p,digest in spec['sha256'].items())
        del model,processor;torch.cuda.empty_cache()
    write(OUT/'inference_status.json',{'status':'COMPLETE','successful_P8_calls':12})
    print('All12 preregistered native calls complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');run(parser.parse_args().resume)
