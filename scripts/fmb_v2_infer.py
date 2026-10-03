"""Sealed BF16 greedy FMB H3/H1 inference with complete text/parser auditing."""
from pathlib import Path
import argparse
from decimal import Decimal
import json
import re
import resource
import time
import traceback
import numpy as np
import torch
from PIL import Image
from berkeley_preprocess import sha256, write_json, read_json
from berkeley_infer import freeze_scene_inputs
from berkeley_input_audit import validate_group_payloads

ROOT=Path(__file__).resolve().parents[1]

def strict_parse(raw,history,horizon):
    match=re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*',raw)
    if match is None:raise ValueError('Expected one complete tracks block')
    frames=match.group(1).split(';')
    if len(frames)!=horizon:raise ValueError(f'Expected {horizon} frames, found {len(frames)}')
    delta=np.empty((8,horizon,3),np.float32)
    for t,frame in enumerate(frames):
        tokens=frame.split()
        if len(tokens)!=33 or Decimal(tokens[0])!=Decimal(t+history):raise ValueError(f'Invalid time ID/record count at {t}')
        seen=set()
        for off in range(1,33,4):
            record=tokens[off:off+4]
            if any(re.fullmatch(r'[+-]?\d+',token) is None for token in record):raise ValueError('Noninteger coordinate')
            p,x,y,z=map(int,record)
            if p not in range(1,9) or p in seen:raise ValueError('Duplicate/out-of-range point ID')
            seen.add(p);delta[p-1,t]=np.array([x,y,z])/1000
        if seen!=set(range(1,9)):raise ValueError('Missing point ID')
    if not np.isfinite(delta).all():raise ValueError('Nonfinite output')
    return delta

def run(scenes,mode):
    h,f=(3,30) if mode=='h3' else (1,32)
    model_id=f'allenai/MolmoMotion-4B-H{h}-F{f}'
    checkpoint=ROOT/'data/checkpoints'/model_id.split('/')[1]
    dest_name='predictions' if mode=='h3' else 'predictions_h1'
    for scene in scenes:
        validate_group_payloads(scene)
        assert read_json(scene/'geometry/input_audit.json')['success']
        assert len(np.load(scene/'observed/selected_point_ids.npy'))==24
        if mode=='h1':assert read_json(scene/'predictions/model_run.json')['success']
        meta=read_json(scene/'metadata.json')
        freeze_scene_inputs(scene,meta,meta['instruction'])
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    torch.manual_seed(0);start=time.monotonic()
    processor=MolmoMotionProcessor.from_pretrained(str(checkpoint))
    previous=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:model=MolmoMotion.from_pretrained(str(checkpoint))
    finally:torch.set_default_dtype(previous)
    model._internal=model._internal.cuda().eval();torch.cuda.synchronize()
    load_seconds=time.monotonic()-start
    revision='3f5e790a511ff2cdf21c8d2a14cb4d8409c94629' if mode=='h3' else read_json(scenes[0].parent/'h1_checkpoint_hub.json')['sha']
    info={'model_id':model_id,'checkpoint_revision':revision,'config_sha256':sha256(checkpoint/'config.yaml'),
          'checkpoint_bytes':(checkpoint/'model.pt').stat().st_size,'dtype':'bfloat16','seed':0,
          'decoding':'official default greedy','model_load_seconds':load_seconds,
          'max_new_tokens':160*f,
          'inference_script_sha256':sha256(Path(__file__)),'history_size':h,'future_horizon':f,
          'torch_version':torch.__version__}
    for scene in scenes:
        meta=read_json(scene/'metadata.json');dest=scene/dest_name;dest.mkdir(exist_ok=True)
        freeze=read_json(scene/'predictions/input_freeze.json')
        rgb=np.load(scene/'observed/history_rgb.npy')[-h:]
        frames=[Image.fromarray(frame) for frame in rgb];statuses=[]
        for group in sorted((scene/'groups').glob('group_*')):
            out=dest/group.name;out.mkdir(exist_ok=True)
            xy=np.load(group/'points_2d_at_t0.npy');xyz=np.load(group/'points_3d_history.npy')[-h:]
            status_path=out/'model_run.json'
            if status_path.exists():
                prior=read_json(status_path)
                if (out/'raw_model_output.txt').exists() and (out/'future_3d.npy').exists():
                    delta=strict_parse((out/'raw_model_output.txt').read_text(),h,f)
                    stored=np.load(out/'future_3d.npy')
                    assert np.allclose(stored,delta+xyz[-1,0],atol=1e-4)
                    prior.update(success=True,parse_status=f'FULL_8x{f}x3')
                    write_json(status_path,prior);statuses.append(prior);continue
                if prior.get('generation_started'):raise RuntimeError('Incomplete prior generation: inspect receipt before explicit retry')
            batch=processor(history_frames=frames,points_2d_at_t0=torch.from_numpy(xy).float(),
                   points_3d_history=torch.from_numpy(xyz).float(),action=meta['instruction'],future_horizon=f)
            torch.save(batch,out/'processor_inputs.pt')
            anchor=batch['anchor_3d'].detach().cpu().float().numpy().reshape(3)
            assert np.array_equal(anchor,xyz[-1,0])
            np.save(out/'anchor.npy',anchor)
            write_json(out/'input_hashes.json',{'frozen_input_receipt_sha256':sha256(scene/'predictions/input_freeze.json'),
                       'processor_inputs_sha256':sha256(out/'processor_inputs.pt'),
                       'history_size':h,'source_history_indices':meta['history_source_indices'][-h:],
                       'actual_xyz':xyz.tolist(),'action':meta['instruction'],'future_used':False})
            status={**info,'group':group.name,'success':False,'generation_started':False}
            write_json(status_path,status)
            try:
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                status['generation_started']=True;write_json(status_path,status)
                print(scene.name,mode,group.name,'generating',flush=True)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model.predict_trajectory(**batch)
                torch.cuda.synchronize();status['prediction_seconds']=time.monotonic()-start
                (out/'raw_model_output.txt').write_text(output.future_text,encoding='utf-8')
                future=output.future_3d.detach().cpu().float().numpy();np.save(out/'future_3d.npy',future)
                delta=strict_parse(output.future_text,h,f)
                assert future.shape==(8,f,3) and np.isfinite(future).all() and np.any(future)
                assert np.allclose(future,delta+anchor,atol=1e-4)
                status.update(success=True,parse_status=f'FULL_8x{f}x3',parser_max_abs_reconstruction_difference_m=float(np.max(np.abs(future-delta-anchor))),
                    raw_output_sha256=sha256(out/'raw_model_output.txt'),prediction_sha256=sha256(out/'future_3d.npy'))
            except Exception as exc:
                status.update(error=str(exc),traceback=traceback.format_exc());raise
            finally:
                status.update(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
                write_json(status_path,status)
            statuses.append(status);del batch,output;torch.cuda.empty_cache()
        assert all(sha256(scene/name)==digest for name,digest in freeze['sha256'].items()),'Frozen input changed'
        future=np.concatenate([np.load(dest/group.name/'future_3d.npy') for group in sorted((scene/'groups').glob('group_*'))])
        assert future.shape==(24,f,3)
        np.save(dest/'future_3d.npy',future)
        write_json(dest/'model_run.json',{**info,'success':True,'successful_chunks':3,'attempted_chunks':3,
                   'prediction_shape':list(future.shape),'groups':statuses,
                   'prediction_seconds':sum(row['prediction_seconds'] for row in statuses)})
        print(scene.name,mode,'COMPLETE',future.shape,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene-dir',type=Path,nargs='+',required=True)
    p.add_argument('--mode',choices=['h3','h1'],default='h3');a=p.parse_args();run(a.scene_dir,a.mode)
