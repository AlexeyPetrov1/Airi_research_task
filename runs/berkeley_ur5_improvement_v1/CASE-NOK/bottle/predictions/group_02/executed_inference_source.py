"""Actual author MolmoMotion inference for controlled geometry pilots/full cases.

Retains all 24 canonical IDs. --groups 0 generates only the first P8 pilot;
--groups 0 1 2 reuses a successful pilot and generates the remaining groups.
Never silently repeats a generation after a crash. Raw outputs are recoverable.
"""
from pathlib import Path
import hashlib
EXECUTED_SCRIPT_BYTES=Path(__file__).read_bytes()
EXECUTED_SCRIPT_SHA256=hashlib.sha256(EXECUTED_SCRIPT_BYTES).hexdigest()
import argparse
import json
import resource
import subprocess
import time
import traceback
import re
from decimal import Decimal
import numpy as np
import torch
from PIL import Image
from berkeley_infer import (ROOT,CHECKPOINT,REVISION,sha,write,strict_parse,
                           freeze_scene_inputs,recover_saved_group,finalize_scene)
from berkeley_input_audit import validate_group_payloads


def parse_tracks(raw,history_size,horizon):
    if history_size==3 and horizon==30:return strict_parse(raw)
    match=re.fullmatch(r'<tracks coords="([^"]+)">[^<]*</tracks>\s*',raw)
    if match is None:raise ValueError('Expected exactly one complete tracks block')
    frames=match.group(1).split(';')
    if len(frames)!=horizon:raise ValueError('Incomplete future horizon')
    delta=np.empty((8,horizon,3),np.float32)
    for step,frame in enumerate(frames):
        tokens=frame.split()
        if len(tokens)!=33 or Decimal(tokens[0])!=Decimal(step+history_size):raise ValueError('Wrong time ID')
        seen=set()
        for off in range(1,33,4):
            record=tokens[off:off+4]
            if any(re.fullmatch(r'[+-]?\d+',x) is None for x in record):raise ValueError('Noninteger coordinate')
            point,x,y,z=map(int,record)
            if point not in range(1,9) or point in seen:raise ValueError('Wrong point ID')
            seen.add(point);delta[point-1,step]=np.array([x,y,z])/1000
    return delta


def recover_selected(out,group,prior,history_size,horizon):
    if history_size==3:return recover_saved_group(out,group,prior)
    if prior.get('generation_started') is not True:raise ValueError('No actual generation was recorded')
    raw=(out/'raw_model_output.txt').read_text()
    xyz=np.load(out/'future_3d.npy')
    anchor=np.load(group/'points_3d_history.npy')[-1,0]
    if xyz.shape!=(8,horizon,3) or not np.isfinite(xyz).all() or not np.allclose(xyz,parse_tracks(raw,history_size,horizon)+anchor,atol=1e-4):
        raise ValueError('Saved H1 output is incomplete or inconsistent')
    for name,key in [('future_3d.npy','prediction_sha256'),('raw_model_output.txt','raw_output_sha256')]:
        if prior.get(key) and prior[key]!=sha(out/name):raise ValueError('Saved output changed')
    result=dict(prior,success=True,recovered_from_saved_artifacts=True,recovery_additional_forward_calls=0)
    write(out/'model_run.json',result);return result


def finalize_selected(scene,groups,receipts,info,elapsed):
    if len(groups)==3:
        finalize_scene(scene,groups,receipts,info,elapsed)
        return
    xyz=np.load(scene/'predictions/group_00/future_3d.npy')
    ids=np.load(scene/'groups/group_00/point_ids.npy')
    if xyz.shape!=(8,info.get('future_horizon',30),3) or not all(x.get('success') for x in receipts):
        raise ValueError('Pilot is incomplete')
    np.save(scene/'predictions/future_3d.npy',xyz)
    np.save(scene/'predictions/point_ids.npy',ids)
    write(scene/'predictions/model_run.json',dict(info,success=True,groups=receipts,
        successful_chunks=1,prediction_completeness=1.,prediction_shape=list(xyz.shape),
        canonical_point_count=24,evaluated_point_count=8,pilot_only=True,
        prediction_sha256=sha(scene/'predictions/future_3d.npy'),runtime_seconds=elapsed))
    print(scene.parent.name,scene.name,'PILOT COMPLETE',xyz.shape,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes',type=Path,nargs='+',required=True)
    p.add_argument('--groups',type=int,nargs='+',default=[0])
    p.add_argument('--recover-only',action='store_true')
    p.add_argument('--checkpoint',choices=['H3','H1'],default='H3')
    a=p.parse_args()
    checkpoint=CHECKPOINT if a.checkpoint=='H3' else ROOT/'data/checkpoints/MolmoMotion-4B-H1-F32'
    history_size,horizon=(3,30) if a.checkpoint=='H3' else (1,32)
    revision=REVISION if a.checkpoint=='H3' else '14e69b0d5dd55b2f09c885030b8756cd053ea6a4'
    if a.checkpoint=='H1' and a.groups!=[0]:raise ValueError('H1 is a first-eight-point control')
    if a.groups not in ([0],[0,1,2]):raise ValueError('Use first P8 pilot or complete fixed 24-point set')
    plans=[]
    for scene in a.scenes:
        provenance=json.loads((scene/'geometry/case_provenance.json').read_text())
        if provenance.get('future_used') is not False:raise ValueError('Geometry provenance is not causal')
        if not np.allclose(np.load(scene/'geometry/K_median.npy'),provenance['K']):
            raise ValueError('Case K does not match completed geometry provenance')
        validate_group_payloads(scene)
        audit=json.loads((scene/'geometry/input_audit.json').read_text())
        if audit.get('success') is not True:raise ValueError('Observed input audit failed')
        meta=json.loads((scene/'metadata.json').read_text())
        action=meta.get('instruction',meta.get('task'))
        frozen=freeze_scene_inputs(scene,meta,action)
        groups=[scene/f'groups/group_{i:02d}' for i in a.groups]
        plans.append((scene,action,frozen,groups))
    if a.recover_only:
        for scene,action,frozen,groups in plans:
            receipts=[]
            for group in groups:
                out=scene/'predictions'/group.name
                receipts.append(recover_selected(out,group,json.loads((out/'model_run.json').read_text()),history_size,horizon))
            info=dict(receipts[0],action=action,selected_groups=a.groups,
                      mode='pilot_first_8' if a.groups==[0] else 'full_fixed_24',recovery_additional_forward_calls=0)
            if not all(sha(scene/name)==digest for name,digest in frozen.items()):raise ValueError('Frozen inputs changed')
            finalize_selected(scene,groups,receipts,info,0.)
        return
    ready=False
    while not ready:
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        ready=used<1500
        if not ready:print('Waiting for existing GPU work; memory MiB',used,flush=True);time.sleep(10)
    from molmo_motion import MolmoMotion,MolmoMotionProcessor
    started=time.monotonic();torch.manual_seed(0)
    processor=MolmoMotionProcessor.from_pretrained(str(checkpoint))
    if processor.config.history_size!=history_size:raise ValueError('Checkpoint history size differs')
    dtype=torch.get_default_dtype();torch.set_default_dtype(torch.bfloat16)
    try:model=MolmoMotion.from_pretrained(str(checkpoint))
    finally:torch.set_default_dtype(dtype)
    model._internal=model._internal.cuda().eval();torch.cuda.synchronize()
    info=dict(model_id=f'allenai/MolmoMotion-4B-H{history_size}-F{horizon}',checkpoint_hf_revision=revision,
        history_size=history_size,future_horizon=horizon,
        config_sha256=sha(checkpoint/'config.yaml'),model_bytes=(checkpoint/'model.pt').stat().st_size,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        inference_script_sha256=EXECUTED_SCRIPT_SHA256,torch_version=torch.__version__,dtype='bfloat16',seed=0,
        decoding='official default greedy',max_new_tokens=4800,model_load_seconds=time.monotonic()-started,
        selected_groups=a.groups,mode='pilot_first_8' if a.groups==[0] else 'full_fixed_24')
    for scene,action,frozen,groups in plans:
        scene_start=time.monotonic();receipts=[]
        frames=[Image.fromarray(f).convert('RGB') for f in np.load(scene/'observed/history_rgb.npy')[-history_size:]]
        for group in groups:
            out=scene/'predictions'/group.name;out.mkdir(parents=True,exist_ok=True)
            statusfile=out/'model_run.json'
            if statusfile.exists():
                prior=json.loads(statusfile.read_text())
                if prior.get('generation_started'):
                    receipts.append(recover_selected(out,group,prior,history_size,horizon));continue
            (out/'executed_inference_source.py').write_bytes(EXECUTED_SCRIPT_BYTES)
            points2=torch.from_numpy(np.load(group/'points_2d_at_t0.npy')).float()
            history3=torch.from_numpy(np.load(group/'points_3d_history.npy')[-history_size:]).float()
            batch=processor(history_frames=frames,points_2d_at_t0=points2,points_3d_history=history3,
                            action=action,future_horizon=horizon)
            torch.save(batch,out/'processor_inputs.pt')
            status=dict(info,group=group.name,success=False,generation_started=False)
            write(statusfile,status)
            try:
                batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                status['generation_started']=True;write(statusfile,status)
                print(scene.parent.name,scene.name,group.name,'actual generation',flush=True)
                call=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model.predict_trajectory(**batch)
                torch.cuda.synchronize()
                status['prediction_seconds']=time.monotonic()-call
                (out/'raw_model_output.txt').write_text(output.future_text,encoding='utf8')
                xyz=output.future_3d.detach().cpu().float().numpy();np.save(out/'future_3d.npy',xyz)
                parsed=parse_tracks(output.future_text,history_size,horizon)
                anchor=np.load(group/'points_3d_history.npy')[-1,0]
                if xyz.shape!=(8,horizon,3) or not np.isfinite(xyz).all() or not np.allclose(xyz,parsed+anchor,atol=1e-4):
                    raise ValueError('Incomplete output or anchor/text disagreement')
                np.savez_compressed(out/'prediction.npz',future_3d=xyz,parsed_visibility=np.ones((8,horizon),bool))
                status.update(success=True,parse_status=f'FULL_8x{horizon}x3',parsed_point_times=8*horizon,
                    prediction_shape=list(xyz.shape),prediction_sha256=sha(out/'future_3d.npy'),
                    raw_output_sha256=sha(out/'raw_model_output.txt'))
            except Exception as e:
                status.update(error=str(e),traceback=traceback.format_exc());raise
            finally:
                status.update(peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_cuda_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    peak_ram_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
                write(statusfile,status)
            receipts.append(status);del batch,output;torch.cuda.empty_cache()
        if not all(sha(scene/name)==digest for name,digest in frozen.items()):raise ValueError('Observed inputs changed')
        finalize_selected(scene,groups,receipts,dict(info,action=action),time.monotonic()-scene_start)


if __name__=='__main__':main()
