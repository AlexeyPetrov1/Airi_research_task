"""Extract verified Berkeley native metric depth; future decoding is a separate command."""
from pathlib import Path
import argparse,hashlib,io,json,struct
import numpy as np
import pandas as pd
import av
from PIL import Image
from probe_fmb_rlds_record import find_feature_values

ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')
SHARD='berkeley_autolab_ur5-train.tfrecord-00004-of-00412'
SOURCE='https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/d2590280290484e2a7eb53a91bba32ac2ff669a0/'+SHARD
SCENES=[('bottle',9,47,40,0),('cup',10,63,56,1)]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evaluation',action='store_true');ap.add_argument('--run-root',type=Path,default=Path(__file__).resolve().parents[1]/'runs/berkeley_ur5_molmomotion');args=ap.parse_args()
    if args.evaluation:
        for scene,_,_,_,_ in SCENES:
            gate=args.run_root/scene/'predictions/model_run.json'
            if not gate.exists() or not json.loads(gate.read_text()).get('success'):
                raise RuntimeError(f'Evaluation extraction requires successful real predictions: {gate}')
    recs=[]
    with (ROOT/SHARD).open('rb')as f:
        while header:=f.read(12):
            n=struct.unpack('<Q',header[:8])[0];recs.append(f.read(n));f.read(4)
    receipt={'source':SOURCE,'shard_bytes':(ROOT/SHARD).stat().st_size,'shard_sha256':hashlib.sha256((ROOT/SHARD).read_bytes()).hexdigest(),'depth_dtype':'float32','depth_units':'meters','depth_decode':'PIL RGBA PNG -> uint8 array.view(<f4); TFDS _FloatImageEncoder lossless bitcast','decoder_reference':'https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/features/image_feature.py','scenes':{}}
    for scene,ep,t0,start,recidx in SCENES:
        record=recs[recidx];features=dict(find_feature_values(record))
        def payload(key,index=0):
            s,e=features[key][index];return record[s:e]
        def rgb(i):return np.array(Image.open(io.BytesIO(payload('steps/observation/image',i))))
        df=pd.read_parquet(ROOT/f'data/chunk-000/episode_{ep:06d}.parquet')
        state=np.frombuffer(payload('steps/observation/robot_state'),dtype='<f4').reshape(-1,15)
        lrstate=np.stack(df['observation.state']);rawpose=state[:,6:14]
        state_err=float(np.abs(rawpose-lrstate).max());assert state_err<1e-7,(ep,state_err)
        world=np.frombuffer(payload('steps/action/world_vector'),dtype='<f4').reshape(-1,3)
        action_err=float(np.abs(world-np.stack(df['action'])[:,:3]).max());assert action_err<1e-7,(ep,action_err)
        frames=[f.to_ndarray(format='rgb24')for f in av.open(ROOT/f'videos/chunk-000/observation.images.image/episode_{ep:06d}.mp4').decode(video=0)]
        audit_indices=sorted(set([0,start,t0]))
        rgb_rmse={str(i):float(np.sqrt(np.mean((rgb(i).astype(float)-frames[i].astype(float))**2)))for i in audit_indices}
        inds=list(range(t0,t0+11))if args.evaluation else list(range(start,t0+1))
        split='evaluation'if args.evaluation else 'observed'
        out=ROOT/'native'/scene/split;out.mkdir(parents=True,exist_ok=True)
        depths=[]
        for i in inds:
            dbytes=payload('steps/observation/image_with_depth',i)
            rgba=np.asarray(Image.open(io.BytesIO(dbytes)),dtype=np.uint8)
            assert rgba.shape==(480,640,4),rgba.shape
            depth=rgba.view('<f4').reshape(480,640)
            assert np.isfinite(depth).all()
            np.save(out/f'depth_{i:06d}.npy',depth)
            (out/f'depth_float_bits_{i:06d}.png').write_bytes(dbytes)
            (out/f'rgb_{i:06d}.png').write_bytes(payload('steps/observation/image',i))
            depths.append({'frame':i,'valid_fraction':float((depth>0).mean()),'positive_percentiles':np.percentile(depth[depth>0],[0,1,50,99,100]).tolist()})
        np.save(out/'source_frame_indices.npy',np.array(inds));np.save(out/'source_timestamps.npy',df.timestamp.to_numpy()[inds])
        r={'episode_id':ep,'t0_source_frame':t0,'record_in_shard':recidx,'episode_length':len(df),'native_robot_pose_gripper_max_abs_error':state_err,'native_action_translation_max_abs_error':action_err,'native_vs_lerobot_rgb_rmse_0_255':rgb_rmse,'instruction':payload('steps/observation/natural_language_instruction').decode(),'split':split,'frames':inds,'depth_stats':depths}
        receipt['scenes'][scene]=r
        (out/'receipt.json').write_text(json.dumps(r,indent=2))
        print(scene,json.dumps(r),flush=True)
    (ROOT/f'native_{"evaluation"if args.evaluation else "observed"}_receipt.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
