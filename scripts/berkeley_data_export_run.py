"""Export future native depth after completed predictions, without overwriting artifacts."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
import numpy as np
from berkeley_data_extract_native import ROOT,SOURCE,SHARD,SCENES
REPO=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,default=REPO/'runs/berkeley_ur5_molmomotion');ap.add_argument('--future',action='store_true');args=ap.parse_args()
    if not args.future:raise RuntimeError('Observed depth is attached by berkeley_scene.py; this command requires --future.')
    for scene,_,_,_,_ in SCENES:
        gate=args.run_root/scene/'predictions/model_run.json'
        if not gate.exists() or not json.loads(gate.read_text()).get('success'):
            raise RuntimeError(f'Successful real model predictions required: {gate}')
        for name in ('native_depth_future.npy','native_depth_metadata.json'):
            if (args.run_root/scene/'evaluation'/name).exists():
                raise FileExistsError(f'Refusing to overwrite existing artifact: {scene}/{name}')
    subprocess.run([sys.executable,str(REPO/'scripts/berkeley_data_extract_native.py'),'--evaluation','--run-root',str(args.run_root)],check=True)
    for scene,ep,t0,start,rec in SCENES:
        split='evaluation'if args.future else 'observed';inds=list(range(t0+1,t0+11))if args.future else list(range(start,t0+1))
        source=ROOT/'native'/scene/split;dest=args.run_root/scene/split;dest.mkdir(parents=True,exist_ok=True)
        depths=np.stack([np.load(source/f'depth_{i:06d}.npy')for i in inds])
        name='native_depth_future.npy'if args.future else 'native_depth.npy'
        np.save(dest/name,depths)
        metadata={'source':SOURCE,'source_feature':'steps.observation.image_with_depth','depth_source':'native_realsense_tfds','measured_metric_depth':True,'units':'meters','dtype':'float32','shape':list(depths.shape),'source_episode_id':ep,'source_record_in_shard':rec,'source_frame_indices':inds,'raw_shard':str(ROOT/SHARD),'raw_shard_sha256':hashlib.sha256((ROOT/SHARD).read_bytes()).hexdigest(),'identity_verification':'All episode robot pose/gripper states and translation actions match LeRobot exactly; observed RGB RMSE 2.95-4.12/255 due to AV1 compression.','decode_method':'lossless TFDS RGBA uint8 PNG bitcast to little-endian float32, no rescaling','invalid_values':'zero is invalid; saturation and distant-background spikes can occur, use robust valid local median','future_decoded_after_predictions':args.future}
        (dest/'native_depth_metadata.json').write_text(json.dumps(metadata,indent=2))
        print(scene,name,depths.shape,flush=True)
if __name__=='__main__':main()
