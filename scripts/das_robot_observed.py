"""Export only already observed UR5 joints from the native TFDS record."""
from pathlib import Path
import hashlib, json, struct
import numpy as np
from probe_fmb_rlds_record import find_feature_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/berkeley_ur5_molmomotion/cup/das_robot_cup/observed'
OUT.mkdir(parents=True, exist_ok=True)
shard = Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes/berkeley_autolab_ur5-train.tfrecord-00004-of-00412')
with shard.open('rb') as f:
    for record_id in range(2):
        header = f.read(12)
        record = f.read(struct.unpack('<Q', header[:8])[0])
        f.read(4)
features = dict(find_feature_values(record))
s, e = features['steps/observation/robot_state'][0]
# The record buffer contains the episode; only steps <= t0 enter this export.
history = np.frombuffer(record[s:e], dtype='<f4').reshape(-1, 15)[56:64].copy()
np.savez_compressed(OUT/'robot_state_history.npz', source_indices=np.arange(56,64),
                    joints=history[:,:6], ee_pose=history[:,6:13],
                    gripper_closed=history[:,13], action_blocked=history[:,14])
receipt = {'source_shard_sha256':hashlib.sha256(shard.read_bytes()).hexdigest(),
           'record_index':1, 'episode':10, 'source_indices':list(range(56,64)),
           'future_used':False, 'state_layout':'joints[0:6], xyz[6:9], quaternion_xyzw[9:13], gripper_closed[13], action_blocked[14]',
           'layout_source':'https://sites.google.com/view/berkeley-ur5/home',
           't0_state':history[-1].tolist()}
(OUT/'robot_state_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
