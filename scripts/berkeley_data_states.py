from pathlib import Path
import json
import pandas as pd
import numpy as np
ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')
allrows={}
for idx,start,stop in [(10,52,74),(9,28,53)]:
    df=pd.read_parquet(ROOT/f'data/chunk-000/episode_{idx:06d}.parquet')
    states=np.stack(df['observation.state']); acts=np.stack(df['action'])
    rows=[{'frame_index':i,'timestamp':float(df.timestamp.iloc[i]),'observation_state':states[i].tolist(),'action':acts[i].tolist()}for i in range(start,stop+1)]
    allrows[idx]=rows
    print('EPISODE',idx,'GRIPPER_CHANGE',np.where(np.diff(states[:,-1])!=0)[0]+1)
    for row in rows:print(row)
(ROOT/'detailed_states.json').write_text(json.dumps(allrows,indent=2))
