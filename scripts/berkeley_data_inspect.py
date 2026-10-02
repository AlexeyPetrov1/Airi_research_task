from pathlib import Path
import json
import numpy as np
import pandas as pd
import av
from PIL import Image,ImageDraw
import requests

ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')
def main():
    rows=[]
    for p in sorted((ROOT/'data/chunk-000').glob('*.parquet')):
        idx=int(p.stem.split('_')[-1]); df=pd.read_parquet(p)
        print('EPISODE',idx,'COLUMNS',list(df),flush=True)
        act=np.stack(df['action']);state=np.stack(df['observation.state'])
        print('ACTION GRIPPER',[(int(i),float(act[i,-1]))for i in np.where(abs(act[:,-1])>0.1)[0]],flush=True)
        print('STATE_LAST',[(int(i),state[i,-2:].tolist())for i in range(0,len(df),5)],flush=True)
        v=ROOT/'videos/chunk-000/observation.images.image'/f'episode_{idx:06d}.mp4'
        frames=[f.to_ndarray(format='rgb24') for f in av.open(v).decode(video=0)]
        inds=np.unique(np.linspace(0,len(frames)-1,20).round().astype(int))
        out=Image.new('RGB',(320*5,260*4),'white'); draw=ImageDraw.Draw(out)
        for k,i in enumerate(inds):
            x=(k%5)*320;y=(k//5)*260
            out.paste(Image.fromarray(frames[i]).resize((320,240)),(x,y));draw.text((x+5,y+242),f'episode {idx} frame {i} t={i/5:.1f}s',fill='black')
        out.save(ROOT/f'candidate_{idx:06d}.jpg')
        rows.append({'episode_index':idx,'frames':len(frames),'gripper_action_events':[(int(i),float(act[i,-1]))for i in np.where(abs(act[:,-1])>0.1)[0]]})
    (ROOT/'candidate_gripper_events.json').write_text(json.dumps(rows,indent=2))
    raw='https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/main/'
    for f in ['dataset_info.json','features.json']:
        r=requests.get(raw+f,timeout=30);print('RAW',f,r.status_code,r.text[:10000],flush=True)
        if r.ok:(ROOT/f'raw_{f}').write_bytes(r.content)
    r=requests.get('https://huggingface.co/api/datasets/lerobot-raw/berkeley_autolab_ur5_raw/tree/main',timeout=30)
    print('RAW TREE',r.text[:3000],flush=True);(ROOT/'raw_tree.json').write_bytes(r.content)
if __name__=='__main__':main()
