"""Bounded TFRecord retrieval and exact RGB/state identity audit."""
from pathlib import Path
import json,struct,hashlib,time,io,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import requests
import numpy as np
import pandas as pd
import av
from PIL import Image,ImageDraw
ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')
def sheets():
    for idx,start,stop in [(10,55,74),(9,34,53)]:
        video=ROOT/'videos/chunk-000/observation.images.image'/f'episode_{idx:06d}.mp4'
        frames=[f.to_ndarray(format='rgb24')for f in av.open(video).decode(video=0)]
        out=Image.new('RGB',(320*5,260*4),'white');d=ImageDraw.Draw(out)
        for k,i in enumerate(range(start,stop+1)):
            x=k%5*320;y=k//5*260;out.paste(Image.fromarray(frames[i]).resize((320,240)),(x,y));d.text((x+5,y+242),f'episode {idx} frame {i} t={i/5:.1f}',fill='black')
        out.save(ROOT/f'detailed_{idx:06d}.jpg')
        (ROOT/f'episode_{idx:06d}_frames').mkdir(exist_ok=True)
        for i in range(start,stop+1):Image.fromarray(frames[i]).save(ROOT/f'episode_{idx:06d}_frames/frame_{i:06d}.png')
def main():
    sheets()
    name='berkeley_autolab_ur5-train.tfrecord-00004-of-00412';out=ROOT/name
    tree=json.loads((ROOT/'raw_tree.json').read_text());entry=next(e for e in tree if e['path']==name)
    print('CHECKED SHARD',entry,flush=True)
    if not out.exists():
        url='https://huggingface.co/datasets/lerobot-raw/berkeley_autolab_ur5_raw/resolve/d2590280290484e2a7eb53a91bba32ac2ff669a0/'+name
        with requests.get(url,stream=True,timeout=(30,120))as r:
            r.raise_for_status()
            with out.with_suffix('.partial').open('wb')as f:
                count=0
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk);count+=len(chunk)
                    if count>entry['size']:raise ValueError('Size mismatch')
        if count!=entry['size']:raise ValueError('Size mismatch')
        out.with_suffix('.partial').replace(out)
    print('SAVED',out,out.stat().st_size,flush=True)
    from probe_fmb_rlds_record import find_feature_values
    with out.open('rb')as f:
        i=0
        while True:
            header=f.read(12)
            if not header:break
            length=struct.unpack('<Q',header[:8])[0];record=f.read(length);f.read(4)
            features=dict(find_feature_values(record));summary={}
            for k,vals in features.items():
                summary[k]={'count':len(vals),'first_lengths':[e-s for s,e in vals[:3]]}
                if 'instruction' in k:
                    s,e=vals[0];summary[k]['first']=record[s:e].decode()
                if k.endswith('/image'):
                    s,e=vals[0];img=Image.open(io.BytesIO(record[s:e]));img.save(ROOT/f'raw_shard4_record{i}_first.png')
            (ROOT/f'raw_shard4_record{i}_summary.json').write_text(json.dumps(summary,indent=2))
            print('RECORD',i,'SUMMARY',summary,flush=True)
            i+=1
if __name__=='__main__':main()
