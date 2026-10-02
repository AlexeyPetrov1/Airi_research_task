"""Bounded, reproducible Berkeley UR5 candidate retrieval (no depth-video use)."""
from pathlib import Path
import json, hashlib
from concurrent.futures import ThreadPoolExecutor
import requests

DEST=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes')
REV='6306aa91c00b9b0de28831b7fceaa42b8096bb4f'
BASE=f'https://huggingface.co/datasets/lerobot/berkeley_autolab_ur5/resolve/{REV}/'
def get(path,base=BASE):
    out=DEST/path
    out.parent.mkdir(parents=True,exist_ok=True)
    if not out.exists():
        r=requests.get(base+path,timeout=90);r.raise_for_status()
        if len(r.content)>30_000_000: raise ValueError('Unexpected large download')
        out.write_bytes(r.content)
    return out
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    for f in ['meta/info.json','meta/tasks.jsonl','meta/episodes.jsonl']:get(f)
    tasks=[json.loads(x) for x in (DEST/'meta/tasks.jsonl').read_text().splitlines()]
    episodes=[json.loads(x) for x in (DEST/'meta/episodes.jsonl').read_text().splitlines()]
    print('TASKS',tasks,flush=True)
    candidates={}
    for scene,phrase in [('cup','blue cup'),('bottle','ranch bottle')]:
        es=[e for e in episodes if any(phrase in t.lower() for t in e['tasks'])]
        candidates[scene]=es
        print(scene,len(es),es[:12],flush=True)
    (DEST/'candidates_all.json').write_text(json.dumps(candidates,indent=2))
    # First four each, selected only by minimum data duration and source index.
    chosen=[e for es in candidates.values() for e in es if e['length']>=40][:0]
    chosen=sum([[e for e in es if e['length']>=40][:4] for es in candidates.values()],[])
    paths=[]
    for e in chosen:
        i=e['episode_index'];paths.extend([f'data/chunk-000/episode_{i:06d}.parquet',f'videos/chunk-000/observation.images.image/episode_{i:06d}.mp4'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p in pool.map(get,paths):print('SAVED',p,p.stat().st_size,flush=True)
    receipt={'repo':'lerobot/berkeley_autolab_ur5','revision':REV,'selected_for_visual_inspection':chosen,'files':[{'path':p,'size':(DEST/p).stat().st_size,'sha256':hashlib.sha256((DEST/p).read_bytes()).hexdigest()}for p in paths]}
    (DEST/'retrieval_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('DONE',flush=True)
if __name__=='__main__':main()
