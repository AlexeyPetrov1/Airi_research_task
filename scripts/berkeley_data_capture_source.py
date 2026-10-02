"""Save original Berkeley capture code establishing depth alignment and units."""
from pathlib import Path
import requests,json,hashlib
ROOT=Path('/mnt/f/AIRI_task/data/berkeley_ur5_two_scenes/capture_source');ROOT.mkdir(exist_ok=True)
s=requests.Session();api='https://api.github.com/repos/yunliangchen/ur5bc'
commit=s.get(api+'/commits/main',timeout=30).json()['sha']
tree=s.get(api+f'/git/trees/{commit}?recursive=1',timeout=30).json()['tree']
paths=[e['path']for e in tree if e['path'].endswith('.py')and ('camera' in e['path'].lower()or 'realsense' in e['path'].lower()or 'robot_env' in e['path'].lower())]
print('COMMIT',commit,'FILES',paths,flush=True)
receipt=[]
for p in paths:
    url=f'https://raw.githubusercontent.com/yunliangchen/ur5bc/{commit}/{p}'
    r=s.get(url,timeout=30);r.raise_for_status();out=ROOT/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(r.content)
    matches=[{'line':i,'text':line}for i,line in enumerate(r.text.splitlines(),1)if any(q in line for q in ['depth_scale','align','third_person_image','get_distance'])]
    receipt.append({'path':p,'source':url,'sha256':hashlib.sha256(r.content).hexdigest(),'relevant_lines':matches});print(p,matches[:35],flush=True)
(ROOT/'receipt.json').write_text(json.dumps({'commit':commit,'repo':'https://github.com/yunliangchen/ur5bc','files':receipt},indent=2))
