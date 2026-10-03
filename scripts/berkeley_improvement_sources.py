"""Fetch pinned official calibration and DELTA sources; inspect ShareRobot mapping.

Provenance only. Does not modify or recompute the original ML experiment.
"""
from pathlib import Path
import hashlib
import json
import sys
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'
OUT.mkdir(parents=True,exist_ok=True)


def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=float)+'\n',encoding='utf8')


def fetch(url,path):
    r=requests.get(url,timeout=(20,120));r.raise_for_status()
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(r.content)
    return dict(url=url,sha256=hashlib.sha256(r.content).hexdigest(),bytes=len(r.content),path=str(path.relative_to(OUT)))


def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'sources'
    if mode=='sources':
        receipts=[]
        auge='0added8b6645fef1c7de2ad5d5bd7e9c1bde0d2e'
        receipts.append(fetch(f'https://raw.githubusercontent.com/BerkeleyAutomation/AugE-Toolkit/{auge}/config.py',OUT/'auge_config.py'))
        delta=requests.get('https://api.github.com/repos/snap-research/DELTA_densetrack3d/commits/main',timeout=30).json()['sha']
        tree=requests.get(f'https://api.github.com/repos/snap-research/DELTA_densetrack3d/git/trees/{delta}?recursive=1',timeout=30).json()
        write(OUT/'delta_tree.json',tree)
        names=['densetrack3d/models/predictor/predictor.py','README.md']
        names += [x['path'] for x in tree['tree'] if x['path'].endswith('.py') and ('model_utils' in x['path'] or 'geometry_utils' in x['path'])]
        for name in names:
            receipts.append(fetch(f'https://raw.githubusercontent.com/snap-research/DELTA_densetrack3d/{delta}/{name}',OUT/'delta'/name))
        write(OUT/'official_sources.json',dict(auge_commit=auge,delta_commit=delta,files=receipts))
        print(json.dumps(dict(auge_commit=auge,delta_commit=delta)),flush=True)
    elif mode=='share':
        info=requests.get('https://huggingface.co/api/datasets/BAAI/ShareRobot',timeout=40).json()
        rev=info['sha'];write(OUT/'sharerobot_metadata.json',info)
        receipts=[]; matches=[]
        paths=[x['rfilename'] for x in info['siblings'] if x['rfilename'].endswith('.json') and ('trajectory/' in x['rfilename'] or 'planning/jsons/' in x['rfilename'])]
        write(OUT/'sharerobot_search_scope.json',dict(revision=rev,paths=paths,target_episodes=[10,9],
            warning='Episode numbering alone does not prove mapping; compare selected frames by pixels.'))
        for name in paths:
            cached=OUT/('share_'+Path(name).stem+'.json')
            if cached.exists():
                prior=json.loads(cached.read_text())
                matches.append(dict(path=name,rows_examined=prior['rows_examined'],berkeley_rows=len(prior['berkeley_rows'])))
                continue
            print('ShareRobot reading',name,flush=True)
            source=f'https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{rev}/{name}'
            # Stream large planning JSON; preserve matching rows, not a 200MB cache.
            sys.path.insert(0,str(ROOT.parent/'.tools/ijson'))
            import ijson
            r=requests.get(source,stream=True,timeout=(20,120));r.raise_for_status();r.raw.decode_content=True
            count=0;found=[]
            try:
                for row in ijson.items(r.raw,'item',use_float=True):
                    count+=1
                    text=json.dumps(row)
                    if 'berkeley_autolab_ur5' in text:
                        found.append(dict(row_index=count-1,row=row))
            finally:r.close()
            write(OUT/('share_'+Path(name).stem+'.json'),dict(source=source,rows_examined=count,berkeley_rows=found))
            matches.append(dict(path=name,rows_examined=count,berkeley_rows=len(found)))
            print(json.dumps(matches[-1]),flush=True)
        write(OUT/'sharerobot_manifest_search.json',dict(revision=rev,complete=True,files=matches))


if __name__=='__main__':main()
