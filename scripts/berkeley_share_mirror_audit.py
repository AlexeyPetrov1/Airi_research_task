"""Find bounded access to the exact ShareRobot planning images in Hub mirrors."""
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/berkeley_ur5_improvement_v1/sources'


def main():
    names=json.loads((OUT/'sharerobot_archive_headers.json').read_text())['dataset_mirrors']
    def probe(name):
        r=requests.get(f'https://huggingface.co/api/datasets/{name}',timeout=45)
        r.raise_for_status();info=r.json();files=[x['rfilename'] for x in info.get('siblings',[])]
        row=dict(repo=name,revision=info['sha'],file_count=len(files),
            target_paths=[x for x in files if '43_berkeley_autolab_ur5' in x and ('#episode_9/' in x or '#episode_10/' in x)],
            planning_archives=[x for x in files if 'planning' in x and ('.tar' in x or '.zip' in x)],
            parquet_paths=[x for x in files if x.endswith('.parquet')],
            file_sample=files[:20])
        print(json.dumps({k:v for k,v in row.items() if k!='file_sample'}),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,names))
    (OUT/'sharerobot_mirror_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
