"""Save current primary-source receipts for the improvement review."""
import json,urllib.request
from datetime import datetime,timezone
from fmb_wrist_prepare import ROOT,write,sha
from fmb_wrist_v4_diagnose import OUT

def run():
    dest=OUT/'sources';dest.mkdir(exist_ok=True);receipts=[]
    requests=[
        ('hf_model_inventory.json','https://huggingface.co/api/models?author=allenai&search=MolmoMotion&sort=lastModified&direction=-1&limit=10'),
        ('upstream_commit.json','https://api.github.com/repos/allenai/molmo-motion/commits/main'),
    ]
    for name,url in requests:
        target=dest/name
        if not target.exists():target.write_bytes(urllib.request.urlopen(url,timeout=40).read())
        receipts.append({'url':url,'file':name,'sha256':sha(target)})
    commit=json.loads((dest/'upstream_commit.json').read_text())['sha']
    for path in ['README.md','src/molmo_motion/processor.py','src/molmo_motion/data/trajectory_3d_dataset.py','data_generation/README.md']:
        url=f'https://raw.githubusercontent.com/allenai/molmo-motion/{commit}/{path}';target=dest/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(urllib.request.urlopen(url,timeout=40).read())
        receipts.append({'url':url,'file':path,'revision':commit,'sha256':sha(target),'local_file_sha256':sha(ROOT/path)})
    write(dest/'source_receipts.json',{'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sources':receipts,
          'connector_search_failure':'Hugging Face model_search returned tool not found; official public Hub API used as fallback',
          'assessment_VLM_downloaded_or_used':False})
    inventory=json.loads((dest/'hf_model_inventory.json').read_text())
    print('Official Hub inventory',[(r['id'],r.get('sha')) for r in inventory],flush=True)

if __name__=='__main__':run()
