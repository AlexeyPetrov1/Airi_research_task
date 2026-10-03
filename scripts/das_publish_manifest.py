"""Create a compact index and hashes for the deliverable, excluding model weights."""
import argparse
import json
from pathlib import Path
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,required=True);a=p.parse_args()
out=a.scene/'das_wanfun'
files=[f for f in out.rglob('*') if f.is_file() and f.name not in ['artifact_manifest.json']]
result={'scope':'Only this DaS experiment; source Berkeley inputs and checkpoints remain separate',
        'files':{str(f.relative_to(out)):{'bytes':f.stat().st_size,'sha256':sha256(f)} for f in sorted(files)},
        'total_bytes':sum(f.stat().st_size for f in files)}
write_json(out/'artifact_manifest.json',result)
print(json.dumps({'files':len(files),'bytes':result['total_bytes']},indent=2))
