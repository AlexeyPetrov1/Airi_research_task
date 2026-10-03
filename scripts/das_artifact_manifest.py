"""Inventory a finished experiment without hashing its own manifest recursively."""
import argparse,json
from pathlib import Path
from das_prepare_control import sha256,write_json

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--verify',action='store_true');a=p.parse_args()
destination=a.root/'artifact_manifest.json'
if a.verify:
    manifest=json.loads(destination.read_text(encoding='utf-8'))
    failures=[name for name,record in manifest['files'].items()
              if not (a.root/name).is_file() or (a.root/name).stat().st_size!=record['bytes']
              or sha256(a.root/name)!=record['sha256']]
    print(json.dumps({'status':'PASS' if not failures else 'FAIL','files':len(manifest['files']),
                      'failed':failures},indent=2))
    raise SystemExit(bool(failures))
files={str(f.relative_to(a.root)).replace('\\','/'):{'bytes':f.stat().st_size,'sha256':sha256(f)}
       for f in sorted(a.root.rglob('*')) if f.is_file() and f!=destination and '__pycache__' not in f.parts}
write_json(destination,{'scope':'All experiment files, including baseline, variants, failure/recovery receipts; manifest excludes itself.',
                        'files':files,'total_bytes':sum(v['bytes'] for v in files.values())})
print({'files':len(files),'bytes':sum(v['bytes'] for v in files.values())})
