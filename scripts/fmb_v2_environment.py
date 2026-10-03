"""Read-only environment/checkpoint probe; no secrets are printed."""
import json
from pathlib import Path
import torch, transformers, requests
root=Path(__file__).resolve().parents[1]
print(json.dumps({'torch':torch.__version__,'transformers':transformers.__version__,
                  'cuda':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0)}),flush=True)
data=requests.get('https://huggingface.co/api/models/allenai/MolmoMotion-4B-H1-F32',timeout=60).json()
out=root/'runs/fmb_v2_berkeley_matched'
out.mkdir(exist_ok=True)
safe={k:data.get(k) for k in ['id','sha','siblings']}
(out/'h1_checkpoint_hub.json').write_text(json.dumps(safe,indent=2)+'\n')
print(json.dumps(safe),flush=True)
