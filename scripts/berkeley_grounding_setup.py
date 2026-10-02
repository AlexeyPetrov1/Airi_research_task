"""Inspect official grounding access and obtain user-approved official SAM2.1."""
from pathlib import Path
import json
import time
import requests
from huggingface_hub import HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/berkeley_ur5_molmomotion"
OUT.mkdir(parents=True, exist_ok=True)
record = {"checked_unix_time": time.time(), "models": {}}
api = HfApi()
for name in ("allenai/MolmoPoint-Vid-4B", "facebook/sam3"):
    info = api.model_info(name, files_metadata=True)
    weights = [{"file": f.rfilename, "bytes": f.size} for f in info.siblings
               if f.rfilename.endswith(".safetensors")]
    record["models"][name] = {"commit": info.sha, "gated": info.gated,
                              "safetensors": weights,
                              "total_safetensors_bytes": sum(f["bytes"] or 0 for f in weights)}
token = get_token()
headers = {"Authorization": "Bearer " + token} if token else {}
response = requests.get("https://huggingface.co/facebook/sam3/resolve/main/sam3.pt", headers=headers,
                        stream=True, timeout=60)
record["models"]["facebook/sam3"]["actual_weight_http_status"] = response.status_code
record["models"]["facebook/sam3"]["actual_error_code"] = response.headers.get("X-Error-Code")
if response.status_code != 200:
    record["models"]["facebook/sam3"]["actual_error_message"] = response.text[:1500]
response.close()
url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
dest = ROOT.parent / "models/sam2.1_hiera_large.pt"
record["official_sam2"] = {"url": url, "path": str(dest),
                          "hf_repo": "facebook/sam2.1-hiera-large",
                          "hf_revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
                          "selection_reason": "User explicitly approved official SAM2.1 earlier release after SAM3 access403"}
(OUT / "grounding_availability.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
if not dest.exists():
    hf_hub_download("facebook/sam2.1-hiera-large", "sam2.1_hiera_large.pt",
                    revision="665f8e2ad61cf5f53d65644ff27c8ee525124610", local_dir=str(dest.parent))
print("Official SAM2.1 present", dest.stat().st_size, flush=True)
print(json.dumps(record, indent=2), flush=True)
