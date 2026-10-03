"""Inspect the single selected WorldTrack source and benchmark map entry."""

import json
from pathlib import Path

import numpy as np

root = Path(__file__).parent
key = "adt_mini/Apartment_release_clean_seq131_0_clip02_obj1_t108-149"
entry = json.loads((root / "source/worldtrack_index_map.json").read_text())[key]
print("KEY", key)
for k, v in entry.items():
    if k == "point_indices":
        print(k, "count", len(v), "first", v[:20])
    elif k == "frame_indices":
        print(k, "count", len(v), "first", v[:10], "last", v[-10:])
    else:
        print(k, v)
with np.load(root / "source/Apartment_release_clean_seq131_0.npz", allow_pickle=True) as data:
    for k in data.files:
        x = data[k]
        print("ARRAY", k, x.shape, x.dtype, "sample", str(x.ravel()[0])[:120] if x.size else "")
