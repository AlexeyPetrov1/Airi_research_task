"""Summarize candidate author WorldTrack clips before loading source data."""

import json
from collections import Counter
from pathlib import Path

path = Path(__file__).parent / "source/worldtrack_index_map.json"
data = json.loads(path.read_text())
print("entries", len(data), "split counts", Counter(k.split("/")[0] for k in data))
for key, entry in data.items():
    if not key.startswith("adt_mini/"):
        continue
    fi = entry["frame_indices"]
    print(
        key, "source=", entry["source"], "frames=", len(fi),
        "first_last=", (fi[0], fi[-1]), "points=", len(entry["point_indices"]),
        "caption=", str(entry.get("caption"))[:100],
    )
