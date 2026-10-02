"""List complete HoNY RGB-D captures from the existing split-ZIP index.

No network request is made. The ranking is a screening aid; inspect RGB and
camera motion after selectively downloading a candidate.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "runs/plex_dobbe_preflight/dobbe_depth_nested_index.json"
NEEDED = {
    "compressed_video_h264.mp4",
    "compressed_np_depth_float32.bin",
    "labels.json",
}


def main() -> None:
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for item in index["entries"]:
        path = Path(item["name"])
        if path.name in NEEDED:
            groups[str(path.parent)][path.name] = item
    candidates = []
    for path, members in groups.items():
        if set(members) == NEEDED and "Home15/Env1/2023-04-25--02-05-30" not in path:
            video = members["compressed_video_h264.mp4"]
            depth = members["compressed_np_depth_float32.bin"]
            labels = members["labels.json"]
            candidates.append({
                "path": path,
                "part": (105 + video["local_offset"]) // 1073741824 + 1,
                "video_bytes": video["uncompressed_bytes"],
                "depth_bytes": depth["uncompressed_bytes"],
                "labels_bytes": labels["uncompressed_bytes"],
            })
    candidates = [c for c in candidates
                  if "/Pick_and_Place/" in c["path"]
                  and 10_000_000 <= c["depth_bytes"] <= 40_000_000
                  and "/Home15/" not in c["path"]]
    candidates.sort(key=lambda x: x["video_bytes"], reverse=True)
    print("Complete Pick_and_Place captures in other homes (depth 10-40 MB):", len(candidates))
    print(json.dumps(candidates[:30], indent=2))


if __name__ == "__main__":
    main()
