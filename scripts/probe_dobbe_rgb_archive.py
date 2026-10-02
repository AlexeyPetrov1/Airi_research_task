"""Inventory the official HoNY RGB ZIP over HTTP ranges without downloading it."""

from __future__ import annotations

import collections
import json
import sys
import zipfile
from pathlib import Path

from inspect_remote_zip import HTTPRangeFile


URL = "https://dl.dobb-e.com/datasets/homes_of_new_york.zip"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "plex_dobbe_preflight"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    remote = HTTPRangeFile(URL, block_size=65536)
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        names = [m.filename for m in members if not m.is_dir()]
        tasks = collections.Counter(
            path.split("/")[1] for path in names
            if path.startswith("iphone_data/") and len(path.split("/")) > 3
        )
        suffixes = collections.Counter(Path(path).name for path in names)
        target = "iphone_data/r3d_files.txt"
        with archive.open(target) as file:
            (OUT / "dobbe_r3d_files.txt").write_bytes(file.read())
        result = {
            "url": URL, "archive_size_bytes": remote.length,
            "transferred_bytes": remote.transferred,
            "members": len(members), "files": len(names),
            "task_file_counts": dict(sorted(tasks.items())),
            "cup_task_names": [t for t in tasks if "cup" in t.lower() or "mug" in t.lower()],
            "metadata_members": [p for p in names if Path(p).name == "metadata"],
            "depth_members": [p for p in names if p.endswith(".depth") or "compressed_np_depth" in p],
            "known_file_count": {k: suffixes[k] for k in
                                 ("compressed_video_h264.mp4", "labels.json", "r3d_files.txt")},
            "sample_cup_files": [p for p in names if "/cup" in p.lower()][:30],
        }
    (OUT / "dobbe_rgb_archive_inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
