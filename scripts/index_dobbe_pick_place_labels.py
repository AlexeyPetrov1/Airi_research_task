"""Index only Pick_and_Place HoNY gripper traces using HTTP ZIP ranges."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from inspect_remote_zip import HTTPRangeFile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dobbe_oxe" / "pick_place_label_index.jsonl"
URL = "https://dl.dobb-e.com/datasets/homes_of_new_york.zip"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as file:
            done = {json.loads(line)["member"] for line in file if line.strip()}
    remote = HTTPRangeFile(URL, block_size=65536)
    with zipfile.ZipFile(remote) as archive, OUT.open("a", encoding="utf-8") as output:
        members = [m for m in archive.infolist()
                   if m.filename.startswith("iphone_data/Pick_and_Place/") and
                   m.filename.endswith("/labels.json")]
        members.sort(key=lambda m: abs(m.file_size - 242 * 210))
        for i, member in enumerate(members, 1):
            if member.filename in done:
                continue
            labels = json.loads(archive.read(member))
            n = len(labels)
            record = {"member": member.filename, "frames": n,
                      "gripper": [labels[str(j)]["gripper"] for j in range(n)],
                      "first_xyz": labels["0"]["xyz"]}
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
            if i % 50 == 0:
                print("indexed", i, "/", len(members),
                      "range_bytes", remote.transferred, flush=True)
    print("indexed_total", len(members), "transferred", remote.transferred)


if __name__ == "__main__":
    main()
