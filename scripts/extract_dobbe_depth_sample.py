"""Fetch one HoNY RGB-D record from the last split ZIP with HTTP ranges.

The outer split archive stores the inner ZIP verbatim. The index created by
index_dobbe_depth_zip.py maps inner local-header offsets into the final part.
Only members wholly inside that part are eligible here.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import requests

from index_dobbe_depth_zip import BASE, OUT, confirmed_params, range_get


def fetch_range(session: requests.Session, params: dict, start: int, count: int) -> bytes:
    return b"".join(
        range_get(session, params, pos, min(start + count, pos + 1024 * 1024) - 1)
        for pos in range(start, start + count, 1024 * 1024)
    )


def main() -> None:
    index = json.loads((OUT / "dobbe_depth_nested_index.json").read_text(encoding="utf-8"))
    entries = index["entries"]
    target = "iphone_data_depth/Drawer_Closing/Home10/Env2/2022-12-22--00-09-08/"
    selected = [e for e in entries if e["name"].startswith(target) and
                Path(e["name"]).name in {
                    "compressed_np_depth_float32.bin", "compressed_video_h264.mp4", "labels.json"
                }]
    if len(selected) != 3:
        raise RuntimeError("Expected the three sample members")
    params = confirmed_params(index["final_part_file_id"])
    session = requests.Session()
    out_dir = OUT / "dobbe_rgbd_sample"
    out_dir.mkdir(exist_ok=True)
    receipt = {"source": target, "members": []}
    for entry in selected:
        local = index["inner_start_relative_final_part"] + entry["local_offset"]
        header = fetch_range(session, params, local, 30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Local ZIP header absent at {local}")
        method, name_bytes, extra_bytes = (
            struct.unpack_from("<H", header, 8)[0],
            struct.unpack_from("<H", header, 26)[0],
            struct.unpack_from("<H", header, 28)[0],
        )
        data_start = local + 30 + name_bytes + extra_bytes
        compressed_bytes = entry["compressed_bytes"]
        if data_start < 0 or data_start + compressed_bytes > index["final_part_bytes"]:
            raise RuntimeError(f"Member crosses a split-part boundary: {entry['name']}")
        compressed = fetch_range(session, params, data_start, compressed_bytes)
        if method != entry["method"]:
            raise RuntimeError(f"Compression method mismatch for {entry['name']}")
        content = zlib.decompress(compressed, -15) if method == 8 else compressed
        if len(content) != entry["uncompressed_bytes"]:
            raise RuntimeError(f"Uncompressed length mismatch for {entry['name']}")
        path = out_dir / Path(entry["name"]).name
        path.write_bytes(content)
        receipt["members"].append({"name": entry["name"], "bytes": len(content),
                                   "final_part_offset": data_start, "method": method})
    (OUT / "dobbe_rgbd_sample_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
