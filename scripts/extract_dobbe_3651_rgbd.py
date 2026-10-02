"""Fetch the gripper-matched HoNY RGB-D capture from one split ZIP part."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path

import requests

from index_dobbe_depth_zip import BASE, OUT, confirmed_params, range_get


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "dobbe_oxe" / "target_raw"
INDEX = OUT / "dobbe_depth_nested_index.json"
GDOWN_TARGET = ROOT.parent / ".tools" / "gdown"
FOLDER_ID = "1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO"
SOURCE = "iphone_data_depth/Pick_and_Place/Home15/Env1/2023-04-25--02-05-30/"
PART_SIZE = 1073741824
INNER_START_GLOBAL = 105  # Verified from .z01 split marker and local ZIP header.


def read_range(session: requests.Session, params: dict, start: int, count: int) -> bytes:
    return b"".join(
        range_get(session, params, p, min(start + count, p + 1024 * 1024) - 1)
        for p in range(start, start + count, 1024 * 1024)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE,
                        help="Inner ZIP capture directory, including iphone_data_depth/")
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--receipt", type=Path,
                        default=OUT / "dobbe_3651_rgbd_receipt.json")
    args = parser.parse_args()
    source = args.source.rstrip("/") + "/"
    dest = args.dest
    sys.path.insert(0, str(GDOWN_TARGET))
    import gdown

    index = json.loads(INDEX.read_text(encoding="utf-8"))
    file_list = gdown.download_folder(id=FOLDER_ID, skip_download=True,
                                      quiet=True, timeout=30)
    if not file_list:
        raise RuntimeError("Could not list split ZIP parts")
    files = {Path(item.path).suffix: item.id for item in file_list}
    selected = [entry for entry in index["entries"]
                if entry["name"].startswith(source) and Path(entry["name"]).name in {
                    "compressed_np_depth_float32.bin", "compressed_video_h264.mp4", "labels.json"
                }]
    if len(selected) != 3:
        raise RuntimeError("Target RGB-D trio absent from ZIP index")
    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    receipt = {"source": source, "members": [],
               "exact_camera_K_in_archive": False}
    params_cache = {}
    for entry in selected:
        global_offset = INNER_START_GLOBAL + entry["local_offset"]
        part_zero = global_offset // PART_SIZE
        suffix = f".z{part_zero + 1:02d}" if part_zero < 76 else ".zip"
        if suffix not in files:
            raise RuntimeError(f"Split part {suffix} missing")
        params = params_cache.setdefault(suffix, confirmed_params(files[suffix]))
        local = global_offset % PART_SIZE
        header = read_range(session, params, local, 30)
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Missing member local header: {entry['name']}")
        method, name_len, extra_len = (
            struct.unpack_from("<H", header, 8)[0],
            struct.unpack_from("<H", header, 26)[0],
            struct.unpack_from("<H", header, 28)[0],
        )
        full_name = read_range(session, params, local + 30, name_len).decode("utf-8")
        if full_name != entry["name"] or method != entry["method"]:
            raise RuntimeError("ZIP central/local header mismatch")
        data_start = local + 30 + name_len + extra_len
        size = entry["compressed_bytes"]
        if data_start + size > PART_SIZE:
            raise RuntimeError(f"Member crosses split boundary: {entry['name']}")
        compressed = read_range(session, params, data_start, size)
        data = zlib.decompress(compressed, -15) if method == 8 else compressed
        if len(data) != entry["uncompressed_bytes"]:
            raise RuntimeError("ZIP member length mismatch")
        target = dest / Path(entry["name"]).name
        target.write_bytes(data)
        receipt["members"].append({"name": entry["name"], "part": suffix,
                                   "uncompressed_bytes": len(data),
                                   "sha256": hashlib.sha256(data).hexdigest()})
        print("extracted", target, len(data), flush=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
