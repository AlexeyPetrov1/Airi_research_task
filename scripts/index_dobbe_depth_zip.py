"""Index the inner HoNY RGB-D ZIP from only the tail of its final split part.

The outer split ZIP stores one uncompressed nested ZIP. This script reads its
central directory by HTTP Range and downloads no RGB/depth payload.
"""

from __future__ import annotations

import collections
import json
import re
import struct
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "plex_dobbe_preflight"
GDOWN_TARGET = ROOT.parent / ".tools" / "gdown"
FOLDER_ID = "1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO"
BASE = "https://drive.usercontent.google.com/download"


def confirmed_params(file_id: str) -> dict[str, str]:
    response = requests.get(BASE, params={"id": file_id, "export": "download"},
                            timeout=20)
    response.raise_for_status()
    match = re.search(r'name="uuid" value="([^"]+)"', response.text)
    if not match:
        raise RuntimeError("Google Drive confirmation form missing")
    return {"id": file_id, "export": "download", "confirm": "t",
            "uuid": match.group(1)}


def range_get(session: requests.Session, params: dict, start: int, end: int) -> bytes:
    for _ in range(3):
        response = session.get(BASE, params=params,
                               headers={"Range": f"bytes={start}-{end}"},
                               timeout=(10, 30), stream=True)
        valid = (response.status_code == 206 and
                 response.headers.get("Content-Range", "").startswith(
                     f"bytes {start}-{end}/"))
        if valid:
            content = response.content
            response.close()
            if len(content) != end - start + 1:
                raise RuntimeError("Incomplete Google Drive range")
            return content
        response.close()
    raise RuntimeError(f"Bad Google Drive range response at {start}-{end}")


def main() -> None:
    sys.path.insert(0, str(GDOWN_TARGET))
    import gdown

    files = gdown.download_folder(id=FOLDER_ID, skip_download=True,
                                  quiet=True, timeout=30)
    if not files:
        raise RuntimeError("Google Drive folder listing unavailable")
    last = next(item for item in files if item.path.endswith(".zip"))
    params = confirmed_params(last.id)
    session = requests.Session()
    probe = session.get(BASE, params=params, headers={"Range": "bytes=-65536"},
                        timeout=(10, 30))
    probe.raise_for_status()
    if probe.status_code != 206:
        raise RuntimeError("Last split part does not support ranges")
    file_length = int(probe.headers["Content-Range"].split("/")[1])
    # The nested central directory is 4.7 MB; fetch a little extra headroom.
    start = file_length - 6 * 1024 * 1024
    parts = [range_get(session, params, offset,
                       min(file_length - 1, offset + 1024 * 1024 - 1))
             for offset in range(start, file_length, 1024 * 1024)]
    tail = b"".join(parts)
    outer_eocd = tail.rfind(b"PK\x05\x06")
    nested_eocd = tail.rfind(b"PK\x05\x06", 0, outer_eocd)
    zip64 = tail.rfind(b"PK\x06\x06", 0, nested_eocd)
    if min(outer_eocd, nested_eocd, zip64) < 0:
        raise RuntimeError("Expected nested ZIP / ZIP64 footer not found")
    fields = struct.unpack_from("<4sQHHIIQQQQ", tail, zip64)
    count, central_bytes, central_offset = fields[7], fields[8], fields[9]
    # The outer split archive stores the inner ZIP without compression. This
    # relative start also lets us fetch members living entirely in the final
    # split part without downloading the 77 GB delivery.
    inner_start_relative_final_part = (start + zip64) - (central_offset + central_bytes)
    cd_start = zip64 - central_bytes
    if tail[cd_start:cd_start + 4] != b"PK\x01\x02":
        raise RuntimeError("Inner central-directory start not found")
    entries = []
    cursor = cd_start
    for _ in range(count):
        if tail[cursor:cursor + 4] != b"PK\x01\x02":
            raise RuntimeError(f"Unexpected central-directory signature at {cursor}")
        method = struct.unpack_from("<H", tail, cursor + 10)[0]
        compressed, uncompressed = struct.unpack_from("<II", tail, cursor + 20)
        n_name, n_extra, n_comment = struct.unpack_from("<HHH", tail, cursor + 28)
        disk_start = struct.unpack_from("<H", tail, cursor + 34)[0]
        local_offset = struct.unpack_from("<I", tail, cursor + 42)[0]
        name = tail[cursor + 46:cursor + 46 + n_name].decode("utf-8", errors="replace")
        extra = tail[cursor + 46 + n_name:cursor + 46 + n_name + n_extra]
        if local_offset == 0xffffffff or compressed == 0xffffffff or uncompressed == 0xffffffff:
            at = 0
            while at + 4 <= len(extra):
                kind, size = struct.unpack_from("<HH", extra, at)
                body = extra[at + 4:at + 4 + size]
                if kind == 1:
                    value_at = 0
                    if uncompressed == 0xffffffff:
                        uncompressed = struct.unpack_from("<Q", body, value_at)[0]
                        value_at += 8
                    if compressed == 0xffffffff:
                        compressed = struct.unpack_from("<Q", body, value_at)[0]
                        value_at += 8
                    if local_offset == 0xffffffff:
                        local_offset = struct.unpack_from("<Q", body, value_at)[0]
                    break
                at += 4 + size
            else:
                raise RuntimeError(f"Missing ZIP64 extra for {name}")
        entries.append({"name": name, "method": method,
                        "compressed_bytes": compressed,
                        "uncompressed_bytes": uncompressed,
                        "local_offset": local_offset, "disk_start": disk_start})
        cursor += 46 + n_name + n_extra + n_comment
    if cursor != zip64:
        raise RuntimeError("Central-directory length mismatch")
    suffixes = collections.Counter(Path(e["name"]).name for e in entries)
    result = {"source_folder_id": FOLDER_ID,
              "final_part_file_id": last.id,
              "final_part_bytes": file_length,
              "bytes_downloaded": len(tail) + len(probe.content),
              "inner_archive_file_count": len(entries),
              "central_directory_bytes": central_bytes,
              "central_directory_offset": central_offset,
              "inner_start_relative_final_part": inner_start_relative_final_part,
              "sample_names": [e["name"] for e in entries[:20]],
              "basename_counts": dict(suffixes.most_common(20)),
              "metadata_count": sum(Path(e["name"]).name == "metadata" for e in entries),
              "depth_file_count": sum(e["name"].endswith(".depth") for e in entries),
              "entries": entries}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dobbe_depth_nested_index.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "entries"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
