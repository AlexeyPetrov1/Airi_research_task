"""Probe byte-range access to HoNY's final split ZIP member, no bulk download."""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "dobbe_depth_delivery_probe.json"
GDOWN_TARGET = ROOT.parent / ".tools" / "gdown"
FOLDER_ID = "1o8c6b6hSKfId8EzemVGf8c7DQoZ2IHAO"


def main() -> None:
    sys.path.insert(0, str(GDOWN_TARGET))
    import gdown

    files = gdown.download_folder(id=FOLDER_ID, skip_download=True,
                                  quiet=True, timeout=30)
    if not files:
        raise RuntimeError("Google Drive folder listing unavailable")
    last = next(item for item in files if item.path.endswith(".zip"))
    base = "https://drive.usercontent.google.com/download"
    response = requests.get(base, params={"id": last.id, "export": "download"},
                            timeout=20)
    response.raise_for_status()
    match = re.search(r'name="uuid" value="([^"]+)"', response.text)
    if not match:
        raise RuntimeError("Google Drive confirmation token unavailable")
    request = requests.get(base, params={"id": last.id, "export": "download",
                                         "confirm": "t", "uuid": match.group(1)},
                           headers={"Range": "bytes=-65536"}, stream=True,
                           timeout=(10, 20))
    result = {"folder_id": FOLDER_ID, "split_file_count": len(files),
              "first_file": files[0].path, "last_file": last.path,
              "last_file_id": last.id, "range_status": request.status_code,
              "content_range": request.headers.get("Content-Range"),
              "content_length": request.headers.get("Content-Length"),
              "content_type": request.headers.get("Content-Type")}
    if request.status_code == 206 and int(result["content_length"]) <= 65536:
        tail = request.content
        result["tail"] = tail[-32:].hex()
        at = tail.rfind(b"PK\x05\x06")
        if at >= 0 and at + 22 <= len(tail):
            fields = struct.unpack_from("<4s4H2IH", tail, at)
            result["zip_eocd"] = {"this_disk": fields[1],
                                  "central_directory_disk": fields[2],
                                  "records_this_disk": fields[3],
                                  "records_total": fields[4],
                                  "central_directory_bytes": fields[5],
                                  "central_directory_offset_on_disk": fields[6]}
            cd_at = tail.rfind(b"PK\x01\x02", 0, at)
            if cd_at >= 0:
                result["central_directory_first_method"] = struct.unpack_from(
                    "<H", tail, cd_at + 10)[0]
                result["central_directory_first_compressed_bytes_32"] = struct.unpack_from(
                    "<I", tail, cd_at + 20)[0]
                result["central_directory_first_uncompressed_bytes_32"] = struct.unpack_from(
                    "<I", tail, cd_at + 24)[0]
                name_length = struct.unpack_from("<H", tail, cd_at + 28)[0]
                name = tail[cd_at + 46:cd_at + 46 + name_length]
                result["central_directory_first_name"] = name.decode(
                    "utf-8", errors="replace")
                result["central_directory_entry_offset_in_tail"] = cd_at
                nested_eocd_at = tail.rfind(b"PK\x05\x06", 0, cd_at)
                if nested_eocd_at >= 0 and nested_eocd_at + 22 <= cd_at:
                    nested = struct.unpack_from("<4s4H2IH", tail, nested_eocd_at)
                    result["nested_zip_eocd"] = {
                        "offset_in_tail": nested_eocd_at,
                        "this_disk": nested[1],
                        "central_directory_disk": nested[2],
                        "records_total_16": nested[4],
                        "central_directory_bytes_32": nested[5],
                        "central_directory_offset_32": nested[6],
                    }
    request.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
