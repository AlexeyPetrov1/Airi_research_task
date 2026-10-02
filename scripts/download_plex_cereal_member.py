"""Download only the PickPlaceCereal HDF5 member from Microsoft's PLEX ZIP."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from inspect_remote_zip import HTTPRangeFile


URL = (
    "https://download.microsoft.com/download/e/5/1/"
    "e5106eb4-0f53-4d65-afd1-58c03d60bf98/robosuite_demo_data%20%281%29.zip"
)
MEMBER = (
    "robosuite_demo_data/PickPlaceCereal/Panda/raw/"
    "PickPlaceCereal_demo_act_norm.hdf5"
)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "plex" / Path(MEMBER).name


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    remote = HTTPRangeFile(URL)
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(MEMBER)
        if OUT.exists() and OUT.stat().st_size == info.file_size:
            status = "existing_size_match"
        else:
            partial = OUT.with_suffix(".partial")
            with archive.open(info) as src, partial.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            if partial.stat().st_size != info.file_size:
                raise RuntimeError("extracted member has unexpected size")
            partial.replace(OUT)
            status = "downloaded"
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    result = {"status": status, "source_url": URL, "member": MEMBER,
              "archive_bytes": remote.length, "transferred_bytes": remote.transferred,
              "member_compressed_bytes": info.compress_size,
              "member_uncompressed_bytes": info.file_size,
              "output": str(OUT), "sha256": digest}
    (OUT.parent / "download_receipt.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
