"""Fetch only the selected 34 MB WorldTrack NPZ and 68 MB author index."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent / "source"
REV = "564ffa2e3cdb0ba443db8590b40e9691f487436c"
INDEX_URL = f"https://huggingface.co/datasets/allenai/PointMotionBench/resolve/{REV}/worldtrack/worldtrack_index_map.json"
SCRIPT_URL = f"https://huggingface.co/datasets/allenai/PointMotionBench/resolve/{REV}/worldtrack/reconstruct_worldtrack.py"
FILE_ID = "1dqQGUTajJ8tvfrC_FDV3xKulLX0GN748"
EXPECTED = {
    "worldtrack_index_map.json": (68124748, "9a3649138ad6493104ed3e1b1be8091086115f3ca7d1eea5df31c0c82d7c7668"),
    "Apartment_release_clean_seq131_0.npz": (33815492, "05d3c40e0fc76073899a58a6dba520c7fb3889eb9bcf3f9fd9224287d50bd323"),
}


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(path):
    if not path.exists():
        return False
    size, digest = EXPECTED[path.name]
    return path.stat().st_size == size and file_sha(path) == digest


def download_response(response, path):
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise RuntimeError("Download returned an HTML page")
    temp = path.with_suffix(path.suffix + ".download")
    with temp.open("wb") as f:
        for chunk in response.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    size, digest = EXPECTED[path.name]
    if temp.stat().st_size != size or file_sha(temp) != digest:
        raise RuntimeError(f"Wrong size or digest for {path.name}")
    temp.replace(path)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    index = ROOT / "worldtrack_index_map.json"
    if not valid(index):
        download_response(session.get(INDEX_URL, stream=True, timeout=120), index)
    source = ROOT / "Apartment_release_clean_seq131_0.npz"
    if not valid(source):
        warning = session.get(f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t", timeout=30)
        warning.raise_for_status()
        match = re.search(r'name="uuid" value="([^"]+)"', warning.text)
        if not match:
            raise RuntimeError("Google Drive confirmation token missing")
        url = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t&uuid={match.group(1)}"
        download_response(session.get(url, stream=True, timeout=120), source)
    script = ROOT / "reconstruct_worldtrack.py"
    if not script.exists():
        response = session.get(SCRIPT_URL, timeout=30)
        response.raise_for_status()
        script.write_bytes(response.content)
    for path in (index, source, script):
        print(path, path.stat().st_size, file_sha(path))


if __name__ == "__main__":
    main()
