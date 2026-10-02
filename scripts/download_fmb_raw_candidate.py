"""Download one original FMB NPY, verifying its Hugging Face SHA-256.

Usage: python scripts/download_fmb_raw_candidate.py FILENAME.npy
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import requests


REVISION = "f99fd55c072eea5573523c96aa527aed3c665690"
BASE = (
    "https://huggingface.co/datasets/charlesxu0124/"
    f"functional-manipulation-benchmark/resolve/{REVISION}/"
    "single_object_manipulation_dataset/"
)
DEST = Path(__file__).resolve().parents[2] / "data/fmb/single_object_manipulation_dataset"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2 or Path(sys.argv[1]).name != sys.argv[1] or not sys.argv[1].endswith(".npy"):
        raise SystemExit("Specify one NPY basename")
    name = sys.argv[1]
    url = BASE + name
    head = requests.head(url, allow_redirects=False, timeout=30)
    head.raise_for_status()
    expected_size = int(head.headers["X-Linked-Size"])
    expected_hash = head.headers["X-Linked-ETag"].strip('"')
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / name
    if target.exists():
        assert target.stat().st_size == expected_size and digest(target) == expected_hash
        print(f"VERIFIED_EXISTING {target} {expected_size} {expected_hash}", flush=True)
        return
    partial = target.with_suffix(".npy.part")
    # A partial file is only a transport checkpoint; the final path is atomically
    # published after both the source size and hash have been checked.
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            raise RuntimeError("Server did not honor resume range")
        with partial.open("ab" if offset else "wb") as output:
            for block in response.iter_content(chunk_size=8 * 1024 * 1024):
                if block:
                    output.write(block)
                    print(f"{name}: {output.tell()}/{expected_size}", flush=True)
    assert partial.stat().st_size == expected_size, "Incomplete download"
    actual_hash = digest(partial)
    assert actual_hash == expected_hash, f"SHA-256 mismatch: {actual_hash} != {expected_hash}"
    os.replace(partial, target)
    print(f"VERIFIED_DOWNLOADED {target} {expected_size} {actual_hash}", flush=True)


if __name__ == "__main__":
    main()
