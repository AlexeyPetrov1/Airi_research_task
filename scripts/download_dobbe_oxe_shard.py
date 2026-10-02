"""Download only the OXE/TFDS Dobb-e shard containing a candidate ordinal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dobbe_oxe"
REPO = "lerobot-raw/dobbe_raw"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CANDIDATE_ORDINAL = 3651  # Hypothesis only: ShareRobot episode number may not be TFDS index.


def main() -> None:
    session = requests.Session()
    info_response = session.get(f"{BASE}/dataset_info.json", timeout=30)
    info_response.raise_for_status()
    info = info_response.json()
    train = next(split for split in info["splits"] if split["name"] == "train")
    lengths = [int(n) for n in train["shardLengths"]]
    if CANDIDATE_ORDINAL >= sum(lengths):
        raise ValueError("Candidate ordinal outside published train split")
    start = 0
    for shard_index, length in enumerate(lengths):
        if start <= CANDIDATE_ORDINAL < start + length:
            break
        start += length
    name = f"dobbe-train.array_record-{shard_index:05d}-of-{len(lengths):05d}"
    target = OUT / name
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_info.json").write_bytes(info_response.content)
    head = session.head(f"{BASE}/{name}", allow_redirects=True, timeout=30)
    head.raise_for_status()
    expected = int(head.headers["Content-Length"])
    if not target.exists() or target.stat().st_size != expected:
        response = session.get(f"{BASE}/{name}", stream=True, timeout=(20, 120))
        response.raise_for_status()
        temporary = target.with_suffix(".partial")
        with temporary.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        if temporary.stat().st_size != expected:
            raise RuntimeError("Incomplete shard download")
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    receipt = {"repo": REPO, "format": info["fileFormat"],
               "candidate_ordinal_assumption": CANDIDATE_ORDINAL,
               "shard_index": shard_index, "shard_start_ordinal": start,
               "candidate_local_index": CANDIDATE_ORDINAL - start,
               "shard_records": length, "filename": name,
               "bytes": expected, "sha256": digest,
               "mapping_status": "UNVERIFIED_ORDINAL_HYPOTHESIS"}
    (OUT / "download_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
