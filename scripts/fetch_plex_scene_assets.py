"""Fetch only asset files referenced by a recorded PLEX Robosuite scene."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import posixpath
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
HDF5 = ROOT / "data" / "plex" / "PickPlaceCereal_demo_act_norm.hdf5"
ASSETS = ROOT / "data" / "plex" / "robosuite_assets"
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "plex_asset_receipt.json"
COMMIT = "c2fca799bb8ebd42bad9c4c5fa4157fea4251f98"
OLD_PREFIX = "/mnt/d/code/robosuite/robosuite/models/assets/"
BASE = f"https://raw.githubusercontent.com/ARISE-Initiative/robosuite/{COMMIT}/robosuite/models/assets/"


def relative_asset(old: str) -> str:
    if not old.startswith(OLD_PREFIX):
        raise ValueError(f"Unexpected recorded asset path: {old}")
    relative = posixpath.normpath(old[len(OLD_PREFIX):])
    if relative.startswith("../") or relative.startswith("/"):
        raise ValueError(f"Unsafe recorded asset path: {old}")
    return relative


def fetch(relative: str) -> dict:
    dest = ASSETS / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        data = dest.read_bytes()
        status = "existing"
    else:
        url = BASE + relative
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            return {"relative_path": relative, "status": "http_error",
                    "http_status": exc.code, "url": url}
        dest.write_bytes(data)
        status = "downloaded"
    return {"relative_path": relative, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "status": status}


def main() -> None:
    with h5py.File(HDF5, "r") as file:
        xmls = [episode.attrs["model_file"] for episode in file["data"].values()]
    names = set()
    for xml in xmls:
        root = ET.fromstring(xml)
        for element in root.findall('.//*[@file]'):
            names.add(relative_asset(element.attrib["file"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        receipts = list(pool.map(fetch, sorted(names)))
    result = {"official_robosuite_commit": COMMIT,
              "recorded_xml_count": len(xmls),
              "asset_count": len(receipts),
              "asset_bytes": sum(x.get("bytes", 0) for x in receipts),
              "missing_assets": [x for x in receipts if x["status"] == "http_error"],
              "assets": receipts}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "assets"}, indent=2))


if __name__ == "__main__":
    main()
