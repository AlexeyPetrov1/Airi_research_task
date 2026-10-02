"""Collect the official camera and object assets without altering their bytes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fmb_second_scene"
BASE = "https://functional-manipulation-benchmark.github.io/static/"
ASSETS = {
    "calibration/raw/side_1": "files/side_1",
    "calibration/raw/side_2": "files/side_2",
    "calibration/raw/wrist_1": "files/wrist_1",
    "calibration/raw/wrist_2": "files/wrist_2",
    "object/peg_official.step": "files/peg.step",
    "object/shape_color_reference.pdf": "doc/FMB%20Shape%20and%20Color%20Number%20Reference%20Sheet%20-%20Google%20Docs.pdf",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = []
    for rel, uri in ASSETS.items():
        target = DATA / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(BASE + uri, timeout=60)
        response.raise_for_status()
        if not target.exists():
            target.write_bytes(response.content)
        elif target.read_bytes() != response.content:
            raise RuntimeError(f"Existing raw asset differs: {target}")
        manifest.append({"local_path": str(target.relative_to(ROOT)), "source_url": BASE + uri,
                         "bytes": target.stat().st_size, "sha256": sha256(target)})
        print(rel, target.stat().st_size, sha256(target), flush=True)
    for rel in ("envs/franka_fmb_env.py", "camera/rs_capture.py"):
        target = DATA / "calibration/raw" / Path(rel).name
        url = "https://raw.githubusercontent.com/rail-berkeley/fmb/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/" + rel
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        # Replace a local text copy if checkout newline conversion changed its
        # bytes; the final file must match the pinned upstream Git blob exactly.
        if not target.exists() or target.read_bytes() != response.content:
            target.write_bytes(response.content)
        manifest.append({"local_path": str(target.relative_to(ROOT)),
                         "source_url": "https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/robot_infra/" + rel,
                         "bytes": target.stat().st_size, "sha256": sha256(target)})
    (DATA / "asset_provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
