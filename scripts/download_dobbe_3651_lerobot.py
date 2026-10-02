"""Download only the LeRobot RGB video and states for Dobb-E episode 3651."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "dobbe_oxe"
OUT = ROOT / "runs" / "plex_dobbe_preflight"
BASE = "https://huggingface.co/datasets/IPEC-COMMUNITY/dobbe_lerobot/resolve/main/"
FILES = {
    "lerobot_episode_003651.mp4": (
        "videos/chunk-003/observation.images.wrist_image/episode_003651.mp4",
        "495dbe8c0a5d6472fd515eb5d1325a66116721a476393a80a22b8f3ad940b51d",
    ),
    "lerobot_episode_003651.parquet": (
        "data/chunk-003/episode_003651.parquet",
        "54e1fad45d81b58512a2d66aa88ba401285b8c487d0de80845e53df56ad08bed",
    ),
}


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = []
    session = requests.Session()
    for name, (remote_path, expected_sha) in FILES.items():
        target = DEST / name
        if not target.exists() or checksum(target) != expected_sha:
            temporary = target.with_suffix(target.suffix + ".part")
            with session.get(BASE + remote_path, stream=True, timeout=60) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if checksum(temporary) != expected_sha:
                raise RuntimeError(f"Unexpected content hash for {remote_path}")
            temporary.replace(target)
        receipt.append({"source": BASE + remote_path, "path": str(target),
                        "bytes": target.stat().st_size, "sha256": checksum(target)})
    result = {"files": receipt}
    (OUT / "dobbe_3651_lerobot_receipt.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
