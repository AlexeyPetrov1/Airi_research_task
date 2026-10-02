"""Repeat the read-only ShareRobot transfer preflight from saved evidence frames.

Only metadata and the chosen ShareRobot trajectory rows are fetched. No model
weights or source video archives are downloaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "sharerobot_transfer_berkeley_ur5_episode_26"
SHAREROBOT_REVISION = "3266d92902b038ce40e7fb8aac5bfd9287eb3e45"
FMB_LEROBOT_REVISION = "76c86f111d328229773288bf60d5b5b50e047e8d"
TRAJECTORY_URL = (
    f"https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{SHAREROBOT_REVISION}/"
    "trajectory/trajectory.json"
)
FMB_ZIP_URL = "https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip"
FMB_LEROBOT_INFO = (
    f"https://huggingface.co/datasets/lerobot/fmb/resolve/{FMB_LEROBOT_REVISION}/meta/info.json"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_info(name: str, image_path: str, session: requests.Session) -> dict:
    path = OUT / name
    url = (
        f"https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{SHAREROBOT_REVISION}/"
        f"trajectory/images/{quote(image_path, safe='/')}"
    )
    response = session.get(url, timeout=30)
    response.raise_for_status()
    local_hash = sha256(path)
    remote_hash = hashlib.sha256(response.content).hexdigest()
    assert remote_hash == local_hash, (name, local_hash, remote_hash)
    with Image.open(path) as im:
        return {
            "file": name,
            "source_url": url,
            "source_bytes_match": True,
            "width": im.width,
            "height": im.height,
            "mode": im.mode,
            "bytes": path.stat().st_size,
            "sha256": local_hash,
        }


def main():
    session = requests.Session()
    response = session.get(TRAJECTORY_URL, timeout=60)
    response.raise_for_status()
    rows = response.json()
    chosen = {
        "primary": [x for x in rows if "/43_berkeley_autolab_ur5#episode_26/" in x["image_path"]],
        "reserve": [x for x in rows if "/57_fmb#episode_5201/" in x["image_path"]],
    }
    assert len(chosen["primary"]) == 1, chosen["primary"]
    assert {x["image_path"].rsplit("/", 1)[-1] for x in chosen["reserve"]} == {
        "frame_0.png", "frame_15.png"
    }

    archive_head = session.head(FMB_ZIP_URL, timeout=30)
    archive_head.raise_for_status()
    lerobot = session.get(FMB_LEROBOT_INFO, timeout=30)
    lerobot.raise_for_status()
    cfg = ROOT / "data" / "checkpoints" / "MolmoMotion-4B-H3-F30" / "config.yaml"
    result = {
        "status": "BLOCKED/PARTIAL",
        "repository_head": "8091aa98df93495dfa9e27dd998e1932c2806f57",
        "sharerobot_revision": SHAREROBOT_REVISION,
        "sharerobot_trajectory_url": TRAJECTORY_URL,
        "selected_rows": chosen,
        "images": [
            image_info("sharerobot_frame_15.png", chosen["primary"][0]["image_path"], session),
            image_info("reserve_fmb_episode_5201_frame_0.png", chosen["reserve"][0]["image_path"], session),
            image_info("reserve_fmb_episode_5201_frame_15.png", chosen["reserve"][1]["image_path"], session),
        ],
        "fmb_intrinsics_file": {
            "url": "https://functional-manipulation-benchmark.github.io/static/files/side_1",
            "file": "reserve_fmb_side1_intrinsics",
            "bytes": (OUT / "reserve_fmb_side1_intrinsics").stat().st_size,
            "sha256": sha256(OUT / "reserve_fmb_side1_intrinsics"),
        },
        "fmb_original_zip": {
            "url": FMB_ZIP_URL,
            "bytes": int(archive_head.headers["Content-Length"]),
            "accept_ranges": archive_head.headers.get("Accept-Ranges"),
            "etag": archive_head.headers.get("ETag"),
        },
        "fmb_lerobot": {
            "url": FMB_LEROBOT_INFO,
            "total_episodes": lerobot.json()["total_episodes"],
            "fps": lerobot.json()["fps"],
            "note": "Conversion metadata; no verified mapping to ShareRobot episode_5201.",
        },
        "checkpoint_config": {"file": str(cfg.relative_to(ROOT)), "sha256": sha256(cfg)},
        "generation_calls": 0,
        "validated_history_frames": 0,
        "validated_future_frames": 0,
        "metric_pairs": 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "primary_rows": len(chosen["primary"]),
        "reserve_rows": len(chosen["reserve"]),
        "archive_bytes": result["fmb_original_zip"]["bytes"],
        "generation_calls": 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
