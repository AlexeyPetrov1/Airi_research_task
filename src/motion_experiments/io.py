from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf8")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            result.update(part)
    return result.hexdigest()


def video_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"Cannot decode {path}")
    return np.stack(frames)


class Source:
    """Read prepared native artifacts and remember precisely which files were read."""
    def __init__(self, root):
        self.root = Path(root)
        self.files = []

    def path(self, name):
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        self.files.append(path)
        return path

    def array(self, name):
        return np.load(self.path(name), allow_pickle=False)

    def json(self, name):
        return read_json(self.path(name))

    def rgb(self, name):
        from PIL import Image
        return np.asarray(Image.open(self.path(name)).convert("RGB"))

    def tensor(self, name):
        import torch
        return torch.load(self.path(name), map_location="cpu", weights_only=True).numpy()
