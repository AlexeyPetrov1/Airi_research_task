"""Inspect one saved FMB TFDS TFRecord example without loading TensorFlow.

The input is the TFRecord payload (the 12-byte record header is omitted). This
reads protobuf offsets directly so the large RGB-D example need not be copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from pathlib import Path


def varint(buf: mmap.mmap, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7
        if shift > 70:
            raise ValueError("invalid protobuf varint")


def fields(buf: mmap.mmap, start: int, end: int):
    pos = start
    while pos < end:
        tag, pos = varint(buf, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 2:
            size, pos = varint(buf, pos)
            value = (pos, pos + size)
            pos += size
        elif wire == 0:
            value, pos = varint(buf, pos)
        elif wire == 1:
            value = (pos, pos + 8)
            pos += 8
        elif wire == 5:
            value = (pos, pos + 4)
            pos += 4
        else:
            raise ValueError(f"unknown protobuf wire type {wire}")
        yield number, wire, value
    if pos != end:
        raise ValueError("protobuf length mismatch")


def find_feature_values(buf: mmap.mmap):
    # Example.features -> Features.feature map -> Feature.bytes_list -> values
    example = list(fields(buf, 0, len(buf)))
    if len(example) != 1 or example[0][:2] != (1, 2):
        raise ValueError("expected tf.train.Example")
    for n, w, entry in fields(buf, *example[0][2]):
        if (n, w) != (1, 2):
            continue
        parts = list(fields(buf, *entry))
        key = next(bytes(buf[s:e]).decode() for num, wire, (s, e) in parts if (num, wire) == (1, 2))
        feature = next((s, e) for num, wire, (s, e) in parts if (num, wire) == (2, 2))
        values = []
        for _, wire, list_range in fields(buf, *feature):
            if wire != 2:
                continue
            values.extend((s, e) for num, value_wire, (s, e) in fields(buf, *list_range) if (num, value_wire) == (1, 2))
        yield key, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with args.record.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        features = dict(find_feature_values(buf))
        summary = {key: {"count": len(values), "first_lengths": [e-s for s, e in values[:3]]} for key, values in features.items()}
        for key in ("episode_metadata/file_path", "episode_metadata/episode_language_instruction"):
            if key in features:
                summary[key]["value"] = [bytes(buf[s:e]).decode(errors="replace") for s, e in features[key]]
        image_keys = [key for key in features if key.startswith("steps/observation/image_") and "depth" not in key]
        for key in image_keys:
            values = features[key]
            indices = sorted({0, len(values)//4, len(values)//2, 3*len(values)//4, len(values)-1})
            summary[key]["sample_indices"] = indices
            for i in indices:
                s, e = values[i]
                content = bytes(buf[s:e])
                name = f"{key.rsplit('/', 1)[-1]}_{i:03d}.jpg"
                (args.out / name).write_bytes(content)
                summary[key].setdefault("samples", []).append({"step": i, "file": name, "sha256": hashlib.sha256(content).hexdigest()})
        (args.out / "rlds_record_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
        print(json.dumps({"metadata": {k: v for k, v in summary.items() if k.startswith("episode_metadata/")}, "images": {k: summary[k]["count"] for k in image_keys}}, indent=2))


if __name__ == "__main__":
    main()
