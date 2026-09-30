"""Strictly decode 30 frames of eight unique point IDs, without gap filling."""

import re

import numpy as np


def strict_decode(text):
    blocks = re.findall(r'<tracks\s+coords="([^"]*)"', text)
    candidates = [b for b in blocks if len([x for x in b.split(";") if x.strip()]) == 30]
    if not candidates:
        raise ValueError("No complete 30-frame <tracks> block")
    frames = candidates[-1].split(";")
    delta = np.empty((8, 30, 3), dtype=np.float32)
    for step, line in enumerate(frames):
        fields = line.strip().split()
        if len(fields) != 33:
            raise ValueError(f"Frame {step} has {len(fields)} fields, expected 33")
        if float(fields[0]) != step + 3:
            raise ValueError(f"Frame {step} timestamp {fields[0]} differs from {step+3}")
        ids = []
        for p in range(8):
            base = 1 + p * 4
            ids.append(int(fields[base]))
            delta[p, step] = [int(v) / 1000 for v in fields[base + 1:base + 4]]
        if ids != list(range(1, 9)):
            raise ValueError(f"Frame {step} IDs {ids} differ from 1..8")
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite delta")
    return delta
