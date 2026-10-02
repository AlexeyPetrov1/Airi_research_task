"""Freeze frame indices before geometry estimation and prediction."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_rgbd_study/protocol.json"


def main() -> None:
    protocol = {
        "version": 1,
        "main": {
            "source": "Pick_and_Place/Home15/Env1/2023-04-25--02-05-30",
            "calibration_past": list(range(96)),
            "history": [96, 97, 98],
            "future_evaluation_only": list(range(99, 129)),
        },
        "second": {
            "source": "Pick_and_Place/Home7/Env1/2023-04-27--10-47-40",
            "calibration_past": list(range(121)),
            "history": [121, 122, 123],
            "future_evaluation_only": list(range(124, 154)),
        },
        "rule": "Only calibration_past and history may enter camera/pose/mapping/point selection; future_evaluation_only is sealed until prediction is frozen.",
    }
    for scene in ("main", "second"):
        part = protocol[scene]
        assert max(part["calibration_past"]) < min(part["history"])
        assert max(part["history"]) < min(part["future_evaluation_only"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and json.loads(OUT.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError(f"Protocol already exists and differs: {OUT}")
    OUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
