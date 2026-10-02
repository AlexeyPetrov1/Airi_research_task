"""Match a LeRobot episode to a HoNY capture by exact gripper time series.

Run index_dobbe_pick_place_labels.py once before this offline comparison.
An ordinal or natural-language match is never treated as proof by itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "plex_dobbe_preflight"
DATA = ROOT / "data" / "dobbe_oxe"


def main() -> None:
    states = pq.read_table(DATA / "lerobot_episode_003651.parquet",
                           columns=["observation.state"])
    target = np.asarray([state[-1] for state in states.column(0).to_pylist()],
                        dtype=np.float32)
    records = [json.loads(line) for line in
               (DATA / "pick_place_label_index.jsonl").read_text(
                   encoding="utf-8").splitlines() if line]
    matches = []
    rankings = []
    for record in records:
        values = np.asarray(record["gripper"], dtype=np.float32)
        if len(values) < len(target):
            continue
        best = None
        for offset in range(len(values) - len(target) + 1):
            diff = np.abs(values[offset:offset + len(target)] - target)
            metric = (float(np.max(diff)), float(np.mean(diff)), offset)
            if best is None or metric < best:
                best = metric
        rankings.append({"member": record["member"],
                         "frames": len(values), "max_abs_error": best[0],
                         "mean_abs_error": best[1], "offset": best[2]})
        if best[0] == 0.0:
            matches.append(rankings[-1])
    rankings.sort(key=lambda item: (item["max_abs_error"],
                                    item["mean_abs_error"]))
    result = {"candidate": "IPEC-COMMUNITY/dobbe_lerobot episode_003651",
              "candidate_frames": len(target),
              "source_records_compared": len(records),
              "exact_matches": matches,
              "next_best_matches": rankings[:5],
              "criterion": "all gripper values are bitwise equal after float32 conversion"}
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "dobbe_3651_source_match.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
