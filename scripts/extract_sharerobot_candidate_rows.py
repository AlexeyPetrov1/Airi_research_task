"""Stream one ShareRobot planning JSON and retain only two target episode rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "sharerobot_candidate_rows.json"
IJSON_TARGET = ROOT.parent / ".tools" / "ijson"
SOURCE = (
    "https://huggingface.co/datasets/BAAI/ShareRobot/resolve/main/"
    "planning/jsons/planning_task.json"
)
TARGETS = ("plex_robosuite#episode_46", "dobbe#episode_3651")


def main() -> None:
    sys.path.insert(0, str(IJSON_TARGET))
    import ijson

    rows = []
    response = requests.get(SOURCE, stream=True, timeout=(15, 60))
    response.raise_for_status()
    response.raw.decode_content = True
    count = 0
    try:
        for row in ijson.items(response.raw, "item"):
            count += 1
            if any(target in row.get("id", "") for target in TARGETS):
                rows.append({"row_index": count - 1, "row": row})
            if len({target for item in rows for target in TARGETS
                    if target in item["row"].get("id", "")}) == len(TARGETS):
                break
    finally:
        response.close()
    result = {"source": SOURCE, "source_file_bytes": 227881806,
              "rows_examined": count, "targets": TARGETS,
              "found": len(rows), "records": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"rows_examined": count,
                      "records": [{"row_index": item["row_index"],
                                   "id": item["row"].get("id"),
                                   "selected_step": item["row"].get("selected_step"),
                                   "image_count": len(item["row"].get("image", [])),
                                   "first_image": item["row"].get("image", [None])[0]}
                                  for item in rows]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
