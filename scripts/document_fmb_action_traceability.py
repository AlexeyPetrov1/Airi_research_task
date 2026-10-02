"""Record exact source primitive and ShareRobot action wording for frozen run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from probe_fmb_cad_pnp import SOURCE


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/sharerobot_fmb_episode_5201"
RUN = BASE / "quantitative_2d_sensor_t126"


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    rows = json.loads((BASE / "planning_rows.json").read_text(encoding="utf8"))
    relevant = []
    for row in rows:
        prompt = row["conversations"][0]["value"].splitlines()[-1]
        if "insert the rectangular object into the square slot" in prompt.lower():
            relevant.append({"share_robot_row_id": row["id"], "task": row["task"],
                             "prompt_exact_last_line": prompt})
    config = json.loads((RUN / "experiment_config.json").read_text(encoding="utf8"))
    result = {"source_FMB_primitive_exact_at_t0": str(source["primitive"][126]),
              "source_FMB_primitive_exact_over_evaluation_window": sorted(set(
                  map(str, source["primitive"][124:147]))),
              "source_FMB_object_info": source["object_info"],
              "ShareRobot_exact_prompts": relevant,
              "model_action_exact": config["action_text_exact_passed_to_model"],
              "model_action_is_verbatim_FMB_metadata": False,
              "model_action_is_verbatim_ShareRobot_prompt": False,
              "relation": "The frozen English action paraphrases the FMB primitive insert and ShareRobot's rectangular-object/square-slot task; color, peg, and blue board are from object metadata and RGB scene.",
              "action_frozen_before_prediction": True}
    (RUN / "action_traceability.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
