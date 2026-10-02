"""Check whether a second run with exactly the same frozen inputs is identical."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


RUN = Path(os.environ.get(
    "FMB_QUANT_RUN",
    Path(__file__).resolve().parents[1] /
    "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    repeat = RUN / "variants/nominal_repeat"
    first_run = json.loads((RUN / "model_run.json").read_text(encoding="utf8"))
    second_run = json.loads((repeat / "model_run.json").read_text(encoding="utf8"))
    first = np.load(RUN / "prediction_15hz.npy")
    second = np.load(repeat / "prediction_15hz.npy")
    status = {
        "purpose": "same-input repeat before attributing differences between K variants to inputs",
        "same_model_revision": first_run["model_revision"] == second_run["model_revision"],
        "same_action": first_run["action_text"] == second_run["action_text"],
        "same_history_hash": first_run["geometry_sha256"] == second_run["geometry_sha256"],
        "same_raw_text_hash": sha256(RUN / "raw_model_output.txt") == sha256(repeat / "raw_model_output.txt"),
        "raw_text_sha256": sha256(RUN / "raw_model_output.txt"),
        "same_prediction_array_exactly": bool(np.array_equal(first, second)),
        "max_abs_prediction_difference_m": float(np.max(np.abs(first-second))),
    }
    status["status"] = "PASS" if all((status["same_model_revision"],
                                         status["same_action"],
                                         status["same_history_hash"],
                                         status["same_raw_text_hash"],
                                         status["same_prediction_array_exactly"])) else "FAIL"
    (RUN / "nominal_repeatability.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf8")
    print(json.dumps(status, indent=2))
    if status["status"] != "PASS":
        raise ValueError("Nominal prediction was not reproduced exactly")


if __name__ == "__main__":
    main()
