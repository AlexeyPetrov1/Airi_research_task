"""Record the source-backed FMB reference and visual candidate decision."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fmb_second_scene"
SOURCE_DIR = ROOT.parent / "data/fmb/single_object_manipulation_dataset"
REFERENCE = "1_M_L_3_vertical_n_2.npy"
SELECTED = "1_L_L_4_vertical_n_0.npy"
HF = "https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/tree/f99fd55c072eea5573523c96aa527aed3c665690/single_object_manipulation_dataset"
NAMING = "https://functional-manipulation-benchmark.github.io/dataset/index.html"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def red_peg_quality(path: Path) -> dict:
    """Sample the red object interior, kept separate from full-run selection."""
    data = np.load(path, allow_pickle=True).item()
    indices = np.unique(np.linspace(0, len(data["primitive"]) - 1, 16, dtype=int))
    out = {}
    for camera in ("side_1", "side_2"):
        areas, fractions = [], []
        for i in indices:
            hsv = cv2.cvtColor(data[f"obs/{camera}"][i], cv2.COLOR_BGR2HSV)
            low = cv2.inRange(hsv, (0, 80, 35), (14, 255, 255))
            high = cv2.inRange(hsv, (170, 80, 35), (179, 255, 255))
            mask = cv2.erode(low | high, np.ones((3, 3), np.uint8)) > 0
            area = int(mask.sum())
            areas.append(area)
            if area >= 30:
                fractions.append(float(np.mean(data[f"obs/{camera}_depth"][i][mask] > 0)))
        out[camera] = {"frames_with_30plus_mask_pixels": sum(a >= 30 for a in areas),
                       "sampled_frames": len(indices), "median_mask_pixels": float(np.median(areas)),
                       "median_depth_valid_fraction": float(np.median(fractions)) if fractions else None}
    return out


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    ref_path = SOURCE_DIR / REFERENCE
    ref_data = np.load(ref_path, allow_pickle=True).item()
    ref_info = ref_data["object_info"]
    reference = {
        "share_robot_episode": "57_fmb#episode_5201", "source_file": REFERENCE,
        "trajectory_id": 2, "trajectory_id_provenance": NAMING + "#file-naming; final filename token",
        "N": len(ref_data["primitive"]), "object_info": ref_info,
        "primitive_sequence": list(dict.fromkeys(map(str, ref_data["primitive"]))),
        "source_archive": "https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip",
        "single_file_mirror": HF + "/" + REFERENCE,
        "source_sha256": sha256(ref_path),
        "share_robot_mapping_evidence": "runs/sharerobot_fmb_episode_5201/preflight.json",
    }
    write_json(RUN / "reference_5201.json", reference)

    rows = json.loads((RUN / "candidate_survey.json").read_text(encoding="utf-8"))
    for row in rows:
        name = row["source_file"]
        row["source_file"] = "single_object_manipulation_dataset/" + name
        row["accepted"] = name == SELECTED
        if name == SELECTED:
            row["side_1_quality"] = "GOOD: yellow peg clear through grasp and insertion"
            row["side_2_quality"] = "GOOD_EARLY, PARTIAL_LATE: peg near top edge during insertion"
            row["depth_quality"] = "GOOD: median object-interior valid fraction 0.933 side_1, 0.981 side_2"
            row["rejection_reason"] = ""
        elif name == REFERENCE:
            row["side_1_quality"] = "KNOWN_GOOD_REFERENCE"
            row["side_2_quality"] = "NOT_ASSESSED_HERE"
            row["depth_quality"] = "KNOWN_NONZERO_REFERENCE"
            row["rejection_reason"] = "Same source demonstration and trajectory_id as episode_5201"
        elif name.startswith("2_M_L_7_vertical_n_0"):
            row["side_1_quality"] = "FAIR: dark-blue round peg merges with blue board late"
            row["side_2_quality"] = "FAIR_EARLY, POOR_LATE: peg small near board and gripper"
            row["depth_quality"] = "PRESENT, object-ROI validity not yet quantified"
            row["rejection_reason"] = "Weaker point visibility and contrast than selected yellow rectangular peg"
        elif row["config_difference_count"] == 0:
            row["rejection_reason"] = "Same object configuration as episode_5201; lower preference"
        else:
            row["rejection_reason"] = "Only one configuration field differs; lower preference"
        if name != SELECTED and not name.startswith("2_M_L_7_vertical_n_0"):
            quality = red_peg_quality(SOURCE_DIR / name)
            for camera in ("side_1", "side_2"):
                q = quality[camera]
                row[f"{camera}_quality"] = (f"red-mask visible {q['frames_with_30plus_mask_pixels']}/{q['sampled_frames']} sampled; "
                                             f"median interior area {q['median_mask_pixels']:.0f} px")
            row["depth_quality"] = ("sampled red-mask median valid fraction: "
                                    + ", ".join(f"{camera}={quality[camera]['median_depth_valid_fraction']:.3f}"
                                                if quality[camera]["median_depth_valid_fraction"] is not None else f"{camera}=UNKNOWN"
                                                for camera in ("side_1", "side_2")))
    fields = ["source_file", "trajectory_id", "N", "shape", "size", "length", "color", "angle",
              "distractor", "primitive_sequence", "side_1_quality", "side_2_quality", "depth_quality",
              "config_difference_count", "accepted", "rejection_reason"]
    with (RUN / "candidate_selection.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)
    selection = {
        "selected_source_file": SELECTED, "trajectory_id": 0,
        "selected_source_sha256": sha256(SOURCE_DIR / SELECTED),
        "source_path": str(SOURCE_DIR / SELECTED), "single_file_mirror": HF + "/" + SELECTED,
        "naming_rule_source": NAMING, "reference_source_file": REFERENCE, "reference_trajectory_id": 2,
        "independence": "Distinct original NPY path, SHA-256, and FMB filename trajectory_id; full source run, not a window",
        "config_differences": {"size": [ref_info["size"], "L"], "color": [ref_info["color"], 4]},
        "config_difference_count": 2,
        "comparable_manipulation": "Both vertical starts with grasp > move_up > go_to_board > insert",
        "shortlist": ["1_L_L_4_vertical_n_0.npy", "2_M_L_7_vertical_n_0.npy"],
        "selection_reason": "Yellow large rectangular peg is visibly separable from green board on side_1, has a broad face for multiple points, and its color-segmented interior retains raw depth; side_2 is also usable early. The dark-blue round alternative becomes less separable from its blue board and gripper late in insertion.",
        "tie_break_seed": 42, "tie_break_used": False,
        "selection_inputs": ["original raw RGB/depth", "object_info", "primitive", "TCP"],
        "model_forecast_used": False,
        "candidate_depth_evidence": "runs/fmb_second_scene/candidate_depth_roi.json",
        "visual_evidence": ["runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_1.png",
                            "runs/fmb_second_scene/candidate_previews/1_L_L_4_vertical_n_0_side_2.png"],
    }
    write_json(RUN / "selection.json", selection)
    print("SELECTED", SELECTED, "candidates", len(rows))


if __name__ == "__main__":
    main()
