"""Write method readiness, candidate t0 points, and ShareRobot search status."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fmb_second_scene"
RUN = ROOT / "runs/fmb_second_scene"


def save(name: str, value: object) -> None:
    (RUN / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def center_and_area(bgr: np.ndarray) -> tuple[list[float] | None, int]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (5, 100, 80), (38, 255, 255))
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return None, len(xs)
    return [float(xs.mean()), float(ys.mean())], len(xs)


def main() -> None:
    source = np.load(DATA / "raw/source_demo.npy", allow_pickle=True).item()
    n = len(source["primitive"])
    depth_roi = json.loads((RUN / "candidate_depth_roi.json").read_text(encoding="utf-8"))
    calibration = json.loads((DATA / "calibration/calibration.json").read_text(encoding="utf-8"))
    object_meta = json.loads((DATA / "object/object_metadata.json").read_text(encoding="utf-8"))
    readiness = {
        "sensor_rgbd": {"status": "PARTIAL", "available": ["four synchronized RGB streams", "four exact raw depth streams",
                            "published 640x480 rectified camera profiles", "object-interior nonzero raw depth"],
                         "missing": ["independently confirmed depth scale for this run", "active 256x256 RGB/depth intrinsics",
                                     "verified RGB-to-depth pixel correspondence after unpublished depth processing"],
                         "evidence": ["data/fmb_second_scene/calibration/calibration.json", "runs/fmb_second_scene/candidate_depth_roi.json"]},
        "monocular_depth": {"status": "READY", "available": ["all 142 full-resolution published RGB PNGs per camera",
                              "byte-identical original BGR NPY preserved"], "missing": [],
                             "models_prepared_only": ["MoGe-2", "MoGe-3", "UniDepthV2", "Metric3Dv2", "Depth Anything V2 Metric"],
                             "models_run": False},
        "cad_pnp": {"status": "PARTIAL", "available": ["official 54-solid peg STEP collection", "shape/size/length metadata",
                     "142 RGB frames per camera"],
                    "missing": ["proven per-body CAD ID for this run", "validated visible 2D-to-CAD feature correspondences",
                                "active camera intrinsics for metric PnP"],
                    "evidence": ["data/fmb_second_scene/object/object_metadata.json"]},
        "tcp_consistency": {"status": "READY", "available": ["exact per-frame tcp_pose", "exact per-frame gripper_pose",
                             "exact per-frame primitive", "common source frame index"], "missing": [],
                            "scope": "robot-state temporal and magnitude consistency; camera-base pose remains unknown"},
        "stereo": {"status": "PARTIAL", "available": ["synchronized side_1 RGB", "synchronized side_2 RGB"],
                   "missing": ["active K1 at 256x256", "active K2 at 256x256", "T_side1_to_side2"]},
        "future_2d": {"status": "READY", "available": ["all 142 chronological side_1 RGB frames",
                              "high contrast visible rigid peg for several pre-insertion and insertion frames"],
                      "missing": [], "scope": "future point labels can be created later, independently of model output"},
    }
    save("method_readiness.json", readiness)

    candidates = []
    for t0 in (94, 98, 102, 106, 110):
        center0, area0 = center_and_area(source["obs/side_1"][t0])
        center30, area30 = center_and_area(source["obs/side_1"][t0 + 30])
        centroid_motion = float(np.linalg.norm(np.subtract(center30, center0))) if center0 and center30 else None
        tcp_motion = float(np.linalg.norm(source["obs/tcp_pose"][t0 + 30, :3] - source["obs/tcp_pose"][t0, :3]))
        depth_fraction = depth_roi["cameras"]["side_1"]["per_frame"][t0]["valid_depth_fraction_in_mask"]
        candidates.append({
            "frame_index": t0, "primitive": str(source["primitive"][t0]),
            "history_frames_available": t0 + 1, "future_frames_available": n - t0 - 1,
            "H3_history_indices": [t0 - 2, t0 - 1, t0],
            "F30_possible": n - t0 - 1 >= 30,
            "object_visibility": {"side_1_yellow_interior_pixels_at_t0": area0,
                                  "side_1_yellow_interior_pixels_at_t0_plus_30": area30,
                                  "manual_review": "visible broad yellow face at t0; top edge/gripper coverage grows late"},
            "motion_degree": {"yellow_centroid_displacement_px_to_t0_plus_30": centroid_motion,
                              "tcp_xyz_displacement_m_to_t0_plus_30": tcp_motion},
            "occlusion": "partial gripper coverage during insertion; inspect frame pair before final point selection",
            "depth_quality": {"side_1_object_mask_valid_fraction_at_t0": depth_fraction,
                              "method": "yellow HSV mask; no depth scale assigned"},
            "reason_interesting": "Visible rigid peg during the move toward the board and insertion, with at least 30 actual later frames",
        })
    save("t0_candidates.json", {"status": "CANDIDATES_ONLY", "N": n, "history_target": 3,
                                "future_target": 30, "selection_after_this_goal": True, "candidates": candidates,
                                "future_leakage_rule": "Later model input and metric scale calibration may use frames <= frozen t0 only"})

    mapping = {
        "status": "BLOCKED", "selected_FMB_source": "1_L_L_4_vertical_n_0.npy",
        "selected_FMB_trajectory_id": 0,
        "ShareRobot_episode": None, "camera": None, "verified_frame_pairs": [],
        "attempted_sources": [
            {"source": "https://huggingface.co/datasets/BAAI/ShareRobot", "finding": "ShareRobot names FMB episodes by converted dataset ordinal, not raw NPY filename"},
            {"source": "https://github.com/rail-berkeley/fmb/blob/d4da6ce044a9806f41e58bf7423b5a3c05289925/fmb_dataset_builder/fmb_dataset/fmb_dataset_dataset_builder.py",
             "finding": "TFDS source glob includes raw NPY paths but published builder has no filename-to-ordinal manifest"},
            {"source": "runs/sharerobot_fmb_episode_5201/preflight.json", "finding": "Proven mapping for 5201 applies only to reference source; cannot transfer ID to new source"},
            {"source": "https://huggingface.co/datasets/BAAI/ShareRobot/tree/main/planning/images",
             "finding": "Planning RGB archive is a many-part solid gzip; selected run has no individually established image path"},
        ],
        "blocking_gap": "No source-backed ShareRobot episode ID or independently retrievable full 30-frame ShareRobot sequence for this raw NPY",
        "not_inferred": ["ShareRobot ID from raw filename trajectory_id", "30-frame linspace sampling", "image match from one or two frames"],
        "would_verify_if_found": "Match each of all 30 ShareRobot RGBs against all 142 source steps, report monotonic indices and per-pair error",
    }
    trajectory_search = RUN / "sharerobot_trajectory_search.json"
    if trajectory_search.exists():
        search = json.loads(trajectory_search.read_text(encoding="utf-8"))
        best = search["results"][0]
        mapping["attempted_sources"].append({
            "source": search["manifest_url"],
            "finding": f"Checked all {search['successful_images']}/{search['unique_images']} individually retrievable FMB Trajectory RGBs against all 142 source frames, four cameras and both channel interpretations at 32x32. The coarse-best pair has full-resolution MAE={best['full_resolution_MAE']:.3f}; no confirmed pixel match.",
            "evidence": "runs/fmb_second_scene/sharerobot_trajectory_search.json",
            "best_pair_visual": "runs/fmb_second_scene/visuals/sharerobot_search_best_pair.png",
        })
        mapping["trajectory_subset_search"] = {
            "images_checked": search["successful_images"], "best_image_path": best["image_path"],
            "best_FMB_step": best["best_FMB_step"], "best_camera": best["best_camera"],
            "best_full_resolution_MAE": best["full_resolution_MAE"],
            "conclusion": "NO_CONFIRMED_MATCH_IN_TRAJECTORY_SUBSET; planning archive remains unsearched by pixels",
        }
    save("sharerobot_mapping.json", mapping)
    for subdir in ("observed", "evaluation"):
        (RUN / subdir).mkdir(exist_ok=True)
    (RUN / "observed/README.md").write_text(
        "# Future input partition\n\nFreeze t0 in a later experiment. Copy or link only frames <= t0 and derive any input geometry or scale solely from those frames. No t0 has been selected here.\n",
        encoding="utf-8")
    (RUN / "evaluation/README.md").write_text(
        "# Future evaluation partition\n\nAfter freezing t0, put frames > t0 and independent future point labels here. These data cannot tune input geometry or scale. No t0 has been selected here.\n",
        encoding="utf-8")
    print("READINESS", {key: value["status"] for key, value in readiness.items()}, "t0", len(candidates))


if __name__ == "__main__":
    main()
