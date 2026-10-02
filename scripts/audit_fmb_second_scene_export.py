"""Independently verify the complete FMB second-scene export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/fmb_second_scene"
RUN = ROOT / "runs/fmb_second_scene"
REF = ROOT.parent / "data/fmb/single_object_manipulation_dataset/1_M_L_3_vertical_n_2.npy"
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")
STATE_MAP = {"obs/tcp_pose": "tcp_pose", "obs/tcp_vel": "tcp_vel", "obs/tcp_force": "tcp_force",
             "obs/tcp_torque": "tcp_torque", "obs/q": "q", "obs/dq": "dq", "obs/jacobian": "jacobian",
             "obs/gripper_pose": "gripper_pose", "actions": "action", "primitive": "primitive"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    source_path = DATA / "raw/source_demo.npy"
    source = np.load(source_path, allow_pickle=True).item()
    metadata = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    n = len(source["primitive"])
    checks = {}
    checks["source_checksum"] = (sha256(source_path) == metadata["source_sha256"] == metadata["raw_copy_sha256"])
    checks["independent_source_and_trajectory"] = (
        source_path.name != REF.name and sha256(source_path) != sha256(REF)
        and metadata["source_filename"] != REF.name and metadata["trajectory_id"] != 2)
    checks["N_at_least_33"] = n >= 33
    checks["all_source_arrays_have_N"] = all(len(array) == n for array in source.values() if isinstance(array, np.ndarray))
    checks["metadata_N"] = metadata["N"] == n
    rgb_checks = {}
    depth_checks = {}
    videos = {}
    for camera in CAMERAS:
        raw_rgb = source[f"obs/{camera}"]
        paths = sorted((DATA / "rgb" / camera).glob("frame_*.png"))
        exact = len(paths) == n
        distinct = len({hashlib.sha256(frame.tobytes()).digest() for frame in raw_rgb}) > 1
        for i, path in enumerate(paths):
            actual = np.asarray(Image.open(path).convert("RGB"))
            if actual.shape != raw_rgb[i].shape or not np.array_equal(actual, raw_rgb[i][..., ::-1]):
                exact = False
                break
        rgb_checks[camera] = {"count": len(paths), "shape": list(raw_rgb.shape), "dtype": str(raw_rgb.dtype),
                              "range": [int(raw_rgb.min()), int(raw_rgb.max())],
                              "exact_BGR_to_RGB_every_frame": exact, "changes_over_time": distinct}
        path = DATA / "depth_raw" / f"{camera}.npy"
        exported = np.load(path, allow_pickle=False)
        raw_depth = source[f"obs/{camera}_depth"]
        depth_checks[camera] = {"shape": list(exported.shape), "dtype": str(exported.dtype),
                                "exact_all_values": bool(np.array_equal(exported, raw_depth)),
                                "changes_over_time": bool(np.any(exported[0] != exported[-1])),
                                "nonconstant": bool(np.min(exported) != np.max(exported))}
        video = cv2.VideoCapture(str(RUN / "previews" / f"preview_{camera}.mp4"))
        count = 0
        while True:
            ok, _ = video.read()
            if not ok:
                break
            count += 1
        video.release()
        videos[camera] = {"decoded_frames": count, "matches_N": count == n}
    checks["all_RGB_exact_and_nonconstant"] = all(v["exact_BGR_to_RGB_every_frame"] and v["changes_over_time"]
                                                   for v in rgb_checks.values())
    checks["all_depth_exact_and_nonconstant"] = all(v["exact_all_values"] and v["changes_over_time"] and v["nonconstant"]
                                                 for v in depth_checks.values())
    checks["all_previews_N"] = all(v["matches_N"] for v in videos.values())
    state = np.load(DATA / "robot_state.npz", allow_pickle=False)
    state_checks = {key: bool(np.array_equal(state[target], source[key])) for key, target in STATE_MAP.items()}
    state_checks["object_info_json"] = json.loads(str(state["object_info_json"])) == source["object_info"]
    checks["all_state_arrays_exact"] = all(state_checks.values())
    checks["finite_tcp_pose"] = bool(np.isfinite(source["obs/tcp_pose"]).all())
    checks["finite_action"] = bool(np.isfinite(source["actions"]).all())
    primitives = list(map(str, source["primitive"]))
    ranges = metadata["primitive_ranges"]
    checks["primitive_sequence_complete"] = (
        ranges[0]["first_frame"] == 0 and ranges[-1]["last_frame"] == n - 1
        and all(ranges[i]["last_frame"] + 1 == ranges[i + 1]["first_frame"] for i in range(len(ranges) - 1))
        and all(all(primitives[i] == r["primitive"] for i in range(r["first_frame"], r["last_frame"] + 1)) for r in ranges))
    checks["all_camera_shapes_expected"] = all(source[f"obs/{camera}"].shape == (n, 256, 256, 3)
                                               and source[f"obs/{camera}_depth"].shape == (n, 256, 256) for camera in CAMERAS)
    checks["source_color_and_depth_dtypes"] = all(source[f"obs/{camera}"].dtype == np.uint8
                                                  and source[f"obs/{camera}_depth"].dtype == np.uint16 for camera in CAMERAS)
    result = {"status": "PASS" if all(checks.values()) else "FAIL", "source_file": metadata["source_filename"],
              "source_sha256": sha256(source_path), "N": n, "checks": checks,
              "RGB": rgb_checks, "depth": depth_checks, "previews": videos, "state_keys": state_checks,
              "depth_scale_verified": False, "color_interpretation": "Official FMB says BGR; exact per-frame PNG equality confirms one BGR-to-RGB conversion",
              "limitations": ["Exact RGB/depth pixel registration is separate from export fidelity and remains unresolved",
                              "No metric unit assigned to raw depth"]}
    (RUN / "data_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result["status"], n, sum(checks.values()), "/", len(checks))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
