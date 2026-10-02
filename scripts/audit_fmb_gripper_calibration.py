"""Audit whether published FMB observations identify gripper-based camera K."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs" / "sharerobot_fmb_episode_5201" / "fmb_reference_code" / "robot_infra" / "franka_server.py"
DATA = Path("F:/AIRI_task/data/fmb/single_object_manipulation_dataset")
OUT = ROOT / "runs" / "fmb_effective_k_256_calibration" / "gripper_calibration_audit.json"


def main() -> None:
    code = SOURCE.read_text(encoding="utf-8")
    assert "msg.O_T_EE" in code and "self.gripper_pos = np.sum(msg.position)" in code
    episodes = []
    for path in sorted(DATA.glob("1_M_L_3_*.npy")):
        obs = np.load(path, allow_pickle=True).item()
        pose = obs["obs/tcp_pose"]
        tail = pose[int(.8 * len(pose)):]
        rotations = Rotation.from_quat(tail[:, 3:])
        relative_angle = (rotations * rotations[0].inv()).magnitude()
        episodes.append({"name": path.name, "frames": len(pose),
                         "tail_20pct_frame_count": len(tail),
                         "tail_xyz_range_mm": (1000 * np.ptp(tail[:, :3], axis=0)).tolist(),
                         "tail_max_rotation_deg": float(np.degrees(np.max(relative_angle))),
                         "published_gripper_pose_values": np.unique(obs["obs/gripper_pose"]).tolist(),
                         "raw_keys_containing_calibration": [k for k in obs if any(w in k.lower() for w in ("intrinsic", "extrinsic", "calib", "f_t_ee", "finger", "width"))]})
    result = {"robot_pose_source": str(SOURCE),
              "robot_pose_source_line": "_set_currpos reshapes msg.O_T_EE and exports translation plus quaternion",
              "published_gripper_pose_is_binary": all(e["published_gripper_pose_values"] == [0, 1] for e in episodes),
              "recorded_finger_joint_width": False,
              "recorded_F_T_EE": False,
              "known_physical_3d_gripper_landmarks_relative_to_recorded_EE": False,
              "episodes": episodes,
              "interpretation": "There is robot pose diversity in several episodes, but no recording-time EE-to-physical-gripper transform, finger width, or verified 3D-to-2D gripper landmarks. Standard URDF coordinates cannot be treated as measured FMB landmarks. No gripper-derived K is accepted."}
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
