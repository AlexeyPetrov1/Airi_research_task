"""Recover Cereal free-joint poses directly from recorded PLEX qpos states."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HDF5 = ROOT / "data" / "plex" / "PickPlaceCereal_demo_act_norm.hdf5"
OUT = ROOT / "runs" / "plex_dobbe_preflight" / "plex_cereal_state_geometry.json"
JOINT_DIMS = {"free": (7, 6), "ball": (4, 3), "hinge": (1, 1), "slide": (1, 1)}


def joints_in_xml_order(xml: str) -> tuple[list[dict], int, int]:
    scene = ET.fromstring(xml)
    world = scene.find("worldbody")
    if world is None:
        raise ValueError("Missing worldbody")
    joints = []
    nq = nv = 0
    for element in world.iter():
        if element.tag not in ("joint", "freejoint"):
            continue
        kind = "free" if element.tag == "freejoint" else element.get("type", "hinge")
        qpos_count, qvel_count = JOINT_DIMS[kind]
        joints.append({"name": element.get("name"), "type": kind,
                       "qpos_offset": nq, "qpos_count": qpos_count,
                       "qvel_offset": nv, "qvel_count": qvel_count})
        nq += qpos_count
        nv += qvel_count
    return joints, nq, nv


def main() -> None:
    demos = []
    with h5py.File(HDF5, "r") as file:
        for key in sorted(file["data"], key=lambda s: int(s.split("_")[1])):
            episode = file["data"][key]
            states = episode["states"][:]
            joints, nq, nv = joints_in_xml_order(episode.attrs["model_file"])
            if states.shape[1] != 1 + nq + nv:
                raise ValueError(f"{key}: state length {states.shape[1]} != 1+{nq}+{nv}")
            cereal = next(j for j in joints if j["name"] == "Cereal_joint0")
            pose = states[:, 1 + cereal["qpos_offset"]:1 + cereal["qpos_offset"] + 7]
            xyz = pose[:, :3]
            intervals = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            demos.append({"demo": key, "frames": len(states),
                          "dt_min_max_s": [float(np.min(np.diff(states[:, 0]))),
                                           float(np.max(np.diff(states[:, 0])))],
                          "nq": nq, "nv": nv,
                          "cereal_qpos_offset": cereal["qpos_offset"],
                          "cereal_first_xyz": xyz[0].tolist(),
                          "cereal_last_xyz": xyz[-1].tolist(),
                          "cereal_displacement": float(np.linalg.norm(xyz[-1] - xyz[0])),
                          "cereal_path_length": float(np.sum(intervals)),
                          "cereal_xyz_range": np.ptp(xyz, axis=0).tolist(),
                          "cereal_quat_norm_min_max": [float(np.min(np.linalg.norm(pose[:, 3:], axis=1))),
                                                        float(np.max(np.linalg.norm(pose[:, 3:], axis=1)))]})
    result = {"source": str(HDF5),
              "state_layout": "time + all qpos in MJCF joint order + all qvel; dimension validated for all 75 demos",
              "Cereal_joint0": "free joint, 7 qpos (xyz + wxyz quaternion)",
              "demo_count": len(demos), "demos": demos}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    selected = sorted(demos, key=lambda d: d["cereal_displacement"], reverse=True)[:5]
    print(json.dumps({"demo_count": len(demos), "first": demos[0],
                      "largest_endpoint_displacement": selected}, indent=2))


if __name__ == "__main__":
    main()
