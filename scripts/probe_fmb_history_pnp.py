"""Planar CAD PnP on eight RGB-tracked interior peg points, without depth."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from probe_fmb_cad_pnp import K, SOURCE


OUT = Path(__file__).resolve().parents[1] / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_20261001"


def main() -> None:
    probe = json.loads((OUT / "history_silhouette_pose_probe.json").read_text(encoding="utf8"))
    cad = np.asarray([row["CAD_local_xyz_m"] for row in probe["ray_front_face_checks"]], dtype="float64")
    observed = json.loads((OUT / "tracking_probe.json").read_text(encoding="utf8"))["checks"]
    source = np.load(SOURCE, allow_pickle=True).item()
    tcp = source["obs/tcp_pose"]
    records = {}
    for step in (124, 125, 126):
        uv = np.asarray([row["uv"] for row in observed[str(step)]], dtype="float64")
        n, rvs, tvs, errors = cv2.solvePnPGeneric(cad, uv, K, None, flags=cv2.SOLVEPNP_IPPE)
        branches = []
        for rv, tv in zip(rvs, tvs):
            seed = np.r_[rv.ravel(), tv.ravel()]
            fit = least_squares(lambda p: (cv2.projectPoints(cad, p[:3], p[3:], K, None)[0]
                                           .reshape(-1, 2)-uv).ravel(), seed,
                                loss="soft_l1", f_scale=2., max_nfev=150)
            pose = fit.x
            xyz = (Rotation.from_rotvec(pose[:3]).as_matrix() @ cad.T).T + pose[3:]
            reproj = cv2.projectPoints(cad, pose[:3], pose[3:], K, None)[0].reshape(-1, 2)
            branches.append({"pose": pose.tolist(), "rmse_px": float(np.sqrt(np.mean((reproj-uv)**2))),
                             "median_Z_m": float(np.median(xyz[:, 2])),
                             "mean_Z_m": float(np.mean(xyz[:, 2])),
                             "all_positive_Z": bool((xyz[:, 2] > 0).all())})
        records[str(step)] = {"uv": uv.tolist(), "branches": branches}
        print(step, [(round(b["rmse_px"], 2), round(b["median_Z_m"], 3)) for b in branches])
    reference = np.asarray(records["126"]["branches"][0]["pose"])
    for step in (124, 125):
        for i, branch in enumerate(records[str(step)]["branches"]):
            pose = np.asarray(branch["pose"])
            orientation = (Rotation.from_rotvec(pose[:3]).inv() *
                           Rotation.from_rotvec(reference[:3])).magnitude()*180/np.pi
            object_motion = np.linalg.norm(reference[3:]-pose[3:])*1000
            tcp_motion = np.linalg.norm(tcp[126, :3]-tcp[step, :3])*1000
            print("branch", step, i, "angle_deg", round(orientation, 2),
                  "object_motion_mm", round(object_motion, 1),
                  "tcp_motion_mm", round(tcp_motion, 1))
    (OUT / "history_pnp_probe.json").write_text(json.dumps({"CAD_points_m": cad.tolist(),
        "K": K.tolist(), "frames": records}, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
