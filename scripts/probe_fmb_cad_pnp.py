"""Explicitly labeled planar CAD PnP probe for the FMB medium long rectangle.

The four image corners are manually read from enlarged source RGB frames.
They represent the visible broad face, with a 40.32 mm width. Results are
candidate geometry because the active 256x256 RGB intrinsics are unverified.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/sharerobot_fmb_episode_5201"
SOURCE = OUT / "1_M_L_3_vertical_n_2.npy"

# Order: top left, top right, bottom right, bottom left of the broad face.
ANNOTATIONS = {
    130: [[190, 19], [229, 21], [197, 149], [177, 149]],
    135: [[180, 44], [214, 53], [190, 162], [167, 162]],
    140: [[177, 63], [211, 67], [186, 168], [166, 168]],
}
W, D, H = .04032, .02592, .150
FRONT = np.asarray([[-W/2, -D/2, H], [W/2, -D/2, H],
                    [W/2, -D/2, 0], [-W/2, -D/2, 0]], dtype=np.float64)
# Eight stable CAD landmarks, four front and four rear cuboid corners.
EIGHT = np.asarray([[-W/2, -D/2, H], [W/2, -D/2, H],
                    [W/2, -D/2, 0], [-W/2, -D/2, 0],
                    [-W/2, D/2, H], [W/2, D/2, H],
                    [W/2, D/2, 0], [-W/2, D/2, 0]], dtype=np.float64)
K = np.asarray([[152.0836, 0, 124.2908],
                [0, 202.7781333333, 129.2973333333],
                [0, 0, 1]], dtype=np.float64)


def solve_one(depth: np.ndarray, step: int, K_used: np.ndarray) -> dict:
    image = np.asarray(ANNOTATIONS[step], dtype=np.float64)
    n, rotations, translations, _ = cv2.solvePnPGeneric(
        FRONT, image, K_used, np.zeros(5), flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    for i, (rv, tv) in enumerate(zip(rotations, translations)):
        R, _ = cv2.Rodrigues(rv)
        front_cam = (R @ FRONT.T).T + tv.ravel()
        all_cam = (R @ EIGHT.T).T + tv.ravel()
        reproj, _ = cv2.projectPoints(FRONT, rv, tv, K_used, None)
        reproj = reproj.reshape(-1, 2)
        err = np.linalg.norm(reproj - image, axis=1)
        # Interior surface points are held out from fitting.
        checks = []
        for alpha in (.2, .5, .8):
            upper = image[0] * (1-alpha) + image[1] * alpha
            lower = image[3] * (1-alpha) + image[2] * alpha
            for beta in (.25, .5, .75):
                uv = upper * (1-beta) + lower * beta
                u, v = np.rint(uv).astype(int)
                crop = depth[v-2:v+3, u-2:u+3]
                valid = crop[crop > 0]
                cad_point = np.asarray([(alpha-.5)*W, -D/2, (1-beta)*H])
                z = float(((R @ cad_point) + tv.ravel())[2])
                checks.append({"uv": [int(u), int(v)], "cad_Z_m": z,
                               "sensor_Z_m": float(np.median(valid)*.0001) if len(valid) else None})
        valid_checks = [x for x in checks if x["sensor_Z_m"] is not None]
        depth_error = [x["cad_Z_m"] - x["sensor_Z_m"] for x in valid_checks]
        candidates.append({"solution": i, "reprojection_rmse_px": float(np.sqrt(np.mean(err**2))),
                           "reprojection_errors_px": err.tolist(),
                           "all_positive_Z": bool(np.all(all_cam[:, 2] > 0)),
                           "R": R.tolist(), "t_m": tv.ravel().tolist(),
                           "landmarks_camera_m": all_cam.tolist(),
                           "held_out_depth_checks": checks,
                           "held_out_depth_median_abs_error_m":
                           float(np.median(np.abs(depth_error))) if depth_error else None,
                           "held_out_depth_median_signed_error_m":
                           float(np.median(depth_error)) if depth_error else None})
    # Select using RGB reprojection and positive Z only. Sensor depth remains
    # a held-out diagnostic, so the geometric cross-check is independent.
    candidates.sort(key=lambda c: (not c["all_positive_Z"],
                                   c["reprojection_rmse_px"]))
    return {"step": step, "manual_rgb_corners_uv": ANNOTATIONS[step],
            "num_solutions": n, "candidates": candidates}


def main() -> None:
    source = np.load(SOURCE, allow_pickle=True).item()
    rows = [solve_one(source["obs/side_1_depth"][step], step, K)
            for step in ANNOTATIONS]
    selected = np.asarray([r["candidates"][0]["landmarks_camera_m"] for r in rows])
    tcp = source["obs/tcp_pose"]
    centers = selected.mean(axis=1)
    motion = []
    steps = list(ANNOTATIONS)
    for i in range(len(steps)):
        for j in range(i+1, len(steps)):
            camera_distance = float(np.linalg.norm(centers[j]-centers[i]))
            tcp_distance = float(np.linalg.norm(tcp[steps[j]][:3]-tcp[steps[i]][:3]))
            motion.append({"steps": [steps[i], steps[j]],
                           "cad_pnp_center_displacement_m": camera_distance,
                           "tcp_displacement_m": tcp_distance,
                           "difference_m": camera_distance-tcp_distance})
    robot_rotation = []
    for i in range(len(steps)):
        for j in range(i+1, len(steps)):
            q0 = Rotation.from_quat(tcp[steps[i]][3:])
            q1 = Rotation.from_quat(tcp[steps[j]][3:])
            robot_rotation.append({"steps": [steps[i], steps[j]],
                                   "relative_angle_deg": float((q0.inv()*q1).magnitude()*180/np.pi)})
    rotations = Rotation.from_matrix([r["candidates"][0]["R"] for r in rows])
    initial = np.concatenate([rotations.mean().as_rotvec(),
                              *[r["candidates"][0]["t_m"] for r in rows]])

    def residual(x: np.ndarray) -> np.ndarray:
        rv = x[:3]
        return np.concatenate([
            (cv2.projectPoints(FRONT, rv, x[3+3*i:6+3*i], K, None)[0].reshape(-1,2)
             - np.asarray(ANNOTATIONS[step])).ravel()
            for i, step in enumerate(steps)])

    joint = least_squares(residual, initial, method="lm")
    joint_R = Rotation.from_rotvec(joint.x[:3]).as_matrix()
    joint_landmarks = np.asarray([(joint_R @ EIGHT.T).T + joint.x[3+3*i:6+3*i]
                                  for i in range(len(steps))])
    joint_frames = []
    for i, step in enumerate(steps):
        rv, tv = joint.x[:3], joint.x[3+3*i:6+3*i]
        uv = cv2.projectPoints(FRONT, rv, tv, K, None)[0].reshape(-1,2)
        reproj = np.linalg.norm(uv-np.asarray(ANNOTATIONS[step]),axis=1)
        d = source["obs/side_1_depth"][step]
        depth_residuals = []
        for a in (.2,.5,.8):
            for b in (.25,.5,.75):
                point = np.asarray([(a-.5)*W,-D/2,(1-b)*H])
                xyz = joint_R @ point + tv
                u,v = np.rint(cv2.projectPoints(point[None],rv,tv,K,None)[0][0,0]).astype(int)
                patch = d[v-2:v+3,u-2:u+3]
                valid = patch[patch>0]
                if len(valid):
                    depth_residuals.append(float(xyz[2]-np.median(valid)*.0001))
        joint_frames.append({"step":step,"t_m":tv.tolist(),
                             "reprojection_rmse_px":float(np.sqrt(np.mean(reproj**2))),
                             "held_out_depth_median_abs_error_m":float(np.median(np.abs(depth_residuals))),
                             "center_camera_m":joint_landmarks[i].mean(axis=0).tolist()})
    joint_motion = []
    joint_centers = joint_landmarks.mean(axis=1)
    for item in motion:
        i,j = (steps.index(s) for s in item["steps"])
        distance = float(np.linalg.norm(joint_centers[j]-joint_centers[i]))
        joint_motion.append({"steps":item["steps"],"joint_pnp_distance_m":distance,
                             "tcp_distance_m":item["tcp_displacement_m"],
                             "difference_m":distance-item["tcp_displacement_m"]})
    sensitivity = {}
    for factor in (.9, 1.0, 1.1):
        k = K.copy()
        k[0, 0] *= factor
        k[1, 1] *= factor
        sensitivity[str(factor)] = [
            np.mean(solve_one(source["obs/side_1_depth"][step], step, k)
                    ["candidates"][0]["landmarks_camera_m"], axis=0).tolist()
            for step in steps]
    result = {"status": "CANDIDATE_ONLY_UNVERIFIED_RGB_K_AND_MANUAL_CORNERS",
              "cad_source": "official peg.step solid index 48 of 54",
              "cad_size_m": [W, D, H],
              "K_rgb_candidate": K.tolist(),
              "rgb_corners_source": "manual read from peg_crop_{step}.png; broad-face corners are partly occluded",
              "candidate_history_shape": list(selected.shape),
              "cad_pnp_vs_tcp_motion": motion,
              "tcp_rotation_differences_deg": robot_rotation,
              "joint_constant_orientation_fit": {
                  "assumption": "Gripper holds peg rigidly; observed TCP rotation <= 1.12 deg",
                  "shared_R": joint_R.tolist(), "frames": joint_frames,
                  "joint_vs_tcp_motion": joint_motion,
                  "global_reprojection_rmse_px":float(np.sqrt(np.mean(residual(joint.x)**2))),
                  "all_positive_Z":bool(np.all(joint_landmarks[:,:,2]>0))},
              "camera_centers_m_at_focal_scale_0p9_1p0_1p1": sensitivity,
              "frames": rows}
    (OUT / "cad_pnp_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.save(OUT / "cad_pnp_history_candidate.npy", selected.astype("float32"))
    np.save(OUT / "cad_pnp_history_joint_candidate.npy",joint_landmarks.astype("float32"))
    for row in rows:
        step = row["step"]
        frame = source["obs/side_1"][step].copy()
        uv = row["manual_rgb_corners_uv"]
        projected = np.asarray(row["candidates"][0]["landmarks_camera_m"])
        projected_uv = np.stack([K[0, 0] * projected[:, 0] / projected[:, 2] + K[0, 2],
                                 K[1, 1] * projected[:, 1] / projected[:, 2] + K[1, 2]], axis=1)
        for i, (u, v) in enumerate(projected_uv):
            cv2.circle(frame, (round(u), round(v)), 3,
                       (0, 255, 0) if i < 4 else (255, 0, 0), -1)
            cv2.putText(frame, str(i), (round(u)+4, round(v)+4),
                        cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
        for u, v in uv:
            cv2.drawMarker(frame, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 6)
        cv2.imwrite(str(OUT / f"cad_pnp_overlay_{step}.png"),
                    cv2.resize(frame, (768, 768), interpolation=cv2.INTER_NEAREST))
        print(row["step"], [(c["solution"], round(c["reprojection_rmse_px"],2),
                            round(c["held_out_depth_median_abs_error_m"],3))
                           for c in row["candidates"]])


if __name__ == "__main__":
    main()
