"""Project exploratory future 3D predictions onto real future cup RGB frames."""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from inspect_hony_scenes import ROOT
from validate_dobbe_geometry import CAMERA_TO_LABEL


RUN = ROOT / "runs/dobbe_rgbd_study/approx_history"
RAW = ROOT / "data/dobbe_oxe/target_raw"


def project_world_from_h2(xyz_h2: np.ndarray, ref: np.ndarray,
                          target: np.ndarray, k: np.ndarray) -> np.ndarray:
    label_ref = xyz_h2 @ CAMERA_TO_LABEL.T
    world = label_ref @ ref[:3, :3].T + ref[:3, 3]
    label_target = (world - target[:3, 3]) @ target[:3, :3]
    optical_target = label_target @ CAMERA_TO_LABEL
    uv, _ = cv2.projectPoints(optical_target, np.zeros(3), np.zeros(3),
                              k, np.zeros(4))
    uv = uv.reshape(-1, 2)
    uv[optical_target[:, 2] <= .02] = np.nan
    return uv


def main() -> None:
    evaluation = json.loads((RUN / "future_evaluation.json").read_text(
        encoding="utf-8"))
    model_run = json.loads((RUN / "model_run.json").read_text(encoding="utf-8"))
    assert model_run["status"] == "COMPLETE" and model_run["success"]
    assert evaluation["prediction_sha256"] == model_run["prediction_sha256"]
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    k = np.asarray(manifest["K_256x256"], dtype=float)
    pred = np.load(RUN / "prediction_15hz.npy")
    history = np.load(RUN / "points_3d_history_candidate.npy")
    observed_uv = np.load(RUN / "future_rgb_uv.npy")
    valid = np.load(RUN / "future_gt_valid.npy")
    labels = json.loads((RAW / "labels.json").read_text(encoding="utf-8"))
    poses = np.repeat(np.eye(4)[None], 129, axis=0)
    poses[:, :3, :3] = Rotation.from_quat(
        [labels[str(i)]["quats"] for i in range(129)]).as_matrix()
    poses[:, :3, 3] = [labels[str(i)]["xyz"] for i in range(129)]
    reference = poses[98]
    anchor = history[-1]
    velocity = history[-1] - history[-2]
    video = cv2.VideoCapture(str(RAW / "compressed_video_h264.mp4"))
    methods = {"MolmoMotion": pred,
               "Stationary": np.broadcast_to(anchor[:, None, :], (8, 30, 3)),
               "Constant_velocity": anchor[:, None, :] +
                   np.arange(1, 31)[None, :, None] * velocity[:, None, :]}
    details = {name: [] for name in methods}
    panels = []
    for t, frame_id in enumerate(range(99, 129)):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = video.read()
        if not ok:
            raise RuntimeError(f"Could not decode future frame {frame_id}")
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 65, 25), (14, 255, 255))
        mask |= cv2.inRange(hsv, (167, 65, 25), (179, 255, 255))
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        model_uv = None
        for name, trajectory in methods.items():
            uv = project_world_from_h2(trajectory[:, t], reference,
                                       poses[frame_id], k)
            onscreen = np.isfinite(uv).all(axis=1)
            onscreen &= (uv[:, 0] >= 0) & (uv[:, 0] < 256)
            onscreen &= (uv[:, 1] >= 0) & (uv[:, 1] < 256)
            inside = np.zeros(8, dtype=bool)
            for point in np.flatnonzero(onscreen):
                u, v = np.rint(uv[point]).astype(int)
                u, v = np.clip([u, v], 0, 255)
                inside[point] = mask[v, u] > 0
            details[name].append({"frame": frame_id,
                                  "positive_depth_onscreen": int(onscreen.sum()),
                                  "on_red_cup_mask": int(inside.sum())})
            if name == "MolmoMotion":
                model_uv = uv
        if frame_id in (99, 108, 118, 128):
            canvas = frame.copy()
            for point in range(8):
                if valid[point, t] and np.isfinite(observed_uv[point, t]).all():
                    u, v = np.rint(observed_uv[point, t]).astype(int)
                    cv2.circle(canvas, (u, v), 4, (0, 255, 0), -1)
                if model_uv is not None and np.isfinite(model_uv[point]).all():
                    u, v = np.rint(model_uv[point]).astype(int)
                    if 2 <= u < 254 and 2 <= v < 254:
                        cv2.drawMarker(canvas, (u, v), (255, 0, 255),
                                       cv2.MARKER_TILTED_CROSS, 10, 2)
            info = details["MolmoMotion"][-1]
            cv2.putText(canvas, f"{frame_id}: GT {int(valid[:, t].sum())}/8; "
                        f"pred on cup {info['on_red_cup_mask']}/8",
                        (3, 20), cv2.FONT_HERSHEY_SIMPLEX, .4,
                        (255, 255, 255), 1)
            panels.append(canvas)
    video.release()
    cv2.imwrite(str(RUN / "prediction_vs_future_rgb.jpg"),
                np.hstack(panels))
    summary = {"status": "EXPLORATORY_APPROXIMATE",
               "meaning": "RGB mask overlap of 3D points projected using transferred K and Dobb-E poses",
               "valid_gt_points_overlay_color": "green",
               "model_projection_overlay_color": "magenta",
               "by_method": details,
               "total_on_red_mask": {name: sum(x["on_red_cup_mask"]
                                               for x in rows)
                                     for name, rows in details.items()},
               "possible": 240}
    (RUN / "future_projection_diagnostic.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["total_on_red_mask"], indent=2))


if __name__ == "__main__":
    main()
