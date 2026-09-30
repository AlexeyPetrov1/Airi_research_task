"""Prepare and audit one PointMotionBench/WorldTrack episode (CPU only).

Run with the existing AIRI virtualenv, from any directory. No model weights load.
The eight points are selected using only frames 110..112.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from molmo_motion import MolmoMotionProcessor


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SOURCE = ROOT / "source/Apartment_release_clean_seq131_0.npz"
INDEX = ROOT / "source/worldtrack_index_map.json"
CHECKPOINT = REPO / "data/checkpoints/MolmoMotion-4B-H3-F30"
KEY = "adt_mini/Apartment_release_clean_seq131_0_clip02_obj1_t108-149"
T0 = 112
HISTORY = np.arange(T0 - 2, T0 + 1)
FUTURE = np.arange(T0 + 1, T0 + 31)
ALL = np.r_[HISTORY, FUTURE]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project(xyz, intrinsic):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack(
            (
                intrinsic[0] * xyz[..., 0] / xyz[..., 2] + intrinsic[2],
                intrinsic[1] * xyz[..., 1] / xyz[..., 2] + intrinsic[3],
            ), axis=-1,
        )


def world_to_camera(world, w2c):
    return world @ w2c[:3, :3].T + w2c[:3, 3]


def camera_to_world(cam, w2c):
    return (cam - w2c[:3, 3]) @ w2c[:3, :3]


def select_history_points(entry, cam, visibility, intrinsic):
    clip_ids = np.asarray(entry["point_indices"], dtype=int)
    object_ids = np.asarray(entry["object_ids"], dtype=int)
    assert len(clip_ids) == len(object_ids)
    candidates = clip_ids[object_ids == entry["clip_objects"][0]]
    candidates = candidates[visibility[HISTORY[:, None], candidates].all(axis=0)]
    assert len(candidates) >= 8, "Fewer than eight object points visible in all three history frames"
    uv = project(cam[T0, candidates], intrinsic)
    assert np.isfinite(uv).all()
    chosen = [0]
    while len(chosen) < 8:
        distances = np.min(np.linalg.norm(uv[:, None] - uv[chosen][None], axis=-1), axis=1)
        distances[chosen] = -1
        chosen.append(int(np.argmax(distances)))
    return candidates[chosen], candidates


def error_stats(projected, reference, vis, depth):
    finite = np.isfinite(projected).all(-1) & np.isfinite(reference).all(-1)
    valid = vis & finite & (depth > 0)
    dist = np.linalg.norm(projected - reference, axis=-1)
    d = dist[valid]
    return {
        "mean_px": float(d.mean()),
        "median_px": float(np.median(d)),
        "rmse_px": float(np.sqrt(np.mean(d * d))),
        "max_px": float(d.max()),
        "per_point_px": [float(x) if np.isfinite(x) else None for x in dist],
        "valid_points": int(valid.sum()),
        "nonfinite_points": int((~finite).sum()),
        "nonpositive_depth_points": int((depth <= 0).sum()),
    }


def tensor_equal(a, b):
    if torch.is_tensor(a) and torch.is_tensor(b):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, list) and isinstance(b, list):
        return all(tensor_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(tensor_equal(a[k], b[k]) for k in a)
    return a == b


def main():
    entry = json.loads(INDEX.read_text())[KEY]
    caption = entry["caption"]["caption"]
    fps = float(entry["caption"]["fps"])
    assert fps == 30.0
    assert np.array_equal(np.asarray(entry["frame_indices"]), np.arange(108, 150))
    assert set(ALL).issubset(entry["frame_indices"])

    with np.load(SOURCE, allow_pickle=True) as data:
        cam = np.asarray(data["tracks_XYZ"], dtype=np.float64)
        visibility = np.asarray(data["visibility"], dtype=bool)
        w2c_all = np.asarray(data["extrinsics_w2c"], dtype=np.float64)
        intrinsic = np.asarray(data["fx_fy_cx_cy"], dtype=np.float64)
        queries = np.asarray(data["queries_xyt"], dtype=np.float64)
        image_bytes = [bytes(data["images_jpeg_bytes"][int(t)]) for t in ALL]
    assert cam.shape[:2] == visibility.shape == (300, 825)
    assert w2c_all.shape == (300, 4, 4)
    assert intrinsic.shape == (4,)
    assert queries.shape == (825, 3)
    point_ids, candidates = select_history_points(entry, cam, visibility, intrinsic)
    assert len(set(point_ids)) == 8

    frames_dir = ROOT / "frames"
    frames_dir.mkdir(exist_ok=True)
    frames = []
    for t, blob in zip(ALL, image_bytes):
        file = frames_dir / f"frame_{t:03d}.jpg"
        file.write_bytes(blob)
        frame = Image.open(io.BytesIO(blob)).convert("RGB")
        assert frame.size == (512, 512)
        frames.append(frame)
    assert len({sha(frames_dir / f"frame_{t:03d}.jpg") for t in HISTORY}) == 3

    cam_selected = cam[ALL[:, None], point_ids]
    vis_selected = visibility[ALL[:, None], point_ids]
    cameras = w2c_all[ALL]
    world = np.stack([camera_to_world(cam_selected[j], cameras[j]) for j in range(len(ALL))])
    cam_back = np.stack([world_to_camera(world[j], cameras[j]) for j in range(len(ALL))])
    max_roundtrip = float(np.max(np.abs(cam_back - cam_selected)))
    assert max_roundtrip < 1e-10
    assert vis_selected[:3].all()
    assert vis_selected[3:].all(), "Chosen points lack complete 30-frame reference"
    assert np.isfinite(world).all() and np.isfinite(cam_selected).all()

    w2c_t0 = w2c_all[T0]
    c2w_t0 = np.linalg.inv(w2c_t0)
    history_world = world[:3]
    history_cam_t0 = world_to_camera(history_world, w2c_t0)
    source_uv = project(cam_selected, intrinsic)
    assert np.isfinite(source_uv).all()
    direct_uv = project(history_world, intrinsic)
    correct_uv = project(cam_back[:3], intrinsic)
    projection = []
    for j, t in enumerate(HISTORY):
        projection.append({
            "source_frame": int(t),
            "direct_K_only": error_stats(direct_uv[j], source_uv[j], vis_selected[j], history_world[j, :, 2]),
            "source_w2c_K": error_stats(correct_uv[j], source_uv[j], vis_selected[j], cam_back[j, :, 2]),
        })

    fig, axs = plt.subplots(3, 2, figsize=(12, 17), constrained_layout=True)
    for j, t in enumerate(HISTORY):
        for col, xy in enumerate((direct_uv[j], correct_uv[j])):
            ax = axs[j, col]
            ax.imshow(frames[j])
            ax.scatter(source_uv[j, :, 0], source_uv[j, :, 1], marker="x", color="cyan", s=40, label="source 3D→pixel")
            ax.scatter(xy[:, 0], xy[:, 1], facecolors="none", edgecolors="red" if col == 0 else "lime", s=90, label="K only" if col == 0 else "w2c + K")
            for n, pt in zip(point_ids, source_uv[j]):
                ax.text(pt[0] + 3, pt[1] - 3, str(n), color="yellow", fontsize=9)
            ax.set(xlim=(0, 512), ylim=(512, 0), title=f"Frame {t}: " + ("world→K" if col == 0 else "world→w2c(t)→K"))
            ax.legend(loc="upper left", fontsize=7)
    fig.savefig(ROOT / "history_projections.png", dpi=120)
    plt.close(fig)

    np.savez_compressed(
        ROOT / "episode.npz",
        source_frames=ALL, point_ids=point_ids, camera_xyz=cam_selected,
        world_xyz=world, visibility=vis_selected, source_uv=source_uv,
        cameras_w2c=cameras, w2c_t0=w2c_t0, c2w_t0=c2w_t0,
        intrinsics=intrinsic, fps=fps,
    )
    (ROOT / "projection_history.json").write_text(json.dumps(projection, indent=2) + "\n")
    with (ROOT / "frame_mapping.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "source_frame", "time_seconds", "3d_row", "visibility_row", "pose_row", "intrinsics_row", "rgb_sha256"])
        for t in ALL:
            writer.writerow([f"frames/frame_{t:03d}.jpg", int(t), f"{t/fps:.9f}", int(t), int(t), int(t), "constant fx_fy_cx_cy", sha(frames_dir / f"frame_{t:03d}.jpg")])

    processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
    points2d = torch.from_numpy(source_uv[2].astype(np.float32))
    history_input = torch.from_numpy(history_world.astype(np.float32))
    c2w_input = torch.from_numpy(c2w_t0.astype(np.float32))
    inputs_a = processor(history_frames=frames[:3], points_2d_at_t0=points2d, points_3d_history=history_input, action=caption, future_horizon=30)
    inputs_b = processor(history_frames=frames[:3], points_2d_at_t0=points2d, points_3d_history=history_input, action=caption, future_horizon=30, c2w_at_t0=c2w_input)
    w2c_torch = torch.linalg.inv(c2w_input)
    explicit = history_input.reshape(-1, 3) @ w2c_torch[:3, :3].T + w2c_torch[:3, 3]
    inputs_explicit = processor(history_frames=frames[:3], points_2d_at_t0=points2d, points_3d_history=explicit.reshape(3, 8, 3), action=caption, future_horizon=30)
    assert tensor_equal(inputs_b, inputs_explicit), "Explicit and processor camera transforms disagree"
    assert torch.equal(inputs_a["images"], inputs_b["images"])
    assert not torch.equal(inputs_a["input_ids"], inputs_b["input_ids"])
    torch.save(inputs_a, ROOT / "processor_A.pt")
    torch.save(inputs_b, ROOT / "processor_B.pt")
    decoded_a = processor.tokenizer.decode(inputs_a["input_ids"][0].tolist(), truncate_at_eos=False)
    decoded_b = processor.tokenizer.decode(inputs_b["input_ids"][0].tolist(), truncate_at_eos=False)
    (ROOT / "prompt_A.txt").write_text(decoded_a)
    (ROOT / "prompt_B.txt").write_text(decoded_b)

    A = history_input.numpy()
    B = explicit.reshape(3, 8, 3).numpy()
    delta_a = A - A[-1, 0]
    delta_b = B - B[-1, 0]
    token_a = inputs_a["input_ids"][0]
    token_b = inputs_b["input_ids"][0]
    token_differences = int((token_a != token_b).sum()) if token_a.shape == token_b.shape else None
    R = w2c_t0[:3, :3]
    R_error = float(np.linalg.norm(R @ R.T - np.eye(3), ord="fro"))
    det = float(np.linalg.det(R))
    assert R_error < 1e-10 and abs(det - 1) < 1e-10
    max_inv_err = float(np.max(np.abs(camera_to_world(history_cam_t0, w2c_t0) - history_world)))
    assert max_inv_err < 1e-10
    query_checks = []
    for n in point_ids:
        qt = int(round(queries[n, 2]))
        if 0 <= qt < 300 and visibility[qt, n]:
            q_uv = project(cam[qt, n], intrinsic)
            err = float(np.linalg.norm(q_uv - queries[n, :2]))
            query_checks.append({"source_id": int(n), "query_frame": qt, "query_uv": queries[n, :2].tolist(), "projected_uv": q_uv.tolist(), "error_px": err})

    preflight = {
        "source_clip": KEY, "source_file": entry["source"], "caption": caption,
        "fps": fps, "clip_frames": [108, 149], "history_frames": HISTORY.tolist(),
        "future_frames": FUTURE.tolist(), "source_point_ids": point_ids.tolist(),
        "object_candidates_visible_in_history": candidates.tolist(),
        "future_visible_pairs": int(vis_selected[3:].sum()), "future_total_pairs": 240,
        "image_size_wh": [512, 512], "intrinsics": intrinsic.tolist(),
        "no_distortion_fields_in_source": True,
        "pose_convention": "source extrinsics_w2c is world-to-camera; source tracks_XYZ is per-frame camera XYZ",
        "rotation_orthogonality_fro": R_error, "rotation_determinant": det,
        "w2c_t0": w2c_t0.tolist(), "c2w_t0": c2w_t0.tolist(),
        "max_source_camera_world_camera_roundtrip_m": max_roundtrip,
        "max_history_camera_world_inverse_m": max_inv_err,
        "max_history_delta_difference_m": float(np.max(np.abs(delta_a - delta_b))),
        "max_A_quantization_error_m": float(np.max(np.abs(np.round(delta_a * 1000) / 1000 - delta_a))),
        "max_B_quantization_error_m": float(np.max(np.abs(np.round(delta_b * 1000) / 1000 - delta_b))),
        "input_token_shapes": [list(token_a.shape), list(token_b.shape)],
        "input_token_differences": token_differences,
        "visual_inputs_equal": bool(torch.equal(inputs_a["images"], inputs_b["images"])),
        "processor_B_methods_exactly_equal": True,
        "query_uv_checks": query_checks,
        "source_sha256": sha(SOURCE), "index_sha256": sha(INDEX),
        "checkpoint_config_sha256": sha(CHECKPOINT / "config.yaml"),
    }
    (ROOT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    print(json.dumps({k: preflight[k] for k in ("source_clip", "source_point_ids", "future_visible_pairs", "max_history_delta_difference_m", "input_token_differences", "visual_inputs_equal")}, indent=2))
    print(json.dumps(projection, indent=2))


if __name__ == "__main__":
    main()
