"""Export a complete original FMB demonstration and visual preflight.

The raw NPY remains byte identical. RGB PNGs are derived once from its BGR
arrays; the four depth NPY files retain every original uint16 value.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
NAME = "1_L_L_4_vertical_n_0.npy"
SOURCE = ROOT.parent / "data/fmb/single_object_manipulation_dataset" / NAME
DATA = ROOT / "data/fmb_second_scene"
RUN = ROOT / "runs/fmb_second_scene"
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")
STATE_KEYS = ("obs/tcp_pose", "obs/tcp_vel", "obs/tcp_force", "obs/tcp_torque",
              "obs/q", "obs/dq", "obs/jacobian", "obs/gripper_pose", "actions", "primitive")
SOURCE_URL = ("https://huggingface.co/datasets/charlesxu0124/functional-manipulation-benchmark/"
              "blob/f99fd55c072eea5573523c96aa527aed3c665690/"
              "single_object_manipulation_dataset/" + NAME)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ranges(primitives: np.ndarray) -> list[dict]:
    out = []
    start = 0
    for i in range(1, len(primitives) + 1):
        if i == len(primitives) or primitives[i] != primitives[start]:
            out.append({"primitive": str(primitives[start]), "first_frame": start, "last_frame": i - 1})
            start = i
    return out


def labeled_frame(bgr: np.ndarray, index: int, primitive: str, gripper: object,
                  tcp_xyz: np.ndarray) -> np.ndarray:
    # OpenCV receives source BGR directly and writes encoded RGB video correctly.
    frame = bgr.copy()
    cv2.rectangle(frame, (0, 0), (256, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"#{index:03} {primitive} grip={gripper}", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "TCP " + ",".join(f"{v:.3f}" for v in tcp_xyz), (4, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def export_rgb_and_video(data: dict, n: int) -> None:
    for camera in CAMERAS:
        directory = DATA / "rgb" / camera
        directory.mkdir(parents=True, exist_ok=True)
        video_path = RUN / "previews" / f"preview_{camera}.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (256, 256))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create {video_path}")
        try:
            for i, bgr in enumerate(data[f"obs/{camera}"]):
                # FMB documents BGR storage. PIL's RGB PNG receives one reversal.
                Image.fromarray(bgr[..., ::-1]).save(directory / f"frame_{i:04d}.png")
                writer.write(labeled_frame(bgr, i, str(data["primitive"][i]),
                                           data["obs/gripper_pose"][i], data["obs/tcp_pose"][i, :3]))
        finally:
            writer.release()
        print("RGB+MP4", camera, n, flush=True)


def export_depth(data: dict) -> dict:
    stats = {"depth_unit_status": "UNRESOLVED", "depth_scale": None,
             "meaning": "Raw uint16 sample values; no metric conversion was applied", "streams": {}}
    directory = DATA / "depth_raw"
    directory.mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        array = data[f"obs/{camera}_depth"]
        np.save(directory / f"{camera}.npy", array, allow_pickle=False)
        values = array.astype(np.float64, copy=False)
        finite = np.isfinite(values)
        usable = values[finite]
        stats["streams"][camera] = {
            "dtype": str(array.dtype), "shape": list(array.shape),
            "min": float(np.min(usable)), "max": float(np.max(usable)),
            "median": float(np.median(usable)),
            "percentiles": {str(p): float(np.percentile(usable, p)) for p in (1, 5, 50, 95, 99)},
            "fraction_zero": float(np.mean(array == 0)),
            "fraction_nonfinite": float(np.mean(~finite)),
            "unique_count": int(np.unique(array).size),
            "per_frame_valid_fraction": [float(np.mean(frame > 0)) for frame in array],
        }
        print("DEPTH", camera, array.shape, flush=True)
    write_json(RUN / "depth_statistics.json", stats)
    return stats


def export_state(data: dict) -> None:
    arrays = {key.replace("obs/", ""): data[key] for key in STATE_KEYS if key != "actions"}
    arrays["action"] = data["actions"]
    arrays["object_info_json"] = np.array(json.dumps(data["object_info"], sort_keys=True))
    if "object_id" in data:
        arrays["object_id"] = data["object_id"]
    np.savez(DATA / "robot_state.npz", **arrays)


def make_contact(data: dict, camera: str) -> None:
    n = len(data["primitive"])
    indices = np.unique(np.linspace(0, n - 1, 16, dtype=int))
    sheet = Image.new("RGB", (4 * 256, 4 * 283), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, i in enumerate(indices):
        x, y = slot % 4 * 256, slot // 4 * 283
        sheet.paste(Image.fromarray(data[f"obs/{camera}"][i][..., ::-1]), (x, y))
        draw.text((x + 4, y + 258), f"{i}: {data['primitive'][i]}", fill="black")
    sheet.save(RUN / "visuals" / f"contact_{camera}.png")


def make_synced(data: dict) -> None:
    indices = (0, 56, 65, 94, 103, 112, 122, 131, 141)
    sheet = Image.new("RGB", (4 * 256, len(indices) * 282), "white")
    draw = ImageDraw.Draw(sheet)
    for row, i in enumerate(indices):
        for col, camera in enumerate(CAMERAS):
            x, y = col * 256, row * 282
            sheet.paste(Image.fromarray(data[f"obs/{camera}"][i][..., ::-1]), (x, y))
            draw.text((x + 4, y + 258), f"{camera} #{i} {data['primitive'][i]}", fill="black")
    sheet.save(RUN / "visuals/synchronized_four_cameras.png")


def depth_visual(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    valid = depth > 0
    normalized[valid] = (np.clip((depth[valid].astype(float) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_rgb_depth(data: dict, camera: str) -> None:
    depth = data[f"obs/{camera}_depth"]
    positive = depth[depth > 0]
    low, high = np.percentile(positive, (1, 99))
    indices = (0, 56, 65, 94, 103, 112, 122, 131, 141)
    sheet = Image.new("RGB", (2 * 256, len(indices) * 282), "white")
    draw = ImageDraw.Draw(sheet)
    for row, i in enumerate(indices):
        y = row * 282
        sheet.paste(Image.fromarray(data[f"obs/{camera}"][i][..., ::-1]), (0, y))
        sheet.paste(Image.fromarray(depth_visual(depth[i], low, high)), (256, y))
        draw.text((4, y + 258), f"RGB #{i}", fill="black")
        draw.text((260, y + 258), f"depth raw preview #{i}", fill="black")
    sheet.save(RUN / "visuals" / f"rgb_depth_{camera}.png")


def make_plots(data: dict, depth_stats: dict) -> None:
    n = len(data["primitive"])
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(11, 4))
    for camera in CAMERAS:
        ax.plot(x, depth_stats["streams"][camera]["per_frame_valid_fraction"], label=camera)
    ax.set(xlabel="Frame index", ylabel="Fraction of pixels with raw depth > 0", ylim=(0, 1.02),
           title="Full-frame depth validity (raw samples)")
    ax.legend(); fig.tight_layout(); fig.savefig(RUN / "visuals/valid_depth_fraction.png", dpi=150); plt.close(fig)

    xyz = data["obs/tcp_pose"][:, :3]
    fig, ax = plt.subplots(figsize=(11, 4))
    for j, name in enumerate("XYZ"):
        ax.plot(x, xyz[:, j], label=name)
    ax.set(xlabel="Frame index", ylabel="TCP coordinate (m; FMB robot base frame)", title="TCP position")
    ax.legend(); fig.tight_layout(); fig.savefig(RUN / "visuals/tcp_xyz.png", dpi=150); plt.close(fig)

    quats = data["obs/tcp_pose"][:, 3:7].astype(float)
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    angle = 2 * np.degrees(np.arccos(np.clip(np.abs(quats @ quats[0]), 0, 1)))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, angle)
    ax.set(xlabel="Frame index", ylabel="Quaternion angular difference from frame 0 (deg)",
           title="TCP orientation change")
    fig.tight_layout(); fig.savefig(RUN / "visuals/tcp_orientation_change.png", dpi=150); plt.close(fig)

    primitive = data["primitive"]
    order = list(dict.fromkeys(map(str, primitive)))
    pidx = [order.index(str(item)) for item in primitive]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(x, pidx, where="post")
    ax.set(xlabel="Frame index", ylabel="Primitive", yticks=range(len(order)), yticklabels=order,
           title="Primitive timeline")
    fig.tight_layout(); fig.savefig(RUN / "visuals/primitive_timeline.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    ax.step(x, data["obs/gripper_pose"], where="post")
    ax.set(xlabel="Frame index", ylabel="Gripper state (0=open, 1=closed)", title="Gripper state")
    fig.tight_layout(); fig.savefig(RUN / "visuals/gripper_timeline.png", dpi=150); plt.close(fig)

    image = Image.fromarray(data["obs/side_1"][0][..., ::-1]).resize((768, 768))
    canvas = Image.new("RGB", (768, 930), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    info = data["object_info"]
    lines = ["FMB original object_info:",
             f"shape={info['shape']} rectangle | size={info['size']} large | length={info['length']} long",
             f"color={info['color']} yellow | angle={info['angle']} | distractor={info['distractor']}"]
    for j, line in enumerate(lines):
        draw.text((16, 784 + j * 38), line, fill="black")
    canvas.save(RUN / "visuals/object_configuration.png")


def main() -> None:
    for path in (DATA / "raw", DATA / "rgb", DATA / "depth_raw", RUN / "visuals", RUN / "previews"):
        path.mkdir(parents=True, exist_ok=True)
    data = np.load(SOURCE, allow_pickle=True).item()
    n = len(data["primitive"])
    source_hash = sha256(SOURCE)
    raw = DATA / "raw/source_demo.npy"
    if not raw.exists():
        shutil.copyfile(SOURCE, raw)
    assert sha256(raw) == source_hash
    export_rgb_and_video(data, n)
    depth_stats = export_depth(data)
    export_state(data)
    metadata = {
        "source_filename": NAME, "trajectory_id": 0, "N": n,
        "source_url": SOURCE_URL, "original_archive_url":
        "https://rail.eecs.berkeley.edu/datasets/fmb/single_object_manipulation.zip",
        "source_sha256": source_hash, "source_bytes": SOURCE.stat().st_size,
        "raw_copy_sha256": sha256(raw), "object_info": data["object_info"],
        "ordered_unique_primitives": list(dict.fromkeys(map(str, data["primitive"]))),
        "primitive_ranges": ranges(data["primitive"]),
        "image_resolution_hw": list(data["obs/side_1"].shape[1:3]), "cameras": list(CAMERAS),
        "array_dtypes": {key: str(value.dtype) for key, value in data.items() if isinstance(value, np.ndarray)},
        "array_shapes": {key: list(value.shape) for key, value in data.items() if isinstance(value, np.ndarray)},
        "state_export_key_mapping": {"actions": "action", **{key: key.replace("obs/", "") for key in STATE_KEYS if key != "actions"}},
        "raw_color_order": "BGR per official FMB documentation", "export_png_color_order": "RGB",
        "depth_unit_status": "UNRESOLVED", "depth_scale": None,
    }
    write_json(DATA / "metadata.json", metadata)
    for camera in ("side_1", "side_2"):
        make_contact(data, camera)
        make_rgb_depth(data, camera)
    make_synced(data)
    make_plots(data, depth_stats)
    print("EXPORTED", NAME, n, source_hash)


if __name__ == "__main__":
    main()
