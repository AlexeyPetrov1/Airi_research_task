"""Check the individually available ShareRobot FMB trajectory RGBs by pixels."""

from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/fmb_second_scene"
DATA = ROOT / "data/fmb_second_scene/raw/source_demo.npy"
MANIFEST = RUN / "sharerobot_trajectory_manifest.json"
REV = "3266d92902b038ce40e7fb8aac5bfd9287eb3e45"
BASE = f"https://huggingface.co/datasets/BAAI/ShareRobot/resolve/{REV}/trajectory/images/"
CAMERAS = ("side_1", "side_2", "wrist_1", "wrist_2")


def fetch(path: str) -> dict:
    url = BASE + quote(path, safe="/")
    try:
        response = requests.get(url, timeout=(20, 40))
        if not response.ok:
            return {"image_path": path, "http_status": response.status_code, "error": response.text[:100]}
        image = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
        return {"image_path": path, "http_status": response.status_code,
                "sha256": hashlib.sha256(response.content).hexdigest(), "image": image}
    except Exception as exc:
        return {"image_path": path, "error": repr(exc)}


def main() -> None:
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    rows = [row for row in manifest if "57_fmb" in row.get("image_path", "")]
    paths = list(dict.fromkeys(row["image_path"] for row in rows))
    source = np.load(DATA, allow_pickle=True).item()
    downsampled = []
    labels = []
    for camera in CAMERAS:
        for i, raw in enumerate(source[f"obs/{camera}"]):
            small = cv2.resize(raw, (32, 32), interpolation=cv2.INTER_AREA)
            downsampled.extend((small, small[..., ::-1]))
            labels.extend(((camera, i, "raw_bytes_as_RGB"), (camera, i, "BGR_to_RGB")))
    stack = np.stack(downsampled).astype(np.int16)
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, path): path for path in paths}
        for future in as_completed(futures):
            item = future.result()
            image = item.pop("image", None)
            if image is not None and image.shape == (256, 256, 3):
                small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.int16)
                error = np.mean(np.abs(stack - small), axis=(1, 2, 3))
                index = int(np.argmin(error))
                camera, step, color_mode = labels[index]
                source_frame = source[f"obs/{camera}"][step]
                candidate_rgb = source_frame if color_mode == "raw_bytes_as_RGB" else source_frame[..., ::-1]
                item.update({"best_camera": camera, "best_FMB_step": step, "color_interpretation": color_mode,
                             "downsampled_MAE": float(error[index]),
                             "full_resolution_MAE": float(np.mean(np.abs(image.astype(np.int16) - candidate_rgb.astype(np.int16))))})
            results.append(item)
            print(len(results), "/", len(paths), item["image_path"], item.get("full_resolution_MAE", item.get("error", item.get("http_status"))), flush=True)
    results.sort(key=lambda item: item.get("downsampled_MAE", float("inf")))
    output = {"source_FMB_file": "1_L_L_4_vertical_n_0.npy", "source_frames": len(source["primitive"]),
              "manifest_url": f"https://huggingface.co/datasets/BAAI/ShareRobot/blob/{REV}/trajectory/trajectory.json",
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "manifest_FMB_rows": len(rows),
              "unique_images": len(paths), "successful_images": sum("full_resolution_MAE" in r for r in results),
              "search_scope": "All individually referenced FMB trajectory-image rows in pinned ShareRobot trajectory.json; every source step, four cameras, both source color interpretations",
              "results": results,
              "limitation": "This tests the Trajectory annotation subset, not the larger Planning image archive or all FMB episodes."}
    (RUN / "sharerobot_trajectory_search.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("BEST", results[0] if results else None)


if __name__ == "__main__":
    main()
