"""RGB-only COLMAP calibration for causal HoNY prefixes.

Run via PyCOLMAP (the official COLMAP Python bindings). No depth or labels are
loaded by this script. Each model/subset gets an independent database and SfM.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pycolmap


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/dobbe_rgbd_study"
SCENES = {
    "main": (ROOT / "data/dobbe_oxe/target_raw", 96),
    "second": (ROOT / "data/dobbe_oxe/second_raw", 121),
}


def database_complete(path: Path, frame_count: int) -> bool:
    if not path.exists():
        return False
    try:
        with sqlite3.connect(path) as con:
            images = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            geometry = con.execute(
                "SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        pair_count = frame_count * (frame_count - 1) // 2
        return images == frame_count and matches == geometry == pair_count
    except sqlite3.DatabaseError:
        return False


def remove_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = path.with_name(path.name + suffix)
        if target.exists():
            target.unlink()


def save_rgb_prefix(scene: str, aspect_rectified: bool = False) -> list[int]:
    source, end = SCENES[scene]
    square_folder = STUDY / scene / "rgb_prefix"
    folder = (STUDY / scene / "rgb_prefix_256x192" if aspect_rectified
              else square_folder)
    folder.mkdir(parents=True, exist_ok=True)
    if len(list(folder.glob("*.png"))) == end:
        return list(range(end))
    if aspect_rectified:
        save_rgb_prefix(scene)
        for idx in range(end):
            square = cv2.imread(str(square_folder / f"{idx:04d}.png"))
            if square is None or square.shape[:2] != (256, 256):
                raise RuntimeError(f"Unexpected source RGB frame {idx}")
            rectified = cv2.resize(square, (256, 192),
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(folder / f"{idx:04d}.png"), rectified)
        return list(range(end))
    capture = cv2.VideoCapture(str(source / "compressed_video_h264.mp4"))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read {source}")
    for idx in range(end):
        ok, frame = capture.read()
        if not ok or frame.shape[:2] != (256, 256):
            raise RuntimeError(f"Unexpected RGB frame {idx}")
        cv2.imwrite(str(folder / f"{idx:04d}.png"), frame)
    capture.release()
    return list(range(end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--model", choices=["PINHOLE", "OPENCV",
                                             "SIMPLE_PINHOLE"], required=True)
    parser.add_argument("--subset", choices=["all", "even", "odd"], required=True)
    parser.add_argument("--fixed-pp", action="store_true",
                        help="Keep principal point at the image center")
    parser.add_argument("--init-pair", type=int, nargs=2, metavar=("FRAME_A", "FRAME_B"))
    parser.add_argument("--aspect-rectified", action="store_true",
                        help="RGB-only diagnostic: undo Dobb-E's 256x256 stretch to 256x192")
    args = parser.parse_args()
    if args.aspect_rectified != (args.model == "SIMPLE_PINHOLE"):
        parser.error("SIMPLE_PINHOLE must be paired with --aspect-rectified")
    frames = save_rgb_prefix(args.scene, args.aspect_rectified)
    if args.subset == "even":
        frames = frames[::2]
    elif args.subset == "odd":
        frames = frames[1::2]
    slug = f"colmap_{args.model.lower()}_{args.subset}"
    if args.fixed_pp:
        slug += "_fixedpp"
    if args.aspect_rectified:
        slug += "_rectified"
    if args.init_pair:
        slug += f"_init{args.init_pair[0]}_{args.init_pair[1]}"
    run_dir = STUDY / args.scene / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
        return
    db = run_dir / "database.db"
    if db.exists() and not database_complete(db, len(frames)):
        remove_db_files(db)  # Recover only this run's interrupted database.
    image_path = STUDY / args.scene / (
        "rgb_prefix_256x192" if args.aspect_rectified else "rgb_prefix")
    names = [f"{i:04d}.png" for i in frames]
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.model
    reader.camera_params = ({"PINHOLE": "200,200,128,128",
                             "OPENCV": "200,200,128,128,0,0,0,0",
                             "SIMPLE_PINHOLE": "200,128,96"}[args.model])
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = 6
    extract.sift.max_num_features = 4096
    extract.sift.peak_threshold = 0.002
    if not db.exists():
        candidates = sorted((STUDY / args.scene).glob(
            f"colmap_{args.model.lower()}_{args.subset}*/database.db"))
        for source_db in candidates:
            if source_db == db or not database_complete(source_db, len(frames)):
                continue
            with sqlite3.connect(source_db) as src, sqlite3.connect(db) as dest:
                src.backup(dest)  # Includes committed pages even if source uses WAL.
            break
    if not db.exists():
        pycolmap.extract_features(db, image_path, image_names=names,
                                  camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader,
                                  extraction_options=extract,
                                  device=pycolmap.Device.cpu)
        match = pycolmap.FeatureMatchingOptions()
        match.num_threads = 6
        pycolmap.match_exhaustive(db, matching_options=match,
                                  device=pycolmap.Device.cpu)
    if not database_complete(db, len(frames)):
        raise RuntimeError("RGB-only COLMAP database is incomplete")
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 6
    options.random_seed = 17
    options.multiple_models = False
    options.min_model_size = 2
    options.ba_refine_principal_point = not args.fixed_pp
    options.mapper.init_min_num_inliers = 18 if args.init_pair else 30
    options.mapper.init_min_tri_angle = 3.0
    options.mapper.ba_local_min_tri_angle = 2.0
    options.mapper.abs_pose_min_num_inliers = 15
    options.mapper.abs_pose_min_inlier_ratio = 0.1
    if args.init_pair:
        with sqlite3.connect(db) as con:
            image_ids = dict(con.execute("SELECT name, image_id FROM images"))
        options.init_image_id1 = image_ids[f"{args.init_pair[0]:04d}.png"]
        options.init_image_id2 = image_ids[f"{args.init_pair[1]:04d}.png"]
    reconstructions = pycolmap.incremental_mapping(
        db, image_path, run_dir / "sparse", options=options)
    results = []
    for rec_id, rec in reconstructions.items():
        cameras = []
        for cam_id, cam in rec.cameras.items():
            camera_summary = {"id": cam_id, "model": str(cam.model),
                            "width": cam.width, "height": cam.height,
                            "params": np.asarray(cam.params).tolist(),
                            "fx": cam.focal_length_x, "fy": cam.focal_length_y,
                            "cx": cam.principal_point_x,
                            "cy": cam.principal_point_y}
            if args.aspect_rectified:
                camera_summary["published_256x256_K"] = [
                    cam.focal_length_x, cam.focal_length_y * 256 / 192,
                    cam.principal_point_x,
                    (cam.principal_point_y + .5) * 256 / 192 - .5]
            cameras.append(camera_summary)
        trajectory = []
        for im_id in rec.reg_image_ids():
            image = rec.images[im_id]
            transform = image.cam_from_world()
            mat = np.asarray(transform.matrix())
            c2w = np.linalg.inv(np.vstack([mat, [0, 0, 0, 1]])
                                if mat.shape == (3, 4) else mat)
            trajectory.append({"image": image.name,
                               "camera_center": c2w[:3, 3].tolist()})
        trajectory.sort(key=lambda t: t["image"])
        results.append({"id": int(rec_id), "registered": len(trajectory),
                        "points3D": rec.num_points3D(),
                        "mean_reprojection_px": rec.compute_mean_reprojection_error(),
                        "cameras": cameras, "trajectory": trajectory})
    summary = {"scene": args.scene, "model": args.model,
               "subset": args.subset, "input_frames": frames,
               "rgb_only": True, "pycolmap": pycolmap.__version__,
               "aspect_rectified": args.aspect_rectified,
               "image_size": [256, 192] if args.aspect_rectified else [256, 256],
               "initial_camera_params": reader.camera_params,
               "principal_point_refined": not args.fixed_pp,
               "initial_pair": args.init_pair,
               "reconstructions": results}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"run": str(run_dir),
                      "registered": [r["registered"] for r in results],
                      "cameras": [r["cameras"] for r in results]}, indent=2))


if __name__ == "__main__":
    main()
