"""Clean RGB-only COLMAP controls; oracle and causal outputs stay separate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import sqlite3
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_colmap_clean_rerun_v1"
SOURCE = ROOT / "data/dobbe_oxe/target_raw/compressed_video_h264.mp4"
PAIR_BASE = 2147483647
POLICY = {
    "version": 1,
    "design_rgb_frames_causal_only": [0, 32, 64, 95],
    "camera_border_px": [10, 10, 35, 21],  # left/right/top/bottom
    "apparatus_corridors_xy": [
        [[0, 256], [0, 209], [46, 146], [74, 148], [40, 256]],
        [[44, 146], [76, 146], [134, 256], [87, 256]],
        [[151, 256], [161, 144], [192, 146], [256, 238], [256, 256]],
        [[90, 256], [158, 145], [191, 145], [134, 256]],
    ],
    "red_hsv_hue": [[0, 12], [168, 179]],
    "red_hsv_min_saturation": 70,
    "red_hsv_min_value": 40,
    "red_foreground_dilation_radius_px": 10,
    "interpretation": "Conservative apparatus/hand corridors and RGB red cup; imperfect foreground mask, not learned segmentation",
    "future_used_to_choose_policy": False,
    "labels_or_depth_used": False,
}


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)


def write(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(data), indent=2) + "\n", encoding="utf-8")


def digest(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_rgb(frame):
    mask = np.full((256, 256), 255, np.uint8)
    left, right, top, bottom = POLICY["camera_border_px"]
    mask[:, :left] = mask[:, 256-right:] = 0
    mask[:top] = mask[256-bottom:] = 0
    for polygon in POLICY["apparatus_corridors_xy"]:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168)) &
           (hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 40)).astype(np.uint8)
    red = cv2.dilate(red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    mask[red > 0] = 0
    return mask


def prepare():
    protocol_path = OUT / "clean_protocol.json"
    existing = None
    if protocol_path.exists():
        p = json.loads(protocol_path.read_text())
        assert p["mask_policy"] == POLICY
        assert p["source_sha256"] == digest(SOURCE)
        existing = p
        if all((OUT / f"images/main/{i:04d}.png").exists()
               and (OUT / f"masks/main/{i:04d}.png.png").exists() for i in range(243)):
            return p
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze the mask policy before reading any additional diagnostic RGB.
    write(OUT / "mask_policy_frozen.json", POLICY)
    cap = cv2.VideoCapture(str(SOURCE))
    assert cap.isOpened()
    images = OUT / "images/main"
    masks = OUT / "masks/main"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    records, causal_review, oracle_review = [], [], []
    for idx in range(243):
        ok, frame = cap.read()
        assert ok and frame.shape == (256, 256, 3), idx
        mask = mask_rgb(frame)
        image_file = images / f"{idx:04d}.png"
        mask_file = masks / f"{idx:04d}.png.png"
        assert cv2.imwrite(str(image_file), frame)
        assert cv2.imwrite(str(mask_file), mask)
        records.append({"frame": idx, "rgb_sha256": digest(image_file),
                        "mask_sha256": digest(mask_file),
                        "retained_pixels": int(np.count_nonzero(mask)),
                        "retained_fraction": float(np.mean(mask > 0))})
        if idx in (0, 32, 64, 95, 128, 176, 224, 242):
            overlay = frame.copy()
            overlay[mask == 0] = (.25 * frame[mask == 0] +
                                  .75 * np.array([180, 0, 180])).astype(np.uint8)
            panel = np.concatenate([frame, overlay], axis=1)
            cv2.putText(panel, f"frame {idx}: RGB | excluded (purple)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
            (causal_review if idx < 96 else oracle_review).append(panel)
    assert not cap.read()[0]
    cap.release()
    assert cv2.imwrite(str(OUT / "mask_review_causal.jpg"), np.vstack(causal_review))
    assert cv2.imwrite(str(OUT / "mask_review_oracle.jpg"), np.vstack(oracle_review))
    p = {"source": str(SOURCE.relative_to(ROOT)), "source_sha256": digest(SOURCE),
         "source_frames": 243, "rgb_only": True, "mask_policy": POLICY,
         "full_oracle": {"frames": list(range(243)), "eligible_for_main_K": False},
         "causal_prefix": {"frames": list(range(96)), "eligible_for_main_K": True},
         "causal_stride5": {"frames": list(range(0, 96, 5)), "eligible_for_main_K": True},
         "pixel_convention": "COLMAP centers at 0.5; OpenCV K principal point is COLMAP minus 0.5",
         "runtime_only_overrides": {"num_threads": 6, "random_seed": 17},
         "oracle_rule": "Oracle intrinsics, poses and graph cannot initialize/select causal models",
         "images": records}
    if existing is not None:
        assert p == existing, "Regenerated source images or masks differ from frozen protocol"
    else:
        write(protocol_path, p)
    return p


def components(ids, pairs, threshold):
    neighbors = {i: set() for i in ids}
    count = 0
    for p in pairs:
        if p["verified_inliers"] >= threshold:
            neighbors[p["image_id_a"]].add(p["image_id_b"])
            neighbors[p["image_id_b"]].add(p["image_id_a"])
            count += 1
    remaining, sizes = set(ids), []
    while remaining:
        todo = [remaining.pop()]
        n = 0
        while todo:
            i = todo.pop()
            n += 1
            for j in neighbors[i] & remaining:
                remaining.remove(j)
                todo.append(j)
        sizes.append(n)
    sizes.sort(reverse=True)
    return {"pairs": count, "largest_component": sizes[0] if sizes else 0,
            "component_sizes": sizes, "isolated_images": sum(not v for v in neighbors.values())}


def audit_database(db, frames, mask_path, run):
    with sqlite3.connect(db) as c:
        ids = dict(c.execute("SELECT image_id,name FROM images"))
        raw = {i: n for i, n in c.execute("SELECT pair_id,rows FROM matches")}
        verified = {i: (n, cfg) for i, n, cfg in c.execute(
            "SELECT pair_id,rows,config FROM two_view_geometries")}
        pairs = []
        for pid in sorted(set(raw) | set(verified)):
            a, b = divmod(pid, PAIR_BASE)
            n, cfg = verified.get(pid, (0, 0))
            pairs.append({"image_id_a": a, "image_id_b": b,
                          "frame_a": int(Path(ids[a]).stem),
                          "frame_b": int(Path(ids[b]).stem),
                          "raw_matches": raw.get(pid, 0), "verified_inliers": n,
                          "geometry_config": cfg})
        per_image = []
        violations = 0
        for i, name in sorted(ids.items()):
            n, cols, data = c.execute(
                "SELECT rows,cols,data FROM keypoints WHERE image_id=?", (i,)).fetchone()
            points = (np.frombuffer(data, np.float32).reshape(n, cols)
                      if n else np.empty((0, cols), np.float32))
            bad = 0
            if mask_path is not None and n:
                m = cv2.imread(str(mask_path / (name + ".png")), cv2.IMREAD_GRAYSCALE)
                xy = np.floor(points[:, :2]).astype(int).clip(0, 255)
                bad = int(np.count_nonzero(m[xy[:, 1], xy[:, 0]] == 0))
                violations += bad
            relevant = [p for p in pairs if i in (p["image_id_a"], p["image_id_b"])]
            per_image.append({"frame": int(Path(name).stem), "keypoints": n,
                              "keypoints_in_masked_regions": bad,
                              "connected_images_ge15": sum(p["verified_inliers"] >= 15 for p in relevant),
                              "connected_images_ge30": sum(p["verified_inliers"] >= 30 for p in relevant),
                              "sum_verified_inliers": sum(p["verified_inliers"] for p in relevant)})
    assert sorted(int(Path(n).stem) for n in ids.values()) == frames
    assert len(pairs) == len(frames) * (len(frames) - 1) // 2
    assert not violations, violations
    with pycolmap.Database.open(db) as c:
        cameras = c.read_all_cameras()
    assert len(cameras) == 1
    camera_rows = [{"camera_id": c.camera_id, "params": np.asarray(c.params).tolist(),
                    "has_prior_focal_length": c.has_prior_focal_length}
                   for c in cameras]
    assert not any(c["has_prior_focal_length"] for c in camera_rows)
    for name, rows in (("match_pairs.csv", pairs), ("match_images.csv", per_image)):
        with (run / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    graph = {"images": len(frames), "pair_rows": len(pairs),
             "raw_matches": sum(p["raw_matches"] for p in pairs),
             "verified_inliers": sum(p["verified_inliers"] for p in pairs),
             "nonzero_verified_pairs": sum(p["verified_inliers"] > 0 for p in pairs),
             "geometry_config_counts": dict(Counter(p["geometry_config"] for p in pairs)),
             "by_threshold": {str(t): components(ids, pairs, t) for t in (15, 30, 50, 100)},
             "keypoints_median": float(np.median([p["keypoints"] for p in per_image])),
             "mask_violations": violations, "initial_cameras": camera_rows}
    write(run / "match_graph.json", graph)
    print("MATCH_GRAPH", json.dumps(graph), flush=True)
    return graph


def summarize(recs):
    results = []
    for idx, rec in recs.items():
        cameras = []
        for cam in rec.cameras.values():
            cameras.append({"model": str(cam.model), "params": np.asarray(cam.params).tolist(),
                            "fx": cam.focal_length_x, "fy": cam.focal_length_y,
                            "cx_colmap": cam.principal_point_x, "cy_colmap": cam.principal_point_y,
                            "K_OpenCV": [[cam.focal_length_x, 0, cam.principal_point_x - .5],
                                         [0, cam.focal_length_y, cam.principal_point_y - .5], [0, 0, 1]]})
        trajectory = []
        for image_id in rec.reg_image_ids():
            im = rec.images[image_id]
            w2c = np.vstack([np.asarray(im.cam_from_world().matrix()), [0, 0, 0, 1]])
            c2w = np.linalg.inv(w2c)
            trajectory.append({"frame": int(Path(im.name).stem),
                               "camera_center": c2w[:3, 3].tolist(),
                               "c2w_colmap": c2w.tolist()})
        trajectory.sort(key=lambda t: t["frame"])
        results.append({"id": int(idx), "registered": len(trajectory),
                        "usable_size_ge10": len(trajectory) >= 10,
                        "points3D": rec.num_points3D(),
                        "mean_reprojection_px": rec.compute_mean_reprojection_error(),
                        "cameras": cameras, "trajectory": trajectory})
    return sorted(results, key=lambda r: r["registered"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--scene", choices=["main"], default="main")
    parser.add_argument("--scope", choices=["full-oracle", "causal"], default="causal")
    parser.add_argument("--model", choices=["PINHOLE", "SIMPLE_RADIAL", "OPENCV"], default="PINHOLE")
    parser.add_argument("--subset", choices=["all", "even", "odd"], default="all")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--mask", choices=["masked", "raw"], default="masked")
    parser.add_argument("--enhanced", action="store_true",
                        help="DSP-SIFT + affine shape + guided matching; mapper remains default")
    args = parser.parse_args()
    assert args.stride > 0
    protocol = prepare()
    if args.prepare_only:
        print(json.dumps({"protocol": str(OUT / 'clean_protocol.json'),
                          "mask_reviews": ['mask_review_causal.jpg', 'mask_review_oracle.jpg']}))
        return
    all_frames = list(range(243 if args.scope == "full-oracle" else 96))
    if args.subset == "even":
        all_frames = all_frames[::2]
    elif args.subset == "odd":
        all_frames = all_frames[1::2]
    frames = all_frames[::args.stride]
    slug = f"main_{args.scope}_{args.model.lower()}_{args.mask}_stride{args.stride}_{args.subset}"
    if args.enhanced:
        slug += "_dsp_affine_guided"
    run = OUT / slug
    run.mkdir(parents=True, exist_ok=True)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = args.model
    assert reader.camera_params == ""
    mask_path = OUT / "masks/main" if args.mask == "masked" else None
    if mask_path is not None:
        reader.mask_path = mask_path
    extract = pycolmap.FeatureExtractionOptions()
    extract.num_threads = 6
    match = pycolmap.FeatureMatchingOptions()
    match.num_threads = 6
    if args.enhanced:
        extract.sift.estimate_affine_shape = True
        extract.sift.domain_size_pooling = True
        match.guided_matching = True
    options = pycolmap.IncrementalPipelineOptions()
    options.num_threads = 6
    options.random_seed = 17
    defaults = pycolmap.IncrementalPipelineOptions().todict()
    actual = options.todict()
    assert all(plain(actual[k]) == plain(v) for k, v in defaults.items()
               if k not in {"num_threads", "random_seed"})
    config = {"pycolmap": pycolmap.__version__, "input_frames": frames,
              "eligible_for_main_K": args.scope != "full-oracle",
              "mask": args.mask, "mask_policy_sha256": digest(OUT / "mask_policy_frozen.json"),
              "source_sha256": protocol["source_sha256"],
              "reader": plain(reader.todict()), "extraction": plain(extract.todict()),
              "matching": plain(match.todict()), "mapper": plain(actual),
              "mapper_overrides_only": {"num_threads": 6, "random_seed": 17},
              "rgb_only": True, "enhanced": args.enhanced}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    path = run / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "Existing run has another configuration"
    else:
        write(path, config)
    if (run / "summary.json").exists():
        print((run / "summary.json").read_text())
        return
    t0 = time.monotonic()
    db = run / "database.db"
    state_path = run / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    assert not state or state["config_sha256"] == config_hash
    state["config_sha256"] = config_hash
    images = OUT / "images/main"
    if not state.get("extracted"):
        pycolmap.extract_features(db, images, image_names=[f"{f:04d}.png" for f in frames],
                                  camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extract,
                                  device=pycolmap.Device.cpu)
        state["extracted"] = True
        write(state_path, state)
    if not state.get("matched"):
        pycolmap.match_exhaustive(db, matching_options=match, device=pycolmap.Device.cpu)
        state["matched"] = True
        write(state_path, state)
    graph = audit_database(db, frames, mask_path, run)
    recs = pycolmap.incremental_mapping(db, images, run / "sparse", options=options)
    results = summarize(recs)
    summary = {"run": slug, "status": "COMPLETE", "input_frames": frames,
               "eligible_for_main_K": config["eligible_for_main_K"],
               "config_sha256": config_hash, "rgb_only": True,
               "elapsed_seconds_this_execution": round(time.monotonic() - t0, 2),
               "reconstructions": results, "best_registered": results[0]["registered"] if results else 0,
               "match_graph": "match_graph.json", "initial_cameras": graph["initial_cameras"]}
    write(run / "summary.json", summary)
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
