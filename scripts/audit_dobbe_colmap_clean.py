"""Verify frozen controls against actual SQLite databases and sparse models."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3

import cv2
import numpy as np
import pycolmap

from dobbe_colmap_clean import OUT, ROOT, components, digest, plain, write
from summarize_dobbe_colmap_clean import RUNS, read


def main():
    protocol = read(OUT / "clean_protocol.json")
    assert digest(ROOT / protocol["source"]) == protocol["source_sha256"]
    assert not protocol["full_oracle"]["eligible_for_main_K"]
    assert not protocol["mask_policy"]["future_used_to_choose_policy"]
    assert not protocol["mask_policy"]["labels_or_depth_used"]
    assert protocol["mask_policy"] == read(OUT / "mask_policy_frozen.json")
    assert max(protocol["mask_policy"]["design_rgb_frames_causal_only"]) < 96
    for item in protocol["images"]:
        f = item["frame"]
        assert digest(OUT / f"images/main/{f:04d}.png") == item["rgb_sha256"]
        assert digest(OUT / f"masks/main/{f:04d}.png.png") == item["mask_sha256"]
    defaults = plain(pycolmap.IncrementalPipelineOptions().todict())
    controls = []
    for index, slug in enumerate(RUNS):
        run = OUT / slug
        config, summary, graph = [read(run / f"{name}.json")
                                  for name in ("config", "summary", "match_graph")]
        assert config["pycolmap"] == pycolmap.__version__
        assert summary["status"] == "COMPLETE"
        assert config["input_frames"] == summary["input_frames"]
        assert config["mask_policy_sha256"] == digest(OUT / "mask_policy_frozen.json")
        assert config["source_sha256"] == protocol["source_sha256"]
        assert config["reader"]["camera_params"] == ""
        assert config["reader"]["camera_model"] == "PINHOLE"
        assert config["rgb_only"] and config["mask"] == "masked"
        assert config["eligible_for_main_K"] == (index != 0)
        assert config["enhanced"] == (index == 3)
        extraction = pycolmap.FeatureExtractionOptions()
        extraction.num_threads = 6
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = 6
        if index == 3:
            extraction.sift.estimate_affine_shape = True
            extraction.sift.domain_size_pooling = True
            matching.guided_matching = True
        assert config["extraction"] == plain(extraction.todict())
        assert config["matching"] == plain(matching.todict())
        expected = list(range(243)) if index == 0 else list(range(0, 96, 5 if index == 2 else 1))
        assert config["input_frames"] == expected
        for k, v in defaults.items():
            if k not in {"num_threads", "random_seed"}:
                assert config["mapper"][k] == v, (slug, k)
        confighash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert confighash == summary["config_sha256"]
        with (run / "match_pairs.csv").open() as f:
            pairs = [{k: int(v) for k, v in p.items()} for p in csv.DictReader(f)]
        with sqlite3.connect(run / "database.db") as db:
            ids = dict(db.execute("SELECT image_id,name FROM images"))
            assert sorted(int(n.split('.')[0]) for n in ids.values()) == expected
            raw = dict(db.execute("SELECT pair_id,rows FROM matches"))
            verified = {pid: (n, cfg) for pid, n, cfg in db.execute(
                "SELECT pair_id,rows,config FROM two_view_geometries")}
            assert len(pairs) == len(expected) * (len(expected) - 1) // 2
            for pair in pairs:
                pid = pair["image_id_a"] * 2147483647 + pair["image_id_b"]
                assert raw.get(pid, 0) == pair["raw_matches"]
                assert verified.get(pid, (0, 0)) == (pair["verified_inliers"], pair["geometry_config"])
            for image_id, n, cols, data in db.execute("SELECT image_id,rows,cols,data FROM keypoints"):
                if not n:
                    continue
                xy = np.floor(np.frombuffer(data, np.float32).reshape(n, cols)[:, :2]).astype(int).clip(0, 255)
                mask = cv2.imread(str(OUT / ("masks/main/" + ids[image_id] + ".png")), 0)
                assert np.all(mask[xy[:, 1], xy[:, 0]] > 0)
        with pycolmap.Database.open(run / "database.db") as db:
            cameras = db.read_all_cameras()
        assert len(cameras) == 1 and not cameras[0].has_prior_focal_length
        assert np.allclose(cameras[0].params, [307.2, 307.2, 128, 128])
        for t in (15, 30, 50, 100):
            assert components(ids, pairs, t) == graph["by_threshold"][str(t)]
        for rec in summary["reconstructions"]:
            sparse = pycolmap.Reconstruction(run / "sparse" / str(rec["id"]))
            assert sparse.num_reg_images() == rec["registered"]
            assert sparse.num_points3D() == rec["points3D"]
            assert rec["usable_size_ge10"] == (rec["registered"] >= 10)
            for cam in rec["cameras"]:
                k = np.asarray(cam["K_OpenCV"])
                assert k[0, 2] == cam["cx_colmap"] - .5
                assert k[1, 2] == cam["cy_colmap"] - .5
        controls.append({"run": slug, "verified": True, "config_sha256": confighash})
    first, second = [read(OUT / RUNS[i] / "config.json") for i in (0, 1)]
    for key in ("reader", "extraction", "matching", "mapper", "mask_policy_sha256"):
        assert first[key] == second[key], key
    validation = read(OUT / "causal_geometry_validation.json")
    assert not validation["oracle_K_scored"]
    assert all(max(pair) < 96 for pair in validation["calibration_pairs_only"])
    decision = read(OUT / "clean_decision.json")
    assert decision["validated_main_K"] is None and not decision["oracle_eligible_for_main_K"]
    assert digest(ROOT / "runs/dobbe_rgbd_study/approx_history/points_3d_history_candidate.npy") == decision["approximate_existing_input_sha256"]
    freeze = read(ROOT / "runs/dobbe_rgbd_study/approx_history/input_freeze.json")
    assert decision["approximate_existing_input_sha256"] == freeze["sha256"]["points_3d_history_candidate.npy"]
    result = {"status": "PASS", "checks": ["243 RGB and mask hashes", "causal/oracle split", "full and causal settings identical", "actual focal prior false", "installed mapper defaults", "actual masked keypoints", "CSV versus SQLite verified inliers", "threshold graph components", "sparse model counts", "COLMAP to OpenCV half pixel", "no oracle K validation or transfer", "existing approximate input preserved"], "controls": controls}
    write(OUT / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
