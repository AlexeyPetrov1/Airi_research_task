"""Check saved evidence, causal indices, receipts and report links."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/dobbe_rgbd_study"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    protocol = load(OUT / "protocol.json")
    audit = {"protocol": {}, "colmap_runs": {}, "correspondences": {},
             "download_sha256": {}, "report_links": {}, "decision": {}}
    for scene in ("main", "second"):
        p = protocol[scene]
        calibration = set(p["calibration_past"])
        history = set(p["history"])
        future = set(p["future_evaluation_only"])
        assert calibration and len(history) == 3 and len(future) == 30
        assert calibration.isdisjoint(history | future)
        assert history.isdisjoint(future)
        assert max(calibration) < min(history) < max(history) < min(future)
        audit["protocol"][scene] = {
            "calibration": len(calibration), "history": sorted(history),
            "future": [min(future), max(future)], "disjoint": True}

        run_files = sorted((OUT / scene).glob("colmap_*/summary.json"))
        assert run_files
        for path in run_files:
            run = load(path)
            assert set(run["input_frames"]) <= calibration, path
            assert run["rgb_only"] is True
            with sqlite3.connect(path.parent / "database.db") as con:
                images = con.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                matches = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
                geometry_pairs = con.execute(
                    "SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
            expected_pairs = images * (images - 1) // 2
            assert images == len(run["input_frames"]), path
            assert matches == geometry_pairs == expected_pairs, path
            audit["colmap_runs"][f"{scene}/{path.parent.name}"] = {
                "frames": len(run["input_frames"]),
                "verified_pairs": geometry_pairs,
                "registered": max((m["registered"] for m in run["reconstructions"]),
                                  default=0)}

        points = load(OUT / f"{scene}_static_correspondences.json")
        n = 0
        for group in points:
            assert set(group["pair"]) <= calibration
            for point in group["selected"]:
                assert point["depth_a_m"] > 0 and point["depth_b_m"] > 0
                n += 1
        assert n >= 20
        audit["correspondences"][scene] = n
        geometry = load(OUT / f"{scene}_static_geometry.json")
        assert geometry["optical_to_label"] == [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
        assert geometry["P_new"] == [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        for path in sorted((OUT / scene).glob(
                "colmap_simple_pinhole_*_fixedpp_rectified/summary.json")):
            run = load(path)
            assert run["aspect_rectified"] and run["image_size"] == [256, 192]
            for reconstruction in run["reconstructions"]:
                for camera in reconstruction["cameras"]:
                    f, cx, cy = camera["params"]
                    expected = [f, f * 256 / 192, cx,
                                (cy + .5) * 256 / 192 - .5]
                    assert all(abs(a - b) < 1e-9 for a, b in zip(
                        camera["published_256x256_K"], expected))
        for candidate in geometry["candidates"].values():
            fx, fy, cx, cy = candidate["camera"]["params"][:4]
            depth_k = candidate["camera"]["K_depth"]
            assert abs(depth_k[0] - fx) < 1e-9
            assert abs(depth_k[1] - fy * .75) < 1e-9
            assert abs(depth_k[2] - cx) < 1e-9
            assert abs(depth_k[3] - ((cy + .5) * .75 - .5)) < 1e-9

    required_main = {(model, subset) for model in ("PINHOLE", "OPENCV")
                     for subset in ("all", "even", "odd")}
    with (OUT / "colmap_runs.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    found = {(r["model"], r["subset"]) for r in rows
             if r["scene"] == "main"}
    assert required_main <= found
    for scene in ("main", "second"):
        aspect_runs = {r["subset"] for r in rows if r["scene"] == scene
                       and r["model"] == "SIMPLE_PINHOLE"
                       and "_rectified" in r["run"]
                       and "_init" not in r["run"]}
        assert aspect_runs == {"all", "even", "odd"}, (scene, aspect_runs)

    rotation = load(OUT / "image_rotation_diagnostic.json")
    robustness = load(OUT / "correspondence_robustness.json")
    for scene in ("main", "second"):
        baseline = load(OUT / f"{scene}_static_geometry.json")["candidates"][
            "naive"]["variants"]["c2w_z"]
        corrected = rotation[scene]["exporter_consistent"]["naive"]
        assert abs(baseline["median_3d_m"] - corrected["median_3d_m"]) < 1e-12
        assert abs(baseline["median_reprojection_px"] -
                   corrected["median_reprojection_px"]) < 1e-12
        clean = robustness[scene]
        assert clean["original_observations"] == audit["correspondences"][scene]
        assert 20 <= clean["filtered_observations"] < clean["original_observations"]
        assert all(set(pair["pair"]) <= set(protocol[scene]["calibration_past"])
                   for pair in clean["pairs"])
    main_clean = robustness["main"]["candidates"]
    assert (main_clean["colmap_pinhole_all"]["full"]
            ["median_reprojection_px"] >
            main_clean["naive"]["full"]["median_reprojection_px"])

    for scene, folder, receipt_path in [
        ("main", ROOT / "data/dobbe_oxe/target_raw",
         ROOT / "runs/plex_dobbe_preflight/dobbe_3651_rgbd_receipt.json"),
        ("second", ROOT / "data/dobbe_oxe/second_raw",
         OUT / "second_download_receipt.json"),
    ]:
        receipt = load(receipt_path)
        checks = []
        for member in receipt["members"]:
            target = folder / Path(member["name"]).name
            assert target.stat().st_size == member["uncompressed_bytes"]
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            assert digest == member["sha256"]
            checks.append(target.name)
        audit["download_sha256"][scene] = checks

    decision = load(OUT / "decision.json")
    assert decision["status"] == "BLOCKED" and not decision["molmo_ready"]
    approximate = OUT / "approx_history"
    assert decision["points_3d_history_created"] == (
        approximate / "points_3d_history_candidate.npy").exists()
    assert decision["molmomotion_invoked"] == (
        approximate / "model_run.json").exists()
    assert decision["future_depth_opened_for_gt"] == (
        approximate / "future_evaluation.json").exists()
    assert decision["future_prediction_metrics_computed"] == (
        approximate / "future_evaluation.json").exists()
    assert not decision["validated_points_3d_history_created"]
    assert not decision["validated_full_3d_gt_and_metrics"]
    audit["decision"] = {"status": decision["status"],
                         "molmo_ready": decision["molmo_ready"]}

    reports = [ROOT / "report/README.md",
               ROOT / "report/dobbe_rgbd_colmap_study.md",
               ROOT / "report/dobbe_approx_colmap_f2nerf_molmo.md"]
    local = [(report, link) for report in reports
             for link in re.findall(r"\]\(([^)]+)\)",
                                    report.read_text(encoding="utf-8"))
             if not link.startswith(("https://", "http://", "#"))]
    missing = [f"{report.name}: {link}" for report, link in local
               if not (report.parent / link).exists()]
    assert not missing, missing
    audit["report_links"] = {"local_checked": len(local), "missing": 0}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n",
                                    encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
