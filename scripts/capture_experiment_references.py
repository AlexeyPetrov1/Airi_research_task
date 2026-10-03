"""Copy immutable legacy inputs/results before changing their shared pipeline.

This captures actual saved experiments, rather than fabricating a reference by
running the new implementation. Original inference scripts write into completed
runs, so their entry points must not be rerun there. Regression reruns their pure
functions; new inference and rendering always use a separate output directory.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            digest.update(part)
    return digest.hexdigest()


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.source_repo.resolve()
    if (GOLDEN / "manifest.json").exists():
        raise FileExistsError("The captured legacy reference cannot be overwritten.")
    paths = set()
    def add(pattern):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    for pattern in (
        "runs/author_davis_bmx_trees_f30/*", "runs/author_davis_bmx_trees_f30/inputs/*",
        "examples/data/davis_bmx_trees/*",
        "runs/fmb_wrist_v4_improvement/*.json",
        "runs/fmb_wrist_v4_improvement/wrist_*/*.npy", "runs/fmb_wrist_v4_improvement/wrist_*/*.npz",
        "runs/fmb_wrist_v4_improvement/wrist_*/metrics.json",
        "runs/fmb_wrist_v4_improvement/wrist_*/variants/*/*.npy",
        "runs/fmb_wrist_v4_improvement/wrist_*/variants/*/*.json",
        "runs/fmb_wrist_v4_improvement/wrist_*/variants/*/group_*/*",
        "runs/fmb_wrist_v4_improvement/wrist_*/viz/forecast_comparison.mp4",
        "runs/fmb_wrist_v3/wrist_*/metadata.json", "runs/fmb_wrist_v3/wrist_*/observed/history_rgb.npy",
        "runs/fmb_wrist_v3/wrist_*/observed/points_2d_history.npy",
        "runs/fmb_wrist_v3/wrist_*/branches/C_sensor_tcp_official/observed/points_3d_history.npy",
        "runs/fmb_wrist_v3/wrist_*/branches/C_sensor_tcp_official/groups/group_*/*",
        "runs/fmb_wrist_v3/wrist_*/branches/C_sensor_tcp_official/predictions/**/*",
        "runs/fmb_wrist_v3/wrist_*/evaluation/future_reference.npz",
        "runs/fmb_wrist_v3/wrist_*/evaluation/future_rgb.npy",
        "runs/berkeley_ur5_molmomotion/cup/metadata.json",
        "runs/berkeley_ur5_molmomotion/*/observed/history_rgb.npy",
        "runs/berkeley_ur5_molmomotion/*/observed/points_2d_history.npy",
        "runs/berkeley_ur5_molmomotion/*/observed/history_timestamps.npy",
        "runs/berkeley_ur5_molmomotion/*/observed/selected_point_ids.npy",
        "runs/berkeley_ur5_molmomotion/*/observed/points_3d_history.npy",
        "runs/berkeley_ur5_molmomotion/*/geometry/K_median.npy",
        "runs/berkeley_ur5_molmomotion/*/evaluation/evaluation_results.npz",
        "runs/berkeley_ur5_molmomotion/*/evaluation/ground_truth_3d_est.npz",
        "runs/berkeley_ur5_molmomotion/*/evaluation/future_rgb.npy",
        "runs/berkeley_ur5_molmomotion/cup/predictions/**/*",
        "runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle/metadata.json",
        "runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle/observed/*.npy",
        "runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle/geometry/K_median.npy",
        "runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle/predictions/**/*",
        "runs/berkeley_ur5_improvement_v1/CASE-AUGE/bottle/groups/**/*",
        "runs/berkeley_ur5_improvement_v1/cup/fixed_arrays.npz",
        "runs/berkeley_ur5_improvement_v1/cup/fixed_comparison.mp4",
        "runs/berkeley_ur5_improvement_v1/fixed_summary.json",
        "runs/berkeley_ur5_arc_expansion_v1/bottle/*.npz",
        "runs/berkeley_ur5_arc_expansion_v1/phase_schedule/*.json",
        "runs/berkeley_ur5_arc_expansion_v1/phase_schedule/bottle/*.npz",
        "runs/berkeley_ur5_arc_expansion_v1/phase_schedule/bottle/timing_group_02.mp4",
        "runs/dobbe_vipe_v1/C/protocol.json", "runs/dobbe_vipe_v1/C/geometry_all.npz",
        "runs/dobbe_vipe_v1/C/frames/0002[89].png", "runs/dobbe_vipe_v1/C/frames/00030.png",
        "runs/dobbe_vipe_v1/C/pure_vipe/*", "runs/dobbe_vipe_v1/C/pure_vipe/evaluation/*",
        "runs/dobbe_vipe_v1/C/hybrid/*",
    ):
        add(pattern)
    source_hashes = {}
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        dest = GOLDEN / "source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_hashes[relative.as_posix()] = sha(path)
    code = json.loads((Path(__file__).resolve().parents[1] / "docs/legacy_inventory.json").read_text(encoding="utf8"))
    manifest = dict(reference_kind="actual saved legacy artifacts captured before adaptation",
                    legacy_entrypoints_rerun=False,
                    reason="Completed legacy entry points overwrite old runs; pure functions and fresh inference are exercised separately.",
                    source_sha256=source_hashes,
                    legacy_script_sha256={r["path"]:r["sha256"] for r in code},
                    expected_dobbe_hybrid_status="SKIPPED_GEOMETRY_GATE",
                    expected_dobbe_hybrid_failure="static_median",
                    expected_dobbe_pure_status="COMPLETE", dobbe_metric_3d_ground_truth=False)
    (GOLDEN / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    (GOLDEN / ".gitattributes").write_text("source/** -text\n", encoding="utf8")
    print(f"Captured {len(paths)} files, {sum(p.stat().st_size for p in paths)/2**20:.1f} MiB.")


if __name__ == "__main__":
    main()
