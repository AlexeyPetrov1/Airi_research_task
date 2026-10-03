"""Collect the small reviewable experiment artifacts, without raw data or weights."""
import shutil

from dobbe_vipe_v1 import ROOT, RUN, read, write, sha


def collect():
    dest = ROOT / "report/dobbe_vipe_results"
    copied = {}
    for scene in ["A", "B", "C"]:
        folders = [RUN / scene]
        folders += [RUN / scene / branch for branch in ["default", "rectified", "sift_audited"]
                    if (RUN / scene / branch / "geometry_gate.json").exists()]
        for folder in folders:
            names = ["protocol.json", "frame_map.json", "mask_annotation.json", "mask_t0.png",
                "query_overlay.png", "query_points.npz", "geometry_gate.json", "geometry_diagnostics.png",
                "camera_diagnostics.png", "causal_tracks.mp4", "independent_static_sift.json",
                "hybrid_alignment_review.json", "alltracker_execution.json", "control_protocol.json", "ablation_input_audit.json",
                "paired_selection_protocol.json", "paired_selection_audit.json", "ablation_input_audit_initial.json",
                "paired_depth_comparison.png", "paired_forecast_summary.json",
                "vipe_no_vda_execution.json", "vipe_default_execution.json",
                "vipe_no_vda_gdown6_failure_execution.json", "protocol_action_correction.json",
                "vipe/pose/causal_15hz.npz", "vipe/intrinsics/causal_15hz.npz",
                "vipe/intrinsics/causal_15hz_camera.txt"]
            p = read(folder / "protocol.json")
            names += [f"frames/{i:05d}.png" for i in p["history_vipe_indices"]]
            names += [f"measured_depth_edges_{i:05d}.png" for i in p["history_vipe_indices"]]
            names += [path.name for path in sorted(folder.glob("static_sift_*.npz"))]
            names += [str(path.relative_to(folder)) for path in sorted(folder.glob("rejected_sensor_queries/*/*")) if path.is_file()]
            for variant in ["smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "hybrid", "paired_vipe", "paired_hybrid"]:
                names += [f"{variant}/{name}" for name in [
                    "geometry_gate.json", "points_3d_world.npy", "points_2d_t0.npy", "c2w_t0.npy",
                    "points_3d_camera_t0_diagnostic.npy", "processor_equivalence.json", "model_run.json",
                    "input_freeze.json", "prediction_15hz.npy", "prediction_parsed_visibility.npy",
                    "raw_model_output.txt", "forecast_3d.png",
                    "evaluation/metrics.json", "evaluation/future_access_receipt.json",
                    "evaluation/alltracker_execution.json", "evaluation/observed_future_tracks.npz",
                    "evaluation/conditional_projection.npz", "evaluation/future_comparison.png",
                    "evaluation/conditional_error_over_time.png", "evaluation/future_overlay.mp4"]]
            for name in names:
                source = folder / name
                if not source.exists():
                    continue
                target = dest / source.relative_to(RUN)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                assert sha(source) == sha(target)
                copied[str(target.relative_to(dest))] = {"sha256": sha(target),
                    "bytes": target.stat().st_size, "source": str(source.relative_to(ROOT))}
    for name in ["environment.freeze.txt", "implementation_checks.json", "verification.json"]:
        if (RUN / name).exists():
            shutil.copy2(RUN / name, dest / name)
            copied[name] = {"sha256": sha(dest / name), "bytes": (dest / name).stat().st_size,
                            "source": str((RUN / name).relative_to(ROOT))}
    write(dest / "manifest.json", {"scope": "Local review bundle; no external publication performed",
        "metric_ground_truth": False, "artifacts": copied,
        "excluded": ["raw RGB-D recordings", "model checkpoints", "full per-frame ViPE depth", "full console logs"],
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
                        sorted([*ROOT.glob("scripts/*dobbe_vipe*"), ROOT / "tests/test_dobbe_vipe_v1.py"])
                        if path.is_file()},
        "author_source_sha256": {name: sha(ROOT / name) for name in [
            "src/molmo_motion/processor.py", "src/molmo_motion/modeling.py",
            "data_generation/third_party/vipe/track-filter-smooth.py",
            "data_generation/third_party/alltracker/nets/alltracker.py",
            "data_generation/third_party/alltracker/nets/blocks.py"]}})
    print(len(copied), "artifacts; bytes", sum(v["bytes"] for v in copied.values()), flush=True)


if __name__ == "__main__":
    collect()
