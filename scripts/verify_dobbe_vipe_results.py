"""Verify sealed causal inputs and completed/explicitly rejected executions."""
import numpy as np
from dobbe_vipe_v1 import RUN, ROOT, GATE, read, sha, write


def verify():
    scene_checks, forecasts = {}, {}
    for scene in ["A", "B", "C"]:
        folder = RUN / scene
        p = read(folder / "protocol.json")
        ids = list(range(p["t0"] % 2, p["t0"]+1, 2))
        assert p["observed_raw_ids"] == ids and p["history_raw_ids"] == ids[-3:]
        assert p["future_raw_ids_evaluation_only"] == list(range(p["t0"]+2, p["t0"]+61, 2))
        assert p["gate_thresholds"] == GATE
        fm = read(folder / "frame_map.json")
        assert fm["vipe_to_raw"] == {str(i): raw for i, raw in enumerate(ids)}
        assert sha(folder / "causal_15hz.mp4") == fm["causal_video_sha256"]
        assert all(sha(folder / "frames" / f"{int(i):05d}.png") == digest
                   for i, digest in fm["frames_png_sha256"].items())
        g = read(folder / "geometry_gate.json")
        assert g["thresholds_frozen_before_vipe"] == GATE
        assert g["protocol_sha256"] == sha(folder / "protocol.json")
        assert g["tracks_sha256"] == sha(folder / "tracks.npz")
        assert read(folder / "vipe_no_vda_execution.json")["exit_code"] == 0
        assert len(g["anchor_ids"]) == 16 and len(g["selected_ids"]) == 8
        assert g["molmo_ready"] == (scene == "C")
        scene_checks[scene] = {"observed_frames": len(ids), "history_raw_ids": ids[-3:],
                               "original_gate": g["status"], "hashes_valid": True}
    for name in ["C/pure_vipe", *[f"A/sift_audited/{v}" for v in [
            "smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "paired_vipe", "paired_hybrid"]]]:
        folder = RUN / name
        source = folder.parent
        receipt = read(folder / "model_run.json")
        assert receipt["status"] == "COMPLETE" and receipt["success"]
        f = read(folder / "input_freeze.json")
        assert sha(folder / "input_freeze.json") == receipt["input_freeze_sha256"]
        assert sha(source / "protocol.json") == f["protocol_sha256"]
        assert all(sha(source / file) == digest for file, digest in f["input_sha256"].items())
        assert sha(folder / "prediction_15hz.npy") == receipt["prediction_sha256"]
        prediction = np.load(folder / "prediction_15hz.npy")
        assert prediction.shape == (8, 30, 3) and np.isfinite(prediction).all()
        assert np.load(folder / "prediction_parsed_visibility.npy").all()
        assert read(folder / "geometry_gate.json")["molmo_ready"]
        assert all(read(folder / "processor_equivalence.json")["tensor_equal"].values())
        evaluation = read(folder / "evaluation/future_access_receipt.json")
        assert evaluation["prediction_sha256_before_future_access"] == receipt["prediction_sha256"]
        assert evaluation["model_input_freeze_sha256"] == sha(folder / "input_freeze.json")
        assert evaluation["raw_ids"] == [f["history_raw_ids"][-1], *read(source / "protocol.json")["future_raw_ids_evaluation_only"]]
        forecasts[name] = {"prediction_shape": list(prediction.shape), "parsed_point_frames": 240,
                           "sealed_inputs_valid": True, "evaluation_only_future": True}
    paired = RUN / "A/sift_audited"
    for file, digest in read(paired / "paired_selection_audit.json")["paired_input_sha256"].items():
        assert sha(paired / file) == digest
    assert np.array_equal(np.load(paired / "paired_vipe/points_2d_t0.npy"), np.load(paired / "paired_hybrid/points_2d_t0.npy"))
    assert np.array_equal(np.load(paired / "paired_vipe/c2w_t0.npy"), np.load(paired / "paired_hybrid/c2w_t0.npy"))
    assert read(paired / "hybrid/model_run.json")["status"] == "SKIPPED_GEOMETRY_GATE"
    manifest = ROOT / "report/dobbe_vipe_results/manifest.json"
    if manifest.exists():
        for file, record in read(manifest)["artifacts"].items():
            assert sha(manifest.parent / file) == record["sha256"]
    result = {"success": True, "scene_checks": scene_checks, "forecasts": forecasts,
              "paired_depth_inputs_identical_except_3D": True, "bundle_hashes_valid": manifest.exists(),
              "metric_ground_truth": False}
    write(RUN / "verification.json", result)
    print("verified 3 scenes, 6 full forecasts, unchanged input hashes, causal timing, depth-only pair", flush=True)


if __name__ == "__main__":
    verify()
