"""Freeze sensor-supported common queries before depth-only A inference."""
import importlib.util
import shutil
import cv2
import numpy as np

from dobbe_vipe_v1 import RUN, VIPE, GATE, read, write, freeze, sha
from dobbe_vipe_geometry import lift, sample_depth, rigid_error, select_spread, independent_sift_geometry
from dobbe_vipe_inference import save_variant


def prepare_pair():
    dest = RUN / "A/sift_audited"
    audit_path = dest / "paired_selection_audit.json"
    if audit_path.exists():
        for name, digest in read(audit_path)["paired_input_sha256"].items():
            if sha(dest / name) != digest:
                raise RuntimeError("Frozen paired input changed")
        assert not read(dest / "hybrid/geometry_gate.json")["molmo_ready"], "Unsafe original sensor queries must remain rejected"
        return
    for variant in ["paired_vipe", "paired_hybrid"]:
        assert not (dest / variant / "input_freeze.json").exists(), "Pair must be prepared before either prediction"
    freeze(dest / "paired_selection_protocol.json", {
        "future_used": False, "selection_uses": "observed H3 and original 100 object queries only",
        "rule": "intersection of author ViPE filter, valid H3 depth, sensor patch spread <10% Z, and sensor Z within [0.5,1.5] times H3 object-query median Z in each frame; deterministic quality+3D farthest sampling",
        "sensor_band": [0.5, 1.5], "same_numeric_geometry_gate": GATE,
        "reason": "Two original queries sampled background/mixed sensor depth near the cup rim; choose one common supported pool for both depths before either paired prediction."})
    g = np.load(dest / "geometry_all.npz")
    tr = np.load(dest / "tracks.npz")
    h = read(dest / "protocol.json")["history_vipe_indices"]
    n = int(tr["object_count"])
    measured = np.stack([cv2.resize(d, (256, 256), interpolation=cv2.INTER_NEAREST_EXACT)
                        for d in np.load(dest / "measured_depth_prefix.npy")])
    world, valid, spread = lift(tr["tracks"], measured, g["intrinsics"], g["poses"], tr["visibility"])
    sensor_z = np.stack([sample_depth(measured[t], tr["tracks"][t, :n])[0] for t in h])
    median_z = np.nanmedian(sensor_z, axis=1)
    sensor_ok = valid[h, :n].all(0) & ((sensor_z >= .5*median_z[:, None]) & (sensor_z <= 1.5*median_z[:, None])).all(0)
    sensor_ok &= (spread[h, :n] < .1*sensor_z).all(0)
    spec = importlib.util.spec_from_file_location("author_pair_filter", VIPE / "track-filter-smooth.py")
    filt = importlib.util.module_from_spec(spec); spec.loader.exec_module(filt)
    dropped = filt.filter_tracks_by_trust(g["trust"], g["valid"][:, :n], z_thresh=1.5)
    vipe_spread = np.stack([sample_depth(g["depths"][t], tr["tracks"][t, :n])[1] for t in h])
    ray_distance = np.linalg.norm(g["points_world"][h, :n]-g["poses"][h, None, :3, 3], axis=-1)
    good = sensor_ok & ~dropped & g["valid"][h, :n].all(0) & (vipe_spread < .1*ray_distance).all(0)
    candidates = np.flatnonzero(good)
    if len(candidates) < 8:
        raise RuntimeError("Insufficient jointly supported candidates")
    quality = g["trust"][h].mean(0)*tr["confidence"][h, :n].mean(0)
    ids = select_spread(g["object_smoothed"][-1], candidates, quality)
    for variant in ["pure_vipe_unsmoothed", "hybrid"]:
        archive = dest / "rejected_sensor_queries" / variant
        archive.mkdir(parents=True, exist_ok=True)
        for name in ["geometry_gate.json", "points_3d_world.npy", "points_2d_t0.npy", "c2w_t0.npy"]:
            shutil.copy2(dest / variant / name, archive / name)
    old = read(dest / "rejected_sensor_queries/hybrid/geometry_gate.json")
    old["checks"]["measured_object_queries_supported"] = False
    old.update(molmo_ready=False, status="REJECT_ESTIMATED_GEOMETRY",
               reason="Original selected cup rim queries include background/mixed measured depth")
    write(dest / "rejected_sensor_queries/hybrid/geometry_gate.json", old)
    assert not (dest / "hybrid/input_freeze.json").exists(), "Reject unsupported original sensor queries before inference"
    write(dest / "hybrid/geometry_gate.json", old)
    source = read(dest / "geometry_gate.json")
    static_hybrid = independent_sift_geometry(dest, measured, g["intrinsics"], g["poses"])
    for variant, history, static in [
        ("paired_vipe", g["points_world"][h][:, ids], source["static_cross_frame_px"]),
        ("paired_hybrid", world[h][:, ids], static_hybrid)]:
        rigidity = rigid_error(history)
        checks = dict(source["checks"])
        checks.update(complete_history_eight=bool(np.isfinite(history).all()),
            rigid_history=rigidity["relative_to_t0_median_pair_distance"]["median"] <= GATE["rigid_relative_median_max"],
            static_median=static["median"] <= GATE["static_median_px_max"],
            static_p90=static["p90"] <= GATE["static_p90_px_max"],
            measured_object_queries_supported=bool(sensor_ok[ids].all()))
        if variant == "paired_hybrid":
            checks["rgb_depth_alignment_reviewed"] = read(dest / "hybrid_alignment_review.json")["spatial_alignment_supported"]
        save_variant(dest, variant, history, tr["tracks"][-1, ids], g["poses"][-1], checks,
            {"selected_ids": ids.tolist(), "smoothing": False,
             "depth_source": "ViPE" if variant == "paired_vipe" else "measured DobbE optical-Z hypothesis",
             "rigid_history_unsmoothed": rigidity, "static_cross_frame_px": static,
             "paired_selection_protocol_sha256": sha(dest / "paired_selection_protocol.json"),
             "selection_scope": "one common sensor-supported pool for both sources; differs from main pure_vipe eight",
             "translation_scale": "ViPE retained unchanged"})
    original_ids = source["selected_ids"]
    disagreement = np.linalg.norm(g["points_world"][h][:, ids]-world[h][:, ids], axis=-1)
    write(audit_path, {"future_used": False, "eligible_count": len(candidates), "selected_ids": ids.tolist(),
        "sensor_object_median_Z": median_z.tolist(),
        "original_unsupported_ids": [i for i in original_ids if not sensor_ok[i]],
        "original_selected_measured_Z_t0": sensor_z[-1, original_ids].tolist(),
        "paired_measured_Z_t0": sensor_z[-1, ids].tolist(),
        "paired_world_xyz_disagreement_estimated_units": {"median": float(np.median(disagreement)), "p90": float(np.percentile(disagreement, 90))},
        "paired_input_sha256": {f"{v}/{name}": sha(dest / v / name) for v in ["paired_vipe", "paired_hybrid"]
                               for name in ["points_3d_world.npy", "points_2d_t0.npy", "c2w_t0.npy", "geometry_gate.json"]}})
    if (dest / "ablation_input_audit.json").exists():
        shutil.copy2(dest / "ablation_input_audit.json", dest / "ablation_input_audit_initial.json")
    write(dest / "ablation_input_audit.json", {"future_used": False,
        "same_point_ids": ids.tolist(), "same_uv": True, "same_c2w": True, "both_unsmoothed": True,
        "selection": "paired_selection_audit.json", "initial_unsafe_pair_preserved": "rejected_sensor_queries"})
    print(read(audit_path), flush=True)


if __name__ == "__main__":
    prepare_pair()
