"""Separate post-hoc RGB correspondence control; main AllTracker gate is preserved.

Matches were selected by RGB-only SIFT/fundamental consensus, before any A
forecast or future access. Only late-prefix pairs have enough matches; this
control cannot establish geometry quality over the entire observed prefix.
"""
import shutil
import numpy as np

from dobbe_vipe_v1 import RUN, GATE, read, write, freeze, sha
from dobbe_vipe_geometry import independent_sift_geometry
from dobbe_vipe_inference import export_variants


def prepare():
    source, dest = RUN / "A/rectified", RUN / "A/sift_audited"
    if not dest.exists():
        shutil.copytree(source, dest)
        # Copies contain rejected receipts, but no forecasts. Remove only these
        # named receipts so this control produces its own inference status.
        for variant in ["smoke_vipe", "pure_vipe", "pure_vipe_unsmoothed", "hybrid"]:
            receipt = dest / variant / "model_run.json"
            if receipt.exists():
                assert read(receipt)["status"] == "SKIPPED_GEOMETRY_GATE"
                receipt.unlink()
    if (dest / "control_protocol.json").exists():
        return
    assert not list(dest.glob("*/prediction_15hz.npy"))
    protocol = read(source / "protocol.json")
    protocol.update(geometry_branch="sift_audited",
        source_rectified_protocol_sha256=sha(source / "protocol.json"),
        validation_control="post-hoc independent RGB-only SIFT; same numeric thresholds; late-prefix coverage only")
    write(dest / "protocol.json", protocol)
    g = np.load(source / "geometry_all.npz")
    static = independent_sift_geometry(dest, g["depths"], g["intrinsics"], g["poses"])
    gate = read(source / "geometry_gate.json")
    gate.update(geometry_branch="sift_audited", static_validation="independent_sift",
        source_main_gate_sha256=sha(source / "geometry_gate.json"),
        source_main_gate_status=gate["status"],
        static_cross_frame_px=static, protocol_sha256=sha(dest / "protocol.json"),
        validation_scope="late observed prefix/H3 only; no reliable SIFT matches for early frames",
        selection_caveat="validation method introduced after seeing reconstruction audits; separate exploratory control")
    gate["checks"]["static_evidence"] = static["count"] >= GATE["static_observations_min"] and sum(p["count"] >= 5 for p in static["pairs"]) >= GATE["static_pairs_min"]
    gate["checks"]["static_median"] = static["median"] is not None and static["median"] <= GATE["static_median_px_max"]
    gate["checks"]["static_p90"] = static["p90"] is not None and static["p90"] <= GATE["static_p90_px_max"]
    gate["molmo_ready"] = all(gate["checks"].values())
    gate["status"] = "PASS_ESTIMATED_GEOMETRY" if gate["molmo_ready"] else "REJECT_ESTIMATED_GEOMETRY"
    write(dest / "geometry_gate.json", gate)
    freeze(dest / "control_protocol.json", {
        "future_used": False, "scene": "A", "method_selected_before_any_A_prediction": True,
        "source_main_gate_retained": "A/rectified/geometry_gate.json",
        "geometry_gate_sha256": sha(dest / "geometry_gate.json"),
        "numeric_thresholds_unchanged": GATE,
        "match_npz_sha256": {p.name: sha(p) for p in sorted(dest.glob("static_sift_*.npz"))},
        "reason": "Background AllTracker and independent RGB correspondences disagree; explore validation-method sensitivity without changing original decisions or observing A future."})
    export_variants("A", "sift_audited")
    print(gate["status"], static, flush=True)
    print("hybrid", read(dest / "hybrid/geometry_gate.json"), flush=True)


if __name__ == "__main__":
    prepare()
