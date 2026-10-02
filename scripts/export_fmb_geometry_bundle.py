"""Export comparable, explicitly sparse/dense FMB history geometry records."""

from __future__ import annotations

import json

import numpy as np

from run_fmb_ablation_suite import ROOT, RUNS, digest, verify_frozen, write_json


METHODS = {
    "sensor": "history_sensor_3d.npy",
    "planar_pnp": "history_pnp_3d.npy",
    "cad_silhouette_tcp": "history_cad_silhouette_tcp_3d.npy",
    "moge2_raw": "history_moge2_raw_3d.npy",
    "moge2_history_scaled": "history_moge2_history_scaled_3d.npy",
}


def main() -> None:
    for name, run in RUNS.items():
        verify_frozen(run)
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        source_path = ROOT.parent / "data/fmb/single_object_manipulation_dataset" / config["source_file"]
        if digest(source_path) != config["source_sha256"]:
            raise RuntimeError("Source file SHA-256 mismatch")
        source = np.load(source_path, allow_pickle=True).item()
        history_steps = config["history_steps"]
        nominal_k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        uv = np.load(run / "history_points_2d.npy")
        dest = run / "geometry_bundle_v1"
        dest.mkdir(parents=True, exist_ok=True)
        records = {}
        for method, filename in METHODS.items():
            path = run / filename
            if not path.exists():
                continue
            xyz = np.load(path).astype("float32")
            if xyz.shape != (3, 8, 3) or not np.isfinite(xyz).all() or not (xyz[..., 2] > 0).all():
                raise ValueError(f"Invalid XYZ: {path}")
            arrays = {
                "history_steps": np.asarray(history_steps, dtype="int32"),
                "selected_uv_px": uv.astype("float32"),
                "selected_XYZ_m": xyz,
                "selected_depth_m": xyz[..., 2],
                "selected_valid_mask": np.ones((3, 8), dtype=bool),
                "projection_K_px": nominal_k.astype("float32"),
            }
            dense_kind = "none"
            if method == "sensor":
                raw = np.stack([source["obs/side_1_depth"][s] for s in history_steps])
                arrays["dense_depth_m"] = raw.astype("float32") * 1e-4
                arrays["dense_valid_mask"] = raw > 0
                dense_kind = "sensor Z16, 0.0001 m/raw"
            if method.startswith("moge2"):
                maps_path = run / "moge2_history_v1/moge2_history_maps.npz"
                moge_manifest = json.loads((run / "moge2_history_v1/manifest.json").read_text(encoding="utf8"))
                scale = (moge_manifest["single_history_scale_to_sensor"]
                         if method == "moge2_history_scaled" else 1.0)
                with np.load(maps_path) as maps:
                    arrays["dense_depth_m"] = maps["depth"] * scale
                    arrays["dense_valid_mask"] = np.isfinite(maps["depth"]) & (maps["depth"] > 0)
                    arrays["model_output_K_px_given_candidate_FOV"] = maps["K_model"]
                    arrays["dense_points_model_m"] = maps["points"] * scale
                dense_kind = ("RGB-derived MoGe-2 depth, single H3 sensor-fitted scale"
                              if method == "moge2_history_scaled" else
                              "RGB-derived MoGe-2 depth, raw metric output")
            output = dest / f"{method}.npz"
            np.savez_compressed(output, **arrays)
            records[method] = {
                "history_file": filename, "history_file_sha256": digest(path),
                "bundle_file": output.name, "bundle_sha256": digest(output),
                "selected_XYZ_shape": list(xyz.shape),
                "dense_depth_kind": dense_kind,
                "dense_depth_available": "dense_depth_m" in arrays,
                "confidence_available": False,
                "K_status": "SUPPORTED_NOT_EXACT; only nominal projection K is used for this bundle",
                "method_caveat": (
                    "RGB/CAD and TCP derived point positions, no dense depth map" if method == "cad_silhouette_tcp" else
                    "Approximate planar RGB/CAD correspondences, no dense depth map" if method == "planar_pnp" else
                    "Scale fitted to historical sensor depth only" if method == "moge2_history_scaled" else
                    "RGB-only predicted depth conditioned on candidate FOV" if method == "moge2_raw" else
                    "RGB-depth registration of released 256x256 arrays unverified"
                ),
            }
        report = {"episode": name, "source_file": config["source_file"],
                  "source_sha256": config["source_sha256"],
                  "history_steps_only": history_steps, "future_read": False,
                  "common_selected_fields": ["history_steps", "selected_uv_px", "selected_XYZ_m",
                                             "selected_depth_m", "selected_valid_mask", "projection_K_px"],
                  "methods": records}
        write_json(dest / "manifest.json", report)
        print(name, list(records))


if __name__ == "__main__":
    main()
