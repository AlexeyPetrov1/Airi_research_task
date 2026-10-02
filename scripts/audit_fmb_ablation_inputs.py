"""Check the frozen input invariants for the paired FMB ablations."""

from __future__ import annotations

import json

import numpy as np

from run_fmb_ablation_suite import RUNS, SUITE, VARIANTS, digest, verify_frozen, write_json


def main() -> None:
    for name, run in RUNS.items():
        verify_frozen(run)
        base = np.load(run / "history_sensor_3d.npy")
        history_uv = np.load(run / "history_points_2d.npy")
        t0_uv = np.load(run / "points_2d_at_t0.npy")
        config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
        results = {}
        for variant in VARIANTS:
            dest = run / SUITE / variant
            xyz_path, uv_path = dest / "history_3d.npy", dest / "points_2d_at_t0.npy"
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf8"))
            xyz, uv = np.load(xyz_path), np.load(uv_path)
            checks = {
                "shape": xyz.shape == (3, 8, 3) and uv.shape == (8, 2),
                "finite_positive": bool(np.isfinite(xyz).all() and (xyz[..., 2] > 0).all()),
                "input_hashes": digest(xyz_path) == manifest["history_3d_sha256"] and
                digest(uv_path) == manifest["points_2d_at_t0_sha256"],
                "same_episode_t0": manifest["t0"] == config["t0"],
                "same_source_hash": manifest["source_sha256"] == config["source_sha256"],
                "frozen_historical_frames": [digest(run / p) for p in manifest["frames"]] == manifest["frame_sha256"],
            }
            if variant.startswith("scale"):
                factor = .9 if variant.endswith("090") else 1.1
                checks["all_XYZ_scaled_together"] = bool(np.allclose(xyz, base * factor, atol=1e-7))
                checks["same_t0_2D"] = bool(np.array_equal(uv, t0_uv))
            if variant.startswith("focal"):
                k = np.asarray(manifest["projection_K"])
                reproj = np.stack((k[0, 0] * xyz[..., 0] / xyz[..., 2] + k[0, 2],
                                   k[1, 1] * xyz[..., 1] / xyz[..., 2] + k[1, 2]), axis=-1)
                checks["reprojects_to_original_history_2D"] = bool(np.max(np.abs(reproj-history_uv)) < 1e-4)
                checks["sensor_depth_unchanged"] = bool(np.array_equal(xyz[..., 2], base[..., 2]))
            if variant.startswith("perm"):
                order = manifest["point_order_model_to_original"]
                checks["permutation_consistent_XYZ_2D"] = bool(np.allclose(xyz, base[:, order]) and
                                                                  np.array_equal(uv, t0_uv[order]))
                checks["first_anchor_condition"] = (order[0] == 0) if variant == "perm_keep_anchor" else (order[0] != 0)
            if variant == "no_history":
                checks["XYZ_repeated"] = bool(np.array_equal(xyz[0], xyz[1]) and np.array_equal(xyz[1], xyz[2]))
                checks["RGB_repeated"] = len(set(manifest["frame_sha256"])) == 1
            if variant.startswith("text"):
                checks["history_identical"] = bool(np.array_equal(xyz, base))
                checks["text_changed"] = manifest["action"] != config["action_text_exact_passed_to_model"]
            if variant.startswith("moge2"):
                moge = json.loads((run / "moge2_history_v1/manifest.json").read_text(encoding="utf8"))
                raw = np.load(run / "history_moge2_raw_3d.npy")
                scaled = np.load(run / "history_moge2_history_scaled_3d.npy")
                source = scaled if variant == "moge2_scaled" else raw
                checks["moge_source_hash"] = (
                    digest(run / ("history_moge2_history_scaled_3d.npy" if variant == "moge2_scaled"
                                  else "history_moge2_raw_3d.npy")) ==
                    (moge["scaled_xyz_sha256"] if variant == "moge2_scaled" else moge["raw_xyz_sha256"])
                )
                checks["source_depth_unchanged"] = bool(np.array_equal(xyz[..., 2], source[..., 2]))
                checks["same_t0_2D"] = bool(np.array_equal(uv, t0_uv))
                k = np.asarray(manifest["projection_K"])
                reproj = np.stack((k[0, 0] * xyz[..., 0] / xyz[..., 2] + k[0, 2],
                                   k[1, 1] * xyz[..., 1] / xyz[..., 2] + k[1, 2]), axis=-1)
                checks["reprojects_to_original_history_2D"] = bool(np.max(np.abs(reproj-history_uv)) < 1e-4)
                if variant == "moge2_scaled":
                    checks["single_H3_scale_all_XYZ"] = bool(np.allclose(
                        scaled, raw * moge["single_history_scale_to_sensor"], atol=1e-7))
                if variant == "moge2_focal_090":
                    checks["alternative_focal_prescribed"] = bool(np.allclose(
                        [k[0, 0], k[1, 1]],
                        np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])[[0, 1], [0, 1]] * .9))
                else:
                    checks["XYZ_matches_frozen_source"] = bool(np.array_equal(xyz, source))
            results[variant] = checks
        status = "PASS" if all(all(v.values()) for v in results.values()) else "FAIL"
        write_json(run / SUITE / "input_audit.json", {"episode": name, "status": status, "checks": results})
        print(name, status, len(results), "variants")
        if status != "PASS":
            raise RuntimeError(f"Input audit failed: {name}")


if __name__ == "__main__":
    main()
