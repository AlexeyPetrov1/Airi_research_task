"""Frozen, paired FMB ablations. Run in the Linux GPU environment.

Preparation reads only the three historical frames and frozen inputs. Evaluation
reads the future annotations only after inference has saved the raw response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "first": ROOT / "runs/sharerobot_fmb_episode_5201/quantitative_2d_sensor_t126",
    "second": ROOT / "runs/fmb_second_example_1_M_L_3_vertical_n_3/quantitative_2d_sensor_t130",
}
CHECKPOINT = ROOT / "data/checkpoints/MolmoMotion-4B-H3-F30"
MODEL_REVISION = "3f5e790a511ff2cdf21c8d2a14cb4d8409c94629"
SUITE = "ablation_suite_v1"
CORE_VARIANTS = (
    "text_paraphrase", "text_generic", "text_counterfactual", "no_history",
    "scale_090", "scale_110", "perm_keep_anchor", "perm_new_anchor",
    "focal_090", "focal_110",
)
MOGE_VARIANTS = ("moge2_raw", "moge2_scaled", "moge2_focal_090")
VARIANTS = CORE_VARIANTS + MOGE_VARIANTS
# The original point 0 stays first for the first permutation. The second
# permutation changes the processor's point-0 anchor.
PERM_KEEP = np.array([0, 2, 4, 6, 1, 3, 5, 7])
PERM_NEW = np.array([4, 0, 6, 2, 7, 1, 5, 3])
TEXT = {
    "text_paraphrase": "Place the red rectangular piece into the matching opening in the blue board.",
    "text_generic": "Move the object to complete the task.",
    "text_counterfactual": "Move the red rectangular piece away from the blue board.",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf8")


def verify_frozen(run: Path) -> None:
    frozen = json.loads((run / "input_freeze.json").read_text(encoding="utf8"))
    for name, expected in frozen["sha256"].items():
        if digest(run / name) != expected:
            raise RuntimeError(f"Frozen input changed: {run / name}")
    preflight = json.loads((run / "preflight.json").read_text(encoding="utf8"))
    if preflight["decision"] != "PASS_WITH_CALIBRATION_UNCERTAINTY":
        raise RuntimeError(f"Preflight failed: {run}")


def make_variant(run: Path, variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(variant)
    verify_frozen(run)
    config = json.loads((run / "experiment_config.json").read_text(encoding="utf8"))
    xyz = np.load(run / "history_sensor_3d.npy").astype(np.float64)
    uv = np.load(run / "points_2d_at_t0.npy").astype(np.float64)
    hist_uv = np.load(run / "history_points_2d.npy").astype(np.float64)
    k = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"], dtype=float)
    frames = [run / f"frame_{step}.png" for step in config["history_steps"]]
    order = np.arange(8)
    action = config["action_text_exact_passed_to_model"]
    projection_k = k.copy()
    change = variant
    geometry_source = "FMB sensor history"
    geometry_source_sha256 = digest(run / "history_sensor_3d.npy")
    if variant in TEXT:
        action = TEXT[variant]
    elif variant == "no_history":
        xyz = np.repeat(xyz[-1:, :, :], 3, axis=0)
        frames = [frames[-1]] * 3
    elif variant in ("scale_090", "scale_110"):
        xyz = xyz * (0.9 if variant.endswith("090") else 1.1)
    elif variant in ("perm_keep_anchor", "perm_new_anchor"):
        order = PERM_KEEP if variant == "perm_keep_anchor" else PERM_NEW
        xyz, uv = xyz[:, order], uv[order]
    elif variant in ("focal_090", "focal_110"):
        factor = 0.9 if variant.endswith("090") else 1.1
        projection_k[0, 0] *= factor
        projection_k[1, 1] *= factor
        # Preserve measured Z and 2D observations; rebuild X,Y from the
        # altered intrinsics. This is a controlled K hypothesis, not an
        # estimated calibration uncertainty.
        z = xyz[..., 2].copy()
        xyz[..., 0] = (hist_uv[..., 0] - projection_k[0, 2]) * z / projection_k[0, 0]
        xyz[..., 1] = (hist_uv[..., 1] - projection_k[1, 2]) * z / projection_k[1, 1]
    elif variant in MOGE_VARIANTS:
        moge_dir = run / "moge2_history_v1"
        moge_manifest_path = moge_dir / "manifest.json"
        moge_manifest = json.loads(moge_manifest_path.read_text(encoding="utf8"))
        raw_path = run / "history_moge2_raw_3d.npy"
        scaled_path = run / "history_moge2_history_scaled_3d.npy"
        if (digest(raw_path) != moge_manifest["raw_xyz_sha256"] or
            digest(scaled_path) != moge_manifest["scaled_xyz_sha256"] or
            digest(moge_dir / "moge2_history_maps.npz") != moge_manifest["maps_sha256"]):
            raise RuntimeError("MoGe historical geometry changed after preparation")
        selected = scaled_path if variant == "moge2_scaled" else raw_path
        xyz = np.load(selected).astype(np.float64)
        geometry_source = "MoGe-2 ViT-L RGB depth; historical sensor scale" if variant == "moge2_scaled" else "MoGe-2 ViT-L RGB depth; raw"
        geometry_source_sha256 = digest(selected)
        if variant == "moge2_focal_090":
            projection_k[0, 0] *= .9
            projection_k[1, 1] *= .9
            z = xyz[..., 2].copy()
            xyz[..., 0] = (hist_uv[..., 0] - projection_k[0, 2]) * z / projection_k[0, 0]
            xyz[..., 1] = (hist_uv[..., 1] - projection_k[1, 2]) * z / projection_k[1, 1]
    else:
        raise ValueError(variant)
    if xyz.shape != (3, 8, 3) or uv.shape != (8, 2):
        raise ValueError("Unexpected input shape")
    if not np.isfinite(xyz).all() or not (xyz[..., 2] > 0).all():
        raise ValueError("Invalid historical XYZ")
    if set(order.tolist()) != set(range(8)):
        raise ValueError("Invalid point permutation")
    dest = run / SUITE / variant
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "manifest.json").exists():
        previous = json.loads((dest / "manifest.json").read_text(encoding="utf8"))
        if (previous.get("variant") != variant or previous.get("t0") != config["t0"] or
            digest(dest / "history_3d.npy") != previous["history_3d_sha256"] or
            digest(dest / "points_2d_at_t0.npy") != previous["points_2d_at_t0_sha256"] or
            not np.array_equal(np.load(dest / "history_3d.npy"), xyz.astype("float32")) or
            not np.array_equal(np.load(dest / "points_2d_at_t0.npy"), uv.astype("float32")) or
            [digest(p) for p in frames] != previous["frame_sha256"]):
            raise RuntimeError(f"Prepared variant differs from frozen source: {dest}")
        return previous
    np.save(dest / "history_3d.npy", xyz.astype("float32"))
    np.save(dest / "points_2d_at_t0.npy", uv.astype("float32"))
    manifest = {
        "suite": SUITE, "variant": variant, "change": change,
        "source_run": str(run.relative_to(ROOT)), "t0": config["t0"],
        "history_steps_original": config["history_steps"],
        "future_steps_evaluation_only": config["future_steps"],
        "source_file": config["source_file"], "source_sha256": config["source_sha256"],
        "frames": [str(p.relative_to(run)) for p in frames],
        "frame_sha256": [digest(p) for p in frames],
        "action": action, "point_order_model_to_original": order.tolist(),
        "projection_K": projection_k.tolist(),
        "geometry_source": geometry_source,
        "geometry_source_sha256": geometry_source_sha256,
        "K_status": "SUPPORTED_NOT_EXACT; focal changes are prescribed scenarios",
        "model_id": "allenai/MolmoMotion-4B-H3-F30",
        "model_revision": MODEL_REVISION, "seed": 0, "dtype": "bfloat16",
        "future_horizon_steps": 30, "model_hz_assumed": 15,
        "source_hz_nominal": 10,
        "input_only_uses_history": True,
        "history_3d_sha256": digest(dest / "history_3d.npy"),
        "points_2d_at_t0_sha256": digest(dest / "points_2d_at_t0.npy"),
    }
    write_json(dest / "manifest.json", manifest)
    return manifest


def prepare(episodes: list[str], variants: list[str]) -> None:
    for ep in episodes:
        for variant in variants:
            print(ep, variant, make_variant(RUNS[ep], variant)["history_3d_sha256"], flush=True)


def infer(episodes: list[str], variants: list[str]) -> None:
    import torch
    from PIL import Image
    from molmo_motion import MolmoMotion, MolmoMotionProcessor
    from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.manual_seed(0)
    processor = MolmoMotionProcessor.from_pretrained(str(CHECKPOINT))
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = MolmoMotion.from_pretrained(str(CHECKPOINT))
    finally:
        torch.set_default_dtype(old_dtype)
    model._internal = model._internal.cuda().eval()
    for ep in episodes:
        run = RUNS[ep]
        verify_frozen(run)
        for variant in variants:
            dest = run / SUITE / variant
            manifest_path = dest / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            if (dest / "prediction_15hz_model_order.npy").exists():
                completed = dest / "model_run.json"
                checked = dest / "parser_validation.json"
                if (completed.exists() and checked.exists() and
                    json.loads(completed.read_text(encoding="utf8")).get("success") and
                    json.loads(checked.read_text(encoding="utf8")).get("status") == "PASS" and
                    (dest / "prediction_15hz.npy").exists()):
                    print(ep, variant, "already complete", flush=True)
                    continue
                raise RuntimeError(f"Partial forecast artifacts require inspection: {dest}")
            xyz_path, uv_path = dest / "history_3d.npy", dest / "points_2d_at_t0.npy"
            if digest(xyz_path) != manifest["history_3d_sha256"] or digest(uv_path) != manifest["points_2d_at_t0_sha256"]:
                raise RuntimeError(f"Prepared input changed: {dest}")
            frames = [run / name for name in manifest["frames"]]
            if [digest(p) for p in frames] != manifest["frame_sha256"]:
                raise RuntimeError(f"Historical frame changed: {dest}")
            started = time.perf_counter()
            status = {"episode": ep, "variant": variant, "status": "RUNNING", "success": False,
                      "model_revision": MODEL_REVISION, "model_pt_sha256_previously_verified":
                      "506beccd01e9edd4d3ebf0bf88fec00b83530eec525571168187c7c4888ee205",
                      "checkpoint_config_sha256": digest(CHECKPOINT / "config.yaml"),
                      "manifest_sha256": digest(manifest_path), "torch_version": torch.__version__,
                      "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
            def save_stage(stage: str) -> None:
                status["stage"] = stage
                status["elapsed_seconds"] = round(time.perf_counter() - started, 2)
                status["peak_ram_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
                status["peak_cuda_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
                write_json(dest / "model_run.json", status)
                print(ep, variant, stage, status["elapsed_seconds"], flush=True)
            try:
                images = [Image.open(p).convert("RGB") for p in frames]
                xyz = np.load(xyz_path)
                uv = np.load(uv_path)
                batch = processor(history_frames=images,
                                  points_2d_at_t0=torch.from_numpy(uv),
                                  points_3d_history=torch.from_numpy(xyz),
                                  action=manifest["action"], future_horizon=30)
                status["processor_shapes"] = {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
                save_stage("predict")
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model.predict_trajectory(**batch)
                torch.cuda.synchronize()
                raw = output.future_text
                pred = output.future_3d.detach().cpu().numpy()
                (dest / "raw_model_output.txt").write_text(raw, encoding="utf8")
                np.save(dest / "prediction_15hz_model_order.npy", pred)
                status.update({"raw_output_sha256": digest(dest / "raw_model_output.txt"),
                               "prediction_shape": list(pred.shape),
                               "prediction_finite_pairs": int(np.isfinite(pred).all(axis=2).sum())
                               if pred.shape == (8, 30, 3) else 0,
                               "all_positive_Z": bool((pred[..., 2] > 0).all()),
                               "all_zero_tensor": bool(np.all(pred == 0))})
                parsed = parse_tracks_text(raw)
                if parsed is None:
                    raise ValueError("Raw text has no complete tracks block")
                delta, visible = tracks_to_array(parsed, num_points=8, num_frames=30, start_timestamp=3.0)
                reconstructed = np.asarray(delta) + xyz[-1, 0]
                discrepancy = float(np.max(np.abs(reconstructed - pred)))
                parser_check = {"visible_count": int(np.asarray(visible).sum()),
                                "max_abs_raw_vs_tensor_m": discrepancy,
                                "status": "PASS" if pred.shape == (8, 30, 3) and
                                np.isfinite(pred).all() and (pred[..., 2] > 0).all() and
                                not np.all(pred == 0) and np.asarray(visible).all() and
                                discrepancy < 1e-4 else "FAIL"}
                write_json(dest / "parser_validation.json", parser_check)
                if parser_check["status"] != "PASS":
                    raise ValueError(f"Parser validation failed: {parser_check}")
                order = np.asarray(manifest["point_order_model_to_original"])
                np.save(dest / "prediction_15hz.npy", pred[np.argsort(order)])
                status["success"] = True
                status["status"] = "COMPLETE"
                save_stage("complete")
            except Exception as exc:
                status["status"] = "FAILED"
                status["error"] = f"{type(exc).__name__}: {exc}"
                status["traceback"] = traceback.format_exc(limit=8)
                save_stage("failed")
                raise


def evaluate(episodes: list[str], variants: list[str]) -> None:
    # Reuse the primary metric definition and 15 -> 10 Hz interpolation.
    from evaluate_fmb_quantitative_2d import interpolate, measure, project
    all_rows = []
    nominal_by_episode = {}
    for ep in episodes:
        run = RUNS[ep]
        nominal_by_episode[ep] = json.loads((run / "metrics.json").read_text(encoding="utf8"))["metrics"]["MolmoMotion"]
        gt = np.load(run / "gt_2d.npy")
        mask = np.load(run / "validity_mask.npy")
        k_nominal = np.asarray(json.loads((run / "k_nominal.json").read_text(encoding="utf8"))["K_nominal"])
        nominal = np.load(run / "prediction_15hz.npy")
        nominal_uv = project(interpolate(nominal), k_nominal)
        for variant in variants:
            dest = run / SUITE / variant
            if not (dest / "prediction_15hz.npy").exists():
                continue
            status = json.loads((dest / "model_run.json").read_text(encoding="utf8"))
            parser = json.loads((dest / "parser_validation.json").read_text(encoding="utf8"))
            manifest = json.loads((dest / "manifest.json").read_text(encoding="utf8"))
            if not status["success"] or parser["status"] != "PASS":
                raise ValueError(f"Failed forecast cannot be scored: {dest}")
            pred = np.load(dest / "prediction_15hz.npy")
            pred10 = interpolate(pred)
            k = np.asarray(manifest["projection_K"])
            projected = project(pred10, k)
            result, per_pair_error = measure(projected, gt, mask)
            alternative = np.load(run / "gt_dense_flow_alternative.npy")
            alt_result, _ = measure(projected, alternative, mask)
            np.save(dest / "prediction_10hz.npy", pred10.astype("float32"))
            np.save(dest / "prediction_2d.npy", projected.astype("float32"))
            outside = ((projected[..., 0] < 0) | (projected[..., 0] >= 256) |
                       (projected[..., 1] < 0) | (projected[..., 1] >= 256))
            mean_change = float(np.linalg.norm(projected - nominal_uv, axis=2)[mask].mean())
            report = {"episode": ep, "variant": variant, "status": "EVALUATED",
                      "metrics": result, "alternative_dense_flow_metrics": alt_result,
                      "mean_forecast_2d_change_vs_nominal_px": mean_change,
                      "outside_image_pairs": int(outside.sum()),
                      "per_pair_error_px": per_pair_error.tolist(),
                      "evaluation_mask_sha256": digest(run / "validity_mask.npy"),
                      "gt_sha256": digest(run / "gt_2d.npy"),
                      "time_assumption": "F30 j/15 s interpolated to native FMB j/10 s",
                      "counterfactual_ADE_is_not_goal_success": variant == "text_counterfactual"}
            write_json(dest / "evaluation.json", report)
            all_rows.append({"episode": ep, "variant": variant,
                             "ADE_2D_px": result["ADE_2D_px"],
                             "FDE_2D_px": result["FDE_2D_px"],
                             "dense_flow_ADE_2D_px": alt_result["ADE_2D_px"],
                             "dense_flow_FDE_2D_px": alt_result["FDE_2D_px"],
                             "mean_forecast_change_px": mean_change,
                             "outside_image_pairs": int(outside.sum()),
                             "valid_pairs": result["valid_pairs"]})
    path = ROOT / "runs" / "fmb_ablation_suite_v1_summary.csv"
    with path.open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]) if all_rows else ["episode", "variant"])
        writer.writeheader()
        writer.writerows(all_rows)
    if all_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(len(episodes), 2, figsize=(18, 5.8 * len(episodes)), squeeze=False)
        labels = {"text_paraphrase": "Paraphrase", "text_generic": "Generic text",
                  "text_counterfactual": "Other goal*", "no_history": "No history",
                  "scale_090": "Scale 0.9", "scale_110": "Scale 1.1",
                  "perm_keep_anchor": "Reorder / same anchor",
                  "perm_new_anchor": "Reorder / new anchor",
                  "focal_090": "Focal 0.9", "focal_110": "Focal 1.1",
                  "moge2_raw": "MoGe-2 raw", "moge2_scaled": "MoGe-2 scaled",
                  "moge2_focal_090": "MoGe-2 / focal 0.9"}
        for row_index, ep in enumerate(episodes):
            available = {row["variant"]: row for row in all_rows if row["episode"] == ep}
            names = [v for v in variants if v in available]
            if not names:
                continue
            colors = ["#999999" if v == "text_counterfactual" else
                      "#4e79a7" if v.startswith("text") else
                      "#f28e2b" if v.startswith("focal") else
                      "#59a14f" for v in names]
            for col, metric in enumerate(("ADE_2D_px", "FDE_2D_px")):
                delta = [available[v][metric] - nominal_by_episode[ep][metric] for v in names]
                ax = axes[row_index, col]
                ax.bar(np.arange(len(names)), delta, color=colors)
                ax.axhline(0, color="black", linewidth=.8)
                ax.set(xticks=np.arange(len(names)), xticklabels=[labels.get(v, v) for v in names],
                       ylabel=f"Delta {metric} vs nominal (px)", title=f"{ep}: {metric}")
                ax.tick_params(axis="x", rotation=42, labelsize=8)
                ax.grid(axis="y", alpha=.25)
        fig.legend(handles=[Patch(color="#4e79a7", label="Text"),
                            Patch(color="#999999", label="Changed goal*"),
                            Patch(color="#59a14f", label="History / geometry"),
                            Patch(color="#f28e2b", label="K")],
                   loc="lower center", bbox_to_anchor=(.5, .005), ncol=4, fontsize=9)
        fig.suptitle("Paired FMB input ablations; * original-future ADE does not measure success at the changed goal",
                     y=.99, fontsize=10)
        fig.tight_layout(rect=(0, .13, 1, .95))
        fig.savefig(ROOT / "runs/fmb_ablation_suite_v1_summary.png", dpi=160)
        plt.close(fig)
    print(json.dumps(all_rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "infer", "evaluate"))
    parser.add_argument("--episodes", nargs="+", choices=tuple(RUNS), default=list(RUNS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(CORE_VARIANTS))
    parser.add_argument("--all-variants", action="store_true",
                        help="Use the ten core variants and the three prepared MoGe-2 variants")
    args = parser.parse_args()
    if args.all_variants:
        args.variants = list(VARIANTS)
    {"prepare": prepare, "infer": infer, "evaluate": evaluate}[args.phase](args.episodes, args.variants)


if __name__ == "__main__":
    main()
