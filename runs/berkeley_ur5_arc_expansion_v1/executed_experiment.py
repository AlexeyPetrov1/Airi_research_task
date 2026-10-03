"""Shared-scale and camera-up controls on frozen, genuine CASE-AUGE forecasts.

This is a postprocessing experiment, not new model inference. All candidates
use the same coefficients in both scenes and retain all 24 original point IDs.
Future data are used for evaluation and visual selection only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from berkeley_temporal_diagnostics import BASE, OUT, load_scene, scale_displacement, write
from berkeley_baseline_comparison import controls, scores
from berkeley_evaluate import ALIGNMENT, TIMES, project, points_on, trails_on, label, save_rgb, write_video

ROOT = Path(__file__).resolve().parents[1]
SOURCE = OUT / "CASE-AUGE"
DEST = ROOT / "runs/berkeley_ur5_arc_expansion_v1"
VARIANTS = {
    "reference_0333": (1/3, 0.),
    "scale_036": (.36, 0.),
    "scale_038": (.38, 0.),
    "scale_040": (.40, 0.),
    "scale_042": (.42, 0.),
    "lift_005": (1/3, .005),
    "lift_010": (1/3, .010),
    "scale_036_lift005": (.36, .005),
    "scale_038_lift005": (.38, .005),
    "scale_036_lift010": (.36, .010),
    "scale_038_lift010": (.38, .010),
}
FAMILIES = {
    "scale": ["reference_0333", "scale_036", "scale_040"],
    "height": ["scale_036", "scale_036_lift005", "scale_036_lift010"],
    "controls": ["reference_0333", "scale_036_lift005", "CV_3D"],
}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(folder):
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}


def expand(prediction, p0, alpha, lift_m, times_s):
    """Own-point displacement scale plus one smooth translation for all points.

    Camera X is right, Y is down, Z is forward. Lift is negative camera Y,
    not world/gravity height. Smoothstep is zero at t0 and reaches lift_m at 2s.
    The lift cannot change any pair distance at the same time; it changes no Z.
    """
    times_s = np.asarray(times_s, dtype=float)
    if times_s.shape != (prediction.shape[1],) or not np.isfinite(times_s).all():
        raise ValueError("One finite timestamp per prediction step is required")
    if np.any((times_s < 0) | (times_s > 2)) or not np.isfinite(lift_m) or lift_m < 0:
        raise ValueError("Lift requires 0..2 second timestamps and a nonnegative finite height")
    result = scale_displacement(prediction, p0, alpha)
    phase = times_s / 2.
    result[..., 1] -= float(lift_m) * (3 * phase**2 - 2 * phase**3)[None]
    if np.any(result[..., 2] <= 0):
        raise ValueError("Candidate lies behind the camera")
    return result


def scene_data(name):
    s = load_scene(name)
    folder = SOURCE / name
    # The root predictions/point_ids.npy in cup is a stale P8 pilot artifact.
    # Validate the authoritative observed IDs against all three actual groups.
    ids = np.load(folder / "observed/selected_point_ids.npy")
    np.testing.assert_array_equal(ids, s["ids"])
    history = np.load(folder / "observed/points_3d_history.npy").astype(float)
    raw = np.load(folder / "predictions/future_3d.npy").astype(float)
    K = np.load(folder / "geometry/K_median.npy").astype(float)
    receipt = json.loads((folder / "predictions/model_run.json").read_text())
    if raw.shape != (24, 30, 3) or history.shape != (3, 24, 3) or not np.isfinite(raw).all():
        raise ValueError("Incomplete real CASE-AUGE result")
    if not receipt["success"] or receipt["successful_chunks"] != 3:
        raise ValueError("Three successful real MolmoMotion groups are required")
    if len(receipt["groups"]) != 3 or not all(g["success"] for g in receipt["groups"]):
        raise ValueError("Individual group receipts disagree")
    for g in range(3):
        group_ids = np.load(folder / f"groups/group_{g:02d}/point_ids.npy")
        np.testing.assert_array_equal(group_ids, ids[g*8:(g+1)*8])
        group_prediction = np.load(folder / f"predictions/group_{g:02d}/future_3d.npy")
        np.testing.assert_allclose(group_prediction, raw[g*8:(g+1)*8], rtol=0, atol=0)
    return s, K, history, raw, receipt


def extent_diagnostics(uv, baseline, uv0, xyz, baseline_xyz):
    distance = np.linalg.norm(uv - uv0[:, None], axis=-1)
    base_distance = np.linalg.norm(baseline - uv0[:, None], axis=-1)
    step = np.diff(np.concatenate([uv0[:, None], uv], axis=1), axis=1)
    base_step = np.diff(np.concatenate([uv0[:, None], baseline], axis=1), axis=1)
    return dict(
        median_farther_px_by_time=np.median(distance-base_distance, axis=0).tolist(),
        median_higher_px_by_time=np.median(baseline[..., 1]-uv[..., 1], axis=0).tolist(),
        farther_pair_fraction=float((distance > base_distance+1e-6).mean()),
        higher_pair_fraction=float((uv[..., 1] < baseline[..., 1]-1e-6).mean()),
        median_path_length_ratio_vs_one_third=float(np.median(
            np.linalg.norm(step, axis=-1).sum(1) / np.linalg.norm(base_step, axis=-1).sum(1))),
        median_endpoint_farther_px=float(np.median(distance[:, -1]-base_distance[:, -1])),
        median_endpoint_higher_px=float(np.median(baseline[:, -1, 1]-uv[:, -1, 1])),
        median_camera_up_difference_mm=float(np.median(baseline_xyz[..., 1]-xyz[..., 1])*1000),
    )


def annotated(s, uv, t, indices, title):
    frame = s["rgb"][t].copy()
    ids = s["ids"][indices]
    start = s["uv0"][indices, None]
    gt = s["gt"][indices]
    mask = s["mask"][indices]
    trails_on(frame, np.concatenate([start, gt[:, :t+1]], axis=1), (40, 245, 100),
              np.c_[np.ones(len(ids), bool), mask[:, :t+1]], 2)
    trails_on(frame, np.concatenate([start, uv[indices, :t+1]], axis=1), (255, 65, 170), thickness=2)
    points_on(frame, gt[:, t], ids, (40, 245, 100), mask[:, t], labels=False)
    points_on(frame, uv[indices, t], ids, (255, 65, 170), marker="cross", labels=False)
    label(frame, f"{title} | +{TIMES[t]:.1f}s")
    label(frame, "green: frozen tracker | pink: forecast", 1)
    return frame


def render_families(s, projections):
    receipts = {}
    for family, names in FAMILIES.items():
        for g in range(3):
            frames = []
            indices = np.arange(g*8, (g+1)*8)
            for t in range(10):
                frames.append(np.concatenate([annotated(s, projections[n], t, indices, n) for n in names], axis=1))
            tag = f"{family}_group_{g:02d}"
            folder = DEST / s["name"]
            receipts[tag] = dict(write_video(folder / f"{tag}.mp4", frames),
                point_ids=s["ids"][indices].tolist(), variants=names, time_s=TIMES.tolist())
            save_rgb(folder / f"{tag}_contact.png", np.concatenate([frames[t] for t in (0, 4, 9)]))
            # Decode the actual encoded movie before assembling every real frame.
            cap = cv2.VideoCapture(str(folder / f"{tag}.mp4"))
            decoded = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                decoded.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            cap.release()
            if len(decoded) != 10:
                raise ValueError("Encoded movie does not contain all ten real moments")
            # Tight fixed ROI shows the whole observed/forecast arc at readable resolution.
            cropped = [cv2.resize(f[110:400].reshape(290, 3, 640, 3).transpose(1, 0, 2, 3)
                       .reshape(870, 640, 3), (640, 870)) for f in decoded]
            # Native three-column frames are rearranged to three rows, then ten columns.
            # Use three timestamps per row of the montage to avoid a 6400-pixel strip.
            padded = cropped + [np.zeros_like(cropped[0]) for _ in range(2)]
            montage = np.concatenate([np.concatenate(padded[i:i+4], axis=1) for i in (0, 4, 8)], axis=0)
            save_rgb(folder / f"{tag}_all10_decoded.jpg", montage)
    write(DEST / s["name"] / "render_receipts.json", receipts)


def grid_plot(s, projections):
    names = list(VARIANTS)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    all_uv = np.concatenate([s["gt"].reshape(-1, 2), *[projections[n].reshape(-1, 2) for n in names]])
    lo = np.minimum(np.nanmin(all_uv, axis=0)-15, [0, 0])
    hi = np.maximum(np.nanmax(all_uv, axis=0)+15, [640, 480])
    for ax, name in zip(axes.flat, names):
        ax.imshow(s["rgb"][-1])
        for p in range(24):
            truth = np.vstack([s["uv0"][p], s["gt"][p]])
            truth[1:][~s["mask"][p]] = np.nan
            forecast = np.vstack([s["uv0"][p], projections[name][p]])
            ax.plot(*truth.T, color="#20b850", lw=1, alpha=.7)
            ax.plot(*forecast.T, color=("#499eff", "#ff9518", "#ed35b3")[p//8], lw=1)
        ax.set(title=name, xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), aspect="equal")
    axes.flat[-1].axis("off")
    axes.flat[-1].text(.02, .8, "All 24 immutable point IDs\nGreen: future tracker reference\nBlue/orange/pink: P8 groups\nSame camera and parameters in both scenes\nNo future fitting of parameters\nFull extent; no curve clipping", fontsize=12)
    fig.tight_layout()
    fig.savefig(DEST / s["name"] / "all24_candidate_grid.png", dpi=120)
    plt.close(fig)


def gallery(data):
    template = (ROOT / "scripts/berkeley_arc_gallery.html").read_text(encoding="utf8")
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    (DEST / "gallery.html").write_text(template.replace("__EXPERIMENT_DATA__", payload), encoding="utf8")


def generate(resume=False):
    existing = None
    if (DEST / "protocol.json").exists():
        if not resume or (DEST / "results.json").exists():
            raise FileExistsError("Experiment already prepared; completed results cannot be overwritten")
        existing = json.loads((DEST / "protocol.json").read_text())
    before = {**snapshot(BASE), **snapshot(OUT)}
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / ".gitattributes").write_text("** -text\n", encoding="utf8")
    script_hash = sha(Path(__file__))
    if existing:
        if existing["source_sha256"] != before or existing["variants"] != {k:dict(alpha=a,lift_m=h) for k,(a,h) in VARIANTS.items()}:
            raise ValueError("Resume cannot change source files or the frozen parameter grid")
    write(DEST / "protocol.json", dict(created_utc=existing["created_utc"] if existing else datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_sha256=before, executed_script_sha256=script_hash, variants={k:dict(alpha=a, lift_m=h) for k,(a,h) in VARIANTS.items()},
        parameter_formula="p0_i + alpha * (raw_i(t) - p0_i) - [0, lift_m*smoothstep(t/2), 0]",
        camera_axes="X right; Y down; Z forward. Camera-up is negative Y, not calibrated gravity.",
        source="genuine saved CASE-AUGE all-three-P8 MolmoMotion greedy outputs; no new inference",
        shared_parameters=True, point_count_per_scene=24, ML_calls=0,
        prediction_times_s=(np.arange(1,31)/15).tolist(), physical_indices=ALIGNMENT.tolist(), real_frame_times_s=TIMES.tolist(),
        parameter_grid_frozen_before_new_scores=True, previously_inspected_scenes=True, independent_test_set=False,
        input_future_used=False, parameters_fitted_to_future=False, visual_selection_uses_future=True,
        input_geometry_limitation="CASE-AUGE changed rays while retaining baseline smoothed sensor Z; author filtering not rerun",
        K_limitation="AugE effective simulation prior, not factory RealSense calibration",
        resumed_after_preflight=bool(existing),
        previous_preflight_script_sha256=existing["executed_script_sha256"] if existing else None,
        preflight_discovery="Cup root predictions/point_ids.npy has pilot-only P8 IDs; all 24 observed IDs independently validated against three group inputs and outputs"))
    # Preserve exact executed source alongside results.
    (DEST / "executed_experiment.py").write_bytes(Path(__file__).read_bytes())
    all_results = {}
    gallery_data = dict(variants=list(VARIANTS)+["static", "CV_3D", "object_translation_3D"], scenes={})
    common_K = None
    for name in ("cup", "bottle"):
        s, K, history, raw, model_receipt = scene_data(name)
        if common_K is not None:
            np.testing.assert_array_equal(K, common_K)
        common_K = K
        folder = DEST / name
        folder.mkdir(exist_ok=True)
        (folder / "frames").mkdir(exist_ok=True)
        np.save(folder / "K.npy", K)
        np.save(folder / "point_ids.npy", s["ids"])
        np.save(folder / "p0.npy", history[-1])
        full = {key:expand(raw, history[-1], a, h, np.arange(1,31)/15) for key,(a,h) in VARIANTS.items()}
        xyz = {key:value[:,ALIGNMENT] for key,value in full.items()}
        xyz.update(controls(history, history[-1], s["times"]))
        uv = {key:project(value, K) for key,value in xyz.items()}
        np.savez_compressed(folder / "full_30_steps.npz", **full)
        np.savez_compressed(folder / "aligned_3d.npz", **xyz)
        np.savez_compressed(folder / "projected_2d.npz", **uv)
        results = {key:dict(scores(value, s, xyz[key], history[-1], K),
            expansion_vs_one_third=extent_diagnostics(value, uv["reference_0333"], s["uv0"], xyz[key], xyz["reference_0333"])) for key,value in uv.items()}
        old = json.loads((SOURCE / name / "comparison.json").read_text())["methods"]["CASE-AUGE_one_third"]["2D_px"]
        for key in ("ADE_2D_px", "FDE_2D_px"):
            # Previous report mixed float32/float64 projection arithmetic.
            np.testing.assert_allclose(results["reference_0333"][key], old[key], rtol=0, atol=1e-5)
        metadata = json.loads((SOURCE / name / "metadata.json").read_text())
        all_results[name] = dict(episode_id=metadata["episode_id"], t0=metadata["t0"], K=K.tolist(),
            point_ids=s["ids"].tolist(), model_revision=model_receipt["checkpoint_hf_revision"], methods=results)
        for t, rgb in enumerate(s["rgb"]):
            save_rgb(folder / "frames" / f"{t:02d}.jpg", rgb)
        gallery_data["scenes"][name] = dict(ids=s["ids"].tolist(), uv0=s["uv0"].tolist(),
            gt=s["gt"].tolist(), mask=s["mask"].tolist(), uv={k:v.tolist() for k,v in uv.items()})
        render_families(s, uv)
        grid_plot(s, uv)
        print(json.dumps({"scene":name,"done":True,"variants":len(full)}, ensure_ascii=False), flush=True)
    gallery(gallery_data)
    write(DEST / "results.json", all_results)
    if before != {**snapshot(BASE), **snapshot(OUT)}:
        raise ValueError("Prior experiment data were mutated")
    write(DEST / "generation_receipt.json", dict(success=True, previous_files_unchanged=len(before),
        ML_calls=0, genuine_source_groups=6, postprocessed_forecasts=22, movies=18,
        finished_utc=datetime.now(timezone.utc).isoformat()))


def verify():
    protocol = json.loads((DEST / "protocol.json").read_text())
    if protocol["source_sha256"] != {**snapshot(BASE), **snapshot(OUT)}:
        raise ValueError("Frozen source files changed")
    np.testing.assert_equal(sha(DEST / "executed_experiment.py"), protocol["executed_script_sha256"])
    variants = {k:(v["alpha"],v["lift_m"]) for k,v in protocol["variants"].items()}
    assert variants == VARIANTS
    total_movies = 0
    common_K = None
    for name in ("cup", "bottle"):
        s, K, history, raw, _ = scene_data(name)
        np.testing.assert_array_equal(np.load(DEST/name/"K.npy"), K)
        if common_K is not None:
            np.testing.assert_array_equal(K, common_K)
        common_K = K
        np.testing.assert_array_equal(np.load(DEST/name/"point_ids.npy"), s["ids"])
        full = np.load(DEST/name/"full_30_steps.npz")
        aligned = np.load(DEST/name/"aligned_3d.npz")
        uv = np.load(DEST/name/"projected_2d.npz")
        for key,(alpha,lift) in variants.items():
            independently_scaled = history[-1,:,None] + alpha*(raw-history[-1,:,None])
            phase = np.arange(1,31)/30
            independently_scaled[...,1] -= lift*(3*phase*phase-2*phase*phase*phase)
            np.testing.assert_allclose(full[key], independently_scaled, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(aligned[key], full[key][:,ALIGNMENT])
            np.testing.assert_allclose(uv[key], project(aligned[key],K), rtol=0, atol=1e-12)
            if alpha > 1/3 and lift == 0:
                base = full["reference_0333"]-history[-1,:,None]
                new = full[key]-history[-1,:,None]
                assert np.all(np.linalg.norm(new,axis=-1) >= np.linalg.norm(base,axis=-1)-1e-14)
        for movie in (DEST/name).glob("*.mp4"):
            cap = cv2.VideoCapture(str(movie)); decoded = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            while cap.read()[0]:
                decoded += 1
            cap.release()
            assert decoded == 10 and abs(fps-5)<1e-6
            total_movies += 1
    assert total_movies == 18
    if not (DEST/"visual_review.json").exists():
        raise ValueError("Visual review must be recorded before final verification")
    review = json.loads((DEST/"visual_review.json").read_text())
    assert review["completed"] is True
    assert review["shared_candidate"] in VARIANTS
    write(DEST / "verification.json", dict(success=True, previous_source_files_verified=len(protocol["source_sha256"]),
        prediction_shape_per_candidate=[24,30,3], candidates_per_scene=11, common_K_identical=True,
        original_one_third_reproduced=True, original_point_ID_order_retained=True,
        all_projections_recomputed=True, all_18_movies_decoded=True, real_frames_per_movie=10, fps=5,
        visual_review_completed=True, ML_calls=0, finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({"verified":True,"movies":total_movies,"source_files":len(protocol["source_sha256"])}),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("generate", "verify"), required=True)
    parser.add_argument("--resume", action="store_true", help="Resume an incomplete run with the same frozen grid and sources")
    args = parser.parse_args()
    generate(args.resume) if args.stage == "generate" else verify()
