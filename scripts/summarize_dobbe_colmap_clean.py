"""Summarize the four frozen COLMAP controls and independently score causal K.

No intrinsics are fitted to depth/labels. Historical RGB correspondences are
filtered by the frozen foreground masks before applying every candidate K.
"""
from __future__ import annotations

import copy
import csv
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dobbe_colmap_clean import OUT, ROOT, digest, write
from validate_dobbe_geometry import evaluate, load_scene

RUNS = [
    "main_full-oracle_pinhole_masked_stride1_all",
    "main_causal_pinhole_masked_stride1_all",
    "main_causal_pinhole_masked_stride5_all",
    "main_causal_pinhole_masked_stride1_all_dsp_affine_guided",
]
LABELS = ["Full 243 (oracle)", "Causal 96", "Causal stride 5", "Causal DSP/affine/guided"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_causal():
    source = ROOT / "runs/dobbe_rgbd_study/main_static_correspondences.json"
    original = read(source)
    points = copy.deepcopy(original)
    for pair in points:
        masks = [cv2.imread(str(OUT / f"masks/main/{f:04d}.png.png"), 0)
                 for f in pair["pair"]]
        pair["selected"] = [p for p in pair["selected"] if all(
            mask[int(round(p[key][1])), int(round(p[key][0]))] > 0
            for mask, key in zip(masks, ("rgb_a", "rgb_b")))]
    _, _, poses = load_scene("main")
    candidates = {"naive_OpenCV_200": {"model": "PINHOLE", "params": [200, 200, 128, 128]}}
    for slug in RUNS[1:]:
        for rec in read(OUT / slug / "summary.json")["reconstructions"]:
            if not rec["usable_size_ge10"]:
                continue
            cam = rec["cameras"][0]
            k = cam["K_OpenCV"]
            candidates[f"{slug}_model{rec['id']}"] = {
                "model": "PINHOLE", "params": [k[0][0], k[1][1], k[0][2], k[1][2]],
                "registered": rec["registered"], "source": f"{slug}/summary.json",
            }
    scores = {}
    for name, candidate in candidates.items():
        scores[name] = {"camera": candidate, "variants": {
            f"c2w_{depth_kind}": evaluate(points, poses, candidate, "c2w", depth_kind)
            for depth_kind in ("z", "ray")}}
    result = {
        "status": "DIAGNOSTIC_ONLY_NO_K_FIT", "oracle_K_scored": False,
        "correspondences_source": str(source.relative_to(ROOT)),
        "correspondences_source_sha256": digest(source),
        "selection": "Historical RGB RANSAC matches; both endpoints must pass frozen foreground masks; identical samples for all K",
        "calibration_pairs_only": [p["pair"] for p in points],
        "point_counts_before": [len(p["selected"]) for p in original],
        "point_counts_after": [len(p["selected"]) for p in points],
        "limitations": "Small filtered sample; RGB matching is not a manually verified static landmark annotation. Depth Z versus ray remains unresolved. These scores can reject a candidate, not certify calibration.",
        "candidates": scores,
    }
    write(OUT / "causal_geometry_validation.json", result)
    write(OUT / "causal_validation_correspondences.json", points)
    return result


def main():
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for slug, label, ax in zip(RUNS, LABELS, axes.flat):
        run = OUT / slug
        summary, graph = read(run / "summary.json"), read(run / "match_graph.json")
        frames = summary["input_frames"]
        indices = {f: i for i, f in enumerate(frames)}
        adjacency = np.zeros((len(frames), len(frames)))
        with (run / "match_pairs.csv").open() as f:
            for p in csv.DictReader(f):
                a, b = indices[int(p["frame_a"])], indices[int(p["frame_b"])]
                adjacency[a, b] = adjacency[b, a] = int(p["verified_inliers"])
        im = ax.imshow(np.ma.masked_less(adjacency, 1), norm=LogNorm(vmin=1, vmax=150), cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("Selected frame index")
        ax.set_ylabel("Selected frame index")
        recs = summary["reconstructions"]
        row = {"run": slug, "images": len(frames), "model_count": len(recs),
               "model_count_ge10": sum(r["usable_size_ge10"] for r in recs),
               "best_registered": summary["best_registered"],
               "unique_registered": len({t["frame"] for r in recs for t in r["trajectory"]}),
               "keypoints_median": graph["keypoints_median"]}
        for t in (15, 30, 50, 100):
            row[f"pairs_ge{t}"] = graph["by_threshold"][str(t)]["pairs"]
            row[f"largest_component_ge{t}"] = graph["by_threshold"][str(t)]["largest_component"]
        rows.append(row)
    fig.colorbar(im, ax=list(axes.flat), label="Verified inliers per image pair (log scale)", shrink=.8)
    fig.savefig(OUT / "verified_match_graphs.png", dpi=160)
    plt.close(fig)
    with (OUT / "experiment_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    validation = validate_causal()
    write(OUT / "clean_decision.json", {
        "status": "NO_VALIDATED_CAUSAL_K", "formal_geometry_gate": "BLOCKED",
        "claim_scope": "Four recorded masked PINHOLE configurations on this exported sequence; not a proof that COLMAP self-calibration is impossible",
        "legacy_claim_correction": "Legacy 9/96 used guessed focal prior, free principal point and relaxed thresholds; it did not establish general COLMAP failure",
        "oracle_eligible_for_main_K": False, "validated_main_K": None,
        "experiments": rows, "validation": "causal_geometry_validation.json",
        "approximate_existing_run": "../dobbe_rgbd_study/approx_history/manifest.json",
        "approximate_existing_input_sha256": digest(ROOT / "runs/dobbe_rgbd_study/approx_history/points_3d_history_candidate.npy"),
        "approximate_run_replaced": False,
    })
    print(json.dumps({"experiments": rows, "validation_counts": validation["point_counts_after"],
                      "validation_c2w_z": {k: v["variants"]["c2w_z"] for k, v in validation["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    main()
