"""Make the compact evidence figures for the causal HoNY geometry study."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from validate_dobbe_geometry import (OUT, ROOT, CAMERA_TO_LABEL,
                                     get_intrinsics, load_scene, unproject)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def k_stability() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    cases = [
        ("Main: unconstrained PINHOLE", "main",
         ["colmap_pinhole_all", "colmap_pinhole_even", "colmap_pinhole_odd"],
         [96, 48, 48]),
        ("Second: PINHOLE, center fixed", "second",
         ["colmap_pinhole_all_fixedpp_init62_107",
          "colmap_pinhole_even_fixedpp_init74_104",
          "colmap_pinhole_odd_fixedpp_init65_111"],
         [121, 61, 60]),
    ]
    for ax, (title, scene, names, sizes) in zip(axes, cases):
        xs = np.arange(3)
        fx, fy, coverage = [], [], []
        for name, n in zip(names, sizes):
            s = read_json(OUT / scene / name / "summary.json")
            recs = s["reconstructions"]
            if recs:
                camera = recs[0]["cameras"][0]
                fx.append(camera["fx"])
                fy.append(camera["fy"])
                coverage.append(f"{recs[0]['registered']}/{n}")
            else:
                fx.append(np.nan)
                fy.append(np.nan)
                coverage.append(f"0/{n}")
        ax.plot(xs, fx, "o-", label="fx", lw=2)
        ax.plot(xs, fy, "s-", label="fy", lw=2)
        ax.axhline(200, color="0.7", ls="--", label="naive 200 px")
        ax.set_xticks(xs, ["all", "even", "odd"])
        ax.set_yscale("log")
        ax.set_ylabel("Focal length (px, log scale)")
        ax.set_title(title)
        ax.grid(alpha=.25)
        for x, n in zip(xs, coverage):
            ax.text(x, .97, n, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("COLMAP RGB-only calibration: registered frames / input frames")
    fig.savefig(OUT / "colmap_k_stability.png", dpi=170)
    plt.close(fig)


def metric_comparison() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for col, scene, labels in [
        (0, "main", ["naive", "previous_fit_diagnostic",
                     "colmap_pinhole_all", "colmap_opencv_all"]),
        (1, "second", ["naive", "colmap_pinhole_all_fixedpp_init62_107",
                       "colmap_pinhole_odd_fixedpp_init65_111",
                       "colmap_simple_pinhole_all_fixedpp_rectified"]),
    ]:
        s = read_json(OUT / f"{scene}_static_geometry.json")
        metrics = [s["candidates"][k]["variants"]["c2w_z"] for k in labels]
        short = ["naive", "old fit", "PINHOLE", "OPENCV"] if scene == "main" \
            else ["naive", "PINHOLE all", "PINHOLE odd", "aspect prior"]
        x = np.arange(len(labels))
        for row, key, scale, unit, color in [
            (0, "median_3d_m", 100, "Static 3D error (cm)", "#3274a1"),
            (1, "median_reprojection_px", 1, "Reprojection error (px)", "#d88728"),
        ]:
            ax = axes[row, col]
            values = [m[key] * scale for m in metrics]
            ax.bar(x, values, .65, color=color)
            ax.set_xticks(x, short)
            ax.set_ylabel(unit)
            ax.set_title(f"{scene.capitalize()} · same RGB-D matches · c2w + Z")
            ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "static_and_reprojection_comparison.png", dpi=170)
    plt.close(fig)


def draw_matches(scene: str, pair: tuple[int, int]) -> None:
    frames, _, _ = load_scene(scene)
    all_points = read_json(OUT / f"{scene}_static_correspondences.json")
    matches = next(p["selected"] for p in all_points if p["pair"] == list(pair))
    a, b = pair
    canvas = np.hstack((frames[a], frames[b]))
    for n, match in enumerate(matches):
        x1, y1 = np.round(match["rgb_a"]).astype(int)
        x2, y2 = np.round(match["rgb_b"]).astype(int)
        hue = (n * 41) % 180
        color = cv2.cvtColor(np.uint8([[[hue, 190, 230]]]),
                             cv2.COLOR_HSV2BGR)[0, 0].tolist()
        color = tuple(int(c) for c in color)
        cv2.circle(canvas, (x1, y1), 3, color, -1)
        cv2.circle(canvas, (x2+256, y2), 3, color, -1)
        cv2.line(canvas, (x1, y1), (x2+256, y2), color, 1)
    title = f"{scene}: RGB-only static candidates {a} -> {b}, n={len(matches)}"
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (0, 0, 0), 3)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / f"{scene}_rgb_matches_{a}_{b}.png"), canvas)


def reprojection_overlay(scene: str, pair: tuple[int, int]) -> None:
    frames, _, poses = load_scene(scene)
    all_points = read_json(OUT / f"{scene}_static_correspondences.json")
    matches = next(p["selected"] for p in all_points if p["pair"] == list(pair))
    a, b = pair
    k, dist = get_intrinsics({"params": [200, 200, 128, 128]})
    uv_a = np.asarray([p["rgb_a"] for p in matches])
    uv_b = np.asarray([p["rgb_b"] for p in matches])
    depth_a = np.asarray([p["depth_a_m"] for p in matches])
    optical_a = unproject(uv_a, depth_a, k, dist, "z")
    label_a = optical_a @ CAMERA_TO_LABEL.T
    # Published labels are camera-to-world after the documented basis change.
    ta, tb = poses[a], poses[b]
    common = label_a @ ta[:3, :3].T + ta[:3, 3]
    label_b = (common - tb[:3, 3]) @ tb[:3, :3]
    optical_b = label_b @ CAMERA_TO_LABEL
    uv_pred, _ = cv2.projectPoints(optical_b, np.zeros(3), np.zeros(3), k, dist)
    uv_pred = uv_pred.reshape(-1, 2)
    canvas = frames[b].copy()
    for observed, predicted in zip(uv_b, uv_pred):
        u, v = np.round(observed).astype(int)
        pu, pv = np.round(predicted).astype(int)
        cv2.circle(canvas, (u, v), 3, (0, 255, 0), -1)
        if -200 < pu < 456 and -200 < pv < 456:
            cv2.line(canvas, (u, v), (pu, pv), (0, 0, 255), 1)
            cv2.circle(canvas, (pu, pv), 3, (0, 0, 255), 1)
    cv2.putText(canvas, f"{scene} {a}->{b} observed green / projected red",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
    cv2.putText(canvas, f"{scene} {a}->{b} observed green / projected red",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
    cv2.imwrite(str(OUT / f"{scene}_reprojection_{a}_{b}.png"), canvas)


def main() -> None:
    k_stability()
    metric_comparison()
    draw_matches("main", (0, 20))
    draw_matches("main", (80, 95))
    draw_matches("second", (80, 100))
    draw_matches("second", (100, 120))
    reprojection_overlay("main", (0, 20))
    reprojection_overlay("second", (80, 100))
    print("Study figures written to", OUT)


if __name__ == "__main__":
    main()
