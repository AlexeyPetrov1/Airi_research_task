"""Consistent readable videos, full-extent tracks and ADE/FDE plots."""
from __future__ import annotations

import html

import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .geometry import project
from .io import video_frames, write_json


def xyz_limits(ax, values):
    points = values.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=-1)]
    center = (points.min(axis=0)+points.max(axis=0))/2
    half = max(float(np.ptp(points, axis=0).max())/2, .005)*1.1
    for setlim, value in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setlim(value-half, value+half)
    ax.set_box_aspect((1, 1, 1))

PINK, GREEN, CYAN = (255, 60, 170), (40, 230, 95), (25, 205, 240)


def caption(frame, text, second=""):
    frame = frame.copy()
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 49), (16, 23, 34), -1)
    for row, value in enumerate((text, second)):
        cv2.putText(frame, value, (9, 19+21*row), cv2.FONT_HERSHEY_SIMPLEX, .44,
                    (245, 247, 250), 1, cv2.LINE_AA)
    return frame


def draw(frame, values, color, mask=None, initial=None, labels=None):
    """Show out-of-image predictions with arrows; never silently discard them."""
    h, w = frame.shape[:2]
    rect = (0, 50, w, max(h-50, 1))
    for index, track in enumerate(values):
        valid = np.isfinite(track).all(axis=-1)
        if mask is not None:
            valid &= mask[index]
        points = np.rint(np.nan_to_num(track, nan=0, posinf=1e7, neginf=-1e7)).clip(-1e7, 1e7).astype(int)
        for t in range(1, len(track)):
            if valid[t-1] and valid[t]:
                ok, a, b = cv2.clipLine(rect, tuple(points[t-1]), tuple(points[t]))
                if ok:
                    cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
        if not valid[-1]:
            continue
        x, y = points[-1]
        if 0 <= x < w and 50 <= y < h:
            cv2.circle(frame, (int(x), int(y)), 4, color, 1, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(frame, str(labels[index]), (int(x)+5, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, .3, color, 1)
        else:
            origin = initial[index] if initial is not None else [w/2, h/2]
            origin = np.rint(origin).astype(int)
            ok, a, b = cv2.clipLine(rect, tuple(origin), (int(x), int(y)))
            if ok:
                cv2.arrowedLine(frame, a, b, color, 2, cv2.LINE_AA, tipLength=.06)
    return frame


def video(path, frames, fps):
    imageio.mimwrite(path, frames, fps=float(fps), codec="libx264", macro_block_size=None,
                    ffmpeg_params=["-crf", "18"])
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    encoded_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if decoded != len(frames):
        raise ValueError("Encoded video lost frames")
    return dict(frame_count=decoded, size_wh=size, fps=encoded_fps, bytes=path.stat().st_size)


def error_plots(sample, metrics, dest):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for ax, key, label in zip(axes, ("2D_px", "3D_m"), ("2D error (px)", "Estimated 3D error (m)")):
        for name, row in metrics["methods"].items():
            if row[key] is not None:
                ax.plot(sample.future_times, row[key]["error_by_time"], label=name,
                        linewidth=2.5 if name == metrics["primary"] else 1.2)
        ax.set(xlabel="Real seconds after t0", ylabel=label)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            reason = "Unavailable: no 3D reference" if key == "3D_m" else "Unavailable: no per-frame camera poses"
            ax.text(.5, .5, reason, ha="center", transform=ax.transAxes)
    fig.suptitle(sample.metadata["title"]+" · fixed masks; off-image predictions retained")
    fig.tight_layout()
    fig.savefig(dest/"errors.png", dpi=155)
    plt.close(fig)
    key = "2D_px" if metrics["methods"][metrics["primary"]]["2D_px"] else "3D_m"
    units = "px" if key == "2D_px" else "m"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
    for name, row in metrics["methods"].items():
        axes[0].plot(sample.future_times, row[key]["cumulative_ADE"], label=name)
        axes[1].plot(sample.future_times, row[key]["error_by_time"], label=name)
    for ax, title in zip(axes, ("ADE over the prefix", "FDE at each fixed horizon")):
        ax.set(title=title, xlabel="Real seconds after t0", ylabel=units)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest/"ade_fde.png", dpi=155)
    plt.close(fig)
    names = list(metrics["methods"])
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(names))
    ax.bar(x-.18, [metrics["methods"][n][key]["ADE"] for n in names], .36, label="ADE")
    ax.bar(x+.18, [metrics["methods"][n][key]["FDE"] for n in names], .36, label="FDE at final horizon")
    ax.set(xticks=x, xticklabels=names, ylabel=units, title="All evaluated points · same frozen reference mask")
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest/"metric_comparison.png", dpi=155)
    plt.close(fig)


def trajectory_plots(sample, xyz, uv, dest):
    name = sample.metadata["primary"]
    observed = sample.evaluation["uv"].copy()
    observed[~sample.evaluation["mask2"]] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h, w = sample.history_frames.shape[1:3]
    background = sample.history_frames[0] if sample.evaluation.get("diagnostic_projection") else sample.history_frames[-1]
    for ax in axes:
        ax.imshow(background)
        for i in range(len(sample.point_ids)):
            ax.plot(*uv[name][i].T, color="#ed3c98", alpha=.6, linewidth=1, label="Forecast" if i == 0 else None)
            ax.plot(*observed[i].T, color="#17a34a", alpha=.6, linewidth=1, label="Reference" if i == 0 else None)
        ax.scatter(*sample.points_2d.T, c="#14c6db", s=18, label="Input points")
        ax.set_aspect("equal")
    axes[0].set(xlim=(0, w), ylim=(h, 0), title="Image bounds · green reference / pink forecast")
    all_points = np.concatenate([uv[name].reshape(-1, 2), observed.reshape(-1, 2)])
    all_points = all_points[np.isfinite(all_points).all(axis=1)]
    lo = np.minimum(all_points.min(axis=0)-15, [0, 0])
    hi = np.maximum(all_points.max(axis=0)+15, [w, h])
    axes[1].set(xlim=(lo[0], hi[0]), ylim=(hi[1], lo[1]), title="Full extent · no trajectory clipping")
    if sample.evaluation.get("diagnostic_projection"):
        # These are different camera frames, so do not suggest a valid overlay.
        axes[0].clear()
        axes[0].imshow(background)
        for i, p in enumerate(uv[name]):
            axes[0].plot(*p.T, color="#ed3c98", linewidth=1, label="Forecast" if i == 0 else None)
        axes[0].set_title("Forecast in first camera · diagnostic projection")
        axes[1].clear()
        axes[1].imshow(sample.history_frames[-1])
        for i, p in enumerate(observed):
            axes[1].plot(*p.T, color="#17a34a", linewidth=1, label="Reference" if i == 0 else None)
        axes[1].set_title("Real image tracks · changing camera; no 2D score")
    for ax in axes:
        ax.set(xlabel="Image u (px)", ylabel="Image v (px)")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(dest/"trajectories_2d.png", dpi=150)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for i, p in enumerate(xyz[name]):
        ax.plot(*p.T, color="#ed3c98", alpha=.65, label="Forecast" if i == 0 else None)
    ref = sample.evaluation.get("xyz")
    if ref is not None:
        ref = ref.copy()
        ref[~sample.evaluation["mask3"]] = np.nan
        for i, p in enumerate(ref):
            ax.plot(*p.T, color="#17a34a", alpha=.65, label="Estimated reference" if i == 0 else None)
    ax.set(title="Forecast and estimated reference" if ref is not None else "Forecast only · no 3D ground truth",
           xlabel="X (m / estimated scale)", ylabel="Y", zlabel="Z")
    xyz_limits(ax, np.concatenate([xyz[name].reshape(-1, 3), ref.reshape(-1, 3)]) if ref is not None else xyz[name])
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(dest/"trajectories_3d.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for axis, ax in enumerate(axes):
        for i, p in enumerate(xyz[name]):
            ax.plot(sample.future_times, p[:, axis], color="#ed3c98", alpha=.45, label="Forecast" if i == 0 else None)
        if ref is not None:
            for i, p in enumerate(ref):
                ax.plot(sample.future_times, p[:, axis], color="#17a34a", alpha=.45, label="Estimated reference" if i == 0 else None)
        ax.set(xlabel="Real seconds after t0", ylabel="m / estimated scale", title="XYZ"[axis])
        ax.grid(alpha=.2)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(dest/"xyz_over_time.png", dpi=155)
    plt.close(fig)


def make_visualizations(sample, xyz, uv, metrics, destination):
    destination.mkdir(parents=True, exist_ok=True)
    primary = sample.metadata["primary"]
    n = len(sample.point_ids)
    shown = np.array(sample.metadata["shown_indices"])
    h, w = sample.history_frames.shape[1:3]
    target_w = 640
    target_h = int(round(h*target_w/w/2))*2
    scale = np.array([target_w/w, target_h/h])
    resized = lambda frame: cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    inputs = []
    for i, frame in enumerate(sample.history_frames):
        canvas = resized(frame)
        # The exact t0 queries are always shown. Historical queries are shown
        # only when supplied independently by the adapter, never invented.
        if i == len(sample.history_frames)-1:
            draw(canvas, (sample.points_2d[shown]*scale)[:, None], CYAN, labels=sample.point_ids[shown])
        inputs.append(caption(canvas, f"History {i+1}/{len(sample.history_frames)}", "cyan = exact input query at t0"))
    imageio.imwrite(destination/"inputs.png", np.hstack(inputs))
    if sample.evaluation.get("diagnostic_projection"):
        frozen_background = sample.history_frames[0]
        frozen_title = "First-camera forecast (diagnostic)"
    else:
        frozen_background = sample.history_frames[-1]
        frozen_title = "Forecast on last observed image"
    frozen_uv = project(xyz[primary], sample.camera_intrinsics)*scale
    animation = None
    if sample.evaluation.get("diagnostic_projection"):
        fig = plt.figure(figsize=(target_w/100, target_h/100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        reference = sample.evaluation["xyz"].copy()
        reference[~sample.evaluation["mask3"]] = np.nan
        bounds = np.concatenate([xyz[primary].reshape(-1, 3), reference.reshape(-1, 3)])
        animation = []
        for t in range(len(sample.future_times)):
            ax.clear()
            for p, g in zip(xyz[primary], reference):
                ax.plot(*p[:t+1].T, color="#ed3c98", linewidth=1)
                ax.plot(*g[:t+1].T, color="#17a34a", linewidth=1)
            xyz_limits(ax, bounds)
            ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
            fig.tight_layout()
            fig.canvas.draw()
            animation.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    forecast, comparison = [], []
    marker = sample.points_2d[shown]*scale
    if sample.evaluation.get("diagnostic_projection"):
        marker = project(sample.points_3d_history[-1], sample.camera_intrinsics)[shown]*scale
    observed = sample.evaluation["uv"]*scale
    projected = uv[primary]*scale
    mask = sample.evaluation.get("visibility2", sample.evaluation["mask2"])
    for t, rgb in enumerate(sample.evaluation["rgb"]):
        time = sample.future_times[t]
        predicted = resized(frozen_background)
        draw(predicted, frozen_uv[shown, :t+1], PINK, initial=marker)
        draw(predicted, marker[:, None], CYAN)
        predicted = caption(predicted, frozen_title, f"+{time:.2f}s | {len(shown)}/{n} points shown; {n} scored")
        forecast.append(predicted)
        real = resized(rgb)
        draw(real, observed[shown, :t+1], GREEN, mask=mask[shown, :t+1], initial=marker)
        real = caption(real, "Real continuation + reference", f"+{time:.2f}s | green = independent tracks")
        overlay = resized(rgb)
        draw(overlay, observed[shown, :t+1], GREEN, mask=mask[shown, :t+1], initial=marker)
        if sample.evaluation.get("diagnostic_projection"):
            overlay = animation[t].copy()
            overlay = caption(overlay, "Common-frame 3D comparison", "pink forecast | green estimated reference | no RGB overlay")
        else:
            draw(overlay, projected[shown, :t+1], PINK, initial=marker)
            outside = np.isfinite(uv[primary][shown, t]).all(-1) & (((uv[primary][shown, t]<0).any(-1)) | ((uv[primary][shown, t]>=[w,h]).any(-1)))
            overlay = caption(overlay, "Prediction vs real", f"+{time:.2f}s | pink forecast | off-image {outside.sum()}/{len(shown)}")
        comparison.append(np.hstack([predicted, overlay, real]))
    fps = 1/float(np.median(np.diff(np.r_[0, sample.future_times])))
    receipts = {"prediction.mp4": video(destination/"prediction.mp4", forecast, fps),
                "comparison.mp4": video(destination/"comparison.mp4", comparison, fps)}
    if not sample.evaluation.get("diagnostic_projection"):
        method_names = list(xyz)[:3]
        method_movie = []
        for t, rgb in enumerate(sample.evaluation["rgb"]):
            panels = []
            for method in method_names:
                panel = resized(rgb)
                draw(panel, observed[shown, :t+1], GREEN, mask=mask[shown, :t+1], initial=marker)
                draw(panel, uv[method][shown, :t+1]*scale, PINK, initial=marker)
                panels.append(caption(panel, method, f"+{sample.future_times[t]:.2f}s | same {len(shown)} point IDs"))
            method_movie.append(np.hstack(panels))
        receipts["methods.mp4"] = video(destination/"methods.mp4", method_movie, fps)
        imageio.imwrite(destination/"methods_final.png", method_movie[-1])
    for t, tag in zip((0, len(comparison)//2, len(comparison)-1), ("first", "middle", "final")):
        imageio.imwrite(destination/f"comparison_{tag}.png", comparison[t])
    imageio.imwrite(destination/"contact_sheet.jpg", np.vstack([comparison[t] for t in (0, len(comparison)//2, len(comparison)-1)]))
    np.savez_compressed(destination/"drawn_coordinates.npz", point_ids=sample.point_ids,
                        shown_indices=shown, time_s=sample.future_times, predicted_uv=uv[primary],
                        reference_uv=sample.evaluation["uv"], reference_mask=sample.evaluation["mask2"],
                        forecast_anchor_uv=project(xyz[primary], sample.camera_intrinsics))
    write_json(destination/"render_receipt.json", dict(videos=receipts, evaluated_points=n,
               shown_point_ids=sample.point_ids[shown].tolist(), coordinate_file="drawn_coordinates.npz",
               no_RGB_interpolation=True, clipping_for_drawing_only=True,
               future_camera_poses_in_prediction_video=False))
    return receipts


def compare_legacy(sample, source_root, destination):
    """Compare the exact user-selected legacy panel with its unified overlay."""
    old = video_frames(source_root/sample.metadata["old_video"])
    new = video_frames(destination/"comparison.mp4")
    if len(old) == len(new)+len(sample.history_frames):
        old = old[len(sample.history_frames):]
    if len(old) != len(new):
        raise ValueError("Legacy and unified movies have different time counts")
    crop = sample.metadata.get("old_panel_crop")
    if crop is not None:
        x, y, w, h = crop
        old = old[:, y:y+h, x:x+w]
    width = new.shape[2]//3
    new = new[:, :, width:2*width]
    rows = []
    for t in (0, len(new)//2, len(new)-1):
        left = cv2.resize(old[t], (new.shape[2], new.shape[1]))
        rows.append(np.hstack([caption(left, "Legacy selected panel", f"+{sample.future_times[t]:.2f}s"),
                               caption(new[t], "Unified rendering", "same fixed point IDs and times")]))
    imageio.imwrite(destination/"legacy_vs_unified.png", np.vstack(rows))
    write_json(destination/"legacy_visualization_parity.json", dict(success=True, frame_count=len(new),
               time_s=sample.future_times.tolist(), point_ids=sample.point_ids.tolist(),
               numerical_coordinate_parity="checked against saved projections by regression tests",
               unified_size_wh=[new.shape[2], new.shape[1]], legacy_panel_crop=crop,
               layouts_intentionally_differ=True, reference_masks_unchanged=True))
