"""Recompute all metrics and figures from saved files; no model load."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

from decode import strict_decode


ROOT = Path(__file__).resolve().parent
COLORS = {"A": "#e44335", "B": "#008fd3", "Static": "#aa47bc", "Velocity": "#ec9a16", "GT": "#50c469"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_world(world, cameras, intr):
    rotation = cameras[:, :3, :3]
    shift = cameras[:, :3, 3]
    cam = np.einsum("tij,tpj->tpi", rotation, world) + shift[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = np.stack((intr[0] * cam[..., 0] / cam[..., 2] + intr[2], intr[1] * cam[..., 1] / cam[..., 2] + intr[3]), -1)
    return xy, cam[..., 2]


def score(pred, gt, mask, cameras, intr, reference_uv):
    assert pred.shape == gt.shape == (30, 8, 3)
    reference = mask & np.isfinite(gt).all(-1)
    assert reference[:, :].sum() > 0
    finite_pred = np.isfinite(pred).all(-1)
    valid3d = reference & finite_pred
    error3d = np.linalg.norm(pred - gt, axis=-1)
    xy, depth = project_world(pred, cameras, intr)
    finite2d = np.isfinite(xy).all(-1)
    valid2d = reference & finite_pred & finite2d & (depth > 0)
    error2d = np.linalg.norm(xy - reference_uv, axis=-1)
    last_ref = reference[-1]
    def aggregate(error, valid):
        if not valid.any():
            return None
        return float(error[valid].mean())
    return {
        "3d_ADE_m": aggregate(error3d, valid3d) if np.array_equal(valid3d, reference) else None,
        "3d_FDE_m": aggregate(error3d[-1], valid3d[-1]) if np.array_equal(valid3d[-1], last_ref) else None,
        "3d_conditional_ADE_m": aggregate(error3d, valid3d),
        "2d_ADE_px": aggregate(error2d, valid2d) if np.array_equal(valid2d, reference) else None,
        "2d_FDE_px": aggregate(error2d[-1], valid2d[-1]) if np.array_equal(valid2d[-1], last_ref) else None,
        "2d_conditional_ADE_px": aggregate(error2d, valid2d),
        "2d_conditional_FDE_px": aggregate(error2d[-1], valid2d[-1]),
        "3d_valid_pairs": int(valid3d.sum()),
        "2d_valid_pairs": int(valid2d.sum()),
        "reference_pairs": int(reference.sum()),
        "3d_invalid_predictions": int((reference & ~finite_pred).sum()),
        "2d_invalid_projection": int((reference & (~finite2d | (depth <= 0))).sum()),
        "2d_nonpositive_depth": int((reference & (depth <= 0)).sum()),
        "2d_invalid_step_point_indices": [[int(t + 1), int(p)] for t, p in np.argwhere(reference & ~valid2d)],
        "3d_error_by_horizon_m": [aggregate(error3d[t], valid3d[t]) for t in range(30)],
        "2d_error_by_horizon_px": [aggregate(error2d[t], valid2d[t]) for t in range(30)],
    }, error3d, error2d, xy, depth


def make_input_figure(point_ids, uv):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for j, frame in enumerate((110, 111, 112)):
        ax = axes[j]
        ax.imshow(Image.open(ROOT / f"frames/frame_{frame:03d}.jpg"))
        ax.scatter(uv[j, :, 0], uv[j, :, 1], s=38, c=COLORS["GT"], edgecolors="black")
        for i, xy in zip(point_ids, uv[j]):
            ax.annotate(str(int(i)), xy, xytext=(3, -6), textcoords="offset points", color="yellow", fontsize=8)
        ax.set(xlim=(0, 512), ylim=(512, 0), title=f"History: source frame {frame}")
    fig.savefig(ROOT / "input_frames.png", dpi=140)
    plt.close(fig)


def make_future_figure(point_ids, uv_gt, predictions, cameras, intr):
    steps = (0, 7, 17, 29)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for ax, step in zip(axes.flat, steps):
        t = 113 + step
        ax.imshow(Image.open(ROOT / f"frames/frame_{t:03d}.jpg"))
        ax.scatter(uv_gt[step, :, 0], uv_gt[step, :, 1], c=COLORS["GT"], s=30, label="GT", edgecolors="black")
        for label in ("A", "B"):
            uv, depth = project_world(predictions[label][step:step+1], cameras[step:step+1], intr)
            good = np.isfinite(uv[0]).all(-1) & (depth[0] > 0)
            ax.scatter(uv[0, good, 0], uv[0, good, 1], facecolors="none", edgecolors=COLORS[label], s=70, label=label)
        for i, xy in zip(point_ids, uv_gt[step]):
            ax.annotate(str(int(i)), xy, xytext=(3, -6), textcoords="offset points", color="yellow", fontsize=8)
        ax.set(xlim=(0, 512), ylim=(512, 0), title=f"Future source frame {t} (step {step+1})")
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(ROOT / "future_overlays.png", dpi=140)
    plt.close(fig)


def make_3d_figure(point_ids, history, gt, predictions):
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    for p in range(8):
        ax = fig.add_subplot(2, 4, p+1, projection="3d")
        ax.plot(history[:, p, 0], history[:, p, 1], history[:, p, 2], color="gray", marker="o", markersize=2, label="history")
        for label, values in (("GT", gt), ("A", predictions["A"]), ("B", predictions["B"])):
            track = values[:, p]
            ax.plot(track[:, 0], track[:, 1], track[:, 2], color=COLORS[label], label=label, linewidth=1.6)
            ax.scatter(*track[-1], color=COLORS[label], s=10)
        ax.set_title(f"Source point {point_ids[p]}")
        ax.set_xlabel("X m", fontsize=7)
        ax.set_ylabel("Y m", fontsize=7)
        ax.set_zlabel("Z m", fontsize=7)
        ax.tick_params(labelsize=6)
        if p == 0:
            ax.legend(fontsize=6)
    fig.savefig(ROOT / "trajectories_common_3d.png", dpi=135)
    plt.close(fig)


def make_error_figure(metrics):
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    x = np.arange(1, 31)
    for name in ("A", "B", "Static", "Velocity"):
        axes[0].plot(x, metrics[name]["3d_error_by_horizon_m"], label=name, color=COLORS[name])
        axes[1].plot(x, metrics[name]["2d_error_by_horizon_px"], label=name, color=COLORS[name])
    axes[0].set(ylabel="3D mean error, m", title="Same 8 points and 30 source frames")
    axes[1].set(xlabel="Future step (1–30)", ylabel="2D mean error, px")
    axes[1].set_yscale("symlog", linthresh=80)
    axes[1].set_title("2D projection error; symlog scale (velocity: 2/240 projections invalid)")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(ROOT / "error_by_horizon.png", dpi=140)
    plt.close(fig)


def control_tests(history, gt, mask, cameras, intr, uv_gt, w2c_t0):
    same, *_ = score(gt, gt, mask, cameras, intr, uv_gt)
    assert same["3d_ADE_m"] < 1e-12 and same["3d_FDE_m"] < 1e-12
    shifted = gt + np.array([0.1, 0, 0])
    shift, *_ = score(shifted, gt, mask, cameras, intr, uv_gt)
    assert abs(shift["3d_ADE_m"] - 0.1) < 1e-12
    half = mask.copy()
    half[::2] = False
    partial, *_ = score(shifted, gt, half, cameras, intr, uv_gt)
    assert partial["reference_pairs"] == 120
    empty_raised = False
    try:
        score(gt, gt, np.zeros_like(mask), cameras, intr, uv_gt)
    except AssertionError:
        empty_raised = True
    assert empty_raised
    broken = gt.copy()
    broken[0, 0] = np.nan
    invalid, *_ = score(broken, gt, mask, cameras, intr, uv_gt)
    assert invalid["3d_invalid_predictions"] == 1 and invalid["3d_ADE_m"] is None
    assert invalid["2d_ADE_px"] is None
    # A known synthetic camera: point [1,2,4] in world, camera shifted by +1 X.
    camera = np.eye(4)[None]
    camera[0, 0, 3] = -1
    uv, depth = project_world(np.array([[[1., 2., 4.]]]), camera, np.array([100., 100., 50., 60.]))
    assert np.allclose(uv[0, 0], [50., 110.]) and depth[0, 0] == 4
    R = w2c_t0[:3, :3]
    tvec = w2c_t0[:3, 3]
    H_cam = history @ R.T + tvec
    H_back = (H_cam - tvec) @ R
    assert np.max(np.abs(H_back - history)) < 1e-12
    translation = np.array([10., -5., 2.])
    deltas_original = history - history[-1, 0]
    deltas_translated = (history + translation) - (history[-1, 0] + translation)
    assert np.max(np.abs(deltas_original - deltas_translated)) < 1e-12
    return {"identity_ADE_m": same["3d_ADE_m"], "shifted_ADE_m": shift["3d_ADE_m"], "partial_mask_pairs": partial["reference_pairs"], "empty_mask_rejected": empty_raised, "invalid_prediction_detected": invalid["3d_invalid_predictions"], "synthetic_projection_pass": True, "rigid_inverse_pass": True, "anchor_translation_cancel_pass": True}


def main():
    run_status = json.loads((ROOT / "run_status.json").read_text())
    assert run_status["success"] and run_status["labels_run"] == ["A", "B"]
    preflight = json.loads((ROOT / "preflight.json").read_text())
    with np.load(ROOT / "episode.npz") as ep:
        point_ids = ep["point_ids"].copy()
        world = ep["world_xyz"].copy()
        mask = ep["visibility"][3:].copy()
        cameras = ep["cameras_w2c"][3:].copy()
        intr = ep["intrinsics"].copy()
        source_uv = ep["source_uv"].copy()
        w2c_t0 = ep["w2c_t0"].copy()
        fps = float(ep["fps"])
    gt = world[3:]
    assert gt.shape == (30, 8, 3) and mask.shape == (30, 8) and mask.all()
    predictions = {}
    raw_checks = {}
    for label in ("A", "B"):
        with np.load(ROOT / f"prediction_{label}.npz") as p:
            predictions[label] = p["future_world"].transpose(1, 0, 2).copy()
            native = p["native_xyz"].copy()
            stored_delta = p["quantized_deltas"].copy()
            assert np.array_equal(p["source_point_ids"], point_ids)
            assert np.array_equal(p["source_future_frames"], np.arange(113, 143))
        decoded = strict_decode((ROOT / f"model_output_{label}_raw.txt").read_text())
        assert np.array_equal(decoded, stored_delta)
        proc = torch.load(ROOT / f"processor_{label}.pt", map_location="cpu", weights_only=False)
        assert np.allclose(native, decoded + proc["anchor_3d"].numpy(), atol=1e-6, rtol=0)
        if label == "A":
            assert np.max(np.abs(predictions[label] - native.transpose(1, 0, 2))) < 1e-12
        else:
            check = (native.transpose(1, 0, 2) - w2c_t0[:3, 3]) @ w2c_t0[:3, :3]
            assert np.max(np.abs(predictions[label] - check)) < 1e-12
        raw_checks[label] = {"frames": 30, "unique_ids_per_frame": 8, "positions": 240, "finite": bool(np.isfinite(native).all()), "raw_sha256": sha(ROOT / f"model_output_{label}_raw.txt")}
    hist = world[:3]
    predictions["Static"] = np.repeat(hist[-1:,:,:], 30, axis=0)
    dt = 1.0 / fps
    steps = (np.arange(1, 31)[:, None, None] * dt)
    velocity = (hist[-1] - hist[-2]) / dt
    predictions["Velocity"] = hist[-1] + steps * velocity
    metrics = {}
    error_arrays = {}
    for label, prediction in predictions.items():
        m, e3, e2, uv, depth = score(prediction, gt, mask, cameras, intr, source_uv[3:])
        metrics[label] = m
        error_arrays[label + "_3d_m"] = e3
        error_arrays[label + "_2d_px"] = e2
        error_arrays[label + "_projected_uv"] = uv
        error_arrays[label + "_depth_m"] = depth

    # A rigid pose applied to prediction and reference must preserve every 3D L2 error.
    R = w2c_t0[:3, :3]
    t = w2c_t0[:3, 3]
    invariance = {}
    for label in ("A", "B", "Static", "Velocity"):
        err_world = np.linalg.norm(predictions[label] - gt, axis=-1)
        err_cam = np.linalg.norm((predictions[label] @ R.T + t) - (gt @ R.T + t), axis=-1)
        invariance[label] = float(np.max(np.abs(err_world - err_cam)))
        assert invariance[label] < 1e-12
    controls = control_tests(hist, gt, mask, cameras, intr, source_uv[3:], w2c_t0)

    proc_a = torch.load(ROOT / "processor_A.pt", map_location="cpu", weights_only=False)
    proc_b = torch.load(ROOT / "processor_B.pt", map_location="cpu", weights_only=False)
    ids_a = proc_a["input_ids"][0].tolist()
    ids_b = proc_b["input_ids"][0].tolist()
    opcodes = difflib.SequenceMatcher(None, ids_a, ids_b, autojunk=False).get_opcodes()
    changed_tokens = sum(max(i2-i1, j2-j1) for op, i1, i2, j1, j2 in opcodes if op != "equal")
    preflight["input_token_differences"] = changed_tokens
    preflight["input_token_differences_aligned"] = changed_tokens
    preflight["input_token_equal_count_aligned"] = sum(i2-i1 for op, i1, i2, _, _ in opcodes if op == "equal")
    preflight["token_difference_method"] = "SequenceMatcher alignment; sum max(replaced/deleted A span, replaced/inserted B span)"
    preflight["processor_image_shape"] = list(proc_a["images"].shape)
    preflight["processor_image_sha256"] = hashlib.sha256(proc_a["images"].numpy().tobytes()).hexdigest()
    assert torch.equal(proc_a["images"], proc_b["images"])
    patches = proc_a["images"][0].numpy()
    assert patches.shape == (3, 729, 588) and patches.dtype == np.uint8
    pixels = patches.reshape(3, 27, 27, 14, 14, 3).transpose(0, 1, 3, 2, 4, 5).reshape(3, 378, 378, 3)
    sheet = Image.new("RGB", (3 * 378, 410), "#111111")
    for j, frame in enumerate((110, 111, 112)):
        sheet.paste(Image.fromarray(pixels[j]), (j * 378, 32))
        ImageDraw.Draw(sheet).text((j * 378 + 8, 8), f"Processor pixels: source frame {frame}", fill="white")
    sheet.save(ROOT / "processor_images.png")
    preflight["processor_image_size_wh"] = [378, 378]
    (ROOT / "preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")

    make_input_figure(point_ids, source_uv[:3])
    make_future_figure(point_ids, source_uv[3:], predictions, cameras, intr)
    make_3d_figure(point_ids, hist, gt, predictions)
    make_error_figure(metrics)
    np.savez_compressed(ROOT / "error_arrays.npz", **error_arrays, gt_visibility=mask, gt_world=gt)
    output = {
        "episode": preflight["source_clip"], "history_frames": [110, 111, 112],
        "future_frames": list(range(113, 143)), "point_ids": point_ids.tolist(),
        "fps": fps, "metrics": metrics, "raw_checks": raw_checks,
        "rigid_error_invariance_max_m": invariance,
        "controls": controls,
        "prediction_difference_world_mean_m": float(np.linalg.norm(predictions["A"] - predictions["B"], axis=-1).mean()),
        "prediction_difference_world_final_mean_m": float(np.linalg.norm(predictions["A"][-1] - predictions["B"][-1], axis=-1).mean()),
        "input_token_differences_aligned": changed_tokens,
        "files_sha256": {name: sha(ROOT / name) for name in ("episode.npz", "prediction_A.npz", "prediction_B.npz", "model_output_A_raw.txt", "model_output_B_raw.txt")},
    }
    (ROOT / "metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"metrics": {label: {k: m[k] for k in ("3d_ADE_m", "3d_FDE_m", "2d_ADE_px", "2d_FDE_px", "3d_valid_pairs", "2d_valid_pairs")} for label, m in metrics.items()}, "aligned_changed_tokens": changed_tokens}, indent=2))


if __name__ == "__main__":
    main()
