"""Trajectory visualization helpers exposed as `from molmo_motion.viz import ...`.

Two entry points:

* `overlay_trajectory_on_image` — project a predicted 3D trajectory back
  onto the t₀ image plane using camera intrinsics and draw per-point
  polylines. Pillow-only, no matplotlib dependency.
* `render_trajectory_3d` — matplotlib 3D scatter of the predicted
  trajectory in camera frame. Requires the `[viz]` extras
  (`pip install -e .[viz]`).
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


_DEFAULT_PALETTE = (
    (255, 30, 30), (30, 200, 30), (40, 100, 255), (255, 200, 0),
    (200, 0, 200), (0, 200, 200), (180, 100, 0), (100, 100, 180),
)


def _to_numpy(t) -> np.ndarray:
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def project_camera_xyz_to_pixel(xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole projection of (..., 3) camera-frame XYZ to (..., 2) pixels."""
    eps = 1e-6
    z = np.clip(xyz[..., 2], eps, None)
    u = K[0, 0] * (xyz[..., 0] / z) + K[0, 2]
    v = K[1, 1] * (xyz[..., 1] / z) + K[1, 2]
    return np.stack([u, v], axis=-1)


def overlay_trajectory_on_image(
    t0_image: Image.Image,
    *,
    points_2d_at_t0: torch.Tensor,
    future_3d: torch.Tensor,
    intrinsics: torch.Tensor,
    palette: Optional[Iterable[tuple[int, int, int]]] = None,
    line_width: int = 2,
    dot_radius: int = 3,
) -> Image.Image:
    """Project `future_3d` into the `t0_image` plane and draw per-point
    polylines connecting the anchor 2D query to the projected 3D path.

    Args:
        t0_image: PIL.Image, the t₀ RGB frame.
        points_2d_at_t0: (P, 2) tensor of pixel coords at t₀ (the anchor
            for each polyline).
        future_3d: (P, F, 3) tensor of camera-frame XYZ in meters — the
            output of `MolmoMotion.predict_trajectory(...).future_3d`.
        intrinsics: (3, 3) tensor with `[[fx,0,cx],[0,fy,cy],[0,0,1]]`.
        palette: optional iterable of (r,g,b) tuples cycled across points.

    Returns:
        A new PIL.Image with the overlay drawn (the input image is not
        mutated).
    """
    img = t0_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    anchor = _to_numpy(points_2d_at_t0)            # (P, 2)
    pred = _to_numpy(future_3d)                    # (P, F, 3)
    K = _to_numpy(intrinsics).astype(np.float32)   # (3, 3)
    palette = tuple(palette) if palette is not None else _DEFAULT_PALETTE

    pixels = project_camera_xyz_to_pixel(pred, K)  # (P, F, 2)
    P = pred.shape[0]
    for pi in range(P):
        col = palette[pi % len(palette)]
        line = [tuple(anchor[pi].tolist())]
        line.extend(tuple(p) for p in pixels[pi].tolist())
        if len(line) >= 2:
            draw.line(line, fill=col, width=line_width)
        for px, py in pixels[pi]:
            if 0 <= px < W and 0 <= py < H:
                draw.ellipse(
                    (px - dot_radius, py - dot_radius,
                     px + dot_radius, py + dot_radius),
                    fill=col,
                )
    return img


def render_trajectory_3d(
    future_3d: torch.Tensor,
    *,
    output_path: str,
    history_3d: Optional[torch.Tensor] = None,
    palette: Optional[Iterable[tuple[int, int, int]]] = None,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> None:
    """Matplotlib 3D scatter of the predicted trajectory.

    The scatter uses camera-frame XYZ with the +Z axis pointing forward
    (away from the camera), so views look like a top-down/perspective
    plot of motion in front of the camera.

    Args:
        future_3d: (P, F, 3) tensor — predicted future, meters.
        output_path: where to save the rendered PNG.
        history_3d: optional (H, P, 3) tensor of camera-frame history
            (the anchor side of the polyline). When given, history is
            drawn in a muted color to distinguish from predicted future.
        palette: optional (r,g,b) tuples in 0–255 cycled across points.
        figsize: matplotlib figure size in inches.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    pred = _to_numpy(future_3d)               # (P, F, 3)
    hist = _to_numpy(history_3d) if history_3d is not None else None
    palette = tuple(palette) if palette is not None else _DEFAULT_PALETTE

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    P, F, _ = pred.shape
    for pi in range(P):
        rgb = tuple(c / 255.0 for c in palette[pi % len(palette)])
        xs, ys, zs = pred[pi].T
        # Plot in (X, Z, -Y) so +Z (forward) is into the screen and -Y is up.
        ax.plot(xs, zs, -ys, color=rgb, alpha=0.85, label=f"point {pi + 1}")
        ax.scatter(xs[-1], zs[-1], -ys[-1], color=rgb, s=35)
        if hist is not None:
            hxs, hys, hzs = hist[:, pi, :].T
            ax.plot(hxs, hzs, -hys, color=rgb, alpha=0.3, linestyle="--")
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.set_zlabel("-Y (up, m)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Predicted trajectory — {P} points × {F} frames")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
