"""Public processor — builds prompts + tokenizes inputs for `MolmoMotion`.

Mirrors the training-time prompt format from `trajectory_3d_dataset.py`:

    Predict the future 3D point coordinates of P points over F timestamps,
    given action: "<caption>", and history 3d point coordinates: "<tracks
    coords="...">3d object history</tracks>".

The model expects:
  - `input_ids` — tokenized prompt with placeholder image-tokens
  - `images` / `image_masks` — H history frames preprocessed for the SigLIP2+
    DINOv2 vision encoder
  - optional `metadata[].coords_2d` — per-point (P, 2) pixel coords for the
    2D-feature conditioning path

The processor handles all that and returns a dict ready to splat into
`MolmoMotion.predict_trajectory(**batch)`.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from molmo_motion.public_config import MolmoMotionConfig


_LABEL_HISTORY = "3d object history"
_COORD_SCALE = 1000


def _quantize(v: float) -> int:
    """Match `Trajectory3DDataset._quantize` — the on-the-wire integer
    representation of a real-valued delta coordinate."""
    return int(round(float(v) * _COORD_SCALE))


def _format_history_tracks(points_3d_history: np.ndarray, anchor: np.ndarray) -> str:
    """Serialize `(H, P, 3)` camera-frame history into `<tracks coords="...">`
    text using the same delta-from-anchor + ×1000 quantization the training
    pipeline uses. `anchor` is a single shared (3,) reference point — point 0
    at the last history frame — matching `Trajectory3DDataset._build_example`."""
    H, P, _ = points_3d_history.shape
    deltas = points_3d_history - anchor[None, None, :]
    frame_strings = []
    for fi in range(H):
        parts = [f"{float(fi):.1f}"]
        for pi in range(P):
            obj_id = pi + 1
            x = _quantize(deltas[fi, pi, 0])
            y = _quantize(deltas[fi, pi, 1])
            z = _quantize(deltas[fi, pi, 2])
            parts.append(f"{obj_id} {x} {y} {z}")
        frame_strings.append(" ".join(parts))
    coords_str = ";".join(frame_strings)
    return f'<tracks coords="{coords_str}">{_LABEL_HISTORY}</tracks>'


class MolmoMotionProcessor:
    """Bundle the text tokenizer, image preprocessor, and prompt builder.

    Load with `MolmoMotionProcessor.from_pretrained("allenai/Molmo-Motion-4B-H3-Pretrain")`.

    The processor builds prompts in the same format the model was trained
    on — meter-scale 3D deltas serialized into a `<tracks>` block, an
    action caption, and `H` history frames preprocessed by the SigLIP2+
    DINOv2 vision pipeline.

    Note: this is a plain class, NOT a `transformers.ProcessorMixin`, because
    the internal Molmo tokenizer is a thin wrapper (not a `PreTrainedTokenizerBase`).
    A HuggingFace-compatible auto-loadable processor is provided separately
    via the `hf_model/` adapter (Phase 4).
    """

    def __init__(self, tokenizer, config: MolmoMotionConfig,
                 mm_preprocessor=None, data_formatter=None):
        self.tokenizer = tokenizer
        self.config = config
        self._mm_preprocessor = mm_preprocessor
        self._data_formatter = data_formatter

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):  # type: ignore[override]
        """Load the processor that pairs with a `MolmoMotion` checkpoint.

        Reads the same `config.yaml` the model loads, builds the internal
        Molmo2 multimodal preprocessor + tokenizer, and wraps them.
        """
        from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig
        from molmo_motion.util import resource_path, resolve_checkpoint_dir

        pretrained_model_name_or_path = resolve_checkpoint_dir(pretrained_model_name_or_path)
        cfg_path = resource_path(pretrained_model_name_or_path, "config.yaml")
        internal_cfg = Molmo2TrajectoryConfig.load(cfg_path, key="model", validate_paths=False)
        tokenizer = internal_cfg.llm.build_tokenizer()
        # `build_preprocessor` returns an `ExamplePreprocessor`, which is the
        # right callable for processing one user-facing example dict (it wraps
        # the inner `MultimodalPreprocessor` + the `DataFormatter`).
        mm_pre = internal_cfg.build_preprocessor(
            for_inference=True,
            is_training=False,
            text_seq_len=None,
            max_seq_len=internal_cfg.llm.max_sequence_length,
        )
        # `history_size` is the number of RGB history frames the preprocessor
        # expects and the model's `<tracks>` output indexes timestamps from
        # (`start_timestamp = history_size`). The internal Molmo2 video preprocessor
        # exposes it as `mm_preprocessor.video.max_frames`. The H=1 release ckpt
        # has `max_frames=1`; the H=3 ckpt has `max_frames=3`. The legacy fallback
        # to a custom top-level `history_size` attr (default 3) is kept for
        # internal checkpoints that predate the move to `video.max_frames`.
        video_cfg = getattr(getattr(internal_cfg, "mm_preprocessor", None),
                            "video", None)
        history_size_from_pre = getattr(video_cfg, "max_frames", None)
        history_size = history_size_from_pre if history_size_from_pre is not None \
            else getattr(internal_cfg, "history_size", 3)
        public_cfg = MolmoMotionConfig(
            num_points=getattr(internal_cfg, "num_points", 8),
            history_size=history_size,
            future_size=getattr(internal_cfg, "num_future_frames", 8),
            max_sequence_length=internal_cfg.llm.max_sequence_length,
        )
        return cls(
            tokenizer=tokenizer,
            config=public_cfg,
            mm_preprocessor=mm_pre,
            data_formatter=internal_cfg.data_formatter,
        )

    def __call__(
        self,
        history_frames: List[Image.Image],
        points_2d_at_t0: torch.Tensor,
        points_3d_history: torch.Tensor,
        action: str,
        future_horizon: int = 32,
        c2w_at_t0: Optional[torch.Tensor] = None,
    ) -> dict:
        """Convert user inputs into model-ready tensors.

        Args:
            history_frames: list of PIL.Image, length must equal
                `self.config.history_size`. Ordered earliest → query (t_0).
            points_2d_at_t0: (P, 2) float tensor, pixel coords at t_0.
                Used by the SigLIP2 grid-sampler for the 2D-feature
                conditioning path.
            points_3d_history: (H, P, 3) float tensor of camera-frame XYZ
                across history. If `c2w_at_t0` is given, this is treated
                as world frame and converted; otherwise assumed to already
                be in camera-frame-at-t_0.
            action: short caption.
            future_horizon: how many future frames to ask the model to
                predict. Affects only the prompt string (`F timestamps`).
            c2w_at_t0: optional (4, 4) camera-to-world at t_0. If given,
                `points_3d_history` is treated as world-frame and
                multiplied by `w2c = inv(c2w_at_t0)` before serialization.
        """
        H_expected = self.config.history_size
        if len(history_frames) != H_expected:
            raise ValueError(
                f"history_frames length {len(history_frames)} != "
                f"config.history_size {H_expected}"
            )
        P = self.config.num_points

        points_3d_history = torch.as_tensor(points_3d_history, dtype=torch.float32)
        points_2d_at_t0 = torch.as_tensor(points_2d_at_t0, dtype=torch.float32)
        if points_3d_history.shape != (H_expected, P, 3):
            raise ValueError(
                f"points_3d_history must be ({H_expected}, {P}, 3); got {tuple(points_3d_history.shape)}"
            )
        if points_2d_at_t0.shape != (P, 2):
            raise ValueError(
                f"points_2d_at_t0 must be ({P}, 2); got {tuple(points_2d_at_t0.shape)}"
            )

        if c2w_at_t0 is not None:
            c2w = torch.as_tensor(c2w_at_t0, dtype=torch.float32)
            w2c = torch.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            pts = points_3d_history.reshape(-1, 3) @ R.T + t
            points_3d_history = pts.reshape(H_expected, P, 3)

        history_np = points_3d_history.cpu().numpy()
        # Anchor: point 0 at the last history frame (training convention,
        # see `Trajectory3DDataset._build_example`).
        anchor = history_np[-1, 0, :]
        history_tracks = _format_history_tracks(history_np, anchor)

        F = int(future_horizon)
        prompt = (
            f"Predict the future 3D point coordinates of {P} points over "
            f"{F} timestamps, given action: \"{action}\", and history 3d "
            f"point coordinates: \"{history_tracks}\"."
        )

        if self._mm_preprocessor is None or self._data_formatter is None:
            raise RuntimeError(
                "Processor is missing internal mm_preprocessor / data_formatter; "
                "load it via `MolmoMotionProcessor.from_pretrained(...)`."
            )

        # Match the training-time example shape from `Trajectory3DDataset`:
        # a `VideoFrames` plus a single `video_qa`-style message-list entry.
        from molmo_motion.data.video_loader import VideoFrames
        frames_rgb = np.stack([np.asarray(img.convert("RGB")) for img in history_frames])
        timestamps = np.arange(H_expected, dtype=np.float64) * 1.0
        video = VideoFrames(frames=frames_rgb, timestamps=timestamps, target_fps=1.0)

        example = {
            "video": video,
            "message_list": [{
                "style": "video_qa",
                "question": prompt,
                "answer": "",  # inference: empty target
            }],
            "metadata": {
                "coords_2d": points_2d_at_t0.cpu().numpy().astype(np.float32),
                "task": "3d_trajectory",
            },
        }
        batch = self._mm_preprocessor(example)

        # Add a batch dimension to every tensor (single-example inference).
        # Note: the ExamplePreprocessor outputs `input_tokens` but the model
        # expects `input_ids` (the collator does this rename when batching;
        # we replicate it here since we don't go through the collator).
        out = {}
        for k, v in batch.items():
            target_key = "input_ids" if k == "input_tokens" else k
            if isinstance(v, torch.Tensor):
                out[target_key] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                out[target_key] = torch.from_numpy(v).unsqueeze(0)
            else:
                out[target_key] = v
        out.setdefault("metadata", [example["metadata"]])
        out["future_horizon"] = F
        # Stash the anchor (camera-frame XYZ at t_0) so `predict_trajectory`
        # can convert the model's delta-from-anchor output back to absolute
        # camera-frame coordinates. Single (3,) point shared across all P points.
        out["anchor_3d"] = torch.from_numpy(anchor.astype(np.float32)).unsqueeze(0)  # (1, 3)
        # And the history size (the start_timestamp of the first future
        # frame is `H` — the next integer after the last history frame at
        # H-1).
        out["history_size"] = H_expected
        return out
