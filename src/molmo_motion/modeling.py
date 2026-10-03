"""Public model class — thin wrapper around the internal `Molmo2Trajectory`.

User-facing surface:

    from molmo_motion import MolmoMotion, MolmoMotionProcessor

    proc = MolmoMotionProcessor.from_pretrained("allenai/Molmo-Motion-4B-H3-Pretrain")
    model = MolmoMotion.from_pretrained("allenai/Molmo-Motion-4B-H3-Pretrain").cuda().eval()

    batch = proc(history_frames, points_2d_at_t0, points_3d_history, action,
                 future_horizon=32)
    out = model.predict_trajectory(**batch)
    print(out.future_3d.shape)   # (P, 32, 3) in meters, camera-frame-at-t0

The wrapper hides the internal `Molmo2Trajectory.generate(batch=...)` /
`OLMoGenerateOutput` token stream behind a single `predict_trajectory`
call that returns parsed 3D coordinates. Internally the model is the
same fully-trained `Molmo2Trajectory`; we just package its inputs and
parse its `<tracks coords="...">` text output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import PreTrainedModel

from molmo_motion.eval.egodex_3d_evaluator import parse_tracks_text, tracks_to_array
from molmo_motion.models.molmo2.molmo2_trajectory import (
    Molmo2Trajectory,
    Molmo2TrajectoryConfig,
)
from molmo_motion.public_config import MolmoMotionConfig
from molmo_motion.train.checkpointer import load_model_state


@dataclass
class MolmoMotionOutput:
    """Return type of `MolmoMotion.predict_trajectory`."""

    future_3d: torch.Tensor
    """Predicted 3D coordinates in camera-frame-at-t0, shape `(P, F, 3)`,
    units = meters. F = `future_horizon` requested by the caller (or as
    long as the model emitted before EOS, whichever is shorter).
    """

    future_text: str
    """Raw decoded `<tracks coords="...">` block from the model. Useful for
    debugging tokenization issues."""


class MolmoMotion(PreTrainedModel):
    """4B-parameter trajectory prediction model.

    Loads with `MolmoMotion.from_pretrained("allenai/Molmo-Motion-4B-H{1,3}-Pretrain")`.
    Generation entry point is `predict_trajectory(**inputs)` — runs the LLM
    decoder once and returns the parsed trajectory.

    Internally this is `molmo_motion.models.molmo2.molmo2_trajectory.Molmo2Trajectory`
    with the standard training-time `Molmo2TrajectoryConfig`. The public
    `MolmoMotionConfig` is a small projection of that config exposing only
    the fields a user needs to read.
    """

    config_class = MolmoMotionConfig
    base_model_prefix = "molmo_motion"

    def __init__(
        self,
        config: MolmoMotionConfig,
        internal_config: Optional[Molmo2TrajectoryConfig] = None,
    ):
        super().__init__(config)
        self.config: MolmoMotionConfig = config
        if internal_config is None:
            raise ValueError(
                "MolmoMotion must be constructed via `from_pretrained` so the "
                "full internal `Molmo2TrajectoryConfig` is available."
            )
        self._internal_config = internal_config
        self._internal: Molmo2Trajectory = internal_config.build_model()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):  # type: ignore[override]
        """Load a checkpoint produced by the public release training script.

        `pretrained_model_name_or_path` is a directory (local path or HF Hub
        repo) containing both a `config.yaml` (with a top-level `model:`
        key) and the saved model state shards.
        """
        from molmo_motion.models.molmo2.molmo2_trajectory import Molmo2TrajectoryConfig as _Cfg
        from molmo_motion.util import resource_path, resolve_checkpoint_dir

        pretrained_model_name_or_path = resolve_checkpoint_dir(pretrained_model_name_or_path)
        cfg_path = resource_path(pretrained_model_name_or_path, "config.yaml")
        internal_cfg = _Cfg.load(cfg_path, key="model", validate_paths=False)

        public_cfg = MolmoMotionConfig(
            num_points=getattr(internal_cfg, "num_points", 8),
            history_size=getattr(internal_cfg, "history_size", 3),
            future_size=getattr(internal_cfg, "num_future_frames", 8),
            max_sequence_length=internal_cfg.llm.max_sequence_length,
        )
        model = cls(public_cfg, internal_config=internal_cfg)
        load_model_state(pretrained_model_name_or_path, model._internal)
        model._internal.eval()
        return model

    def predict_trajectory(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        coords_2d: Optional[torch.Tensor] = None,
        metadata: Optional[list] = None,
        max_new_tokens: Optional[int] = None,
        future_horizon: int = 32,
        **batch_extras,
    ) -> MolmoMotionOutput:
        """Run a single forward+generate pass and return parsed trajectories.

        Args (all batch-1; for batched inference see :doc:`/inference`):
            input_ids, attention_mask, images, image_masks, coords_2d,
                metadata: as produced by `MolmoMotionProcessor`.
            max_new_tokens: override the auto-derived generation budget.
                Defaults to a heuristic of `48 * future_horizon` tokens.
            future_horizon: how many future frames the user wants back.
                Used both as the budget heuristic and to truncate the
                parsed trajectory tensor to a known shape.
        """
        if max_new_tokens is None:
            # Match the production eval budget: every future frame emits up to
            # ~160 quantized-coord tokens (P=8 points × {obj_id, x, y, z} +
            # frame timestamp + separators). 48 × F was a guesstimate that
            # truncates the `<tracks>` block before the closing quote, leaving
            # the regex parser unable to recover any frames. 160 × F leaves
            # comfortable slack and matches the cached predictions byte-for-byte.
            max_new_tokens = 160 * future_horizon

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if images is not None:
            batch["images"] = images
        if image_masks is not None:
            batch["image_masks"] = image_masks
        if metadata is not None:
            batch["metadata"] = metadata
        # Pass through any other processor-emitted tensors (token_pooling,
        # position_ids, etc.) so the internal model.generate() can use them.
        # `target_tokens` and `loss_masks` are training-only; skip them.
        for k, v in batch_extras.items():
            if k in ("target_tokens", "loss_masks"):
                continue
            batch[k] = v

        with torch.inference_mode():
            gen_out = self._internal.generate(
                batch=batch,
                max_steps=max_new_tokens,
                is_distributed=False,
            )

        token_ids = gen_out.token_ids[:, 0].detach().cpu().numpy()[0]
        tokenizer = self._internal.config.llm.build_tokenizer()
        future_text = tokenizer.decode(token_ids[token_ids >= 0])

        parsed = parse_tracks_text(future_text)
        if parsed is None:
            # Model produced no valid `<tracks>` block — return zeros + the
            # raw text so the user can debug.
            P = self.config.num_points
            future_3d = torch.zeros((P, future_horizon, 3), dtype=torch.float32)
            return MolmoMotionOutput(future_3d=future_3d, future_text=future_text)

        # First future timestamp is `H` (history occupies frames 0..H-1).
        H = batch_extras.get("history_size", self.config.history_size)
        delta, _vis = tracks_to_array(
            parsed,
            num_points=self.config.num_points,
            num_frames=future_horizon,
            start_timestamp=float(H),
        )
        future_3d = torch.from_numpy(np.asarray(delta, dtype=np.float32))
        # Add the anchor (camera-frame XYZ at t_0) back to recover absolute
        # camera-frame coords. The processor stashes this in `batch_extras`.
        # `anchor_3d` is a single shared (3,) point — see processor for the
        # training-convention anchor choice.
        anchor_3d = batch_extras.get("anchor_3d")
        if anchor_3d is not None:
            anchor = anchor_3d.detach().cpu().squeeze(0)  # (3,)
            future_3d = future_3d + anchor  # (P, F, 3) + (3,) broadcasts
        return MolmoMotionOutput(future_3d=future_3d, future_text=future_text)

    def forward(self, *args, **kwargs):  # noqa: D401 — required by PreTrainedModel
        """Delegate to the internal model's `forward`. Most users should call
        `predict_trajectory` instead, which handles generation + parsing."""
        return self._internal(*args, **kwargs)
