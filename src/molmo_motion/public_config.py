"""Public configuration class for MolmoMotion.

A user-facing subset of the model's full internal config. Mirrors the
training-time settings so that:
  - `MolmoMotion.from_pretrained(...)` knows P, H, F to expect at inference;
  - `MolmoMotionProcessor.from_pretrained(...)` reads `history_size`,
    `image_size`, etc. to assemble inputs;
  - users who want to train from scratch have a documented config surface.

(The lower-level `BaseConfig`/`StrEnum`/`DType` infrastructure used by the
internal model machinery lives in `molmo_motion.config`. End users do not
typically interact with that.)
"""

from __future__ import annotations

from typing import Optional

from transformers import PretrainedConfig


class MolmoMotionConfig(PretrainedConfig):
    """Configuration for the MolmoMotion model.

    Inherits `PretrainedConfig` so `save_pretrained` / `from_pretrained`
    serialize this to / from `config.json` in the HF cache directory.

    Attributes:
        num_points: P — number of points the model predicts the trajectory
            of per example.
        history_size: H — number of history RGB frames the model consumes.
            Fixed per checkpoint; the H=1 and H=3 release variants differ
            only in this field (and the matching `max_history_frames` in
            the vision preprocessor).
        future_size: Nominal F used at training time. At inference the user
            picks any `future_horizon`; the model's max generation length
            is bounded by `max_sequence_length` regardless.
        image_size: Input resolution to the SigLIP2+DINOv2 vision encoder.
        patch_size: ViT patch size. With image_size=378 this gives
            27×27=729 patches.
        vision_embed_dim: SigLIP2 (1152) + DINOv2 (1024) concatenated.
        max_sequence_length: Context window. The pretrain v3 release is
            2560; finetuned long-horizon variants may bump to 6144.
        max_new_tokens: If None, derived at inference time from
            (max_sequence_length - input_token_budget).
        coord_scale: Internal text format multiplies meter-deltas by this
            before rounding to ints. Users don't see this —
            `predict_trajectory` un-scales for them.
    """

    model_type = "molmo_motion"

    def __init__(
        self,
        num_points: int = 8,
        history_size: int = 3,
        future_size: int = 8,
        image_size: int = 378,
        patch_size: int = 14,
        vision_embed_dim: int = 2176,
        llm_hidden_size: int = 2560,
        llm_num_layers: int = 36,
        llm_num_heads: int = 20,
        max_sequence_length: int = 2560,
        max_new_tokens: Optional[int] = None,
        coord_scale: int = 1000,
        **kwargs,
    ):
        if history_size not in (1, 3):
            raise ValueError(
                f"history_size must be 1 or 3 (release variants only); "
                f"got {history_size}"
            )
        self.num_points = num_points
        self.history_size = history_size
        self.future_size = future_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_embed_dim = vision_embed_dim
        self.llm_hidden_size = llm_hidden_size
        self.llm_num_layers = llm_num_layers
        self.llm_num_heads = llm_num_heads
        self.max_sequence_length = max_sequence_length
        self.max_new_tokens = max_new_tokens
        self.coord_scale = coord_scale
        super().__init__(**kwargs)

    @property
    def patch_grid_size(self) -> int:
        return self.image_size // self.patch_size
