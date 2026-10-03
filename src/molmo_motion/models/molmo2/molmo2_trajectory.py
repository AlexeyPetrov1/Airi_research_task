"""
Molmo2 extended with 2D point feature injection for trajectory prediction.

Anchor-frame pipeline (no cross-attention):
1. Cache raw ViT features during vision backbone forward.
2. The dataloader feeds exactly H history frames `[t-H+1, …, t]` to the ViT.
   The 2D-feature anchor is the LAST history frame `t` (= paper's t_0); no
   extra "anchor frame" is prepended to the video. Its ViT patches are the
   LAST crop in the cached feature tensor.
3. Grid-sample the last crop at the per-point 2D coords (also sampled at
   frame t), then project via the pretrained connector MLP to get P
   per-point features.
4. Inject into the P `<|point_feat|>` token positions (one per point, no
   per-frame duplication).
"""

import dataclasses
import logging
from dataclasses import field
from typing import ClassVar, Optional, Tuple, Sequence

import torch
import torch.nn as nn

from molmo_motion import tokenizer
from molmo_motion.models.molmo2.molmo2 import Molmo2, Molmo2Config
from molmo_motion.nn.point_features import PointFeatureConditioner

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Molmo2TrajectoryConfig(Molmo2Config):
    """Config for Molmo2 with point feature conditioning."""

    # Point feature conditioner (reuses pretrained connector weights)
    use_2d_point_features: bool = False
    point_feat_patch_grid_size: int = 27  # SigLIP2: 378 / 14 = 27

    _model_name: ClassVar[str] = "video_olmo_trajectory"

    def build_model(self, device=None):
        return Molmo2Trajectory(self, device)


class Molmo2Trajectory(Molmo2):
    """Molmo2 with 2D point feature injection for trajectory prediction.

    When `config.use_2d_point_features` is True:
    - Caches raw ViT features during vision forward
    - On forward, finds <|point_feat|> tokens and injects grid-sampled + refined features
    """

    def __init__(self, config: Molmo2TrajectoryConfig, device=None):
        super().__init__(config, device)

        self._point_feat_token_id = self.special_ids.get(tokenizer.POINT_FEATURE_TOKEN)
        self._cached_raw_vit_features = None
        self._current_coords_2d = None

        if config.use_2d_point_features:
            # Reuse the pretrained connector MLP (image_projector) so the
            # per-point features start from an informative initialization.
            # No cross-attention — we only grid-sample the anchor frame.
            assert self.vision_backbone is not None, "Vision backbone required for 2D point features"
            self.point_conditioner = PointFeatureConditioner(
                image_projector=self.vision_backbone.image_projector,
                patch_grid_size=config.point_feat_patch_grid_size,
            )
            log.info(f"PointFeatureConditioner: anchor-frame grid-sample + pretrained "
                     f"projector, grid_size={config.point_feat_patch_grid_size}")
        else:
            self.point_conditioner = None

    def _cache_vit_features_hook(self, image_features_raw):
        """Called after encode_image to cache raw ViT features."""
        # image_features_raw: (B, T_crops, num_patches, vit_dim)
        self._cached_raw_vit_features = image_features_raw

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings=None,
        attention_mask=None,
        attention_bias=None,
        response_mask=None,
        subsegment_ids=None,
        position_ids=None,
        labels=None,
        loss_masks=None,
        images=None,
        image_masks=None,
        token_pooling=None,
        response_logits_only=False,
        past_key_values=None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states=None,
        append_last_valid_logits=None,
        # New: 2D point coordinates for feature injection
        **kwargs,
    ):
        """Extended forward that injects point features.

        coords_2d is extracted from metadata (list of dicts, each with optional "coords_2d" key).
        """
        # Extract coords_2d from metadata if available
        coords_2d = None
        metadata = kwargs.get("metadata")
        if self.point_conditioner is not None and metadata is not None:
            coords_list = [m.get("coords_2d") for m in metadata if isinstance(m, dict)]
            if coords_list and coords_list[0] is not None:
                import numpy as np
                coords_2d = torch.tensor(
                    np.stack([c for c in coords_list if c is not None]),
                    dtype=torch.float32,
                    device=input_ids.device if input_ids is not None else "cuda",
                )

        self._current_coords_2d = coords_2d

        # If we have point conditioning and images, we need to intercept the ViT features.
        # We monkey-patch the vision backbone to cache raw features.
        if (self.point_conditioner is not None and images is not None
                and self._current_coords_2d is not None):
            orig_encode = self.vision_backbone.encode_image

            def encode_and_cache(imgs):
                features = orig_encode(imgs)
                self._cache_vit_features_hook(features)
                return features

            self.vision_backbone.encode_image = encode_and_cache
            try:
                output = self._forward_with_point_injection(
                    input_ids=input_ids,
                    input_embeddings=input_embeddings,
                    attention_mask=attention_mask,
                    attention_bias=attention_bias,
                    response_mask=response_mask,
                    subsegment_ids=subsegment_ids,
                    position_ids=position_ids,
                    labels=labels,
                    loss_masks=loss_masks,
                    images=images,
                    image_masks=image_masks,
                    token_pooling=token_pooling,
                    response_logits_only=response_logits_only,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    last_logits_only=last_logits_only,
                    output_hidden_states=output_hidden_states,
                    append_last_valid_logits=append_last_valid_logits,
                    **kwargs,
                )
            finally:
                self.vision_backbone.encode_image = orig_encode
                self._cached_raw_vit_features = None
                self._current_coords_2d = None
            return output
        else:
            # No point features — standard Molmo2 forward
            return super().forward(
                input_ids=input_ids,
                input_embeddings=input_embeddings,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                response_mask=response_mask,
                subsegment_ids=subsegment_ids,
                position_ids=position_ids,
                labels=labels,
                loss_masks=loss_masks,
                images=images,
                image_masks=image_masks,
                token_pooling=token_pooling,
                response_logits_only=response_logits_only,
                past_key_values=past_key_values,
                use_cache=use_cache,
                last_logits_only=last_logits_only,
                output_hidden_states=output_hidden_states,
                append_last_valid_logits=append_last_valid_logits,
                **kwargs,
            )

    def _forward_with_point_injection(self, **kwargs):
        """Run the standard forward but inject point features after embedding."""
        # We need to manually replicate the relevant parts of Molmo2.forward()
        # up to the embedding step, inject point features, then continue.
        #
        # However, this is fragile since Molmo2.forward() is complex.
        # Instead, we use a simpler approach: register a forward hook on the
        # embedding layer that modifies x after image feature injection.
        #
        # The parent forward does:
        #   1. x = self.transformer.wte(input_ids)
        #   2. x[is_image_patch] += image_features
        #   3. ... (transformer layers)
        #
        # We intercept after step 2 by using a temporary hook.

        input_ids = kwargs['input_ids']
        coords_2d = self._current_coords_2d
        device = input_ids.device

        # We'll inject after the parent forward completes its embedding + image injection.
        # Use a pre-forward hook on the first transformer block to catch x after embedding.
        _injected = [False]

        def inject_hook(module, args):
            """Hook on first transformer block — x is args[0] after embedding + image features."""
            if _injected[0]:
                return args
            _injected[0] = True

            x = args[0]  # (B, seq_len, d_model)

            if self._cached_raw_vit_features is None or coords_2d is None:
                return args

            # Find <|point_feat|> token positions
            is_point_feat = (input_ids == self._point_feat_token_id)  # (B, seq_len)
            n_point_tokens = is_point_feat.sum().item()
            if n_point_tokens == 0:
                return args

            # Raw ViT features: (B, T_crops, num_patches, vit_dim). The anchor
            # The 2D-feature anchor is the LAST history frame `t` (paper t_0).
            # The dataloader feeds exactly H history frames to the ViT (no
            # extra prepended anchor frame), so frame `t` is the LAST crop.
            raw_vit = self._cached_raw_vit_features

            B_coord, P, _ = coords_2d.shape
            # New contract: exactly P <|point_feat|> tokens per example.
            if n_point_tokens != B_coord * P:
                log.warning(f"Point token count mismatch: {n_point_tokens} tokens, "
                            f"expected B={B_coord} × P={P}")
                return args

            T_crops = raw_vit.shape[1]
            if T_crops < 1:
                log.warning(f"No ViT crops available for anchor-frame point features")
                return args
            anchor_patches = raw_vit[:, -1, :, :]  # (B, num_patches, vit_dim) — frame t

            # Run point conditioner (grid-sample + projector, no cross-attn)
            point_features = self.point_conditioner(
                anchor_patches.to(x.dtype),
                coords_2d.to(x.dtype).to(device),
            )  # (B, P, d_model)

            # Inject into x at <|point_feat|> positions
            x_flat = x.view(-1, x.shape[-1])
            is_flat = is_point_feat.view(-1)
            x_flat[is_flat] = x_flat[is_flat] + point_features.reshape(-1, x.shape[-1]).to(x.dtype)

            return (x,) + args[1:] if len(args) > 1 else (x,)

        # Register hook on first transformer block
        handle = self.transformer.blocks[0].register_forward_pre_hook(inject_hook)
        try:
            output = super().forward(**kwargs)
        finally:
            handle.remove()

        return output
