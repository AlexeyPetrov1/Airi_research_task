"""
Point feature extraction for 2D-grounded trajectory prediction.

Pipeline (no cross-attention variant):
1. MultiScalePointFeatureExtractor: grid_sample from the ANCHOR frame's ViT
   patches at 2D coords → (B, P, vit_dim).
2. Reuse the pretrained Molmo2 connector MLP (image_projector) to project to
   the LLM embedding dim → (B, P, d_model).
3. The result is injected into the `<|point_feat|>` token positions (P per
   example — one per point, no per-frame duplication).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScalePointFeatureExtractor(nn.Module):
    """Extract per-point features via multi-scale bilinear grid sampling.

    No learnable parameters — pure geometric operations.

    Args:
        patch_grid_size: Spatial size of the patch grid (27 for SigLIP2 378/14).
    """

    def __init__(self, patch_grid_size: int = 27):
        super().__init__()
        self.patch_grid_size = patch_grid_size
        self.scale2_size = (patch_grid_size + 1) // 2  # 27 -> 14
        self.scale3_size = (patch_grid_size + 3) // 4  # 27 -> 7

    def forward(self, patch_features, point_coords_2d):
        """
        Args:
            patch_features: (B, num_patches, D) where num_patches = H*W
            point_coords_2d: (B, P, 2) in [0, 1], order (x, y)
        Returns:
            (B, P, D)
        """
        B, N, D = patch_features.shape
        H = W = self.patch_grid_size

        feat_map = patch_features.reshape(B, H, W, D).permute(0, 3, 1, 2)

        scale1 = feat_map
        scale2 = F.adaptive_avg_pool2d(feat_map, (self.scale2_size, self.scale2_size))
        scale3 = F.adaptive_avg_pool2d(feat_map, (self.scale3_size, self.scale3_size))

        scale2_up = F.interpolate(scale2, size=(H, W), mode='bilinear', align_corners=True)
        scale3_up = F.interpolate(scale3, size=(H, W), mode='bilinear', align_corners=True)

        fused = (scale1 + scale2_up + scale3_up) / 3.0

        grid = point_coords_2d * 2 - 1  # [0,1] → [-1,1]
        grid = grid.unsqueeze(1)  # (B, 1, P, 2)

        sampled = F.grid_sample(
            fused, grid, mode='bilinear', align_corners=True,
            padding_mode='border',
        )  # (B, D, 1, P)

        return sampled.squeeze(2).permute(0, 2, 1)  # (B, P, D)


class PointFeatureConditioner(nn.Module):
    """Anchor-frame point feature pipeline (no cross-attention).

    Given the ANCHOR frame's raw ViT patches and per-point 2D coords:
    - Grid-sample at the coords to get a per-point feature.
    - Project to LLM dim via the pretrained connector MLP.

    Args:
        image_projector: The pretrained MLP projector from vision_backbone
            (reused so training starts from informative initialization).
        patch_grid_size: ViT patch grid size (27 for 378/14).
    """

    def __init__(self, image_projector, patch_grid_size: int = 27):
        super().__init__()
        self.extractor = MultiScalePointFeatureExtractor(patch_grid_size)
        # Store as a plain Python attribute (NOT an nn.Module child) — the
        # projector is already owned and FSDP-sharded by vision_backbone.
        object.__setattr__(self, '_projector', image_projector)

    def forward(self, anchor_patches, point_coords_2d):
        """
        Args:
            anchor_patches: (B, num_patches, vit_dim) — ViT features of the
                anchor frame only.
            point_coords_2d: (B, P, 2) in [0, 1].
        Returns:
            (B, P, d_model)
        """
        point_feats = self.extractor(anchor_patches, point_coords_2d)  # (B, P, vit_dim)
        return self._projector(point_feats)  # (B, P, d_model)
