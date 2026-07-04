"""Vision branch — ViT patch-grid feature extractor.

Keeps the 2D patch grid structure (row, col per patch) instead of pooling to
a single vector, because RPRCoAttention (src/models/rpr_coattention.py) needs
each patch's grid coordinates to compute 2D relative-position bias.
"""

import timm
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """ViT-Base/16 patch feature extractor.

    Args:
        vit_name: timm model name (config `model.vit_name`), default
            "vit_base_patch16_224" to match the paper's ViT-Base/16 backbone.
        test_mode: if True, builds the same architecture but with random
            (untrained) weights for fast local/CPU shape testing — no
            pretrained checkpoint download. If False (real runs), loads
            ImageNet-pretrained weights, per the paper.
    """

    def __init__(self, vit_name: str = "vit_base_patch16_224", test_mode: bool = False):
        super().__init__()
        pretrained = not test_mode
        self.vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.grid_size = self.vit.patch_embed.grid_size  # (14, 14) for 224/16
        self.d = self.vit.embed_dim  # 768 for ViT-Base

    def patch_coords(self) -> torch.Tensor:
        """Grid (row, col) coordinate of each patch, in raster order — [N_p, 2]."""
        gh, gw = self.grid_size
        rows = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
        cols = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
        return torch.stack([rows, cols], dim=-1)

    def forward(self, images: torch.Tensor):
        """Eq. (4): H_img = ViT_patch(images) -> [B, N_p, d].

        Returns:
            patch_features: [B, N_p, d] (CLS token dropped; grid kept intact).
            patch_coords: [N_p, 2] (row, col) grid coordinates, shared across the batch.
        """
        tokens = self.vit.forward_features(images)  # [B, 1 + N_p, d]
        patch_features = tokens[:, 1:, :]  # drop CLS token, keep the 2D patch grid
        return patch_features, self.patch_coords()


def _self_test() -> bool:
    torch.manual_seed(0)
    images = torch.randn(2, 3, 224, 224)

    model = VisionEncoder(vit_name="vit_base_patch16_224", test_mode=True)
    patch_features, coords = model(images)

    ok = True
    if patch_features.shape != (2, 196, 768):
        print(f"FAIL: patch_features shape {tuple(patch_features.shape)} != (2, 196, 768)")
        ok = False
    if coords.shape != (196, 2):
        print(f"FAIL: coords shape {tuple(coords.shape)} != (196, 2)")
        ok = False
    if torch.isnan(patch_features).any():
        print("FAIL: NaN in patch_features")
        ok = False
    if coords.min().item() < 0 or coords.max().item() > 13:
        print(f"FAIL: coords out of [0,13] range: min={coords.min().item()}, max={coords.max().item()}")
        ok = False

    print("PASS: vision_encoder" if ok else "FAIL: vision_encoder")
    return ok


if __name__ == "__main__":
    _self_test()
