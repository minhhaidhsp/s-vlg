"""RPR-CoAttention: 2D relative-position-aware co-attention between question
tokens and image patches (Shaw et al., 2018, extended to a 2D patch grid).

Design note on where the relative-position bias applies: the bias (Eq. 9-14)
is defined between PATCH pairs (i, j), since only patches carry 2D grid
coordinates — question tokens don't. So the module runs in two stages:

  Stage 1 (RPR2DSelfAttention): patches attend to patches, with the 2D
    relative-position bias baked into both the key and value branches
    (Eq. 12-14). This produces spatially-aware patch features.
  Stage 2 (plain cross-attention): question tokens (query) attend to the
    Stage-1 refined patch features (key/value). No relative-position bias is
    applied here — the query side has no image coordinate, and the patches'
    spatial relations were already folded in during Stage 1.

Stage 2's output is pooled over the question-token dimension to produce the
final spatially-aligned visual feature h_vis [B, d].
"""

import torch
import torch.nn as nn


class RPR2DSelfAttention(nn.Module):
    """Patch self-attention with a learned 2D relative-position bias.

    Args:
        d: feature/embedding dimension (shared by q, k, v and the bias tables).
        k: clipping radius for relative row/col offsets (Eq. 9-10), default 8.
    """

    def __init__(self, d: int, k: int = 8):
        super().__init__()
        self.d = d
        self.k = k
        num_buckets = (2 * k + 1) ** 2  # e.g. 289 for k=8

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

        # Eq. (11): learned lookup tables W_K, W_V producing a_ij_K, a_ij_V.
        self.rel_k_table = nn.Embedding(num_buckets, d)
        self.rel_v_table = nn.Embedding(num_buckets, d)

    def _relative_position_index(self, patch_coords: torch.Tensor) -> torch.Tensor:
        """Eq. (9)-(10): clipped relative offsets -> single bucket index for Eq. (11).

        Args:
            patch_coords: [N_p, 2] (row, col) grid coordinates.

        Returns:
            LongTensor [N_p, N_p] with index[i, j] = bucket((r_j - r_i), (c_j - c_i)).
        """
        r = patch_coords[:, 0]
        c = patch_coords[:, 1]
        N_p = patch_coords.shape[0]

        # delta[i, j] = coord_j - coord_i
        delta_r = (r.view(1, N_p) - r.view(N_p, 1)).clamp(-self.k, self.k) + self.k  # Eq. (9), shifted to [0, 2k]
        delta_c = (c.view(1, N_p) - c.view(N_p, 1)).clamp(-self.k, self.k) + self.k  # Eq. (10), shifted to [0, 2k]
        index = delta_r * (2 * self.k + 1) + delta_c
        return index.long()

    def forward(self, patch_tokens: torch.Tensor, patch_coords: torch.Tensor) -> torch.Tensor:
        """Returns spatially-aware patch features o [B, N_p, d]."""
        B, N_p, d = patch_tokens.shape
        q = self.q_proj(patch_tokens)  # [B, N_p, d]
        k_ = self.k_proj(patch_tokens)
        v_ = self.v_proj(patch_tokens)

        rel_index = self._relative_position_index(patch_coords)  # [N_p, N_p]
        a_K = self.rel_k_table(rel_index)  # [N_p, N_p, d], Eq. (11)
        a_V = self.rel_v_table(rel_index)  # [N_p, N_p, d], Eq. (11)

        # Eq. (12): e_ij = q_i . (k_j + a_ij_K)^T / sqrt(d).
        # Expanded as q_i.k_j + q_i.a_ij_K to avoid materializing a [B,N_p,N_p,d]
        # tensor (a_K/a_V have no batch dim, so this is exactly equivalent but
        # far more memory-efficient than broadcasting k_ + a_K across the batch).
        e_qk = torch.einsum("bid,bjd->bij", q, k_) / (d ** 0.5)
        e_qa = torch.einsum("bid,ijd->bij", q, a_K) / (d ** 0.5)
        e = e_qk + e_qa  # [B, N_p, N_p]

        # Eq. (13)
        alpha = torch.softmax(e, dim=-1)

        # Eq. (14): o_i = sum_j alpha_ij * (v_j + a_ij_V), same expand-avoiding trick.
        o_v = torch.einsum("bij,bjd->bid", alpha, v_)
        o_a = torch.einsum("bij,ijd->bid", alpha, a_V)
        o = o_v + o_a  # [B, N_p, d]

        return o


class RPRCoAttention(nn.Module):
    """Full co-attention block: RPR patch self-attention, then question->patch cross-attention.

    Args:
        d: feature dimension (must match ViT patch embedding dim, e.g. 768).
        k: relative-position clipping radius (Eq. 9-10), default 8.
    """

    def __init__(self, d: int, k: int = 8):
        super().__init__()
        self.d = d
        self.patch_self_attn = RPR2DSelfAttention(d, k=k)

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        patch_coords: torch.Tensor,
        question_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, N_p, d] ViT patch features (VisionEncoder output).
            patch_coords: [N_p, 2] (row, col) grid coordinates (VisionEncoder output).
            question_tokens: [B, L, d] question token embeddings.

        Returns:
            h_vis: [B, d] spatially-aligned visual feature.
        """
        # Stage 1 — Eq. (9)-(14): patch-patch self-attention with 2D RPR bias.
        patch_refined = self.patch_self_attn(patch_tokens, patch_coords)  # [B, N_p, d]

        # Stage 2: plain cross-attention, question tokens attend to refined patches.
        # No RPR bias on this side (see module docstring).
        q = self.q_proj(question_tokens)   # [B, L, d]
        k_ = self.k_proj(patch_refined)    # [B, N_p, d]
        v_ = self.v_proj(patch_refined)    # [B, N_p, d]

        scores = torch.einsum("bld,bnd->bln", q, k_) / (self.d ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.einsum("bln,bnd->bld", attn, v_)  # [B, L, d]

        # Eq. (Σ over question tokens): pool the per-question-token context into h_vis.
        h_vis = context.mean(dim=1)  # [B, d]
        return h_vis


def _self_test() -> bool:
    torch.manual_seed(0)
    B, N_p, d, L, k = 2, 196, 768, 20, 8

    patch_tokens = torch.randn(B, N_p, d)
    gh, gw = 14, 14
    rows = torch.arange(gh).view(-1, 1).expand(gh, gw).reshape(-1)
    cols = torch.arange(gw).view(1, -1).expand(gh, gw).reshape(-1)
    patch_coords = torch.stack([rows, cols], dim=-1)  # [196, 2]
    question_tokens = torch.randn(B, L, d)

    model = RPRCoAttention(d=d, k=k)
    h_vis = model(patch_tokens, patch_coords, question_tokens)

    ok = True
    if h_vis.shape != (B, d):
        print(f"FAIL: h_vis shape {tuple(h_vis.shape)} != {(B, d)}")
        ok = False
    if torch.isnan(h_vis).any():
        print("FAIL: NaN in h_vis")
        ok = False

    expected_buckets = (2 * k + 1) ** 2  # 289 for k=8
    n_k = model.patch_self_attn.rel_k_table.num_embeddings
    n_v = model.patch_self_attn.rel_v_table.num_embeddings
    if n_k != expected_buckets:
        print(f"FAIL: W_K table size {n_k} != {expected_buckets}")
        ok = False
    if n_v != expected_buckets:
        print(f"FAIL: W_V table size {n_v} != {expected_buckets}")
        ok = False

    print("PASS: rpr_coattention" if ok else "FAIL: rpr_coattention")
    return ok


if __name__ == "__main__":
    _self_test()
