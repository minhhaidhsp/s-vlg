"""SU-MedVQA (Version 2): vision + text only Medical VQA model, reusing the
SAME core modules as SVLG (Version 1) — VisionEncoder, RPRCoAttention,
DisentangledFusion, VQADecoder — with the tabular (MLTM) and graph
(GraphSAGE) branches simply absent.

DisentangledFusion runs with num_branches=1 (vision only): it still splits
the fused representation into a shared/specific pair and exposes the
uncertainty score U (Eq. 28) and gate (decoder Eq. 32), even though there is
only one modality to disentangle against. RPR-CoAttention is the primary
contribution for this version; uncertainty-controlled generation is
secondary — see PROJECT_STATE.md for the two-paper split rationale.
"""

import torch
import torch.nn as nn

from src.models.decoder import VQADecoder
from src.models.disentangled_fusion import DisentangledFusion
from src.models.rpr_coattention import RPRCoAttention
from src.models.vision_encoder import VisionEncoder


class SU_MedVQA(nn.Module):
    """Assembles the vision, fusion, and decoder branches (no tabular/graph).

    Expects a config dict shaped like configs/config_v2.yaml merged over
    configs/config.yaml (see src/utils/config.py: load_version_config).
    """

    def __init__(
        self,
        config: dict,
        test_mode: bool = True,
        use_rel_pos_bias: bool = True,
        disentangle_deterministic: bool = False,
    ):
        """
        Args:
            config: see configs/config_v2.yaml merged over configs/config.yaml.
            test_mode: tiny non-quantized decoder (local/CPU) vs real LLM+QLoRA.
            use_rel_pos_bias: ablation switch (Table 9 "no_rpr" variant) — see
                RPRCoAttention docstring.
            disentangle_deterministic: ablation switch (Table 9
                "no_disentangle" variant) — see DisentangledFusion docstring.
        """
        super().__init__()
        model_cfg = config.get("model", {})
        uncertainty_cfg = config.get("uncertainty", {})

        self.test_mode = test_mode

        d_vis = 768  # ViT-Base/16 patch embedding dim (Eq. 4)
        fusion_d_shared = model_cfg.get("fusion_d_shared", 64)
        fusion_d_vis = model_cfg.get("fusion_d_vis", 64)
        question_vocab_size = model_cfg.get("question_vocab_size", 30522)

        n_prefix = model_cfg.get("n_prefix", 8)
        gamma = uncertainty_cfg.get("gamma_threshold")
        if gamma is None:
            gamma = 1.0  # placeholder until chosen via validation (config: uncertainty.gamma_threshold)

        # --- Vision branch: Eq. (4) ViT patches, then Eq. (9)-(14) RPR co-attention ---
        self.vision_encoder = VisionEncoder(
            vit_name=model_cfg.get("vit_name", "vit_base_patch16_224"), test_mode=test_mode
        )
        self.rpr_coattention = RPRCoAttention(d=d_vis, k=8, use_rel_pos_bias=use_rel_pos_bias)
        # TODO: same placeholder embedding as svlg.py — replace with the real
        # question tokenizer/embedding once wired to the actual text pipeline.
        self.question_embedding = nn.Embedding(question_vocab_size, d_vis)

        # --- Fusion: Eq. (21)-(28), num_branches=1 (vision only) ---
        branch_dims = [fusion_d_vis]
        self.fusion = DisentangledFusion(
            in_dim=d_vis, d_shared=fusion_d_shared, branch_dims=branch_dims,
            deterministic=disentangle_deterministic,
        )
        z_final_dim = fusion_d_shared + sum(branch_dims)  # Eq. (27)

        # --- Decoder: Eq. (29)-(33) soft-prefix LLM with the uncertainty gate ---
        self.decoder = VQADecoder(
            z_final_dim=z_final_dim,
            llm_name=model_cfg.get("llm_name", "Qwen/Qwen2.5-3B-Instruct"),
            test_mode=test_mode,
            n_prefix=n_prefix,
            gamma=gamma,
            lora_r=model_cfg.get("lora_r", 16),
            lora_alpha=model_cfg.get("lora_alpha", 32),
        )

    def forward(
        self,
        images: torch.Tensor,
        question_input_ids: torch.Tensor,
        question_text,
        system_text: str = "You are a helpful medical VQA assistant.",
        answer_text=None,
        verbose: bool = False,
        return_attn: bool = False,
    ):
        """End-to-end forward pass (vision + text only, no evidence retrieval).

        Args:
            images: [B, 3, 224, 224] input images.
            question_input_ids: [B, L] question token ids (co-attention branch).
            question_text: single string or list[str] of length B — question, for the decoder prompt.
            answer_text: (training only) single string or list[str] of length B.
            verbose: if True, print the shape after each step for debugging.
            return_attn: if True, also return the RPR-CoAttention question->patch
                attention map [B, L, N_p] (Figure 10 attention heatmap data).

        Returns:
            z_final: [B, d_shared + d_vis], Eq. (27).
            U: [B] per-sample uncertainty score, Eq. (28).
            fusion_losses: dict with "L_MI" (Eq. 24) and "L_KL" (Eq. 25).
            decoder_out: (loss, logits) if answer_text is given, else None.
            attn (only if return_attn=True): [B, L, N_p] attention weights.
        """
        # Step 1 — Eq. (4): ViT patch features.
        patch_features, patch_coords = self.vision_encoder(images)
        if verbose:
            print(f"[1] patch_features: {tuple(patch_features.shape)}, patch_coords: {tuple(patch_coords.shape)}")

        # Step 2 — Eq. (9)-(14): RPR co-attention -> h_vis.
        question_tokens = self.question_embedding(question_input_ids)
        attn = None
        if return_attn:
            h_vis, attn = self.rpr_coattention(patch_features, patch_coords, question_tokens, return_attn=True)
        else:
            h_vis = self.rpr_coattention(patch_features, patch_coords, question_tokens)
        if verbose:
            print(f"[2] h_vis: {tuple(h_vis.shape)}")

        # No tabular/graph branches — h is just h_vis (num_branches=1).
        h = h_vis

        # Step 3 — Eq. (21)-(28): disentangled fusion.
        z_final, U, fusion_losses = self.fusion(h)
        if verbose:
            print(f"[3] z_final: {tuple(z_final.shape)}, U: {tuple(U.shape)}, "
                  f"L_MI: {fusion_losses['L_MI'].item():.4f}, L_KL: {fusion_losses['L_KL'].item():.4f}")

        # Step 4 — Eq. (29)-(33): decoder.
        decoder_out = None
        if answer_text is not None:
            decoder_out = self.decoder(z_final, system_text, "", question_text, answer_text)
            if verbose:
                print(f"[4] decoder loss: {decoder_out[0].item():.4f}, logits: {tuple(decoder_out[1].shape)}")

        if return_attn:
            return z_final, U, fusion_losses, decoder_out, attn
        return z_final, U, fusion_losses, decoder_out

    def compute_total_loss(
        self,
        L_gen: torch.Tensor,
        fusion_losses: dict,
        epoch: int,
        alpha: float = None,
        beta_max: float = None,
        warmup_epochs: int = None,
        config: dict = None,
    ) -> torch.Tensor:
        """Eq. (34): L_total = L_gen + alpha*L_MI + beta_t*L_KL + L_MI_fit
        (no L_MLTM — V2 has no tabular branch).

        beta_t follows the KL-annealing schedule of Eq. (35), same as svlg.py.
        L_MI_fit (Eq. 24b) is the vCLUB estimator's own MLE fitting loss —
        always included at weight 1.0 (not annealed): omitting it lets the
        estimator's parameters degrade and L_MI diverge (see
        disentangled_fusion.VCLUB's docstring).
        """
        train_cfg = (config or {}).get("train", {})
        alpha = train_cfg.get("mi_alpha", 1.0) if alpha is None else alpha
        beta_max = train_cfg.get("kl_beta_max", 1.0) if beta_max is None else beta_max
        warmup_epochs = train_cfg.get("kl_warmup_epochs", 10) if warmup_epochs is None else warmup_epochs

        # Eq. (35): linear KL annealing schedule.
        beta_t = beta_max * min(1.0, epoch / max(1, warmup_epochs))

        L_total = (
            L_gen
            + alpha * fusion_losses["L_MI"]
            + beta_t * fusion_losses["L_KL"]
            + fusion_losses["L_MI_fit"]
        )
        return L_total


def _self_test() -> bool:
    torch.manual_seed(0)
    ok = True

    config = {
        "model": {
            "vit_name": "vit_base_patch16_224",
            "llm_name": "Qwen/Qwen2.5-3B-Instruct",
            "n_prefix": 4,
            "lora_r": 16,
            "lora_alpha": 32,
            "fusion_d_shared": 16,
            "fusion_d_vis": 16,
            "question_vocab_size": 1000,
        },
        "uncertainty": {"gamma_threshold": 1.0},
        "train": {"mi_alpha": 1.0, "kl_beta_max": 1.0, "kl_warmup_epochs": 10},
    }

    model = SU_MedVQA(config, test_mode=True)

    B = 4
    L_question = 12
    images = torch.randn(B, 3, 224, 224)
    question_input_ids = torch.randint(0, config["model"]["question_vocab_size"], (B, L_question))
    question_text = "What organ is likely affected?"
    answer_text = ["The kidney is likely affected." for _ in range(B)]

    print("--- forward (steps 1-4) ---")
    z_final, U, fusion_losses, decoder_out = model(
        images, question_input_ids, question_text, answer_text=answer_text, verbose=True
    )

    expected_z_dim = config["model"]["fusion_d_shared"] + config["model"]["fusion_d_vis"]
    if z_final.shape != (B, expected_z_dim):
        print(f"FAIL: z_final shape {tuple(z_final.shape)} != {(B, expected_z_dim)}")
        ok = False
    if U.shape != (B,):
        print(f"FAIL: U shape {tuple(U.shape)} != {(B,)}")
        ok = False
    if torch.isnan(z_final).any() or torch.isnan(U).any():
        print("FAIL: NaN in z_final or U")
        ok = False

    print("--- compute_total_loss ---")
    L_total = model.compute_total_loss(decoder_out[0], fusion_losses, epoch=3, config=config)
    print(f"L_total: {L_total.item():.4f}")
    if L_total.dim() != 0 or torch.isnan(L_total):
        print("FAIL: L_total is not a valid scalar")
        ok = False

    print("--- generate (inference, uncertainty gate) ---")
    U_fake = torch.tensor([0.1, 0.2, 2.0, 3.0])
    results = model.decoder.generate(
        z_final, U_fake, system_text="You are a helpful medical VQA assistant.",
        evidence_text="", question_text=question_text, gamma=1.0, max_new_tokens=5,
    )
    for i, r in enumerate(results):
        print(f"  sample {i}: U={U_fake[i].item():.2f}, needs_expert_review={r['needs_expert_review']}, answer={r['answer']!r}")
    if results[0]["needs_expert_review"] is not False or results[1]["needs_expert_review"] is not False:
        print("FAIL: confident samples incorrectly flagged for expert review")
        ok = False
    if results[2]["needs_expert_review"] is not True or results[3]["needs_expert_review"] is not True:
        print("FAIL: uncertain samples not flagged for expert review")
        ok = False

    print("PASS: su_medvqa" if ok else "FAIL: su_medvqa")
    return ok


if __name__ == "__main__":
    _self_test()
