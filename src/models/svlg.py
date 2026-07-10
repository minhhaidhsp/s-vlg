"""SVLG: full Vision-Language-Graph model for Medical VQA — assembles every
branch module into a single nn.Module and end-to-end forward pass.

Branches:
  - VisionEncoder + RPRCoAttention -> h_vis   (image + question, spatially aware)
  - MLTM                            -> h_tab   (lab-test / tabular branch)
  - GraphSAGE                       -> h_graph (patient-similarity graph branch)
  - DisentangledFusion               -> z_final, U, {L_MI, L_KL} (Eq. 21-28)
  - VQADecoder                       -> generated answer / training loss (Eq. 29-33)
"""

import torch
import torch.nn as nn

from src.models.decoder import VQADecoder
from src.models.disentangled_fusion import DisentangledFusion
from src.models.graph_sage import GraphSAGE
from src.models.mltm import MLTM
from src.models.rpr_coattention import RPRCoAttention
from src.models.vision_encoder import VisionEncoder


class SVLG(nn.Module):
    """Assembles the vision, tabular, graph, fusion, and decoder branches.

    Expects a config dict shaped like configs/config.yaml (see load_config in
    src/utils/config.py). Hyperparameters not yet present in config.yaml
    (branch widths for MLTM/GraphSAGE/fusion, question-embedding vocab) fall
    back to the defaults below — TODO: promote these to config.yaml once the
    paper fixes exact widths for these branches.
    """

    def __init__(self, config: dict, test_mode: bool = True):
        super().__init__()
        model_cfg = config.get("model", {})
        graph_cfg = config.get("graph", {})
        uncertainty_cfg = config.get("uncertainty", {})

        self.test_mode = test_mode

        d_vis = 768  # ViT-Base/16 patch embedding dim (Eq. 4)
        lab_input_dim = model_cfg.get("lab_input_dim", 50)
        mltm_bottleneck_dim = model_cfg.get("mltm_bottleneck_dim", 32)
        graph_in_dim = model_cfg.get("graph_in_dim", 64)
        graph_hidden_dim = model_cfg.get("graph_hidden_dim", 64)
        graph_out_dim = model_cfg.get("graph_out_dim", 64)
        fusion_d_shared = model_cfg.get("fusion_d_shared", 64)
        fusion_d_vis = model_cfg.get("fusion_d_vis", 64)
        fusion_d_tab = model_cfg.get("fusion_d_tab", 32)
        question_vocab_size = model_cfg.get("question_vocab_size", 30522)  # BERT-sized placeholder vocab

        n_prefix = model_cfg.get("n_prefix", 8)
        graph_layers_K = model_cfg.get("graph_layers_K", 2)
        gamma = uncertainty_cfg.get("gamma_threshold")
        if gamma is None:
            gamma = 1.0  # placeholder until chosen via validation (config: uncertainty.gamma_threshold)

        # --- Vision branch: Eq. (4) ViT patches, then Eq. (9)-(14) RPR co-attention ---
        self.vision_encoder = VisionEncoder(
            vit_name=model_cfg.get("vit_name", "vit_base_patch16_224"), test_mode=test_mode
        )
        self.rpr_coattention = RPRCoAttention(d=d_vis, k=8)
        # TODO: replace with the real question tokenizer/embedding once wired to
        # the actual text pipeline; this is a dedicated embedding table (separate
        # from the decoder LLM's own vocabulary/embedding, which may have a very
        # different dimension, e.g. tiny-gpt2 in test_mode).
        self.question_embedding = nn.Embedding(question_vocab_size, d_vis)

        # --- Tabular branch: Eq. (7)-(8) masked lab-test reconstruction ---
        self.mltm = MLTM(input_dim=lab_input_dim, bottleneck_dim=mltm_bottleneck_dim)

        # --- Graph branch: Eq. (17)-(19) hand-rolled GraphSAGE ---
        self.graph_sage = GraphSAGE(
            in_dim=graph_in_dim,
            hidden_dim=graph_hidden_dim,
            out_dim=graph_out_dim,
            num_layers=graph_layers_K,
            num_samples=graph_cfg.get("neighbor_sample_size", 10),
        )

        # --- Fusion: Eq. (21)-(28) disentangled shared/specific latents ---
        # V1 = 3 branches (vision, tabular, graph) -> num_branches=3.
        fusion_in_dim = d_vis + mltm_bottleneck_dim + graph_out_dim  # concat(h_vis, h_tab, h_graph)
        branch_dims = [fusion_d_vis, fusion_d_tab, model_cfg.get("fusion_d_graph", 32)]
        self.fusion = DisentangledFusion(
            in_dim=fusion_in_dim, d_shared=fusion_d_shared, branch_dims=branch_dims
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
        lab_x: torch.Tensor,
        lab_mask: torch.Tensor,
        graph_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        batch_node_idx: torch.Tensor,
        evidence_text,
        question_text,
        system_text: str = "You are a helpful medical VQA assistant.",
        answer_text=None,
        verbose: bool = False,
    ):
        """End-to-end forward pass.

        Args:
            images: [B, 3, 224, 224] input images.
            question_input_ids: [B, L] question token ids (for the vision
                co-attention branch — see question_embedding TODO above).
            lab_x: [B, D] lab-test value vector.
            lab_mask: [B, D] binary observed-lab mask (1=observed, 0=missing).
            graph_x: [N, d_in] node features for the whole patient graph.
            edge_index: [2, E] graph edges (src, dst).
            edge_weight: [E] Jaccard edge weights.
            batch_node_idx: [B] index into graph_x/node_emb for each sample's patient node.
            evidence_text: single string (broadcast) or list[str] of length B — Graph-RAG evidence.
            question_text: single string (broadcast) or list[str] of length B — question, for the decoder prompt.
            answer_text: (training only) single string or list[str] of length B — ground-truth answer.
            verbose: if True, print the shape after each step (1-7) for debugging.

        Returns:
            z_final: [B, d_shared + sum(branch_dims)], Eq. (27).
            U: [B] per-sample uncertainty score, Eq. (28).
            fusion_losses: dict with "L_MI" (Eq. 24) and "L_KL" (Eq. 25).
            mltm_out: (x_hat, h_tab) from the MLTM branch, for L_MLTM (Eq. 8).
            decoder_out: (loss, logits) if answer_text is given (training),
                else None — call self.decoder.generate(...) directly for inference.
        """
        # Step 1 — Eq. (4): ViT patch features.
        patch_features, patch_coords = self.vision_encoder(images)
        if verbose:
            print(f"[1] patch_features: {tuple(patch_features.shape)}, patch_coords: {tuple(patch_coords.shape)}")

        # Step 2 — Eq. (9)-(14): RPR co-attention -> h_vis.
        question_tokens = self.question_embedding(question_input_ids)  # [B, L, d_vis]
        h_vis = self.rpr_coattention(patch_features, patch_coords, question_tokens)
        if verbose:
            print(f"[2] h_vis: {tuple(h_vis.shape)}")

        # Step 3 — Eq. (7)-(8): MLTM tabular branch.
        x_hat, h_tab = self.mltm(lab_x, lab_mask)
        if verbose:
            print(f"[3] x_hat: {tuple(x_hat.shape)}, h_tab: {tuple(h_tab.shape)}")

        # Step 4 — Eq. (17)-(19): GraphSAGE over the whole patient graph, then
        # gather each batch sample's own node embedding.
        node_emb = self.graph_sage(graph_x, edge_index, edge_weight)
        h_graph = node_emb[batch_node_idx]
        if verbose:
            print(f"[4] node_emb: {tuple(node_emb.shape)}, h_graph: {tuple(h_graph.shape)}")

        # Step 5: combine branches. Plain concatenation (no extra learned
        # projection) — DisentangledFusion's own Eq. (21) linear projections
        # act as the necessary transform from this concatenated h.
        h = torch.cat([h_vis, h_tab, h_graph], dim=-1)
        if verbose:
            print(f"[5] h (concat): {tuple(h.shape)}")

        # Step 6 — Eq. (21)-(28): disentangled fusion.
        z_final, U, fusion_losses = self.fusion(h)
        if verbose:
            print(f"[6] z_final: {tuple(z_final.shape)}, U: {tuple(U.shape)}, "
                  f"L_MI: {fusion_losses['L_MI'].item():.4f}, L_KL: {fusion_losses['L_KL'].item():.4f}")

        # Step 7 — Eq. (29)-(33): decoder. Training (answer_text given) runs a
        # teacher-forced forward for the generation loss; otherwise, use
        # self.decoder.generate(...) directly for gated inference.
        decoder_out = None
        if answer_text is not None:
            decoder_out = self.decoder(z_final, system_text, evidence_text, question_text, answer_text)
            if verbose:
                print(f"[7] decoder loss: {decoder_out[0].item():.4f}, logits: {tuple(decoder_out[1].shape)}")

        return z_final, U, fusion_losses, (x_hat, h_tab), decoder_out

    def compute_total_loss(
        self,
        L_gen: torch.Tensor,
        fusion_losses: dict,
        lab_x: torch.Tensor,
        lab_mask: torch.Tensor,
        x_hat: torch.Tensor,
        epoch: int,
        alpha: float = None,
        beta_max: float = None,
        warmup_epochs: int = None,
        config: dict = None,
    ) -> torch.Tensor:
        """Eq. (34) extended with L_MLTM: L_total = L_gen + alpha*L_MI + beta_t*L_KL + L_MLTM + L_MI_fit.

        beta_t follows the KL-annealing schedule of Eq. (35): linear warm-up
        from 0 to beta_max over `warmup_epochs`, so the fusion module doesn't
        collapse the shared latent to the prior before it has learned anything
        useful (standard VAE "KL vanishing" mitigation).

        alpha, beta_max, warmup_epochs default to `config["train"]` if not
        passed explicitly (all currently unset/None in configs/config.yaml —
        to be chosen via validation).

        L_MI_fit (Eq. 24b) is the vCLUB estimators' own MLE fitting loss —
        always included at weight 1.0 (not annealed): omitting it lets the
        estimators' parameters degrade and L_MI diverge (see
        disentangled_fusion.VCLUB's docstring).
        """
        train_cfg = (config or {}).get("train", {})
        alpha = train_cfg.get("mi_alpha", 1.0) if alpha is None else alpha
        beta_max = train_cfg.get("kl_beta_max", 1.0) if beta_max is None else beta_max
        warmup_epochs = train_cfg.get("kl_warmup_epochs", 10) if warmup_epochs is None else warmup_epochs

        # Eq. (35): linear KL annealing schedule.
        beta_t = beta_max * min(1.0, epoch / max(1, warmup_epochs))

        L_MLTM = MLTM.loss(lab_x, lab_mask, x_hat)  # Eq. (8)

        L_total = (
            L_gen
            + alpha * fusion_losses["L_MI"]
            + beta_t * fusion_losses["L_KL"]
            + L_MLTM
            + fusion_losses["L_MI_fit"]
        )
        return L_total


def _make_random_graph(num_nodes: int, num_undirected_edges: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_undirected_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_undirected_edges,), generator=g)
    weight = torch.rand(num_undirected_edges, generator=g)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    edge_weight = torch.cat([weight, weight])
    return edge_index, edge_weight


def _self_test() -> bool:
    torch.manual_seed(0)
    ok = True

    config = {
        "model": {
            "vit_name": "vit_base_patch16_224",
            "llm_name": "Qwen/Qwen2.5-3B-Instruct",
            "n_prefix": 4,
            "graph_layers_K": 2,
            "lora_r": 16,
            "lora_alpha": 32,
            "lab_input_dim": 20,
            "mltm_bottleneck_dim": 16,
            "graph_in_dim": 32,
            "graph_hidden_dim": 32,
            "graph_out_dim": 32,
            "fusion_d_shared": 16,
            "fusion_d_vis": 16,
            "fusion_d_tab": 16,
            "fusion_d_graph": 16,
            "question_vocab_size": 1000,
        },
        "graph": {"neighbor_sample_size": 10},
        "uncertainty": {"gamma_threshold": 1.0},
        "train": {"mi_alpha": 1.0, "kl_beta_max": 1.0, "kl_warmup_epochs": 10},
    }

    model = SVLG(config, test_mode=True)

    B = 4
    num_nodes = 50
    L_question = 12
    lab_dim = config["model"]["lab_input_dim"]

    images = torch.randn(B, 3, 224, 224)
    question_input_ids = torch.randint(0, config["model"]["question_vocab_size"], (B, L_question))
    lab_x = torch.randn(B, lab_dim)
    lab_mask = (torch.rand(B, lab_dim) > 0.3).float()

    graph_x = torch.randn(num_nodes, config["model"]["graph_in_dim"])
    edge_index, edge_weight = _make_random_graph(num_nodes, num_undirected_edges=100, seed=1)
    batch_node_idx = torch.randint(0, num_nodes, (B,))

    evidence_text = ["" for _ in range(B)]
    question_text = "What organ is likely affected?"
    answer_text = ["The kidney is likely affected." for _ in range(B)]

    print("--- forward (steps 1-7) ---")
    z_final, U, fusion_losses, (x_hat, h_tab), decoder_out = model(
        images,
        question_input_ids,
        lab_x,
        lab_mask,
        graph_x,
        edge_index,
        edge_weight,
        batch_node_idx,
        evidence_text,
        question_text,
        answer_text=answer_text,
        verbose=True,
    )

    d_shared, d_vis_f, d_tab_f, d_graph_f = 16, 16, 16, 16
    expected_z_dim = d_shared + d_vis_f + d_tab_f + d_graph_f
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
    L_gen = decoder_out[0]
    L_total = model.compute_total_loss(
        L_gen, fusion_losses, lab_x, lab_mask, x_hat, epoch=3, config=config
    )
    print(f"L_total: {L_total.item():.4f}")
    if L_total.dim() != 0:
        print(f"FAIL: L_total is not scalar, dim={L_total.dim()}")
        ok = False
    if torch.isnan(L_total):
        print("FAIL: L_total is NaN")
        ok = False

    print("--- generate (inference, uncertainty gate) ---")
    U_fake = torch.tensor([0.1, 0.2, 2.0, 3.0])  # first two confident, last two uncertain
    results = model.decoder.generate(
        z_final, U_fake, question_text=question_text, system_text="You are a helpful medical VQA assistant.",
        evidence_text=["" for _ in range(B)], gamma=1.0, max_new_tokens=5,
    )
    for i, r in enumerate(results):
        print(f"  sample {i}: U={U_fake[i].item():.2f}, needs_expert_review={r['needs_expert_review']}, answer={r['answer']!r}")
    if results[0]["needs_expert_review"] is not False or results[1]["needs_expert_review"] is not False:
        print("FAIL: confident samples incorrectly flagged for expert review")
        ok = False
    if results[2]["needs_expert_review"] is not True or results[3]["needs_expert_review"] is not True:
        print("FAIL: uncertain samples not flagged for expert review")
        ok = False

    print("PASS: svlg" if ok else "FAIL: svlg")
    return ok


if __name__ == "__main__":
    _self_test()
