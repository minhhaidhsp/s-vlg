"""Disentangled fusion of shared/specific representations, with a CLUB-based
mutual-information penalty and a closed-form KL regularizer.

Splits a fused representation h into a variational *shared* factor z_s and
N deterministic *specific* factors (one per input branch), encourages z_s to
be independent of every specific factor via a vCLUB mutual-information upper
bound (Cheng et al., 2020, "CLUB: A Contrastive Log-ratio Upper Bound of
Mutual Information"), and exposes the shared factor's posterior log-variance
as a per-sample uncertainty score U.

Generalized over the number of branches (`num_branches` = len(branch_dims))
so the SAME module serves both project versions:
  - V1 (S-VLG): 3 branches — vision, tabular (MLTM), graph (GraphSAGE).
  - V2 (SU-MedVQA): 1 branch — vision only.
Eq. (24)'s pairwise MI term becomes a sum over however many branches are
given (2 terms for V1, 1 term for V2); every other equation is unchanged.
"""

import torch
import torch.nn as nn


class VCLUB(nn.Module):
    """vCLUB mutual-information upper bound estimator between x and y.

    Learns a variational Gaussian approximation q(y|x) = N(mu_q(x), sigma_q(x)^2)
    via a small MLP, then estimates an upper bound on I(x, y):

        I_CLUB(x, y) <= E_{p(x,y)}[log q(y|x)] - E_{p(x)p(y)}[log q(y|x)]

    The first term is estimated from matched (paired) samples, the second
    from the average over all (i, j) pairs in the batch. The additive
    normalization constant of the Gaussian log-likelihood cancels in the
    difference and is dropped.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, y_dim)
        self.fc_logvar = nn.Linear(hidden_dim, y_dim)

    def _q_params(self, x: torch.Tensor):
        h = self.backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Part of Eq. (24): CLUB upper bound on I(x, y) for one (shared, specific) pair."""
        mu, logvar = self._q_params(x)  # [B, y_dim], q(y|x) for matched pairs

        # Paired (positive) term: log q(y_i | x_i), constant dropped.
        positive = -0.5 * ((y - mu) ** 2 / logvar.exp() + logvar)
        positive = positive.sum(dim=-1).mean()

        # All-pairs (negative) term: log q(y_j | x_i) averaged over i, j.
        mu_i = mu.unsqueeze(1)          # [B, 1, y_dim]
        logvar_i = logvar.unsqueeze(1)  # [B, 1, y_dim]
        y_j = y.unsqueeze(0)            # [1, B, y_dim]
        cross = -0.5 * ((y_j - mu_i) ** 2 / logvar_i.exp() + logvar_i)
        negative = cross.sum(dim=-1).mean()

        return positive - negative


class DisentangledFusion(nn.Module):
    """Splits h into shared + N specific latents, computes L_MI, L_KL, and U.

    Args:
        in_dim: dimension of the input fused representation h.
        d_shared: dimension of the shared latent z_s / z_s_var.
        branch_dims: list of specific-latent dimensions, one entry per input
            branch. len(branch_dims) = num_branches: use a single-element
            list (e.g. [d_vis]) for V2 (vision-only), or three elements
            (e.g. [d_vis, d_tab, d_graph]) for V1 (vision+tabular+graph).
        vclub_hidden: hidden width of each vCLUB approximator network.
    """

    def __init__(self, in_dim: int, d_shared: int, branch_dims: list, vclub_hidden: int = 64):
        super().__init__()
        if len(branch_dims) < 1:
            raise ValueError("branch_dims must have at least one entry (num_branches >= 1)")
        self.num_branches = len(branch_dims)
        self.branch_dims = list(branch_dims)

        # Eq. (21): one linear projection for the shared latent, one per specific branch.
        self.proj_shared = nn.Linear(in_dim, d_shared)
        self.proj_branches = nn.ModuleList([nn.Linear(in_dim, d) for d in branch_dims])

        # Eq. (22): mu and log_sigma^2 heads for the shared latent.
        self.fc_mu = nn.Linear(d_shared, d_shared)
        self.fc_logvar = nn.Linear(d_shared, d_shared)

        # Eq. (24): one vCLUB estimator per (shared, specific-branch) pair.
        self.vclub_branches = nn.ModuleList(
            [VCLUB(d_shared, d, hidden_dim=vclub_hidden) for d in branch_dims]
        )

    def forward(self, h: torch.Tensor):
        """
        Args:
            h: [B, in_dim] fused representation — the concatenation of
               whatever branch embeddings the assembling model (svlg.py or
               su_medvqa.py) produced.

        Returns:
            z_final: [B, d_shared + sum(branch_dims)], Eq. (27).
            U: [B] per-sample uncertainty score, Eq. (28).
            losses: dict with scalar "L_MI" (Eq. 24) and "L_KL" (Eq. 25).
        """
        # Eq. (21)
        z_shared = self.proj_shared(h)
        z_branches = [proj(h) for proj in self.proj_branches]

        # Eq. (22)
        mu = self.fc_mu(z_shared)
        logvar = self.fc_logvar(z_shared).clamp(-10.0, 10.0)

        # Eq. (23): reparameterization trick.
        eps = torch.randn_like(mu)
        z_s_var = mu + torch.exp(0.5 * logvar) * eps

        # Eq. (24): sum of the pairwise CLUB upper bounds, one per branch
        # (2 terms for V1's vis+tab, 1 term for V2's vis-only).
        L_MI = sum(
            vclub(z_s_var, z_b) for vclub, z_b in zip(self.vclub_branches, z_branches)
        )

        # Eq. (25): closed-form KL divergence to a standard normal prior.
        L_KL = 0.5 * torch.sum(mu ** 2 + logvar.exp() - logvar - 1.0, dim=-1).mean()

        # Eq. (28): per-sample uncertainty = mean posterior log-variance over latent dims.
        U = logvar.mean(dim=-1)

        # Eq. (27)
        z_final = torch.cat([z_s_var, *z_branches], dim=-1)

        return z_final, U, {"L_MI": L_MI, "L_KL": L_KL}


def _check_config(branch_dims: list) -> bool:
    torch.manual_seed(0)
    B, in_dim = 8, 32
    d_shared = 16

    model = DisentangledFusion(in_dim=in_dim, d_shared=d_shared, branch_dims=branch_dims)
    h = torch.randn(B, in_dim)
    z_final, U, losses = model(h)

    ok = True
    expected_dim = d_shared + sum(branch_dims)
    if z_final.shape != (B, expected_dim):
        print(f"FAIL: z_final shape {tuple(z_final.shape)} != {(B, expected_dim)}")
        ok = False
    if U.shape != (B,):
        print(f"FAIL: U shape {tuple(U.shape)} != {(B,)}")
        ok = False
    for name in ("L_MI", "L_KL"):
        loss = losses[name]
        if loss.dim() != 0:
            print(f"FAIL: {name} is not scalar, dim={loss.dim()}")
            ok = False
        if torch.isnan(loss):
            print(f"FAIL: {name} is NaN")
            ok = False

    # Semantic check for Eq. (28): higher log_sigma^2 (noisier posterior) -> higher U.
    with torch.no_grad():
        model.fc_logvar.weight.zero_()
        model.fc_logvar.bias.fill_(-5.0)
        _, U_low, _ = model(h)
        model.fc_logvar.bias.fill_(5.0)
        _, U_high, _ = model(h)
    if not (U_high.mean().item() > U_low.mean().item()):
        print(
            f"FAIL: U did not increase with higher log_sigma^2 "
            f"(U_low={U_low.mean().item():.4f}, U_high={U_high.mean().item():.4f})"
        )
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"{status}: disentangled_fusion (num_branches={len(branch_dims)}, branch_dims={branch_dims})")
    return ok


def _self_test() -> bool:
    # V2 (SU-MedVQA): vision-only, num_branches=1.
    ok_v2 = _check_config(branch_dims=[20])
    # V1 (S-VLG): vision+tabular+graph, num_branches=3.
    ok_v1 = _check_config(branch_dims=[20, 12, 16])

    ok = ok_v1 and ok_v2
    print("PASS: disentangled_fusion" if ok else "FAIL: disentangled_fusion")
    return ok


if __name__ == "__main__":
    _self_test()
