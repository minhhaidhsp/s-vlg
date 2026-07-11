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
import torch.nn.functional as F


class VCLUB(nn.Module):
    """vCLUB mutual-information upper bound estimator between x and y.

    Learns a variational Gaussian approximation q(y|x) = N(mu_q(x), sigma_q(x)^2)
    via a small MLP, then estimates an upper bound on I(x, y):

        I_CLUB(x, y) <= E_{p(x,y)}[log q(y|x)] - E_{p(x)p(y)}[log q(y|x)]

    The first term is estimated from matched (paired) samples, the second
    from the average over all (i, j) pairs in the batch. The additive
    normalization constant of the Gaussian log-likelihood cancels in the
    difference and is dropped.

    Per Cheng et al. (2020), q(y|x) must be fit SEPARATELY via maximum
    likelihood on paired samples (`learning_loss`, Eq. 24b) — not via the
    "positive - negative" estimate itself. `forward()` therefore uses this
    module's OWN parameters detached (gradient still flows to `x`, i.e. into
    the encoder, but not into backbone/fc_mu/fc_logvar). Without this split,
    q's own parameters can minimize "positive - negative" without bound
    simply by collapsing logvar toward its clamp floor (making the quadratic
    term explode) — this doesn't reduce true MI, it just games the estimate,
    and was observed empirically to diverge training loss to ~-6e6 within
    one epoch on real data/model scale (toy self-test batches were too small/
    short to reach it).
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

    def _q_params_detached_weights(self, x: torch.Tensor):
        """Same computation as `_q_params`, but using this module's weights
        detached from autograd: gradient flows to `x`, never into this
        module's own parameters. See class docstring for why."""
        linear = self.backbone[0]
        h = F.relu(F.linear(x, linear.weight.detach(), linear.bias.detach()))
        mu = F.linear(h, self.fc_mu.weight.detach(), self.fc_mu.bias.detach())
        logvar = F.linear(h, self.fc_logvar.weight.detach(), self.fc_logvar.bias.detach())
        logvar = logvar.clamp(-10.0, 10.0)
        return mu, logvar

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Part of Eq. (24): CLUB upper bound on I(x, y) for one (shared, specific) pair.

        Used as the disentanglement penalty added to the main loss — gradient
        from this reaches `x` (the encoder's shared latent) but not this
        module's own parameters (see `_q_params_detached_weights`)."""
        mu, logvar = self._q_params_detached_weights(x)  # [B, y_dim], q(y|x) for matched pairs

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

    def learning_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Eq. (24b): fits q(y|x) via maximum likelihood on paired samples —
        trains ONLY this module's own parameters. `x`/`y` are detached so
        gradient here never reaches the encoder/branch projections that
        produced them; must be added into the total loss (see
        su_medvqa.py/svlg.py's `compute_total_loss`) alongside `forward()`'s
        estimate, or q degrades and `forward()` diverges (see class
        docstring)."""
        mu, logvar = self._q_params(x.detach())
        y = y.detach()
        nll = 0.5 * ((y - mu) ** 2 / logvar.exp() + logvar)
        return nll.sum(dim=-1).mean()


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

    def __init__(
        self, in_dim: int, d_shared: int, branch_dims: list, vclub_hidden: int = 64,
        deterministic: bool = False,
    ):
        super().__init__()
        if len(branch_dims) < 1:
            raise ValueError("branch_dims must have at least one entry (num_branches >= 1)")
        self.num_branches = len(branch_dims)
        self.branch_dims = list(branch_dims)
        # Ablation switch (Table 9 "no_disentangle" variant): when True, the
        # shared latent is deterministic (z_s_var = mu, no reparameterization
        # noise), L_KL is dropped (there is no posterior to regularize), and U
        # is undefined (returned as NaN) since there is no variance to read it
        # from — see Eq. (22)-(23), (25), (28).
        self.deterministic = deterministic

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
            losses: dict with scalar "L_MI" (Eq. 24), "L_KL" (Eq. 25), and
                "L_MI_fit" (Eq. 24b) — the vCLUB MLE fitting loss. ALL THREE
                must be added into the caller's total loss (see
                su_medvqa.py/svlg.py's `compute_total_loss`); omitting
                "L_MI_fit" lets the vCLUB estimator's own parameters degrade
                and "L_MI" diverges (see VCLUB's docstring).
        """
        # Eq. (21)
        z_shared = self.proj_shared(h)
        z_branches = [proj(h) for proj in self.proj_branches]

        # Eq. (22). mu is clamped for the same reason logvar is: L_KL's mu**2 term is
        # unbounded above with no natural restoring force while beta_t (Eq. 35) is still
        # small during KL warm-up -- mu can drift to a large magnitude across many epochs
        # without being penalized enough to notice, then once beta_t reaches its full
        # weight post-warmup, 0.5*mu**2 explodes immediately (observed empirically: avg
        # training loss went from ~10 to 1.7e9 to 1.0e11 within a few epochs of warm-up
        # ending on real Colab GPU training with a real 3B-scale model).
        mu = self.fc_mu(z_shared).clamp(-10.0, 10.0)
        logvar = self.fc_logvar(z_shared).clamp(-10.0, 10.0)

        if self.deterministic:
            # Ablation: no reparameterization, no KL, no uncertainty (see __init__ note).
            z_s_var = mu
            L_KL = mu.new_tensor(0.0)
            U = torch.full((mu.shape[0],), float("nan"), device=mu.device, dtype=mu.dtype)
        else:
            # Eq. (23): reparameterization trick.
            eps = torch.randn_like(mu)
            z_s_var = mu + torch.exp(0.5 * logvar) * eps
            # Eq. (25): closed-form KL divergence to a standard normal prior.
            L_KL = 0.5 * torch.sum(mu ** 2 + logvar.exp() - logvar - 1.0, dim=-1).mean()
            # Eq. (28): per-sample uncertainty = mean posterior log-variance over latent dims.
            U = logvar.mean(dim=-1)

        # Eq. (24): sum of the pairwise CLUB upper bounds, one per branch
        # (2 terms for V1's vis+tab, 1 term for V2's vis-only).
        L_MI = sum(
            vclub(z_s_var, z_b) for vclub, z_b in zip(self.vclub_branches, z_branches)
        )
        # Eq. (24b): MLE fitting loss for each vCLUB estimator (see VCLUB.learning_loss).
        L_MI_fit = sum(
            vclub.learning_loss(z_s_var, z_b) for vclub, z_b in zip(self.vclub_branches, z_branches)
        )

        # Eq. (27)
        z_final = torch.cat([z_s_var, *z_branches], dim=-1)

        return z_final, U, {"L_MI": L_MI, "L_KL": L_KL, "L_MI_fit": L_MI_fit}


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
    for name in ("L_MI", "L_KL", "L_MI_fit"):
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


def _check_deterministic() -> bool:
    torch.manual_seed(0)
    B, in_dim, d_shared = 8, 32, 16
    model = DisentangledFusion(in_dim=in_dim, d_shared=d_shared, branch_dims=[20], deterministic=True)
    h = torch.randn(B, in_dim)
    z_final, U, losses = model(h)

    ok = True
    if not torch.isnan(U).all():
        print("FAIL: deterministic=True should return U as NaN (undefined)")
        ok = False
    if losses["L_KL"].item() != 0.0:
        print(f"FAIL: deterministic=True should have L_KL == 0, got {losses['L_KL'].item()}")
        ok = False
    if torch.isnan(z_final).any() or torch.isnan(losses["L_MI"]):
        print("FAIL: deterministic=True produced NaN in z_final or L_MI")
        ok = False

    print("PASS: disentangled_fusion (deterministic ablation)" if ok else "FAIL: disentangled_fusion (deterministic ablation)")
    return ok


def _check_no_divergence() -> bool:
    """Regression test for the divergence observed on real Colab GPU training
    (avg_loss reaching ~-6e6 within one epoch): without `learning_loss`
    (Eq. 24b), the vCLUB estimator's own parameters can minimize L_MI without
    bound by collapsing logvar toward its clamp floor — this doesn't reduce
    true MI, it games the estimate. Single-call shape checks (_check_config)
    can't catch this; it only shows up after many optimizer steps with real
    gradient dynamics, so this runs 200 Adam steps on a toy problem and
    confirms L_MI stays bounded when L_MI + L_MI_fit are optimized together.
    """
    torch.manual_seed(0)
    B, in_dim, d_shared = 16, 32, 8
    model = DisentangledFusion(in_dim=in_dim, d_shared=d_shared, branch_dims=[12])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    h = torch.randn(B, in_dim)
    max_abs_L_MI = 0.0
    ok = True
    for _ in range(200):
        optimizer.zero_grad()
        _, _, losses = model(h)
        loss = losses["L_MI"] + losses["L_MI_fit"]
        if torch.isnan(loss) or torch.isinf(loss):
            print("FAIL: loss became NaN/Inf during training")
            ok = False
            break
        loss.backward()
        optimizer.step()
        max_abs_L_MI = max(max_abs_L_MI, losses["L_MI"].abs().item())

    # Generous bound: the divergence bug this guards against reached ~1e6+
    # magnitude, orders of magnitude past any well-behaved MI estimate here.
    if ok and max_abs_L_MI > 1000.0:
        print(f"FAIL: L_MI diverged over 200 steps (max |L_MI| = {max_abs_L_MI:.2f})")
        ok = False

    print("PASS: disentangled_fusion (no-divergence regression)" if ok else "FAIL: disentangled_fusion (no-divergence regression)")
    return ok


def _check_kl_no_divergence() -> bool:
    """Regression test for the mu-explosion divergence observed on real Colab
    GPU training (avg_loss reaching ~1e11 within a few epochs, starting right
    when KL warm-up finished and beta_t hit its full weight).

    L_KL minimized in isolation is self-stabilizing (it always pulls mu -> 0),
    so it can't reproduce the real failure mode by itself -- the real training
    loop also has L_gen (the decoder's language-modeling loss), which has its
    own incentive to make z_s_var (derived from mu) large/expressive to encode
    information, fighting against L_KL's pull toward 0. This test adds a toy
    stand-in for that competing pressure (`L_task = -mu.pow(2).sum()`, i.e.
    minimizing it explicitly rewards LARGE mu) and runs a low-beta "warm-up"
    phase (mu can drift unchecked, matching real annealing where beta_t is
    still small) followed by a full-beta phase (matching beta_t=1.0 after
    warm-up) -- this is where the unclamped-mu bug actually blew up.
    """
    torch.manual_seed(0)
    B, in_dim, d_shared = 16, 32, 8
    model = DisentangledFusion(in_dim=in_dim, d_shared=d_shared, branch_dims=[12])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    h = torch.randn(B, in_dim)
    warmup_steps = 100
    max_L_KL = 0.0
    ok = True
    for step in range(300):
        beta_t = 0.01 if step < warmup_steps else 1.0  # matches real annealing: small during warm-up, full after
        optimizer.zero_grad()
        z_final, _, losses = model(h)
        L_task = -z_final.pow(2).sum(dim=-1).mean()  # toy stand-in for L_gen's pull toward large/expressive features
        loss = losses["L_MI"] + losses["L_MI_fit"] + beta_t * losses["L_KL"] + 0.1 * L_task
        if torch.isnan(loss) or torch.isinf(loss):
            print("FAIL: loss became NaN/Inf during training")
            ok = False
            break
        loss.backward()
        optimizer.step()
        max_L_KL = max(max_L_KL, losses["L_KL"].item())

    # Generous bound: the divergence bug this guards against reached ~1e9+ magnitude.
    if ok and max_L_KL > 10_000.0:
        print(f"FAIL: L_KL diverged over 300 steps at beta_t=1.0 (max L_KL = {max_L_KL:.2f})")
        ok = False

    print("PASS: disentangled_fusion (KL no-divergence regression)" if ok else "FAIL: disentangled_fusion (KL no-divergence regression)")
    return ok


def _self_test() -> bool:
    # V2 (SU-MedVQA): vision-only, num_branches=1.
    ok_v2 = _check_config(branch_dims=[20])
    # V1 (S-VLG): vision+tabular+graph, num_branches=3.
    ok_v1 = _check_config(branch_dims=[20, 12, 16])
    # Ablation switch: deterministic=True (no reparameterization/KL/U).
    ok_det = _check_deterministic()
    # Regression: vCLUB training must not diverge (see docstring).
    ok_no_div = _check_no_divergence()
    # Regression: mu must not explode once beta_t reaches full weight (see docstring).
    ok_kl_no_div = _check_kl_no_divergence()

    ok = ok_v1 and ok_v2 and ok_det and ok_no_div and ok_kl_no_div
    print("PASS: disentangled_fusion" if ok else "FAIL: disentangled_fusion")
    return ok


if __name__ == "__main__":
    _self_test()
