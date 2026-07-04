"""Masked Lab-Test Modeling (MLTM) — tabular/lab-value branch.

Self-supervised pretext task: reconstruct lab values from a masked view of
themselves, so the encoder bottleneck learns a dense representation h_tab
usable even when some lab tests were never ordered (missing, not just zero).
"""

import torch
import torch.nn as nn


class MLTM(nn.Module):
    """Encoder-decoder MLP for masked lab-value reconstruction.

    Args:
        input_dim: D, number of lab-test channels.
        hidden_dims: hidden layer widths for the encoder (decoder mirrors it).
        bottleneck_dim: size of h_tab, the latent tabular representation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple = (128, 64),
        bottleneck_dim: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        enc_dims = [input_dim, *hidden_dims, bottleneck_dim]
        self.encoder = self._mlp(enc_dims, final_activation=True)

        dec_dims = [bottleneck_dim, *reversed(hidden_dims), input_dim]
        self.decoder = self._mlp(dec_dims, final_activation=False)

    @staticmethod
    def _mlp(dims: list, final_activation: bool) -> nn.Sequential:
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if not is_last or final_activation:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        """Eq. (7): x_hat = Decoder(Encoder(x ⊙ m)).

        Args:
            x: [B, D] raw lab-value vector (missing entries can be any fill value,
               since they are zeroed out by the mask before encoding).
            m: [B, D] binary observation mask (1 = observed, 0 = missing).

        Returns:
            x_hat: [B, D] reconstructed lab values.
            h_tab: [B, d_tab] bottleneck latent representation.
        """
        h_tab = self.encoder(x * m)
        x_hat = self.decoder(h_tab)
        return x_hat, h_tab

    @staticmethod
    def loss(x: torch.Tensor, m: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Eq. (8): masked reconstruction loss.

        L_MLTM = mean over missing positions of ((1 - m) * (x - x_hat))^2,
        i.e. the squared error is computed everywhere but only positions with
        m=0 (missing/held-out) contribute, and the sum is normalized by the
        number of missing positions (not by B*D).
        """
        missing = 1.0 - m
        sq_err = (missing * (x - x_hat)) ** 2
        num_missing = missing.sum().clamp_min(1.0)
        return sq_err.sum() / num_missing


def _self_test() -> bool:
    torch.manual_seed(0)
    B, D = 16, 40
    x = torch.randn(B, D)
    m = (torch.rand(B, D) > 0.3).float()  # ~70% observed, 30% missing

    model = MLTM(input_dim=D, hidden_dims=(64, 32), bottleneck_dim=16)
    x_hat, h_tab = model(x, m)
    loss = model.loss(x, m, x_hat)

    ok = True
    if x_hat.shape != (B, D):
        print(f"FAIL: x_hat shape {x_hat.shape} != {(B, D)}")
        ok = False
    if h_tab.shape != (B, 16):
        print(f"FAIL: h_tab shape {h_tab.shape} != {(B, 16)}")
        ok = False
    if loss.dim() != 0:
        print(f"FAIL: loss is not scalar, dim={loss.dim()}")
        ok = False
    if loss.item() < 0:
        print(f"FAIL: loss is negative ({loss.item()})")
        ok = False
    if torch.isnan(loss):
        print("FAIL: loss is NaN")
        ok = False

    # Edge case: no missing values -> loss must be exactly 0, not NaN/inf.
    m_full = torch.ones(B, D)
    x_hat_full, _ = model(x, m_full)
    loss_full = model.loss(x, m_full, x_hat_full)
    if not torch.isclose(loss_full, torch.tensor(0.0)):
        print(f"FAIL: loss with no missing values should be 0, got {loss_full.item()}")
        ok = False

    print("PASS: mltm" if ok else "FAIL: mltm")
    return ok


if __name__ == "__main__":
    _self_test()
