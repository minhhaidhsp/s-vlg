"""GraphSAGE (inductive), hand-rolled in pure PyTorch — no torch-geometric /
torch-scatter, to avoid their install pain on Windows.

Graph representation: adjacency is a sparse edge list `edge_index [2, E]`
(row 0 = source/neighbor node, row 1 = destination node receiving the
aggregated message) plus a parallel `edge_weight [E]` (e.g. Jaccard patient
similarity). Neighbor aggregation uses `Tensor.index_add_` as a manual
scatter-add, which ships with core PyTorch.

TODO (heterogeneous graph): `node_type [N]` is threaded through the API but
currently unused — the graph is homogeneous (patient nodes only). Once real
MIMIC entity nodes (ICD/CPT codes, etc.) are available, extend SAGEConv with
per-type weight matrices keyed by node_type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_neighbor_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_samples: int = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Fixed-size neighborhood sampling to simulate inductive minibatching.

    For each destination node, keep at most `num_samples` of its incoming
    edges (sampled uniformly without replacement); nodes with fewer than
    `num_samples` neighbors keep all of them. `num_samples=None` disables
    sampling (full-neighborhood aggregation).

    Returns:
        LongTensor of edge indices to keep, into the original `edge_index`.
    """
    if num_samples is None:
        return torch.arange(edge_index.shape[1], device=edge_index.device)

    dst = edge_index[1].tolist()
    adj: dict = {v: [] for v in range(num_nodes)}
    for e, d in enumerate(dst):
        adj[d].append(e)

    keep = []
    for edge_ids in adj.values():
        if len(edge_ids) <= num_samples:
            keep.extend(edge_ids)
        else:
            perm = torch.randperm(len(edge_ids), generator=generator)[:num_samples]
            keep.extend(edge_ids[i] for i in perm.tolist())

    keep.sort()
    return torch.tensor(keep, dtype=torch.long, device=edge_index.device)


class SAGEConv(nn.Module):
    """One GraphSAGE convolution layer: weighted-mean aggregate + concat update + L2 norm."""

    def __init__(self, in_dim: int, out_dim: int, activation=F.relu, eps: float = 1e-8):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)
        self.activation = activation
        self.eps = eps

    def aggregate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Eq. (17): weighted-mean neighbor aggregation.

        h_N(v) = ( sum_{u in N(v)} w_uv * h_u ) / ( sum_{u in N(v)} w_uv )

        Implemented as a manual scatter-add via `index_add_` (no torch-scatter
        dependency): messages are weighted neighbor features scattered into
        per-destination-node accumulators, then divided by the summed weights.
        """
        src, dst = edge_index[0], edge_index[1]
        messages = x[src] * edge_weight.unsqueeze(-1)

        sum_agg = x.new_zeros(num_nodes, x.shape[1])
        sum_agg.index_add_(0, dst, messages)

        weight_sum = x.new_zeros(num_nodes, 1)
        weight_sum.index_add_(0, dst, edge_weight.unsqueeze(-1))

        return sum_agg / weight_sum.clamp_min(self.eps)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        num_nodes = x.shape[0]
        h_neigh = self.aggregate(x, edge_index, edge_weight, num_nodes)

        # Eq. (18): h_v = sigma( W . concat(h_v_prev, h_N(v)) )
        h = self.activation(self.linear(torch.cat([x, h_neigh], dim=-1)))

        # Eq. (19): L2 normalization
        h = F.normalize(h, p=2, dim=-1, eps=self.eps)
        return h


class GraphSAGE(nn.Module):
    """K-layer stack of SAGEConv, with optional per-layer neighbor sampling.

    Args:
        in_dim: input node feature dimension d_in.
        hidden_dim: hidden layer width for intermediate layers.
        out_dim: output embedding dimension d_out.
        num_layers: K, number of stacked SAGEConv layers (config: graph_layers_K).
        num_samples: S, max neighbors sampled per node per layer (None = full neighborhood).
        activation: nonlinearity sigma used in Eq. (18).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        num_samples: int = 10,
        activation=F.relu,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_samples = num_samples

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            [SAGEConv(dims[i], dims[i + 1], activation=activation) for i in range(num_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        node_type: torch.Tensor = None,
    ) -> torch.Tensor:
        """Returns h_v^K, the [N, d_out] node embeddings after K SAGEConv layers.

        `node_type` [N] is accepted but unused today — see module TODO for the
        planned heterogeneous (patient/ICD/CPT) upgrade.
        """
        num_nodes = x.shape[0]
        h = x
        for layer in self.layers:
            keep_ids = sample_neighbor_edges(edge_index, num_nodes, self.num_samples)
            h = layer(h, edge_index[:, keep_ids], edge_weight[keep_ids])
        return h


def _make_random_graph(num_nodes: int, num_undirected_edges: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, num_nodes, (num_undirected_edges,), generator=g)
    dst = torch.randint(0, num_nodes, (num_undirected_edges,), generator=g)
    weight = torch.rand(num_undirected_edges, generator=g)

    # Symmetrize: each undirected edge becomes two directed messages.
    edge_index = torch.stack(
        [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
    )
    edge_weight = torch.cat([weight, weight])
    return edge_index, edge_weight


def _self_test() -> bool:
    torch.manual_seed(0)
    num_nodes, d_in, d_out = 30, 16, 8
    x = torch.randn(num_nodes, d_in)
    edge_index, edge_weight = _make_random_graph(num_nodes, num_undirected_edges=80, seed=1)
    node_type = torch.zeros(num_nodes, dtype=torch.long)  # homogeneous graph placeholder

    model = GraphSAGE(in_dim=d_in, hidden_dim=12, out_dim=d_out, num_layers=2, num_samples=10)
    h = model(x, edge_index, edge_weight, node_type=node_type)

    ok = True
    if h.shape != (num_nodes, d_out):
        print(f"FAIL: embedding shape {h.shape} != {(num_nodes, d_out)}")
        ok = False
    if torch.isnan(h).any():
        print("FAIL: NaN in embeddings")
        ok = False

    # Also sanity-check full-neighborhood mode (num_samples=None).
    model_full = GraphSAGE(in_dim=d_in, hidden_dim=12, out_dim=d_out, num_layers=2, num_samples=None)
    h_full = model_full(x, edge_index, edge_weight, node_type=node_type)
    if h_full.shape != (num_nodes, d_out) or torch.isnan(h_full).any():
        print("FAIL: full-neighborhood mode broken")
        ok = False

    print("PASS: graph_sage" if ok else "FAIL: graph_sage")
    return ok


if __name__ == "__main__":
    _self_test()
