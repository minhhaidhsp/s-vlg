"""Graph-RAG evidence retrieval — Mục 3.4.3.

Two-stage retrieval for a query patient q:
  1. Coarse filter (Eq. 16): Jaccard similarity over historical code sets,
     keep candidates with jaccard(q, u) >= tau.
  2. Fine ranking (Eq. 20): cosine similarity between GraphSAGE node
     embeddings h_q^K, h_u^K; keep the top-M candidates as R_q.
"""

import torch


def jaccard_similarity(S_i: set, S_j: set) -> float:
    """Eq. (16): jaccard(S_i, S_j) = |S_i ∩ S_j| / |S_i ∪ S_j|."""
    union = S_i | S_j
    if not union:
        return 0.0
    return len(S_i & S_j) / len(union)


def coarse_filter(query_patient_id, patient_histories: dict, tau: float) -> list:
    """Coarse candidate set C_q (part of Eq. 16 usage): patients sharing at
    least one historical code with the query AND jaccard(q, u) >= tau.

    Returns:
        [(patient_id, jaccard), ...] for all patients passing the filter.
    """
    S_q = patient_histories[query_patient_id]
    candidates = []
    for pid, S_u in patient_histories.items():
        if pid == query_patient_id:
            continue
        if not (S_q & S_u):  # must share >= 1 code to even be considered
            continue
        sim = jaccard_similarity(S_q, S_u)
        if sim >= tau:
            candidates.append((pid, sim))
    return candidates


def cosine_similarity(h_q: torch.Tensor, h_u: torch.Tensor, eps: float = 1e-8) -> float:
    """Eq. (20): score = (h_q . h_u) / (||h_q|| * ||h_u||)."""
    num = torch.dot(h_q, h_u)
    denom = (h_q.norm() * h_u.norm()).clamp_min(eps)
    return (num / denom).item()


def retrieve(
    query_patient_id,
    patient_histories: dict,
    node_emb: torch.Tensor,
    patient_id_to_idx: dict,
    tau: float,
    M: int,
) -> list:
    """Two-stage Graph-RAG retrieval (Eq. 16 coarse filter, Eq. 20 fine ranking).

    Args:
        query_patient_id: the query patient's id.
        patient_histories: {patient_id: set(codes)}, HISTORY-ONLY (see
            src/graph/patient_graph.py for the anti-leakage constraint).
        node_emb: [N, d_out] GraphSAGE node embeddings h^K for the whole graph.
        patient_id_to_idx: patient_id -> node index into `node_emb`.
        tau: Jaccard threshold for the coarse filter (config: graph.jaccard_threshold_tau).
        M: number of top candidates to keep after fine ranking (config: graph.retrieval_M).

    Returns:
        R_q: list of up to M dicts, sorted by descending cosine score:
            {"patient_id", "jaccard", "cosine_score", "shared_codes"}.
        Empty list if no candidate passes the coarse filter.
    """
    # Stage 1 — Eq. (16) coarse filter.
    candidates = coarse_filter(query_patient_id, patient_histories, tau)
    if not candidates:
        return []

    # Stage 2 — Eq. (20) fine ranking via GraphSAGE embedding cosine similarity.
    q_idx = patient_id_to_idx[query_patient_id]
    h_q = node_emb[q_idx]
    S_q = patient_histories[query_patient_id]

    scored = []
    for pid, jaccard in candidates:
        u_idx = patient_id_to_idx[pid]
        h_u = node_emb[u_idx]
        cos = cosine_similarity(h_q, h_u)
        scored.append({
            "patient_id": pid,
            "jaccard": jaccard,
            "cosine_score": cos,
            "shared_codes": S_q & patient_histories[pid],
        })

    scored.sort(key=lambda d: d["cosine_score"], reverse=True)
    return scored[:M]


def build_similarity_edges(patient_histories: dict, patient_id_to_idx: dict, min_jaccard: float = 0.0):
    """All-pairs patient<->patient similarity edges (Eq. 16), to be added to
    the patient knowledge graph's edge set so GraphSAGE can propagate over
    patient similarity as well as code membership.

    `min_jaccard` controls graph sparsity (pairs with jaccard <= min_jaccard
    are dropped) and is independent of the retrieval-time threshold `tau`
    used by `retrieve()` — this one governs graph construction, not retrieval.

    Returns:
        edge_index [2, E], edge_weight [E] (both directions included, weight
        = jaccard similarity).
    """
    pids = list(patient_histories.keys())
    src, dst, weight = [], [], []
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            sim = jaccard_similarity(patient_histories[pids[i]], patient_histories[pids[j]])
            if sim > min_jaccard:
                i_idx, j_idx = patient_id_to_idx[pids[i]], patient_id_to_idx[pids[j]]
                src += [i_idx, j_idx]
                dst += [j_idx, i_idx]
                weight += [sim, sim]

    if not src:
        return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(weight, dtype=torch.float32)
