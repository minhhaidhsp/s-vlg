"""End-to-end smoke test for the Graph-RAG evidence retrieval pipeline
(patient knowledge graph -> GraphSAGE embeddings -> retrieval -> linearized
evidence), using an entirely FAKE patient graph — no real MIMIC data needed.

Usage:
  python scripts/test_graph_rag.py
"""

import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.graph.linearize import linearize_evidence
from src.graph.patient_graph import PatientKnowledgeGraph
from src.graph.retrieval import build_similarity_edges, retrieve
from src.models.decoder import VQADecoder
from src.models.graph_sage import GraphSAGE


def _make_fake_histories(num_patients: int = 50, num_clusters: int = 5, seed: int = 0) -> dict:
    """Fake HISTORY-ONLY code sets, clustered so some patients share codes
    (nonzero Jaccard) while others (different clusters) mostly don't.
    """
    rng = random.Random(seed)
    icd_vocab = [f"ICD:{i:03d}" for i in range(20)]
    cpt_vocab = [f"CPT:{i:03d}" for i in range(15)]
    full_vocab = icd_vocab + cpt_vocab

    histories = {}
    per_cluster = num_patients // num_clusters
    pid = 0
    for _ in range(num_clusters):
        core = set(rng.sample(icd_vocab, 2)) | set(rng.sample(cpt_vocab, 2))
        for _ in range(per_cluster):
            noise = set(rng.sample(full_vocab, rng.randint(2, 4)))
            histories[f"P{pid:03d}"] = core | noise
            pid += 1
    while pid < num_patients:
        histories[f"P{pid:03d}"] = set(rng.sample(full_vocab, rng.randint(2, 4)))
        pid += 1

    return histories


def main() -> None:
    print("=== Step 1: build fake patient histories (history-only, no leakage) ===")
    histories = _make_fake_histories(num_patients=50, num_clusters=5, seed=0)
    example_pid = "P000"
    print(f"Built {len(histories)} fake patients.")
    print(f"Example history [{example_pid}]: {sorted(histories[example_pid])}")

    print("\n=== Step 2: build PatientKnowledgeGraph (patient<->ICD/CPT edges) ===")
    kg = PatientKnowledgeGraph()
    kg.build_from_history(histories)
    print(f"num_nodes={kg.num_nodes}, num_patients={kg.num_patients}, "
          f"num_codes={kg.num_nodes - kg.num_patients}")
    code_edge_index, code_edge_weight = kg.code_edge_index_and_weight()
    print(f"code (patient<->ICD/CPT) edges: {code_edge_index.shape[1]}")

    print("\n=== Step 3: build patient<->patient similarity edges (Eq. 16) ===")
    sim_edge_index, sim_edge_weight = build_similarity_edges(
        histories, kg.patient_id_to_idx, min_jaccard=0.0
    )
    print(f"similarity edges: {sim_edge_index.shape[1]}, "
          f"weight range [{sim_edge_weight.min().item():.3f}, {sim_edge_weight.max().item():.3f}]")

    edge_index = torch.cat([code_edge_index, sim_edge_index], dim=1)
    edge_weight = torch.cat([code_edge_weight, sim_edge_weight], dim=0)

    print("\n=== Step 4: run GraphSAGE to get node embeddings h^K ===")
    X = kg.node_features()
    graph_model = GraphSAGE(in_dim=X.shape[1], hidden_dim=32, out_dim=32, num_layers=2, num_samples=10)
    with torch.no_grad():
        node_emb = graph_model(X, edge_index, edge_weight)
    print(f"node_emb shape: {tuple(node_emb.shape)}, NaN present: {torch.isnan(node_emb).any().item()}")

    print("\n=== Step 5: retrieval for a query patient (normal tau) ===")
    query_pid = example_pid
    tau, M = 0.2, 3
    results = retrieve(query_pid, histories, node_emb, kg.patient_id_to_idx, tau=tau, M=M)
    print(f"Query patient: {query_pid}, tau={tau}, M={M}")
    print(f"Retrieved {len(results)} candidate(s):")
    for r in results:
        print(f"  {r['patient_id']}: jaccard={r['jaccard']:.3f}, cosine={r['cosine_score']:.3f}, "
              f"shared_codes={sorted(r['shared_codes'])}")

    evidence = linearize_evidence(results)
    print(f"\nLinearized evidence:\n  {evidence!r}")
    assert evidence != "", "expected non-empty evidence when candidates exist"

    print("\n=== Step 6: retrieval with tau too high (no candidates pass) ===")
    tau_high = 0.99
    results_high = retrieve(query_pid, histories, node_emb, kg.patient_id_to_idx, tau=tau_high, M=M)
    evidence_high = linearize_evidence(results_high)
    print(f"Query patient: {query_pid}, tau={tau_high}")
    print(f"Retrieved {len(results_high)} candidate(s), evidence={evidence_high!r}")
    assert results_high == [] and evidence_high == "", "expected empty candidates/evidence at very high tau"

    print("\n=== Step 7: confirm the decoder still runs with empty evidence ===")
    decoder = VQADecoder(z_final_dim=16, test_mode=True, n_prefix=4, gamma=1.0)
    z_final = torch.randn(1, 16)
    loss, logits = decoder(
        z_final,
        system_text="You are a helpful medical VQA assistant.",
        evidence_text=evidence_high,  # empty string, from Step 6
        question_text="What organ is likely affected?",
        answer_text="The kidney is likely affected.",
    )
    print(f"Decoder forward with empty evidence -> loss={loss.item():.4f}")
    assert not torch.isnan(loss), "decoder produced NaN loss with empty evidence"

    print("\nPASS: graph-rag end-to-end test")


if __name__ == "__main__":
    main()
