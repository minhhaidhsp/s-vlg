"""Patient knowledge graph structure — Mục 3.4.3.

Holds patient nodes (each carrying its set of HISTORICAL clinical codes S_i),
ICD code nodes, and CPT code nodes, connected by patient<->code membership
edges. Patient<->patient similarity edges (Eq. 16) are intentionally NOT
built here — see src/graph/retrieval.py, which computes them from the same
`patient_histories` this class stores.
"""

import torch


class PatientKnowledgeGraph:
    """Tripartite graph: patient nodes, ICD code nodes, CPT code nodes.

    IMPORTANT (anti-leakage constraint): `patient_histories` passed to
    `build_from_history` must contain ONLY codes recorded strictly BEFORE the
    current study/encounter being predicted on. Including a code from the
    current or a future encounter would let the retrieved "similar cases"
    evidence leak information about the very case being evaluated — this is
    a hard requirement, not a style preference.
    """

    def __init__(self):
        self.patient_ids: list = []
        self.patient_id_to_idx: dict = {}
        self.code_to_idx: dict = {}
        self.idx_to_code: dict = {}
        self.patient_histories: dict = {}  # patient_id -> set(codes), HISTORY ONLY
        self.edge_list: list = []          # (patient_idx, code_idx) membership pairs
        self.node_type: list = []          # 0=patient, 1=ICD, 2=CPT, aligned to node index

    @staticmethod
    def _code_type(code: str) -> int:
        if code.startswith("ICD:"):
            return 1
        if code.startswith("CPT:"):
            return 2
        raise ValueError(f"Unrecognized code prefix (expected 'ICD:' or 'CPT:'): {code!r}")

    def build_from_history(self, patient_histories: dict) -> None:
        """Build the tripartite graph from historical code sets.

        Args:
            patient_histories: {patient_id: set(codes)}, e.g.
                {"P001": {"ICD:410.9", "CPT:99213"}}. Codes must be prefixed
                "ICD:" or "CPT:". HISTORY-ONLY — see class docstring.
        """
        self.patient_histories = {pid: set(codes) for pid, codes in patient_histories.items()}

        # Patient nodes first: indices 0..P-1.
        self.patient_ids = sorted(self.patient_histories.keys())
        self.patient_id_to_idx = {pid: i for i, pid in enumerate(self.patient_ids)}
        self.node_type = [0] * len(self.patient_ids)

        # Then all distinct codes (ICD/CPT), sorted for determinism.
        self.code_to_idx = {}
        self.idx_to_code = {}
        all_codes = sorted({c for codes in self.patient_histories.values() for c in codes})
        for code in all_codes:
            idx = len(self.node_type)
            self.code_to_idx[code] = idx
            self.idx_to_code[idx] = code
            self.node_type.append(self._code_type(code))

        # Patient<->code membership edges (undirected pair list; symmetrized in code_edge_index_and_weight).
        self.edge_list = []
        for pid, codes in self.patient_histories.items():
            p_idx = self.patient_id_to_idx[pid]
            for code in codes:
                self.edge_list.append((p_idx, self.code_to_idx[code]))

    @property
    def num_nodes(self) -> int:
        return len(self.node_type)

    @property
    def num_patients(self) -> int:
        return len(self.patient_ids)

    def node_features(self) -> torch.Tensor:
        """Bag-of-codes features: patient nodes get a multi-hot vector over the
        code vocabulary; each code node gets a one-hot vector at its own
        vocabulary position. All nodes share dimension = vocab size, as
        required by GraphSAGE's single `in_dim`.
        """
        vocab_size = len(self.code_to_idx)
        X = torch.zeros(self.num_nodes, vocab_size)
        for pid, codes in self.patient_histories.items():
            p_idx = self.patient_id_to_idx[pid]
            for code in codes:
                vocab_pos = self.code_to_idx[code] - self.num_patients
                X[p_idx, vocab_pos] = 1.0
        for code, idx in self.code_to_idx.items():
            vocab_pos = idx - self.num_patients
            X[idx, vocab_pos] = 1.0
        return X

    def code_edge_index_and_weight(self):
        """Symmetrized patient<->code membership edges, uniform weight 1.0."""
        if not self.edge_list:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
        src = [p for p, c in self.edge_list] + [c for p, c in self.edge_list]
        dst = [c for p, c in self.edge_list] + [p for p, c in self.edge_list]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1])
        return edge_index, edge_weight

    def get_patient_node_idx(self, patient_id) -> int:
        return self.patient_id_to_idx[patient_id]
