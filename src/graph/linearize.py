"""Linearize Graph-RAG retrieval results into E_evidence text (Eq. 30 input
to the decoder — see src/models/decoder.py)."""


def linearize_evidence(retrieved: list, max_cases: int = None, max_codes_per_case: int = 5) -> str:
    """Render retrieved similar cases as a short, fixed-template English string.

    Args:
        retrieved: output of src.graph.retrieval.retrieve() — a list of dicts
            with "patient_id", "cosine_score", "shared_codes".
        max_cases: cap the number of cases rendered (None = render all).
        max_codes_per_case: cap the number of diagnosis/procedure codes shown per case.

    Returns:
        e.g. "Retrieved similar cases: Case 1 (similarity 0.62) shares
        diagnoses [001, 002] and procedures [003]. Case 2 (similarity 0.51)
        shares diagnoses [004]." Empty string if `retrieved` is empty — the
        decoder (src/models/decoder.py) accepts an empty evidence string and
        still generates normally.
    """
    if not retrieved:
        return ""

    cases = retrieved[:max_cases] if max_cases is not None else retrieved
    segments = []
    for i, case in enumerate(cases, start=1):
        shared = sorted(case["shared_codes"])
        icd_codes = [c[len("ICD:"):] for c in shared if c.startswith("ICD:")][:max_codes_per_case]
        cpt_codes = [c[len("CPT:"):] for c in shared if c.startswith("CPT:")][:max_codes_per_case]

        segment = f"Case {i} (similarity {case['cosine_score']:.2f})"
        clauses = []
        if icd_codes:
            clauses.append(f"shares diagnoses [{', '.join(icd_codes)}]")
        if cpt_codes:
            clauses.append(f"procedures [{', '.join(cpt_codes)}]")
        segment += " " + " and ".join(clauses) if clauses else " shares no coded diagnoses/procedures on record"
        segments.append(segment + ".")

    return "Retrieved similar cases: " + " ".join(segments)
