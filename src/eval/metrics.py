"""VQA evaluation metrics — shared by V1 and V2 (Table 6/7/8/10 style numbers).

VQA-Acc / Exact Match / BLEU-1 / BLEU-4 over all questions; Precision/
Recall/F1/AUC-ROC restricted to CLOSED (yes/no) questions; per-category
breakdown (by answer_type or question_type); and the risk-coverage curve
built from (uncertainty, correctness) pairs.
"""

import sacrebleu
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def normalize_answer(text: str) -> str:
    return text.strip().lower()


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if normalize_answer(pred) == normalize_answer(ref) else 0.0


def vqa_accuracy(preds: list, refs: list) -> float:
    """Exact-match-based VQA accuracy (fraction of correct answers)."""
    if not preds:
        return float("nan")
    return sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)


def bleu_score(preds: list, refs: list, max_ngram_order: int) -> float:
    """Corpus BLEU-N (sacrebleu, 0-100 scale) — max_ngram_order=1 for BLEU-1,
    4 for BLEU-4."""
    if not preds:
        return float("nan")
    bleu = sacrebleu.BLEU(max_ngram_order=max_ngram_order, effective_order=True)
    result = bleu.corpus_score(preds, [refs])
    return result.score


def closed_question_prf1_auc(preds: list, refs: list, answer_types: list, scores: list = None) -> dict:
    """Precision/Recall/F1 (and AUC-ROC if `scores` is given) over CLOSED
    (yes/no) questions only. Binary positive class = "yes".

    Args:
        scores: per-sample confidence that the answer is "yes" (e.g.
            1 - uncertainty, or a model probability) — needed for AUC-ROC.
    """
    idx = [i for i, t in enumerate(answer_types) if t == "CLOSED"]
    if not idx:
        return {"precision": None, "recall": None, "f1": None, "auc_roc": None}

    y_true = [1 if normalize_answer(refs[i]) == "yes" else 0 for i in idx]
    y_pred = [1 if normalize_answer(preds[i]) == "yes" else 0 for i in idx]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    result = {"precision": precision, "recall": recall, "f1": f1}

    if scores is not None and len(set(y_true)) > 1:
        y_score = [scores[i] for i in idx]
        try:
            result["auc_roc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            result["auc_roc"] = None
    else:
        result["auc_roc"] = None
    return result


def breakdown_by_category(preds: list, refs: list, categories: list) -> dict:
    """Group (preds, refs) by `categories` (e.g. answer_type or question_type
    per sample) into {category: {"vqa_acc", "exact_match", "n"}} — Table 7.
    `None` categories are grouped under the string "unknown".
    """
    groups: dict = {}
    for p, r, c in zip(preds, refs, categories):
        key = c if c is not None else "unknown"
        groups.setdefault(key, {"preds": [], "refs": []})
        groups[key]["preds"].append(p)
        groups[key]["refs"].append(r)

    result = {}
    for cat, data in groups.items():
        acc = vqa_accuracy(data["preds"], data["refs"])
        result[cat] = {"vqa_acc": acc, "exact_match": acc, "n": len(data["preds"])}
    return result


def _trapz(x: list, y: list) -> float:
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return area


def risk_coverage_curve(uncertainties: list, correct_flags: list) -> dict:
    """Risk-coverage curve (Table 10 / Figure 8).

    Samples are sorted by ascending uncertainty (most confident first);
    coverage k/n sweeps from the single most-confident sample up to the
    whole set; risk at each point is the error rate among the covered
    (most-confident-so-far) subset. AUC is the area under this curve
    (trapezoidal rule) — lower is better.

    IMPORTANT: must be computed over the FULL val/test split, never a
    subset of it — see PAPER_DATA_MAP.md, Table 10.

    Returns:
        {"coverage_points": [...], "risk_values": [...], "auc": float}
    """
    n = len(uncertainties)
    if n == 0:
        return {"coverage_points": [], "risk_values": [], "auc": float("nan")}

    order = sorted(range(n), key=lambda i: uncertainties[i])
    coverage_points, risk_values = [], []
    cum_correct = 0.0
    for k, idx in enumerate(order, start=1):
        cum_correct += correct_flags[idx]
        coverage_points.append(k / n)
        risk_values.append(1.0 - cum_correct / k)

    auc = _trapz(coverage_points, risk_values)
    return {"coverage_points": coverage_points, "risk_values": risk_values, "auc": auc}


def _self_test() -> bool:
    ok = True

    preds = ["yes", "no", "yes", "cardiovascular", "mri"]
    refs = ["yes", "no", "yes", "cardiovascular", "ct"]
    answer_types = ["CLOSED", "CLOSED", "CLOSED", "OPEN", "OPEN"]

    acc = vqa_accuracy(preds, refs)
    if abs(acc - 4 / 5) > 1e-9:
        print(f"FAIL: vqa_accuracy {acc} != 0.8")
        ok = False

    em = exact_match("Yes", "yes")
    if em != 1.0:
        print(f"FAIL: exact_match case-insensitive check failed, got {em}")
        ok = False

    b1 = bleu_score(preds, refs, max_ngram_order=1)
    b4 = bleu_score(preds, refs, max_ngram_order=4)
    if not (0.0 <= b1 <= 100.0) or not (0.0 <= b4 <= 100.0):
        print(f"FAIL: BLEU out of [0,100] range: bleu1={b1}, bleu4={b4}")
        ok = False

    prf1 = closed_question_prf1_auc(preds, refs, answer_types, scores=[0.9, 0.4, 0.8, 0.5, 0.5])
    if prf1["precision"] is None or prf1["recall"] is None or prf1["f1"] is None:
        print("FAIL: closed_question_prf1_auc returned None for a batch with CLOSED questions")
        ok = False
    if prf1["auc_roc"] is None:
        print("FAIL: expected a valid AUC-ROC (both classes present)")
        ok = False

    breakdown = breakdown_by_category(preds, refs, answer_types)
    if set(breakdown.keys()) != {"CLOSED", "OPEN"}:
        print(f"FAIL: unexpected breakdown categories {breakdown.keys()}")
        ok = False
    if breakdown["CLOSED"]["n"] != 3 or breakdown["OPEN"]["n"] != 2:
        print(f"FAIL: breakdown group sizes wrong: {breakdown}")
        ok = False

    uncertainties = [0.9, 0.1, 0.5, 0.3, 0.7]
    correct_flags = [0.0, 1.0, 1.0, 1.0, 0.0]
    rc = risk_coverage_curve(uncertainties, correct_flags)
    if len(rc["coverage_points"]) != 5 or len(rc["risk_values"]) != 5:
        print("FAIL: risk_coverage_curve length mismatch")
        ok = False
    if rc["coverage_points"][0] != 0.2 or rc["coverage_points"][-1] != 1.0:
        print(f"FAIL: risk_coverage_curve coverage points wrong: {rc['coverage_points']}")
        ok = False
    # Most confident sample (idx=1, U=0.1) is correct -> risk at coverage=0.2 should be 0.
    if rc["risk_values"][0] != 0.0:
        print(f"FAIL: expected risk=0 at first (most confident) coverage point, got {rc['risk_values'][0]}")
        ok = False
    if not (0.0 <= rc["auc"] <= 1.0):
        print(f"FAIL: risk-coverage AUC out of [0,1]: {rc['auc']}")
        ok = False

    empty_rc = risk_coverage_curve([], [])
    if empty_rc["coverage_points"] != [] or not (empty_rc["auc"] != empty_rc["auc"]):  # NaN check
        print("FAIL: empty risk_coverage_curve should return empty lists and NaN auc")
        ok = False

    print("PASS: metrics" if ok else "FAIL: metrics")
    return ok


if __name__ == "__main__":
    _self_test()
