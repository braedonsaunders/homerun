import json
from pathlib import Path


def _candidate_passes(candidate: dict) -> bool:
    keyword_score = float(candidate.get("keyword_score", 0.0))
    semantic_score = float(candidate.get("semantic_score", 0.0))
    overlap_tokens = int(candidate.get("overlap_tokens", 0))

    if not (keyword_score >= 0.04 or semantic_score >= 0.22):
        return False
    if overlap_tokens < 1:
        return False

    score = 0.25 * keyword_score + 0.45 * semantic_score + 0.30 * float(
        candidate.get("event_score", 0.0)
    )
    return score >= 0.42


def _candidate_score(candidate: dict) -> float:
    return (
        0.25 * float(candidate.get("keyword_score", 0.0))
        + 0.45 * float(candidate.get("semantic_score", 0.0))
        + 0.30 * float(candidate.get("event_score", 0.0))
    )


def test_news_matching_precision_eval_thresholds():
    dataset_path = Path(__file__).with_name("news_eval_dataset.jsonl")
    rows = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows, "dataset must not be empty"

    tp = 0
    fp = 0

    for row in rows:
        candidates = [c for c in row.get("candidates", []) if _candidate_passes(c)]
        if not candidates:
            continue
        top = max(candidates, key=_candidate_score)
        if bool(top.get("is_match", False)):
            tp += 1
        else:
            fp += 1

    assert (tp + fp) > 0, "evaluation produced no positive predictions"
    top1_precision = tp / (tp + fp)
    false_positive_rate = fp / (tp + fp)

    assert top1_precision >= 0.85
    assert false_positive_rate <= 0.15
