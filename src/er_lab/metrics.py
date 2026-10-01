"""Keep retrieval coverage, model quality, and oracle ceilings separate."""
from collections.abc import Mapping, Set
import math


def entity_fbeta(truth: Set[str], predicted: Set[str], beta: float = 0.5) -> float:
    """Score one S1 entity, awarding 1 to correctly predicted singletons."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be positive and finite")
    if not truth and not predicted:
        return 1.0
    beta2 = beta * beta
    return (1.0 + beta2) * len(truth & predicted) / (beta2 * len(truth) + len(predicted))


def _same_entities(truth: Mapping, other: Mapping, name: str) -> None:
    if truth.keys() != other.keys():
        missing = len(truth.keys() - other.keys())
        extra = len(other.keys() - truth.keys())
        raise ValueError(f"{name} must cover exactly the truth entities: {missing} missing, {extra} extra")


def evaluate(
    truth: Mapping[str, Set[str]],
    predictions: Mapping[str, Set[str]],
    candidates: Mapping[str, Set[str]] | None = None,
) -> dict:
    """Macro F0.5 over all entities; optional candidate recall and oracle F0.5.

    Coverage is strict: a missing row cannot silently become a correct singleton.
    Pair-level precision/recall are auxiliary micro metrics, not leaderboard scores.
    A zero denominator yields None for those auxiliary metrics.
    """
    if not truth:
        raise ValueError("truth must contain at least one entity")
    _same_entities(truth, predictions, "predictions")
    if candidates is not None:
        _same_entities(truth, candidates, "candidates")
        if any(not predictions[key] <= candidates[key] for key in truth):
            raise ValueError("Every predicted match must belong to its candidate set")
    tp = sum(len(truth[key] & predictions[key]) for key in truth)
    total_true = sum(map(len, truth.values()))
    total_predicted = sum(map(len, predictions.values()))
    result = {
        "entities": len(truth),
        "macro_f0_5": sum(entity_fbeta(truth[key], predictions[key]) for key in truth) / len(truth),
        "true_positives": tp,
        "false_positives": total_predicted - tp,
        "false_negatives": total_true - tp,
        "micro_precision": tp / total_predicted if total_predicted else None,
        "micro_recall": tp / total_true if total_true else None,
        "true_singletons": sum(not ids for ids in truth.values()),
        "correct_singletons": sum(not truth[key] and not predictions[key] for key in truth),
    }
    if candidates is not None:
        recovered = {key: truth[key] & candidates[key] for key in truth}
        result.update({
            "candidate_pairs": sum(map(len, candidates.values())),
            "candidate_recall": sum(map(len, recovered.values())) / total_true if total_true else None,
            "oracle_macro_f0_5": sum(entity_fbeta(truth[key], recovered[key]) for key in truth) / len(truth),
        })
    return result

