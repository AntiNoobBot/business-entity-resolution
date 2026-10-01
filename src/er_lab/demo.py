"""Synthetic scored pairs illustrate the metric; no challenge data or model."""
from .metrics import evaluate
from .selection import ScoredPair, select_pairs


def demo():
    truth = {
        "S1-demo-a": {"S2-demo-a", "S3-demo-a"},
        "S1-demo-b": {"S2-demo-b"},
        "S1-demo-c": set(),
        "S1-demo-d": {"S3-demo-d"},
    }
    pairs = [
        ScoredPair("S1-demo-a", "S2-demo-a", 0.92),
        ScoredPair("S1-demo-a", "S2-demo-a", 0.80),  # repeated retrieval channel
        ScoredPair("S1-demo-a", "S3-demo-d", 0.75),  # another S1 wins ownership
        ScoredPair("S1-demo-b", "S2-demo-b", 0.83),
        ScoredPair("S1-demo-b", "S3-demo-decoy", 0.80),  # accepted false positive
        ScoredPair("S1-demo-c", "S2-demo-noise", 0.12),
        ScoredPair("S1-demo-d", "S3-demo-d", 0.91),
    ]
    candidates = {key: set() for key in truth}
    for pair in pairs:
        candidates[pair.s1_id].add(pair.rec_id)
    predictions = select_pairs(truth, pairs, threshold=0.65)
    metrics = evaluate(truth, predictions, candidates)
    metrics["scope"] = "synthetic demonstration; not a hackathon measurement"
    return truth, predictions, candidates, metrics

