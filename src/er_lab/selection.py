"""Deterministic thresholding with optional one-owner conflict resolution."""
from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ScoredPair:
    s1_id: str
    rec_id: str
    score: float


def select_pairs(
    entity_ids: Iterable[str], pairs: Iterable[ScoredPair], threshold: float = 0.65,
    *, one_owner: bool = True,
) -> dict[str, set[str]]:
    """Deduplicate by max score, then retain the best S1 owner per target.

    Score ties are resolved by S1 ID, so input iteration order has no effect.
    Each S1 may still own multiple targets. Scores must be finite values in [0, 1].
    """
    entities = list(entity_ids)
    if len(set(entities)) != len(entities) or any(not x.startswith("S1-") for x in entities):
        raise ValueError("entity_ids must be unique S1 identifiers")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    result = {key: set() for key in entities}
    unique = {}
    for pair in pairs:
        if pair.s1_id not in result or not pair.rec_id.startswith(("S2-", "S3-")):
            raise ValueError("Unknown S1 or invalid target-source identifier")
        if not math.isfinite(pair.score) or not 0 <= pair.score <= 1:
            raise ValueError("scores must lie in [0, 1]")
        key = (pair.s1_id, pair.rec_id)
        unique[key] = max(pair.score, unique.get(key, -1.0))
    owners = set()
    for (s1_id, rec_id), score in sorted(unique.items(), key=lambda item: (-item[1], *item[0])):
        if score < threshold or (one_owner and rec_id in owners):
            continue
        result[s1_id].add(rec_id)
        owners.add(rec_id)
    return result

