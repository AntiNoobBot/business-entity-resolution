"""Strict readers and deterministic writers for two-column TSV match lists."""
import csv
from pathlib import Path
from collections.abc import Mapping, Set


def read_pair_lists(path: str | Path, *, candidates: bool = False) -> dict[str, set[str]]:
    column = "candidate_entity_ids" if candidates else "matched_entity_ids"
    result = {}
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader, None) != ["source1_entity_id", column]:
            raise ValueError(f"Expected columns source1_entity_id and {column}")
        for line, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(f"Line {line}: expected exactly two tab-separated columns")
            key, value = row
            if not key.startswith("S1-") or key in result:
                raise ValueError(f"Line {line}: invalid or duplicate S1 identifier")
            ids = value.split(",") if value else []
            if len(set(ids)) != len(ids) or any(
                x != x.strip() or not x.startswith(("S2-", "S3-")) for x in ids
            ):
                raise ValueError(f"Line {line}: invalid or duplicate target identifiers")
            result[key] = set(ids)
    return result


def write_pair_lists(path: Path, values: Mapping[str, Set[str]], *, candidates: bool = False) -> None:
    """Create a new file; existing files are deliberately not overwritten."""
    column = "candidate_entity_ids" if candidates else "matched_entity_ids"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", quoting=csv.QUOTE_NONE, lineterminator="\n")
        writer.writerow(["source1_entity_id", column])
        for key, ids in sorted(values.items()):
            writer.writerow([key, ",".join(sorted(ids))])

