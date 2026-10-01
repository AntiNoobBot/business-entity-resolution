import argparse
import json
from pathlib import Path

from .demo import demo
from .io import read_pair_lists, write_pair_lists
from .metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description="Entity resolution evaluation without dataset downloads")
    commands = parser.add_subparsers(dest="command", required=True)
    demonstration = commands.add_parser("demo", help="Run a tiny synthetic scored-pair example")
    demonstration.add_argument("--output", type=Path, help="Create a NEW directory containing the synthetic TSVs")
    score = commands.add_parser("score", help="Evaluate complete S1 prediction lists against labelled truth")
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            truth, predictions, candidates, metrics = demo()
            if args.output:
                args.output.mkdir(parents=True, exist_ok=False)
                write_pair_lists(args.output / "truth.tsv", truth)
                write_pair_lists(args.output / "matching_results.tsv", predictions)
                write_pair_lists(args.output / "candidate_pairs.tsv", candidates, candidates=True)
        else:
            truth = read_pair_lists(args.truth)
            predictions = read_pair_lists(args.predictions)
            candidates = read_pair_lists(args.candidates, candidates=True) if args.candidates else None
            metrics = evaluate(truth, predictions, candidates)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
