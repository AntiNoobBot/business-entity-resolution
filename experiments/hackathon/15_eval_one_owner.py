"""Evaluate one-owner conflict resolution on development predictions."""

from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

DEV = Path("dev_features_v1.parquet")
GT = Path("parquet/train_gt.parquet")
MODEL = Path("lgbm_baseline.txt")
REPORT = Path("one_owner_report.txt")

FEATURES = [
    "name_core_exact", "name_sorted_exact", "name_compact_exact",
    "name_prefix4_exact", "address_token_exact", "tail_exact",
    "s_addr_missing", "r_addr_missing", "shared_number_count",
    "number_union_count", "number_jaccard", "number_count_exact",
    "name_length_ratio", "address_length_ratio", "name_ratio",
    "name_token_set", "address_ratio", "address_token_set",
]


def encode(df):
    return df.with_columns(
        pl.when(pl.col("country") == "India").then(0).otherwise(1)
        .cast(pl.Int8).alias("country_code"),
        pl.when(pl.col("source") == "S2").then(0).otherwise(1)
        .cast(pl.Int8).alias("source_code"),
    )


def f05(predictions, truth):
    scores = []

    for s1_id, true_ids in truth.items():
        predicted = predictions.get(s1_id, set())

        if not true_ids and not predicted:
            scores.append(1.0)
            continue

        if not predicted:
            scores.append(0.0)
            continue

        tp = len(predicted & true_ids)
        if tp == 0:
            scores.append(0.0)
            continue

        precision = tp / len(predicted)
        recall = tp / len(true_ids)

        scores.append(
            1.25 * precision * recall / (0.25 * precision + recall)
        )

    return float(np.mean(scores))


def predictions_from(df):
    grouped = df.group_by("s1_id").agg(pl.col("rec_id"))

    return {
        s1_id: set(rec_ids)
        for s1_id, rec_ids in grouped.iter_rows()
    }


def score(df, truth):
    return f05(predictions_from(df), truth)


def build_truth(dev_s1_ids):
    gt = pl.read_parquet(GT)

    rows = (
        gt.filter(
            pl.col("source1_entity_id").is_in(dev_s1_ids)
            & pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids") != "")
        )
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.strip_chars().alias("rec_id"),
        )
    )

    truth = {s1_id: set() for s1_id in dev_s1_ids}

    for s1_id, rec_id in rows.iter_rows():
        truth[s1_id].add(rec_id)

    return truth


def select_by_thresholds(scored, thresholds):
    return scored.filter(
        (
            (pl.col("country") == "India")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= thresholds["India_S2"])
        )
        | (
            (pl.col("country") == "India")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= thresholds["India_S3"])
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= thresholds["US_S2"])
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= thresholds["US_S3"])
        )
    )


def apply_one_owner(selected):
    # For each S2/S3 record, retain the S1 with the highest model probability.
    # Deterministic S1-ID ordering breaks exact probability ties.
    return (
        selected.sort(
            ["rec_id", "probability", "s1_id"],
            descending=[False, True, False],
        )
        .group_by("rec_id", maintain_order=True)
        .first()
    )


def main():
    dev = encode(pl.read_parquet(DEV))
    x_dev = dev.select(FEATURES + ["country_code", "source_code"]).to_numpy()

    booster = lgb.Booster(model_file=str(MODEL))
    probabilities = booster.predict(x_dev)

    scored = dev.select("s1_id", "rec_id", "country", "source").with_columns(
        pl.Series("probability", probabilities)
    )

    dev_s1_ids = (
        pl.read_parquet("parquet/train_s1.parquet")
        .filter((pl.col("entity_id").hash(seed=41) % 100) == 7)
        ["entity_id"]
        .to_list()
    )

    truth = build_truth(dev_s1_ids)

    variants = {
        "global_0.600": {
            "India_S2": 0.600,
            "India_S3": 0.600,
            "US_S2": 0.600,
            "US_S3": 0.600,
        },
        "tuned_country_source": {
            "India_S2": 0.600,
            "India_S3": 0.600,
            "US_S2": 0.750,
            "US_S3": 0.575,
        },
    }

    lines = ["== ONE-OWNER CONFLICT RESOLUTION =="]

    for name, thresholds in variants.items():
        selected = select_by_thresholds(scored, thresholds)

        collision_counts = selected.group_by("rec_id").len()
        conflicts = collision_counts.filter(pl.col("len") > 1)

        resolved = apply_one_owner(selected)

        score_before = score(selected, truth)
        score_after = score(resolved, truth)

        lines.extend([
            "",
            f"== {name} ==",
            f"thresholds: {thresholds}",
            f"selected_pairs_before: {selected.height:,}",
            f"conflicting_rec_ids: {conflicts.height:,}",
            f"pairs_in_conflicts: {conflicts['len'].sum() if conflicts.height else 0:,}",
            f"selected_pairs_after: {resolved.height:,}",
            f"macro_f0.5_before: {score_before:.6f}",
            f"macro_f0.5_after: {score_after:.6f}",
            f"improvement: {score_after - score_before:+.6f}",
        ])

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    main()
