"""Inspect LightGBM feature importance and development-set errors."""

from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

DEV = Path("dev_features_v1.parquet")
GT = Path("parquet/train_gt.parquet")
MODEL = Path("lgbm_baseline.txt")
REPORT = Path("error_analysis_report.txt")

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


def select_tuned(scored):
    return scored.filter(
        (
            (pl.col("country") == "India")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= 0.600)
        )
        | (
            (pl.col("country") == "India")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= 0.600)
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= 0.750)
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= 0.575)
        )
    )


def apply_one_owner(selected):
    return (
        selected.sort(
            ["rec_id", "probability", "s1_id"],
            descending=[False, True, False],
        )
        .group_by("rec_id", maintain_order=True)
        .first()
    )


def attach_text(pairs):
    s1 = pl.read_parquet("parquet/train_s1.parquet").select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("s1_name"),
        pl.col("business_address").alias("s1_address"),
    )

    s2 = pl.read_parquet("parquet/train_s2.parquet").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("rec_name"),
        pl.col("business_address").alias("rec_address"),
    )

    s3 = pl.read_parquet("parquet/train_s3.parquet").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("rec_name"),
        pl.col("business_address").alias("rec_address"),
    )

    base = pairs.join(s1, on="s1_id", how="left")

    p2 = (
        base.filter(pl.col("source") == "S2")
        .join(s2, on="rec_id", how="left")
    )

    p3 = (
        base.filter(pl.col("source") == "S3")
        .join(s3, on="rec_id", how="left")
    )

    return pl.concat([p2, p3], how="vertical_relaxed")


def write_samples(lines, title, df, limit=15):
    lines.append("")
    lines.append(title)

    if df.height == 0:
        lines.append("No examples.")
        return

    for row in df.head(limit).iter_rows(named=True):
        probability = row.get("probability")
        score_text = (
            f" probability={probability:.4f}"
            if probability is not None else ""
        )
        lines.append(
            f"{row['country']}/{row['source']}{score_text}"
        )
        lines.append(
            f"S1 {row['s1_id']}: {row['s1_name']} | {row['s1_address']}"
        )
        lines.append(
            f"R  {row['rec_id']}: {row['rec_name']} | {row['rec_address']}"
        )
        lines.append("")


def main():
    dev = encode(pl.read_parquet(DEV))
    x_dev = dev.select(FEATURES + ["country_code", "source_code"]).to_numpy()

    booster = lgb.Booster(model_file=str(MODEL))
    probabilities = booster.predict(x_dev)

    scored = dev.with_columns(
        pl.Series("probability", probabilities)
    )

    selected_before = select_tuned(scored)
    selected = apply_one_owner(selected_before)

    false_positives = (
        selected.filter(pl.col("label") == 0)
        .sort("probability", descending=True)
    )

    false_negatives = (
        scored.filter(pl.col("label") == 1)
        .join(
            selected.select("s1_id", "rec_id"),
            on=["s1_id", "rec_id"],
            how="anti",
        )
        .sort("probability", descending=True)
    )

    dev_s1_ids = (
        pl.read_parquet("parquet/train_s1.parquet")
        .filter((pl.col("entity_id").hash(seed=41) % 100) == 7)
        ["entity_id"]
        .to_list()
    )

    truth_links = (
        pl.read_parquet(GT)
        .filter(
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
        .with_columns(
            pl.col("rec_id").str.slice(0, 2).alias("source")
        )
    )

    blocking_missed = (
        truth_links.join(
            dev.select("s1_id", "rec_id"),
            on=["s1_id", "rec_id"],
            how="anti",
        )
        .join(
            pl.read_parquet("parquet/train_s1.parquet").select(
                pl.col("entity_id").alias("s1_id"),
                "country",
            ),
            on="s1_id",
        )
    )

    importance = pl.DataFrame({
        "feature": FEATURES + ["country_code", "source_code"],
        "gain": booster.feature_importance(importance_type="gain"),
        "split_count": booster.feature_importance(importance_type="split"),
    }).sort("gain", descending=True)

    lines = [
        "== LIGHTGBM ERROR ANALYSIS ==",
        f"development candidates: {dev.height:,}",
        f"accepted pairs before one-owner: {selected_before.height:,}",
        f"accepted pairs after one-owner: {selected.height:,}",
        f"false positives after one-owner: {false_positives.height:,}",
        f"candidate-level false negatives: {false_negatives.height:,}",
        f"true links missed by blocker: {blocking_missed.height:,}",
        "",
        "== FEATURE IMPORTANCE BY GAIN ==",
    ]

    for row in importance.iter_rows(named=True):
        lines.append(
            f"{row['feature']}: gain={row['gain']:.2f}, "
            f"splits={row['split_count']}"
        )

    fp_text = attach_text(false_positives)
    fn_text = attach_text(false_negatives)
    missed_text = attach_text(blocking_missed)

    write_samples(lines, "== HIGHEST-SCORE FALSE POSITIVES ==", fp_text)
    write_samples(lines, "== HIGHEST-SCORE CANDIDATE FALSE NEGATIVES ==", fn_text)
    write_samples(lines, "== TRUE LINKS MISSED BY BLOCKING ==", missed_text)

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    main()
