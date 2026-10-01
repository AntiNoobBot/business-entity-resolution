"""Build pairwise features for the labelled v5 development candidates."""

from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

from norm import add_name_cols, norm_addr, plain, split_last, token_key

P = Path("parquet")
CANDIDATES = Path("dev_candidates_v5.parquet")
OUT = Path("dev_features_v1.parquet")
REPORT = Path("dev_feature_report.txt")
FUZZY_BATCH = 250_000


def rd(name):
    return pl.read_parquet(P / f"{name}.parquet")


def add_pair_features(df):
    df = add_name_cols(df, "s_name", "s_")
    df = add_name_cols(df, "r_name", "r_")

    s_body, s_tail = split_last(pl.col("s_addr"))
    r_body, r_tail = split_last(pl.col("r_addr"))

    df = df.with_columns(
        norm_addr(pl.col("s_addr"), pl.col("country")).alias("s_addr_norm"),
        norm_addr(pl.col("r_addr"), pl.col("country")).alias("r_addr_norm"),
        plain(s_tail).alias("s_tail"),
        plain(r_tail).alias("r_tail"),
        pl.col("s_addr").is_null().alias("s_addr_missing"),
        pl.col("r_addr").is_null().alias("r_addr_missing"),
    )

    df = df.with_columns(
        pl.col("s_addr_norm").str.extract_all(r"\d+").alias("s_numbers"),
        pl.col("r_addr_norm").str.extract_all(r"\d+").alias("r_numbers"),
    )

    df = df.with_columns(
        (pl.col("s_core") == pl.col("r_core")).alias("name_core_exact"),
        (pl.col("s_sorted") == pl.col("r_sorted")).alias("name_sorted_exact"),
        (pl.col("s_compact") == pl.col("r_compact")).alias("name_compact_exact"),
        (
            token_key(pl.col("s_addr_norm"))
            == token_key(pl.col("r_addr_norm"))
        ).fill_null(False).alias("address_token_exact"),
        (pl.col("s_tail") == pl.col("r_tail")).fill_null(False).alias("tail_exact"),
        pl.col("s_core").str.len_chars().alias("s_name_len"),
        pl.col("r_core").str.len_chars().alias("r_name_len"),
        pl.col("s_addr_norm").str.len_chars().fill_null(0).alias("s_addr_len"),
        pl.col("r_addr_norm").str.len_chars().fill_null(0).alias("r_addr_len"),
        pl.col("s_core").str.slice(0, 4).eq(pl.col("r_core").str.slice(0, 4))
        .alias("name_prefix4_exact"),
    )

    df = df.with_columns(
        pl.col("s_numbers")
        .list.set_intersection(pl.col("r_numbers"))
        .list.len()
        .alias("shared_number_count"),
        pl.col("s_numbers")
        .list.set_union(pl.col("r_numbers"))
        .list.len()
        .alias("number_union_count"),
    )

    return df.with_columns(
        pl.when(pl.col("number_union_count") > 0)
        .then(pl.col("shared_number_count") / pl.col("number_union_count"))
        .otherwise(0.0)
        .alias("number_jaccard"),
        (
            pl.col("s_numbers").list.len() == pl.col("r_numbers").list.len()
        ).alias("number_count_exact"),
        (
            pl.col("s_name_len").cast(pl.Float32)
            / pl.max_horizontal(
                pl.col("s_name_len"),
                pl.col("r_name_len"),
                pl.lit(1),
            )
        ).alias("name_length_ratio"),
        (
            pl.col("s_addr_len").cast(pl.Float32)
            / pl.max_horizontal(
                pl.col("s_addr_len"),
                pl.col("r_addr_len"),
                pl.lit(1),
            )
        ).alias("address_length_ratio"),
    )


def add_fuzzy_features(df):
    """RapidFuzz features in chunks to keep memory stable."""
    name_ratio = []
    name_token_set = []
    address_ratio = []
    address_token_set = []

    for start in range(0, df.height, FUZZY_BATCH):
        chunk = df.slice(start, FUZZY_BATCH)

        s_names = chunk["s_core"].fill_null("").to_list()
        r_names = chunk["r_core"].fill_null("").to_list()
        s_addrs = chunk["s_addr_norm"].fill_null("").to_list()
        r_addrs = chunk["r_addr_norm"].fill_null("").to_list()

        name_ratio.extend(
            fuzz.ratio(a, b) / 100.0
            for a, b in zip(s_names, r_names)
        )
        name_token_set.extend(
            fuzz.token_set_ratio(a, b) / 100.0
            for a, b in zip(s_names, r_names)
        )
        address_ratio.extend(
            fuzz.ratio(a, b) / 100.0
            for a, b in zip(s_addrs, r_addrs)
        )
        address_token_set.extend(
            fuzz.token_set_ratio(a, b) / 100.0
            for a, b in zip(s_addrs, r_addrs)
        )

        print(f"fuzzy features: {min(start + FUZZY_BATCH, df.height):,}/{df.height:,}")

    return df.with_columns(
        pl.Series("name_ratio", name_ratio, dtype=pl.Float32),
        pl.Series("name_token_set", name_token_set, dtype=pl.Float32),
        pl.Series("address_ratio", address_ratio, dtype=pl.Float32),
        pl.Series("address_token_set", address_token_set, dtype=pl.Float32),
    )


def main():
    candidates = pl.read_parquet(CANDIDATES)

    s1 = rd("train_s1").select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("s_name"),
        pl.col("business_address").alias("s_addr"),
    )

    s2 = rd("train_s2").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"),
    )

    s3 = rd("train_s3").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"),
    )

    print(f"candidates: {candidates.height:,}")

    c2 = (
        candidates.filter(pl.col("source") == "S2")
        .join(s1, on="s1_id")
        .join(s2, on="rec_id")
    )

    c3 = (
        candidates.filter(pl.col("source") == "S3")
        .join(s1, on="s1_id")
        .join(s3, on="rec_id")
    )

    pairs = pl.concat([c2, c3])
    print("normalizing and building exact/token/number features...")
    pairs = add_pair_features(pairs)

    print("building fuzzy similarity features...")
    pairs = add_fuzzy_features(pairs)

    feature_columns = [
        "s1_id",
        "rec_id",
        "country",
        "source",
        "label",
        "name_core_exact",
        "name_sorted_exact",
        "name_compact_exact",
        "name_prefix4_exact",
        "address_token_exact",
        "tail_exact",
        "s_addr_missing",
        "r_addr_missing",
        "shared_number_count",
        "number_union_count",
        "number_jaccard",
        "number_count_exact",
        "name_length_ratio",
        "address_length_ratio",
        "name_ratio",
        "name_token_set",
        "address_ratio",
        "address_token_set",
    ]

    features = pairs.select(feature_columns)
    features.write_parquet(OUT, compression="zstd")

    lines = []

    def out(*items):
        text = " ".join(str(x) for x in items)
        print(text)
        lines.append(text)

    out("== DEV FEATURE REPORT v1 ==")
    out(f"rows: {features.height:,}")
    out(f"positives: {features['label'].sum():,}")
    out(f"feature file: {OUT}")

    numeric = [
        "shared_number_count",
        "number_jaccard",
        "name_length_ratio",
        "address_length_ratio",
        "name_ratio",
        "name_token_set",
        "address_ratio",
        "address_token_set",
    ]

    flags = [
        "name_core_exact",
        "name_sorted_exact",
        "name_compact_exact",
        "name_prefix4_exact",
        "address_token_exact",
        "tail_exact",
        "s_addr_missing",
        "r_addr_missing",
        "number_count_exact",
    ]

    out("\n== MEAN NUMERIC FEATURES BY LABEL ==")
    out(
        features.group_by("label")
        .agg([pl.col(c).mean().alias(c) for c in numeric])
        .sort("label")
    )

    out("\n== FLAG RATES BY LABEL ==")
    out(
        features.group_by("label")
        .agg([pl.col(c).mean().alias(c) for c in flags])
        .sort("label")
    )

    out("\n== POSITIVE FEATURES BY COUNTRY / SOURCE ==")
    out(
        features.filter(pl.col("label") == 1)
        .group_by(["country", "source"])
        .agg(
            pl.len().alias("positives"),
            pl.col("name_token_set").mean().alias("mean_name_token_set"),
            pl.col("address_token_set").mean().alias("mean_address_token_set"),
            pl.col("number_jaccard").mean().alias("mean_number_jaccard"),
        )
        .sort(["country", "source"])
    )

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
