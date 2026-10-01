"""Add error-driven features to the frozen v5 train/dev candidate tables."""

from pathlib import Path
import polars as pl
from rapidfuzz import fuzz

from norm import add_name_cols, norm_addr

P = Path("parquet")
BATCH = 250_000

SPLITS = [
    ("train", "train_candidates_v5.parquet", "train_features_v1.parquet",
     "train_features_v2.parquet"),
    ("dev", "dev_candidates_v5.parquet", "dev_features_v1.parquet",
     "dev_features_v2.parquet"),
]


def legal_code(c):
    """Coarse raw legal-form category; 0 means no recognised form."""
    x = c.str.to_lowercase()
    return (
        pl.when(x.str.contains(r"\b(p\s*l\s*l\s*c|l\s*l\s*c)\b"))
        .then(pl.lit(1))  # LLC / PLLC
        .when(x.str.contains(r"\bl\s*l\s*p\b"))
        .then(pl.lit(2))  # LLP
        .when(x.str.contains(r"\b(inc|incorporated)\b"))
        .then(pl.lit(3))
        .when(x.str.contains(r"\b(ltd|limited)\b"))
        .then(pl.lit(4))
        .when(x.str.contains(r"\b(corp|corporation)\b"))
        .then(pl.lit(5))
        .when(x.str.contains(r"\b(plc|pc)\b"))
        .then(pl.lit(6))
        .otherwise(pl.lit(0))
        .cast(pl.Int8)
    )


def raw_pairs(candidates):
    s1 = pl.read_parquet(P / "train_s1.parquet").select(
        pl.col("entity_id").alias("s1_id"),
        pl.col("business_name").alias("s_name"),
        pl.col("business_address").alias("s_addr"),
    )

    s2 = pl.read_parquet(P / "train_s2.parquet").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"),
    )

    s3 = pl.read_parquet(P / "train_s3.parquet").select(
        pl.col("entity_id").alias("rec_id"),
        pl.col("business_name").alias("r_name"),
        pl.col("business_address").alias("r_addr"),
    )

    common = candidates.join(s1, on="s1_id", how="left")

    p2 = (
        common.filter(pl.col("source") == "S2")
        .join(s2, on="rec_id", how="left")
    )
    p3 = (
        common.filter(pl.col("source") == "S3")
        .join(s3, on="rec_id", how="left")
    )

    return pl.concat([p2, p3], how="vertical_relaxed")


def add_polars_features(df):
    df = add_name_cols(df, "s_name", "s_")
    df = add_name_cols(df, "r_name", "r_")

    df = df.with_columns(
        norm_addr(pl.col("s_addr"), pl.col("country")).alias("s_addr_norm"),
        norm_addr(pl.col("r_addr"), pl.col("country")).alias("r_addr_norm"),
        legal_code(pl.col("s_name")).alias("s_legal_code"),
        legal_code(pl.col("r_name")).alias("r_legal_code"),
        pl.col("s_core").str.split(" ").alias("s_name_tokens"),
        pl.col("r_core").str.split(" ").alias("r_name_tokens"),
        norm_addr(pl.col("s_addr"), pl.col("country")).str.split(" ").alias("s_addr_tokens"),
        norm_addr(pl.col("r_addr"), pl.col("country")).str.split(" ").alias("r_addr_tokens"),
    )

    df = df.with_columns(
        pl.col("s_name_tokens")
        .list.set_intersection(pl.col("r_name_tokens"))
        .list.len()
        .alias("shared_name_tokens"),
        pl.col("s_name_tokens")
        .list.set_union(pl.col("r_name_tokens"))
        .list.len()
        .alias("name_token_union"),
        pl.col("s_addr_tokens")
        .list.set_intersection(pl.col("r_addr_tokens"))
        .list.len()
        .alias("shared_address_tokens"),
        pl.col("s_addr_tokens")
        .list.set_union(pl.col("r_addr_tokens"))
        .list.len()
        .alias("address_token_union"),
    )

    return df.with_columns(
        pl.when(pl.col("name_token_union") > 0)
        .then(pl.col("shared_name_tokens") / pl.col("name_token_union"))
        .otherwise(0.0)
        .alias("name_token_jaccard"),

        pl.when(pl.col("address_token_union") > 0)
        .then(pl.col("shared_address_tokens") / pl.col("address_token_union"))
        .otherwise(0.0)
        .alias("address_token_jaccard"),

        (
            (pl.col("s_legal_code") > 0)
            & (pl.col("r_legal_code") > 0)
            & (pl.col("s_legal_code") == pl.col("r_legal_code"))
        ).alias("legal_form_agrees"),

        (
            (pl.col("s_legal_code") > 0)
            & (pl.col("r_legal_code") > 0)
            & (pl.col("s_legal_code") != pl.col("r_legal_code"))
        ).alias("legal_form_conflicts"),

        ((pl.col("s_legal_code") > 0) ^ (pl.col("r_legal_code") > 0))
        .alias("legal_form_one_missing"),
    )


def add_fuzzy_features(df):
    partial_name = []
    partial_address = []
    name_contains = []
    address_contains = []

    for start in range(0, df.height, BATCH):
        chunk = df.slice(start, BATCH)

        s_names = chunk["s_core"].fill_null("").to_list()
        r_names = chunk["r_core"].fill_null("").to_list()
        s_addrs = chunk["s_addr_norm"].fill_null("").to_list()
        r_addrs = chunk["r_addr_norm"].fill_null("").to_list()

        for a, b in zip(s_names, r_names):
            partial_name.append(fuzz.partial_ratio(a, b) / 100.0)
            name_contains.append(
                int(min(len(a), len(b)) >= 4 and (a in b or b in a))
            )

        for a, b in zip(s_addrs, r_addrs):
            partial_address.append(fuzz.partial_ratio(a, b) / 100.0)
            address_contains.append(
                int(min(len(a), len(b)) >= 12 and (a in b or b in a))
            )

        print(f"fuzzy v2 features: {min(start + BATCH, df.height):,}/{df.height:,}")

    return df.with_columns(
        pl.Series("name_partial_ratio", partial_name, dtype=pl.Float32),
        pl.Series("address_partial_ratio", partial_address, dtype=pl.Float32),
        pl.Series("name_contains", name_contains, dtype=pl.Int8),
        pl.Series("address_contains", address_contains, dtype=pl.Int8),
    )


def build(split, candidates_path, v1_path, out_path):
    print(f"\n== {split.upper()} ==")
    candidates = pl.read_parquet(candidates_path)
    base = pl.read_parquet(v1_path)

    pairs = raw_pairs(candidates)
    print(f"pairs: {pairs.height:,}")
    print("creating token and legal-form features...")
    pairs = add_polars_features(pairs)

    print("creating partial-ratio and containment features...")
    pairs = add_fuzzy_features(pairs)

    extra_columns = [
        "s1_id", "rec_id",
        "shared_name_tokens", "name_token_union", "name_token_jaccard",
        "shared_address_tokens", "address_token_union",
        "address_token_jaccard",
        "s_legal_code", "r_legal_code",
        "legal_form_agrees", "legal_form_conflicts",
        "legal_form_one_missing",
        "name_partial_ratio", "address_partial_ratio",
        "name_contains", "address_contains",
    ]

    features = base.join(
        pairs.select(extra_columns),
        on=["s1_id", "rec_id"],
        how="left",
        validate="1:1",
    )

    features.write_parquet(out_path, compression="zstd")

    report = (
        features.group_by("label")
        .agg([
            pl.col(c).mean().alias(c)
            for c in extra_columns
            if c not in {"s1_id", "rec_id", "s_legal_code", "r_legal_code"}
        ])
        .sort("label")
    )

    print(f"saved {out_path}")
    print(report)

    return features.height


def main():
    lines = ["== FEATURE v2 BUILD REPORT =="]

    for split, candidates, v1, output in SPLITS:
        n_rows = build(split, candidates, v1, output)
        lines.append(f"{split}_rows={n_rows:,}")
        lines.append(f"{split}_output={output}")

    Path("feature_v2_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\nSaved feature_v2_report.txt")


if __name__ == "__main__":
    main()
