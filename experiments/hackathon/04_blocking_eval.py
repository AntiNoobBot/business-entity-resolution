"""Evaluate conservative exact blocking on a deterministic held-out S1 sample.

This is deliberately a baseline: it measures what the current normalization keys
recover, how many candidates they create, and which source/country slices remain
uncovered.  It does not train a matching model or write submission files.
"""
from pathlib import Path

import polars as pl

from norm import add_name_cols, norm_addr, token_key

P = Path("parquet")
OUT = Path("blocking_report.txt")
SAMPLE_MODULUS = 100  # about 22k S1 entities; safe first measurement on a laptop
SAMPLE_REMAINDER = 7


def rd(name):
    return pl.read_parquet(P / f"{name}.parquet")


def keys(df, entity_col, name_col, addr_col, prefix):
    """Return exact-name and order-insensitive full-address blocking keys."""
    x = add_name_cols(df, name_col, prefix)
    x = x.with_columns(
        token_key(norm_addr(pl.col(addr_col), pl.col("country"))).alias(f"{prefix}addr_key")
    )
    id_expr = pl.col(entity_col).alias("s1_id" if prefix == "s_" else "rec_id")
    names = pl.concat([
        x.select(id_expr, "country", pl.col(f"{prefix}core").alias("key")),
        x.select(id_expr, "country", pl.col(f"{prefix}sorted").alias("key")),
        x.select(id_expr, "country", pl.col(f"{prefix}compact").alias("key")),
    ]).filter(pl.col("key").str.len_chars() >= 3)
    addresses = x.select(id_expr, "country", pl.col(f"{prefix}addr_key").alias("key")).filter(
        pl.col("key").str.len_chars() >= 6
    )
    return names.unique(), addresses.unique()


def main():
    s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")
    dev_s1 = s1.filter((pl.col("entity_id").hash(seed=41) % SAMPLE_MODULUS) == SAMPLE_REMAINDER)
    dev_ids = dev_s1.select(pl.col("entity_id").alias("s1_id"))
    truth = (
        gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("matched_entity_ids").str.strip_chars().alias("rec_id"))
        .join(dev_ids, on="s1_id", how="inner")
        .with_columns(pl.col("rec_id").str.slice(0, 2).alias("src"))
    )

    lines = []
    def out(*parts):
        text = " ".join(str(x) for x in parts)
        print(text)
        lines.append(text)

    out("== EXACT-BLOCKING EVALUATION v1 ==")
    out(f"held-out S1 sample: {dev_s1.height:,} ({SAMPLE_REMAINDER}/{SAMPLE_MODULUS} by stable hash)")
    out(f"true S2/S3 links in sample: {truth.height:,}")

    all_candidates = []
    for source, recs in [("S2", s2), ("S3", s3)]:
        for country in dev_s1["country"].unique().sort().to_list():
            left = dev_s1.filter(pl.col("country") == country).rename(
                {"entity_id": "s1_id", "business_name": "s_name", "business_address": "s_addr"}
            )
            right = recs.filter(pl.col("country") == country).rename(
                {"entity_id": "rec_id", "business_name": "r_name", "business_address": "r_addr"}
            )
            s_name, s_addr = keys(left, "s1_id", "s_name", "s_addr", "s_")
            r_name, r_addr = keys(right, "rec_id", "r_name", "r_addr", "r_")
            name_pairs = s_name.join(r_name, on=["country", "key"], how="inner").select("s1_id", "rec_id")
            addr_pairs = s_addr.join(r_addr, on=["country", "key"], how="inner").select("s1_id", "rec_id")
            pairs = pl.concat([name_pairs, addr_pairs]).unique().with_columns(
                pl.lit(source).alias("src"), pl.lit(country).alias("country")
            )
            out(f"{source}/{country}: {pairs.height:,} unique candidates")
            all_candidates.append(pairs)

    candidates = pl.concat(all_candidates).unique()
    recovered_pairs = candidates.select("s1_id", "rec_id").with_columns(pl.lit(True).alias("recovered"))
    out("\n== TRUE-PAIR CANDIDATE RECALL ==")
    summary = (truth.join(recovered_pairs, on=["s1_id", "rec_id"], how="left")
               .with_columns(pl.col("recovered").fill_null(False))
               .group_by(["src"]).agg(pl.len().alias("true_pairs"), pl.col("recovered").sum().alias("recovered"),
                                      pl.col("recovered").mean().alias("recall")).sort("src"))
    out(summary)
    by_country = (truth.join(dev_s1.select(pl.col("entity_id").alias("s1_id"), "country"), on="s1_id")
                  .join(recovered_pairs, on=["s1_id", "rec_id"], how="left")
                  .with_columns(pl.col("recovered").fill_null(False))
                  .group_by(["country", "src"]).agg(pl.len().alias("true_pairs"), pl.col("recovered").mean().alias("recall"))
                  .sort(["country", "src"]))
    out(by_country)

    counts = candidates.group_by("s1_id").len().rename({"len": "n_candidates"})
    sizes = (dev_ids.join(counts, on="s1_id", how="left").with_columns(pl.col("n_candidates").fill_null(0))
             .select(pl.len().alias("S1"), pl.col("n_candidates").mean().alias("mean"),
                     pl.col("n_candidates").quantile(0.5).alias("p50"),
                     pl.col("n_candidates").quantile(0.95).alias("p95"),
                     pl.col("n_candidates").max().alias("max")))
    out("\n== CANDIDATE-SET SIZE PER S1 ==")
    out(sizes)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
