"""Blocking v2 evaluation on a deterministic held-out S1 sample.

Adds conservative supplemental blocks to the exact-name/address baseline:
- rare full name tokens
- rare 4-character name-token prefixes
- rare address-number tokens
- rare address-word tokens

Reports cumulative true-pair recall and candidate-set size. No submission files
are created.
"""
from pathlib import Path
import polars as pl

from norm import add_name_cols, norm_addr, token_key

P = Path("parquet")
OUT = Path("blocking_v2_report.txt")

# About 22k S1 entities, deterministic and independent of the prior normalizer split.
SAMPLE_MODULUS = 100
SAMPLE_REMAINDER = 7

# Conservative frequency caps: prevent generic tokens from exploding candidates.
MAX_NAME_TOKEN_DF = 200
MAX_NAME_PREFIX_DF = 100
MAX_ADDRESS_NUMBER_DF = 100
MAX_ADDRESS_WORD_DF = 200


def rd(name):
    return pl.read_parquet(P / f"{name}.parquet")


def enrich(df, id_col, name_col, addr_col, prefix):
    """Add conservative normalized name and full-address fields."""
    x = add_name_cols(df, name_col, prefix)
    return x.with_columns(
        norm_addr(pl.col(addr_col), pl.col("country")).alias(f"{prefix}addr_norm")
    )


def exact_keys(x, id_col, output_id, prefix):
    """Exact normalized name variants plus full order-insensitive address key."""
    return pl.concat([
        x.select(
            pl.col(id_col).alias(output_id),
            "country",
            (pl.lit("name:") + pl.col(f"{prefix}core")).alias("key"),
        ),
        x.select(
            pl.col(id_col).alias(output_id),
            "country",
            (pl.lit("name:") + pl.col(f"{prefix}sorted")).alias("key"),
        ),
        x.select(
            pl.col(id_col).alias(output_id),
            "country",
            (pl.lit("name:") + pl.col(f"{prefix}compact")).alias("key"),
        ),
        x.select(
            pl.col(id_col).alias(output_id),
            "country",
            (pl.lit("addr:") + token_key(pl.col(f"{prefix}addr_norm"))).alias("key"),
        ),
    ]).filter(pl.col("key").str.len_chars() >= 7).unique()


def token_keys(x, id_col, output_id, text_col, mode):
    """Explode bounded token keys for a supplemental blocking path."""
    z = (
        x.select(
            pl.col(id_col).alias(output_id),
            "country",
            pl.col(text_col).str.split(" ").alias("tokens"),
        )
        .explode("tokens")
        .rename({"tokens": "token"})
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
    )

    if mode == "name_token":
        z = z.filter(pl.col("token").str.len_chars() >= 5)
        z = z.with_columns((pl.lit("nt:") + pl.col("token")).alias("key"))

    elif mode == "name_prefix":
        z = z.filter(pl.col("token").str.len_chars() >= 6)
        z = z.with_columns(
            (pl.lit("np:") + pl.col("token").str.slice(0, 4)).alias("key")
        )

    elif mode == "address_number":
        z = z.filter(pl.col("token").str.contains(r"^\d{3,}$"))
        z = z.with_columns((pl.lit("an:") + pl.col("token")).alias("key"))

    elif mode == "address_word":
        z = z.filter(
            (pl.col("token").str.len_chars() >= 5)
            & ~pl.col("token").str.contains(r"^\d+$")
        )
        z = z.with_columns((pl.lit("aw:") + pl.col("token")).alias("key"))

    else:
        raise ValueError(mode)

    return z.select(output_id, "country", "key").unique()


def rare_token_pairs(s_keys, r_keys, max_df):
    """Join only tokens occurring in at most max_df source records."""
    frequency = (
        r_keys.group_by(["country", "key"])
        .agg(pl.col("rec_id").n_unique().alias("df"))
        .filter(pl.col("df") <= max_df)
        .select("country", "key")
    )
    return (
        s_keys.join(frequency, on=["country", "key"], how="inner")
        .join(r_keys, on=["country", "key"], how="inner")
        .select("s1_id", "rec_id")
        .unique()
    )


def main():
    s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")

    dev_s1 = s1.filter(
        (pl.col("entity_id").hash(seed=41) % SAMPLE_MODULUS) == SAMPLE_REMAINDER
    )
    dev_ids = dev_s1.select(pl.col("entity_id").alias("s1_id"))

    truth = (
        gt.filter(
            pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids") != "")
        )
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.strip_chars().alias("rec_id"),
        )
        .join(dev_ids, on="s1_id", how="inner")
        .with_columns(pl.col("rec_id").str.slice(0, 2).alias("src"))
    )

    lines = []

    def out(*parts):
        text = " ".join(str(x) for x in parts)
        print(text)
        lines.append(text)

    out("== BLOCKING v2: HELD-OUT EVALUATION ==")
    out(f"Held-out S1 sample: {dev_s1.height:,}")
    out(f"True S2/S3 links: {truth.height:,}")

    paths = {
        "01 exact normalized keys": [],
        "02 + rare full name tokens": [],
        "03 + rare name prefixes": [],
        "04 + rare address numbers": [],
        "05 + rare address words": [],
    }

    for source, records in [("S2", s2), ("S3", s3)]:
        for country in dev_s1["country"].unique().sort().to_list():
            left = (
                dev_s1.filter(pl.col("country") == country)
                .rename({
                    "entity_id": "s1_id",
                    "business_name": "s_name",
                    "business_address": "s_addr",
                })
            )
            right = (
                records.filter(pl.col("country") == country)
                .rename({
                    "entity_id": "rec_id",
                    "business_name": "r_name",
                    "business_address": "r_addr",
                })
            )

            left = enrich(left, "s1_id", "s_name", "s_addr", "s_")
            right = enrich(right, "rec_id", "r_name", "r_addr", "r_")

            exact = (
                exact_keys(left, "s1_id", "s1_id", "s_")
                .join(
                    exact_keys(right, "rec_id", "rec_id", "r_"),
                    on=["country", "key"],
                    how="inner",
                )
                .select("s1_id", "rec_id")
                .unique()
            )

            name_token = rare_token_pairs(
                token_keys(left, "s1_id", "s1_id", "s_core", "name_token"),
                token_keys(right, "rec_id", "rec_id", "r_core", "name_token"),
                MAX_NAME_TOKEN_DF,
            )

            name_prefix = rare_token_pairs(
                token_keys(left, "s1_id", "s1_id", "s_core", "name_prefix"),
                token_keys(right, "rec_id", "rec_id", "r_core", "name_prefix"),
                MAX_NAME_PREFIX_DF,
            )

            address_number = rare_token_pairs(
                token_keys(left, "s1_id", "s1_id", "s_addr_norm", "address_number"),
                token_keys(right, "rec_id", "rec_id", "r_addr_norm", "address_number"),
                MAX_ADDRESS_NUMBER_DF,
            )

            address_word = rare_token_pairs(
                token_keys(left, "s1_id", "s1_id", "s_addr_norm", "address_word"),
                token_keys(right, "rec_id", "rec_id", "r_addr_norm", "address_word"),
                MAX_ADDRESS_WORD_DF,
            )

            pieces = [
                ("01 exact normalized keys", exact),
                ("02 + rare full name tokens", name_token),
                ("03 + rare name prefixes", name_prefix),
                ("04 + rare address numbers", address_number),
                ("05 + rare address words", address_word),
            ]

            for label, pairs in pieces:
                tagged = pairs.with_columns(
                    pl.lit(source).alias("src"),
                    pl.lit(country).alias("country"),
                )
                paths[label].append(tagged)
                out(f"{label} | {source}/{country}: {pairs.height:,} raw candidates")

    cumulative = []
    out("\n== CUMULATIVE TRUE-PAIR RECALL ==")

    for label, frames in paths.items():
        cumulative.extend(frames)
        candidates = pl.concat(cumulative).select("s1_id", "rec_id").unique()
        recovered = candidates.with_columns(pl.lit(True).alias("recovered"))

        scored = (
            truth.join(recovered, on=["s1_id", "rec_id"], how="left")
            .with_columns(pl.col("recovered").fill_null(False))
        )

        overall = scored.select(
            pl.len().alias("true_pairs"),
            pl.col("recovered").sum().alias("recovered"),
            pl.col("recovered").mean().alias("recall"),
        )
        out(f"\n{label}")
        out(overall)

        by_slice = (
            scored.join(
                dev_s1.select(
                    pl.col("entity_id").alias("s1_id"),
                    "country",
                ),
                on="s1_id",
            )
            .group_by(["country", "src"])
            .agg(
                pl.len().alias("true_pairs"),
                pl.col("recovered").mean().alias("recall"),
            )
            .sort(["country", "src"])
        )
        out(by_slice)

        counts = (
            candidates.group_by("s1_id")
            .len()
            .rename({"len": "n_candidates"})
        )
        size = (
            dev_ids.join(counts, on="s1_id", how="left")
            .with_columns(pl.col("n_candidates").fill_null(0))
            .select(
                pl.col("n_candidates").mean().alias("mean"),
                pl.col("n_candidates").quantile(0.5).alias("p50"),
                pl.col("n_candidates").quantile(0.95).alias("p95"),
                pl.col("n_candidates").max().alias("max"),
            )
        )
        out("candidate-size per S1:", size)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
