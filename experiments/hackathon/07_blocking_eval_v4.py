"""Blocking v4: rank and cap the loose one-address-word fallback."""

from pathlib import Path
import polars as pl

from norm import add_name_cols, norm_addr, token_key

P = Path("parquet")
OUT = Path("blocking_v4_report.txt")

SAMPLE_MODULUS = 100
SAMPLE_REMAINDER = 7

MAX_NAME_TOKEN_DF = 200
MAX_ADDRESS_NUMBER_DF = 100
MAX_ADDRESS_WORD_DF = 200
FALLBACK_CAPS = [25, 50, 100]


def rd(name):
    return pl.read_parquet(P / f"{name}.parquet")


def enrich(df, id_col, name_col, addr_col, prefix):
    x = add_name_cols(df, name_col, prefix)
    return x.with_columns(
        norm_addr(pl.col(addr_col), pl.col("country")).alias(f"{prefix}addr_norm")
    )


def exact_keys(x, id_col, output_id, prefix):
    return pl.concat([
        x.select(pl.col(id_col).alias(output_id), "country",
                 (pl.lit("name:") + pl.col(f"{prefix}core")).alias("key")),
        x.select(pl.col(id_col).alias(output_id), "country",
                 (pl.lit("name:") + pl.col(f"{prefix}sorted")).alias("key")),
        x.select(pl.col(id_col).alias(output_id), "country",
                 (pl.lit("name:") + pl.col(f"{prefix}compact")).alias("key")),
        x.select(pl.col(id_col).alias(output_id), "country",
                 (pl.lit("addr:") + token_key(pl.col(f"{prefix}addr_norm"))).alias("key")),
    ]).filter(pl.col("key").str.len_chars() >= 7).unique()


def token_keys(x, id_col, output_id, text_col, kind):
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

    if kind == "name":
        z = z.filter(pl.col("token").str.len_chars() >= 5)
        tag = "name:"
    elif kind == "number":
        z = z.filter(pl.col("token").str.contains(r"^\d{3,}$"))
        tag = "number:"
    elif kind == "address_word":
        z = z.filter(
            (pl.col("token").str.len_chars() >= 5)
            & ~pl.col("token").str.contains(r"^\d+$")
        )
        tag = "address:"
    else:
        raise ValueError(kind)

    return z.with_columns((pl.lit(tag) + pl.col("token")).alias("key")).select(
        output_id, "country", "key"
    ).unique()


def active_keys(r_keys, max_df):
    return (
        r_keys.group_by(["country", "key"])
        .agg(pl.col("rec_id").n_unique().alias("df"))
        .filter(pl.col("df") <= max_df)
        .select("country", "key", "df")
    )


def joined_pairs(s_keys, r_keys, max_df):
    active = active_keys(r_keys, max_df)
    return (
        s_keys.join(active, on=["country", "key"], how="inner")
        .join(r_keys, on=["country", "key"], how="inner")
        .select("s1_id", "rec_id", "key", "df")
        .unique()
    )


def main():
    s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")

    dev_s1 = s1.filter(
        (pl.col("entity_id").hash(seed=41) % SAMPLE_MODULUS) == SAMPLE_REMAINDER
    )
    dev_ids = dev_s1.select(pl.col("entity_id").alias("s1_id"))

    truth = (
        gt.filter(pl.col("matched_entity_ids").is_not_null()
                  & (pl.col("matched_entity_ids") != ""))
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").str.strip_chars().alias("rec_id"),
        )
        .join(dev_ids, on="s1_id", how="inner")
        .with_columns(pl.col("rec_id").str.slice(0, 2).alias("src"))
        .join(
            dev_s1.select(pl.col("entity_id").alias("s1_id"), "country"),
            on="s1_id",
        )
    )

    lines = []

    def out(*items):
        text = " ".join(str(x) for x in items)
        print(text)
        lines.append(text)

    out("== BLOCKING v4: CAPPED LOOSE ADDRESS-WORD FALLBACK ==")
    out(f"Held-out S1 sample: {dev_s1.height:,}")
    out(f"True S2/S3 links: {truth.height:,}")

    tight_frames = []
    loose_frames = []

    for source, records in [("S2", s2), ("S3", s3)]:
        for country in dev_s1["country"].unique().sort().to_list():
            left = dev_s1.filter(pl.col("country") == country).rename({
                "entity_id": "s1_id",
                "business_name": "s_name",
                "business_address": "s_addr",
            })
            right = records.filter(pl.col("country") == country).rename({
                "entity_id": "rec_id",
                "business_name": "r_name",
                "business_address": "r_addr",
            })

            left = enrich(left, "s1_id", "s_name", "s_addr", "s_")
            right = enrich(right, "rec_id", "r_name", "r_addr", "r_")

            exact = (
                exact_keys(left, "s1_id", "s1_id", "s_")
                .join(
                    exact_keys(right, "rec_id", "rec_id", "r_"),
                    on=["country", "key"], how="inner",
                )
                .select("s1_id", "rec_id")
                .unique()
            )

            name_pairs = joined_pairs(
                token_keys(left, "s1_id", "s1_id", "s_core", "name"),
                token_keys(right, "rec_id", "rec_id", "r_core", "name"),
                MAX_NAME_TOKEN_DF,
            )

            number_pairs = joined_pairs(
                token_keys(left, "s1_id", "s1_id", "s_addr_norm", "number"),
                token_keys(right, "rec_id", "rec_id", "r_addr_norm", "number"),
                MAX_ADDRESS_NUMBER_DF,
            )

            address_pairs = joined_pairs(
                token_keys(left, "s1_id", "s1_id", "s_addr_norm", "address_word"),
                token_keys(right, "rec_id", "rec_id", "r_addr_norm", "address_word"),
                MAX_ADDRESS_WORD_DF,
            )

            tight = pl.concat([
                exact,
                name_pairs.select("s1_id", "rec_id"),
                number_pairs.select("s1_id", "rec_id"),
            ]).unique()

            # Loose fallback ranking:
            # More shared rare address words is better; rarer words are better;
            # shared rare name tokens and address numbers break ties strongly.
            address_score = (
                address_pairs.group_by(["s1_id", "rec_id"])
                .agg(
                    pl.col("key").n_unique().alias("shared_address_words"),
                    (1.0 / pl.col("df")).sum().alias("address_rarity"),
                )
            )

            name_evidence = name_pairs.select("s1_id", "rec_id").unique().with_columns(
                pl.lit(1).alias("has_rare_name")
            )
            number_evidence = number_pairs.select("s1_id", "rec_id").unique().with_columns(
                pl.lit(1).alias("has_number")
            )

            loose = (
                address_score
                .join(name_evidence, on=["s1_id", "rec_id"], how="left")
                .join(number_evidence, on=["s1_id", "rec_id"], how="left")
                .with_columns(
                    pl.col("has_rare_name").fill_null(0),
                    pl.col("has_number").fill_null(0),
                    pl.lit(source).alias("src"),
                    pl.lit(country).alias("country"),
                )
            )

            tight_frames.append(
                tight.with_columns(
                    pl.lit(source).alias("src"),
                    pl.lit(country).alias("country"),
                )
            )
            loose_frames.append(loose)

    tight = pl.concat(tight_frames).select("s1_id", "rec_id").unique()
    loose = pl.concat(loose_frames)

    # Never use fallback for candidates already recovered by tight routes.
    fallback = loose.join(tight, on=["s1_id", "rec_id"], how="anti")

    # Rank only within each S1 by evidence, then use deterministic ID tie-break.
    fallback = (
        fallback.sort(
            ["s1_id", "shared_address_words", "has_rare_name", "has_number",
             "address_rarity", "rec_id"],
            descending=[False, True, True, True, True, False],
        )
        .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("rank"))
    )

    out(f"Tight candidates: {tight.height:,}")
    out(f"Loose fallback candidates before cap: {fallback.height:,}")

    variants = {"tight only": tight}

    for cap in FALLBACK_CAPS:
        capped = fallback.filter(pl.col("rank") <= cap).select("s1_id", "rec_id")
        variants[f"tight + top {cap} loose fallback"] = pl.concat([tight, capped]).unique()

    out("\n== VARIANT COMPARISON ==")

    for label, candidates in variants.items():
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

        by_slice = (
            scored.group_by(["country", "src"])
            .agg(
                pl.len().alias("true_pairs"),
                pl.col("recovered").mean().alias("recall"),
            )
            .sort(["country", "src"])
        )

        counts = candidates.group_by("s1_id").len().rename({"len": "n_candidates"})
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

        out(f"\n--- {label} ---")
        out(overall)
        out(by_slice)
        out("candidate-size per S1:", size)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
