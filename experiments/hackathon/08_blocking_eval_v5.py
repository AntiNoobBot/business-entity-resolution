"""Blocking v5: rerank v4 loose address-word fallback with RapidFuzz similarity."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl
from rapidfuzz import fuzz

# Reuse the already-tested helper functions from v4 without running its main().
spec = spec_from_file_location("blocking_v4", "07_blocking_eval_v4.py")
v4 = module_from_spec(spec)
spec.loader.exec_module(v4)

P = Path("parquet")
OUT = Path("blocking_v5_report.txt")

SAMPLE_MODULUS = 100
SAMPLE_REMAINDER = 7
FALLBACK_CAPS = [25, 50, 100]


def add_fuzzy_scores(df):
    """Fast pairwise string scores for only loose fallback candidates."""
    s_names = df["s_core"].fill_null("").to_list()
    r_names = df["r_core"].fill_null("").to_list()
    s_addrs = df["s_addr_norm"].fill_null("").to_list()
    r_addrs = df["r_addr_norm"].fill_null("").to_list()

    name_scores = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(s_names, r_names)
    ]
    addr_scores = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(s_addrs, r_addrs)
    ]

    return df.with_columns(
        pl.Series("name_similarity", name_scores, dtype=pl.Float32),
        pl.Series("address_similarity", addr_scores, dtype=pl.Float32),
    ).with_columns(
        (
            pl.col("name_similarity") * 0.40
            + pl.col("address_similarity") * 0.60
            + pl.col("has_rare_name") * 0.03
            + pl.col("has_number") * 0.02
        ).alias("rank_score")
    )


def main():
    s1, s2, s3, gt = (
        v4.rd("train_s1"),
        v4.rd("train_s2"),
        v4.rd("train_s3"),
        v4.rd("train_gt"),
    )

    dev_s1 = s1.filter(
        (pl.col("entity_id").hash(seed=41) % SAMPLE_MODULUS)
        == SAMPLE_REMAINDER
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

    out("== BLOCKING v5: FUZZY-RANKED LOOSE FALLBACK ==")
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

            left = v4.enrich(left, "s1_id", "s_name", "s_addr", "s_")
            right = v4.enrich(right, "rec_id", "r_name", "r_addr", "r_")

            exact = (
                v4.exact_keys(left, "s1_id", "s1_id", "s_")
                .join(
                    v4.exact_keys(right, "rec_id", "rec_id", "r_"),
                    on=["country", "key"],
                    how="inner",
                )
                .select("s1_id", "rec_id")
                .unique()
            )

            name_pairs = v4.joined_pairs(
                v4.token_keys(left, "s1_id", "s1_id", "s_core", "name"),
                v4.token_keys(right, "rec_id", "rec_id", "r_core", "name"),
                v4.MAX_NAME_TOKEN_DF,
            )

            number_pairs = v4.joined_pairs(
                v4.token_keys(left, "s1_id", "s1_id", "s_addr_norm", "number"),
                v4.token_keys(right, "rec_id", "rec_id", "r_addr_norm", "number"),
                v4.MAX_ADDRESS_NUMBER_DF,
            )

            address_pairs = v4.joined_pairs(
                v4.token_keys(
                    left, "s1_id", "s1_id", "s_addr_norm", "address_word"
                ),
                v4.token_keys(
                    right, "rec_id", "rec_id", "r_addr_norm", "address_word"
                ),
                v4.MAX_ADDRESS_WORD_DF,
            )

            tight = pl.concat([
                exact,
                name_pairs.select("s1_id", "rec_id"),
                number_pairs.select("s1_id", "rec_id"),
            ]).unique()

            address_score = (
                address_pairs.group_by(["s1_id", "rec_id"])
                .agg(
                    pl.col("key").n_unique().alias("shared_address_words"),
                    (1.0 / pl.col("df")).sum().alias("address_rarity"),
                )
            )

            name_evidence = (
                name_pairs.select("s1_id", "rec_id").unique()
                .with_columns(pl.lit(1).alias("has_rare_name"))
            )
            number_evidence = (
                number_pairs.select("s1_id", "rec_id").unique()
                .with_columns(pl.lit(1).alias("has_number"))
            )

            # Add text only to loose candidates, never to the complete source files.
            loose = (
                address_score
                .join(name_evidence, on=["s1_id", "rec_id"], how="left")
                .join(number_evidence, on=["s1_id", "rec_id"], how="left")
                .with_columns(
                    pl.col("has_rare_name").fill_null(0),
                    pl.col("has_number").fill_null(0),
                )
                .join(
                    left.select("s1_id", "s_core", "s_addr_norm"),
                    on="s1_id",
                )
                .join(
                    right.select("rec_id", "r_core", "r_addr_norm"),
                    on="rec_id",
                )
                .with_columns(
                    pl.lit(source).alias("src"),
                    pl.lit(country).alias("country"),
                )
            )

            tight_frames.append(tight)
            loose_frames.append(loose)

    tight = pl.concat(tight_frames).select("s1_id", "rec_id").unique()
    loose = pl.concat(loose_frames).join(
        tight, on=["s1_id", "rec_id"], how="anti"
    )

    out(f"Tight candidates: {tight.height:,}")
    out(f"Loose fallback candidates before ranking: {loose.height:,}")
    out("Computing fuzzy name and address scores for loose fallback candidates...")

    fallback = add_fuzzy_scores(loose)

    fallback = (
        fallback.sort(
            [
                "s1_id",
                "rank_score",
                "shared_address_words",
                "has_rare_name",
                "has_number",
                "address_rarity",
                "rec_id",
            ],
            descending=[False, True, True, True, True, True, False],
        )
        .with_columns(
            pl.col("s1_id").cum_count().over("s1_id").alias("rank")
        )
    )

    out("\n== VARIANT COMPARISON ==")

    variants = {"tight only": tight}
    for cap in FALLBACK_CAPS:
        top = fallback.filter(pl.col("rank") <= cap).select("s1_id", "rec_id")
        variants[f"tight + top {cap} fuzzy-ranked fallback"] = (
            pl.concat([tight, top]).unique()
        )

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

        counts = candidates.group_by("s1_id").len().rename(
            {"len": "n_candidates"}
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

        out(f"\n--- {label} ---")
        out(overall)
        out(by_slice)
        out("candidate-size per S1:", size)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
