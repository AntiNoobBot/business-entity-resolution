"""Build labelled held-out candidates with the frozen v5 blocking recipe."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import polars as pl

spec4 = spec_from_file_location("blocking_v4", "07_blocking_eval_v4.py")
v4 = module_from_spec(spec4)
spec4.loader.exec_module(v4)

spec5 = spec_from_file_location("blocking_v5", "08_blocking_eval_v5.py")
v5 = module_from_spec(spec5)
spec5.loader.exec_module(v5)

P = Path("parquet")
OUT = Path("train_candidates_v5.parquet")
REPORT = Path("train_candidate_report.txt")

# Same deterministic held-out S1 split used in blocking experiments.
SAMPLE_MODULUS = 100
TRAIN_REMAINDERS = [0, 1]
FALLBACK_CAP = 25


def main():
    s1 = v4.rd("train_s1")
    s2 = v4.rd("train_s2")
    s3 = v4.rd("train_s3")
    gt = v4.rd("train_gt")

    dev_s1 = s1.filter(
        (pl.col("entity_id").hash(seed=41) % SAMPLE_MODULUS)
        .is_in(TRAIN_REMAINDERS)
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
        .unique()
    )

    lines = []

    def out(*items):
        text = " ".join(str(x) for x in items)
        print(text)
        lines.append(text)

    out("== BUILD LABELLED TRAIN CANDIDATES v5 ==")
    out(f"training-sample S1 records: {dev_s1.height:,}")
    out(f"training-sample true links: {truth.height:,}")

    tight_frames = []
    loose_frames = []

    for source, records in [("S2", s2), ("S3", s3)]:
        for country in dev_s1["country"].unique().sort().to_list():
            out(f"processing {source}/{country}...")

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
            )

            tight_frames.append(tight)
            loose_frames.append(loose)

    tight = pl.concat(tight_frames).select("s1_id", "rec_id").unique()

    loose = (
        pl.concat(loose_frames)
        .join(tight, on=["s1_id", "rec_id"], how="anti")
    )

    out(f"tight candidates before fallback: {tight.height:,}")
    out(f"loose fallback candidates before ranking: {loose.height:,}")
    out("computing fuzzy fallback scores...")

    fallback = v5.add_fuzzy_scores(loose)

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
        .filter(pl.col("rank") <= FALLBACK_CAP)
        .select("s1_id", "rec_id")
    )

    candidates = (
        pl.concat([tight, fallback])
        .unique()
        .with_columns(
            pl.when(
                pl.struct(["s1_id", "rec_id"]).is_in(
                    truth.select(pl.struct(["s1_id", "rec_id"])).to_series()
                )
            )
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("label"),
            pl.col("rec_id").str.slice(0, 2).alias("source"),
        )
        .join(
            dev_s1.select(
                pl.col("entity_id").alias("s1_id"),
                "country",
            ),
            on="s1_id",
        )
        .select("s1_id", "rec_id", "country", "source", "label")
    )

    candidates.write_parquet(OUT, compression="zstd")

    positives = candidates.filter(pl.col("label") == 1).height
    candidate_recall = positives / truth.height

    counts = (
        candidates.group_by("s1_id")
        .len()
        .rename({"len": "n_candidates"})
    )

    size = (
        dev_ids.join(counts, on="s1_id", how="left")
        .with_columns(pl.col("n_candidates").fill_null(0))
        .select(
            pl.len().alias("s1_records"),
            pl.col("n_candidates").mean().alias("mean_candidates"),
            pl.col("n_candidates").quantile(0.5).alias("p50"),
            pl.col("n_candidates").quantile(0.95).alias("p95"),
            pl.col("n_candidates").max().alias("max"),
        )
    )

    out("\n== RESULT ==")
    out(f"saved candidates: {candidates.height:,}")
    out(f"positive candidates: {positives:,}")
    out(f"candidate recall: {candidate_recall:.4%}")
    out(f"positive rate: {positives / candidates.height:.4%}")
    out(size)

    by_slice = (
        candidates.group_by(["country", "source"])
        .agg(
            pl.len().alias("candidates"),
            pl.col("label").sum().alias("positives"),
            pl.col("label").mean().alias("positive_rate"),
        )
        .sort(["country", "source"])
    )

    out("\n== CANDIDATES BY COUNTRY / SOURCE ==")
    out(by_slice)

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT} and {REPORT}")


if __name__ == "__main__":
    main()
