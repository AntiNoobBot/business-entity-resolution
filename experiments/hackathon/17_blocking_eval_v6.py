"""Blocking v6: add capped rare address-token-prefix candidates to frozen v5."""

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
OUT = Path("blocking_v6_report.txt")

SAMPLE_MODULUS = 100
SAMPLE_REMAINDER = 7
V5_FALLBACK_CAP = 25
MAX_ADDRESS_PREFIX_DF = 75
PREFIX_CAPS = [10, 25, 50]


def address_prefix_keys(df, id_col, output_id, address_col):
    """Rare 4-character prefixes of meaningful normalized address tokens."""
    keys = (
        df.select(
            pl.col(id_col).alias(output_id),
            "country",
            pl.col(address_col).str.split(" ").alias("tokens"),
        )
        .explode("tokens")
        .rename({"tokens": "token"})
        .filter(
            pl.col("token").is_not_null()
            & (pl.col("token").str.len_chars() >= 6)
            & ~pl.col("token").str.contains(r"^\d+$")
        )
        .with_columns(
            (pl.lit("addrprefix:") + pl.col("token").str.slice(0, 4))
            .alias("key")
        )
        .select(output_id, "country", "key")
        .unique()
    )
    return keys


def add_evidence(pair_scores, name_pairs, number_pairs, left, right):
    """Attach text and existing v5 evidence to a candidate-pair table."""
    name_evidence = (
        name_pairs.select("s1_id", "rec_id")
        .unique()
        .with_columns(pl.lit(1).alias("has_rare_name"))
    )
    number_evidence = (
        number_pairs.select("s1_id", "rec_id")
        .unique()
        .with_columns(pl.lit(1).alias("has_number"))
    )

    return (
        pair_scores
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


def rank_top(df, cap, prefix_count_col):
    """Use v5 fuzzy score, then prefix evidence, to take top extra candidates/S1."""
    scored = v5.add_fuzzy_scores(df)

    return (
        scored.sort(
            [
                "s1_id",
                "rank_score",
                prefix_count_col,
                "has_rare_name",
                "has_number",
                "address_rarity",
                "rec_id",
            ],
            descending=[False, True, True, True, True, True, False],
        )
        .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("rank"))
        .filter(pl.col("rank") <= cap)
        .select("s1_id", "rec_id")
    )


def f05(predictions, truth):
    values = []

    for s1_id, true_ids in truth.items():
        predicted = predictions.get(s1_id, set())

        if not true_ids and not predicted:
            values.append(1.0)
            continue
        if not predicted:
            values.append(0.0)
            continue

        tp = len(predicted & true_ids)
        if tp == 0:
            values.append(0.0)
            continue

        precision = tp / len(predicted)
        recall = tp / len(true_ids)
        values.append(
            1.25 * precision * recall / (0.25 * precision + recall)
        )

    return sum(values) / len(values)


def evaluate(name, candidates, truth, dev_ids, dev_s1):
    recovered = candidates.with_columns(pl.lit(True).alias("recovered"))

    scored_truth = (
        truth.join(recovered, on=["s1_id", "rec_id"], how="left")
        .with_columns(pl.col("recovered").fill_null(False))
    )

    predictions = {
        s1_id: set(rec_ids)
        for s1_id, rec_ids in (
            candidates.group_by("s1_id")
            .agg(pl.col("rec_id"))
            .iter_rows()
        )
    }

    recall = scored_truth["recovered"].mean()

    by_slice = (
        scored_truth
        .join(
            dev_s1.select(pl.col("entity_id").alias("s1_id"), "country"),
            on="s1_id",
        )
        .group_by(["country", "src"])
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

    return recall, by_slice, size


def main():
    s1 = v4.rd("train_s1")
    s2 = v4.rd("train_s2")
    s3 = v4.rd("train_s3")
    gt = v4.rd("train_gt")

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
    )

    lines = []

    def out(*items):
        text = " ".join(str(x) for x in items)
        print(text)
        lines.append(text)

    out("== BLOCKING v6: ADDRESS TOKEN PREFIX EXPERIMENT ==")
    out(f"held-out S1 sample: {dev_s1.height:,}")
    out(f"true links: {truth.height:,}")
    out(f"address prefix df cap: {MAX_ADDRESS_PREFIX_DF}")

    tight_frames = []
    loose_frames = []
    prefix_frames = []

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
                v4.token_keys(
                    left, "s1_id", "s1_id", "s_addr_norm", "number"
                ),
                v4.token_keys(
                    right, "rec_id", "rec_id", "r_addr_norm", "number"
                ),
                v4.MAX_ADDRESS_NUMBER_DF,
            )

            word_pairs = v4.joined_pairs(
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

            word_scores = (
                word_pairs.group_by(["s1_id", "rec_id"])
                .agg(
                    pl.col("key").n_unique().alias("shared_address_words"),
                    (1.0 / pl.col("df")).sum().alias("address_rarity"),
                )
            )

            loose = add_evidence(
                word_scores, name_pairs, number_pairs, left, right
            ).with_columns(
                pl.lit(source).alias("source"),
                pl.lit(country).alias("country"),
            )

            s_prefix = address_prefix_keys(
                left, "s1_id", "s1_id", "s_addr_norm"
            )
            r_prefix = address_prefix_keys(
                right, "rec_id", "rec_id", "r_addr_norm"
            )

            prefix_pairs = v4.joined_pairs(
                s_prefix,
                r_prefix,
                MAX_ADDRESS_PREFIX_DF,
            )

            prefix_scores = (
                prefix_pairs.group_by(["s1_id", "rec_id"])
                .agg(
                    pl.col("key").n_unique().alias("shared_address_prefixes"),
                    (1.0 / pl.col("df")).sum().alias("address_rarity"),
                )
            )

            prefix = add_evidence(
                prefix_scores, name_pairs, number_pairs, left, right
            ).with_columns(
                pl.lit(source).alias("source"),
                pl.lit(country).alias("country"),
            )

            tight_frames.append(tight)
            loose_frames.append(loose)
            prefix_frames.append(prefix)

    tight = pl.concat(tight_frames).select("s1_id", "rec_id").unique()

    loose = (
        pl.concat(loose_frames)
        .join(tight, on=["s1_id", "rec_id"], how="anti")
    )

    out(f"tight candidates: {tight.height:,}")
    out(f"v5 loose candidates before cap: {loose.height:,}")
    out("ranking v5 loose fallback...")

    v5_fallback = rank_top(
        loose, V5_FALLBACK_CAP, "shared_address_words"
    )

    v5_candidates = pl.concat([tight, v5_fallback]).unique()

    prefix = (
        pl.concat(prefix_frames)
        .join(v5_candidates, on=["s1_id", "rec_id"], how="anti")
    )

    out(f"new address-prefix candidates before cap: {prefix.height:,}")
    out("ranking new address-prefix candidates...")

    variants = {"v5 reference": v5_candidates}

    for cap in PREFIX_CAPS:
        extra = rank_top(prefix, cap, "shared_address_prefixes")
        variants[f"v5 + top {cap} address-prefix candidates"] = (
            pl.concat([v5_candidates, extra]).unique()
        )

    out("\n== VARIANT COMPARISON ==")

    for name, candidates in variants.items():
        recall, by_slice, size = evaluate(
            name, candidates, truth, dev_ids, dev_s1
        )
        out(f"\n--- {name} ---")
        out(f"candidate_recall={recall:.6f}")
        out(by_slice)
        out("candidate-size per S1:", size)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
