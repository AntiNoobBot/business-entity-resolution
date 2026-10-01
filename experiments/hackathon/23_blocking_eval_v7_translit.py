"""Test transliterated rare-name-token blocking for India on the v5 dev split."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import polars as pl
from text_unidecode import unidecode

spec4 = spec_from_file_location("blocking_v4", "07_blocking_eval_v4.py")
v4 = module_from_spec(spec4)
spec4.loader.exec_module(v4)

spec5 = spec_from_file_location("blocking_v5", "08_blocking_eval_v5.py")
v5 = module_from_spec(spec5)
spec5.loader.exec_module(v5)

P = Path("parquet")
OUT = Path("blocking_v7_translit_report.txt")
MAX_TOKEN_DF = 200
EXTRA_CAPS = [10, 25, 50]

DROP = {
    "private", "pvt", "limited", "ltd", "inc", "incorporated",
    "llc", "llp", "corp", "corporation", "company", "services",
    "service", "centre", "center", "india",
}


def roman_name(value):
    """Deterministic script-to-Latin normalization for candidate retrieval only."""
    value = unidecode(value or "").lower()
    tokens = re.findall(r"[a-z0-9]+", value)
    return " ".join(
        token for token in tokens
        if len(token) >= 3 and token not in DROP
    )


def roman_tokens(df, id_col, name_col, output_id):
    names = [roman_name(value) for value in df[name_col].to_list()]

    return (
        pl.DataFrame({
            output_id: df[id_col].to_list(),
            "roman_name": names,
        })
        .with_columns(pl.col("roman_name").str.split(" ").alias("tokens"))
        .explode("tokens")
        .rename({"tokens": "token"})
        .filter(
            pl.col("token").is_not_null()
            & (pl.col("token").str.len_chars() >= 5)
        )
        .select(output_id, "token")
        .unique()
    )


def main():
    s1 = pl.read_parquet(P / "train_s1.parquet")
    s2 = pl.read_parquet(P / "train_s2.parquet")
    s3 = pl.read_parquet(P / "train_s3.parquet")
    gt = pl.read_parquet(P / "train_gt.parquet")

    dev_s1 = s1.filter(
        (pl.col("entity_id").hash(seed=41) % 100) == 7
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
        .with_columns(pl.col("rec_id").str.slice(0, 2).alias("source"))
    )

    v5_candidates = (
        pl.read_parquet("dev_candidates_v5.parquet")
        .select("s1_id", "rec_id")
        .unique()
    )

    lines = []

    def out(*items):
        text = " ".join(str(x) for x in items)
        print(text)
        lines.append(text)

    out("== BLOCKING v7: INDIA TRANSLITERATED NAME TOKENS ==")
    out(f"v5 reference candidates: {v5_candidates.height:,}")

    left = (
        dev_s1.filter(pl.col("country") == "India")
        .rename({
            "entity_id": "s1_id",
            "business_name": "s_name",
            "business_address": "s_addr",
        })
    )

    extra_frames = []

    for source, records in [("S2", s2), ("S3", s3)]:
        out(f"transliterating India/{source} names...")

        right = (
            records.filter(pl.col("country") == "India")
            .rename({
                "entity_id": "rec_id",
                "business_name": "r_name",
                "business_address": "r_addr",
            })
        )

        s_tokens = roman_tokens(left, "s1_id", "s_name", "s1_id")
        r_tokens = roman_tokens(right, "rec_id", "r_name", "rec_id")

        active = (
            r_tokens.group_by("token")
            .agg(pl.col("rec_id").n_unique().alias("df"))
            .filter(pl.col("df") <= MAX_TOKEN_DF)
        )

        pairs = (
            s_tokens.join(active, on="token", how="inner")
            .join(r_tokens, on="token", how="inner")
            .group_by(["s1_id", "rec_id"])
            .agg(
                pl.col("token").n_unique().alias("shared_translit_tokens"),
                (1.0 / pl.col("df")).sum().alias("translit_rarity"),
            )
        )

        out(f"India/{source} raw transliterated candidates: {pairs.height:,}")

        # Attach existing normalized text, then use v5's proven fuzzy ranker.
        left_rich = v4.enrich(left, "s1_id", "s_name", "s_addr", "s_")
        right_rich = v4.enrich(right, "rec_id", "r_name", "r_addr", "r_")

        pairs = (
            pairs
            .join(
                left_rich.select("s1_id", "s_core", "s_addr_norm"),
                on="s1_id",
            )
            .join(
                right_rich.select("rec_id", "r_core", "r_addr_norm"),
                on="rec_id",
            )
            .with_columns(
                pl.lit(0).alias("has_rare_name"),
                pl.lit(0).alias("has_number"),
                pl.col("translit_rarity").alias("address_rarity"),
            )
        )

        pairs = v5.add_fuzzy_scores(pairs).with_columns(
            pl.lit(source).alias("source")
        )

        extra_frames.append(pairs)

    extra = (
        pl.concat(extra_frames)
        .join(v5_candidates, on=["s1_id", "rec_id"], how="anti")
    )

    out(f"new transliterated candidates after v5 overlap removal: {extra.height:,}")

    ranked = (
        extra.sort(
            [
                "s1_id", "rank_score", "shared_translit_tokens",
                "translit_rarity", "rec_id",
            ],
            descending=[False, True, True, True, False],
        )
        .with_columns(pl.col("s1_id").cum_count().over("s1_id").alias("rank"))
    )

    variants = {"v5 reference": v5_candidates}

    for cap in EXTRA_CAPS:
        extra_top = (
            ranked.filter(pl.col("rank") <= cap)
            .select("s1_id", "rec_id")
        )
        variants[f"v5 + top {cap} transliterated-name candidates"] = (
            pl.concat([v5_candidates, extra_top]).unique()
        )

    out("\n== VARIANT COMPARISON ==")

    for name, candidates in variants.items():
        recovered = candidates.with_columns(pl.lit(True).alias("recovered"))

        scored = (
            truth.join(recovered, on=["s1_id", "rec_id"], how="left")
            .with_columns(pl.col("recovered").fill_null(False))
        )

        counts = candidates.group_by("s1_id").len().rename(
            {"len": "n_candidates"}
        )

        size = (
            dev_ids.join(counts, on="s1_id", how="left")
            .with_columns(pl.col("n_candidates").fill_null(0))
            .select(
                pl.col("n_candidates").mean().alias("mean"),
                pl.col("n_candidates").quantile(0.95).alias("p95"),
                pl.col("n_candidates").max().alias("max"),
            )
        )

        by_slice = (
            scored.join(
                dev_s1.select(pl.col("entity_id").alias("s1_id"), "country"),
                on="s1_id",
            )
            .group_by(["country", "source"])
            .agg(
                pl.len().alias("true_pairs"),
                pl.col("recovered").mean().alias("recall"),
            )
            .sort(["country", "source"])
        )

        out(f"\n--- {name} ---")
        out(f"candidate_recall={scored['recovered'].mean():.6f}")
        out(by_slice)
        out("candidate-size per S1:", size)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
