"""Coordinate-descent threshold tuning by country and source on the dev split."""

from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

DEV = Path("dev_features_v1.parquet")
GT = Path("parquet/train_gt.parquet")
MODEL = Path("lgbm_baseline.txt")
REPORT = Path("threshold_tuning_report.txt")

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

    return float(np.mean(values))


def truth_for_dev(dev_s1_ids):
    gt = pl.read_parquet(GT)

    rows = (
        gt.filter(
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
    )

    truth = {s1_id: set() for s1_id in dev_s1_ids}

    for s1_id, rec_id in rows.iter_rows():
        truth[s1_id].add(rec_id)

    return truth


def score(scored, truth, thresholds):
    selected = scored.filter(
        (
            (pl.col("country") == "India")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= thresholds["India_S2"])
        )
        | (
            (pl.col("country") == "India")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= thresholds["India_S3"])
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S2")
            & (pl.col("probability") >= thresholds["US_S2"])
        )
        | (
            (pl.col("country") == "US")
            & (pl.col("source") == "S3")
            & (pl.col("probability") >= thresholds["US_S3"])
        )
    )

    grouped = selected.group_by("s1_id").agg(pl.col("rec_id"))

    predictions = {
        s1_id: set(rec_ids)
        for s1_id, rec_ids in grouped.iter_rows()
    }

    return f05(predictions, truth)


def main():
    dev = encode(pl.read_parquet(DEV))
    x_dev = dev.select(FEATURES + ["country_code", "source_code"]).to_numpy()

    booster = lgb.Booster(model_file=str(MODEL))
    probabilities = booster.predict(x_dev)

    scored = dev.select("s1_id", "rec_id", "country", "source").with_columns(
        pl.Series("probability", probabilities)
    )

    dev_s1_ids = (
        pl.read_parquet("parquet/train_s1.parquet")
        .filter((pl.col("entity_id").hash(seed=41) % 100) == 7)
        ["entity_id"]
        .to_list()
    )

    truth = truth_for_dev(dev_s1_ids)

    keys = ["India_S2", "India_S3", "US_S2", "US_S3"]
    thresholds = {key: 0.600 for key in keys}

    baseline = score(scored, truth, thresholds)

    # Conservative search range around the known global optimum.
    grid = [round(float(x), 3) for x in np.arange(0.450, 0.751, 0.025)]

    lines = []
    lines.append("== COUNTRY / SOURCE THRESHOLD TUNING ==")
    lines.append(f"global_threshold=0.600 macro_f0.5={baseline:.6f}")
    lines.append("")

    for pass_number in range(1, 4):
        improved = False
        lines.append(f"PASS {pass_number}")

        for key in keys:
            best_value = thresholds[key]
            best_score = score(scored, truth, thresholds)

            for value in grid:
                trial = dict(thresholds)
                trial[key] = value
                trial_score = score(scored, truth, trial)

                if trial_score > best_score:
                    best_score = trial_score
                    best_value = value

            if best_value != thresholds[key]:
                improved = True

            thresholds[key] = best_value
            lines.append(
                f"{key}: threshold={best_value:.3f} "
                f"macro_f0.5={best_score:.6f}"
            )

        lines.append("")

        if not improved:
            break

    final_score = score(scored, truth, thresholds)

    lines.append("== FINAL ==")
    for key in keys:
        lines.append(f"{key}_threshold={thresholds[key]:.3f}")
    lines.append(f"tuned_macro_f0.5={final_score:.6f}")
    lines.append(f"improvement={final_score - baseline:+.6f}")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {REPORT}")


if __name__ == "__main__":
    main()
