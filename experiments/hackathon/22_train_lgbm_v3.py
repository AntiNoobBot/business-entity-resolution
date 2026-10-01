"""Train and evaluate a LightGBM baseline for entity matching."""

from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb

TRAIN = Path("train_features_v2.parquet")
DEV = Path("dev_features_v2.parquet")
GT = Path("parquet/train_gt.parquet")
MODEL_OUT = Path("lgbm_v3.txt")
REPORT_OUT = Path("lgbm_v3_report.txt")

FEATURES = [
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
    "shared_name_tokens",
    "name_token_union",
    "name_token_jaccard",
    "shared_address_tokens",
    "address_token_union",
    "address_token_jaccard",
    "s_legal_code",
    "r_legal_code",
    "legal_form_agrees",
    "legal_form_conflicts",
    "legal_form_one_missing",
    "name_partial_ratio",
    "address_partial_ratio",
    "name_contains",
    "address_contains",
]


def encode(df):
    return df.with_columns(
        pl.when(pl.col("country") == "India").then(0).otherwise(1).cast(pl.Int8).alias("country_code"),
        pl.when(pl.col("source") == "S2").then(0).otherwise(1).cast(pl.Int8).alias("source_code"),
    )


def f05(pred, truth):
    scores = []

    for s1_id, true_ids in truth.items():
        predicted = pred.get(s1_id, set())

        if not true_ids and not predicted:
            scores.append(1.0)
            continue

        if not predicted:
            scores.append(0.0)
            continue

        tp = len(predicted & true_ids)
        if tp == 0:
            scores.append(0.0)
            continue

        precision = tp / len(predicted)
        recall = tp / len(true_ids)
        scores.append(
            1.25 * precision * recall / (0.25 * precision + recall)
        )

    return float(np.mean(scores))


def build_truth(gt, dev_s1_ids):
    rows = (
        gt.filter(
            pl.col("source1_entity_id").is_in(dev_s1_ids)
            & pl.col("matched_entity_ids").is_not_null()
            & (pl.col("matched_entity_ids") != "")
        )
        .with_columns(pl.col("matched_entity_ids").str.split(","))
        .explode("matched_entity_ids")
        .select(
            pl.col("source1_entity_id"),
            pl.col("matched_entity_ids").str.strip_chars().alias("rec_id"),
        )
    )

    truth = {x: set() for x in dev_s1_ids}

    for row in rows.iter_rows():
        truth[row[0]].add(row[1])

    return truth


def main():
    train = encode(pl.read_parquet(TRAIN))
    dev = encode(pl.read_parquet(DEV))

    x_train = train.select(FEATURES + ["country_code", "source_code"]).to_numpy()
    y_train = train["label"].to_numpy()

    x_dev = dev.select(FEATURES + ["country_code", "source_code"]).to_numpy()
    y_dev = dev["label"].to_numpy()

    feature_names = FEATURES + ["country_code", "source_code"]

    print(f"training rows: {len(y_train):,}")
    print(f"training positives: {int(y_train.sum()):,}")
    print(f"development rows: {len(y_dev):,}")
    print(f"development positives: {int(y_dev.sum()):,}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1400,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        n_jobs=8,
        verbosity=-1,
    )

    model.fit(
        x_train,
        y_train,
        feature_name=feature_names,
        eval_set=[(x_dev, y_dev)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(60, verbose=True),
            lgb.log_evaluation(25),
        ],
    )

    booster = model.booster_
    booster.save_model(str(MODEL_OUT))

    probabilities = model.predict_proba(x_dev)[:, 1]

    dev_ids = dev["s1_id"].to_list()
    dev_rec_ids = dev["rec_id"].to_list()

    dev_s1_ids = (
        pl.read_parquet("parquet/train_s1.parquet")
        .filter(
            (pl.col("entity_id").hash(seed=41) % 100) == 7
        )["entity_id"]
        .to_list()
    )

    truth = build_truth(pl.read_parquet(GT), dev_s1_ids)

    scored = pl.DataFrame({
        "s1_id": dev_ids,
        "rec_id": dev_rec_ids,
        "probability": probabilities,
    })

    lines = []
    lines.append("== LIGHTGBM v3: SLOWER LEARNING RATE ==")
    lines.append(f"best_iteration: {model.best_iteration_}")
    lines.append(f"training_rows: {len(y_train):,}")
    lines.append(f"development_rows: {len(y_dev):,}")
    lines.append(f"development_s1_entities: {len(truth):,}")

    best_score = -1.0
    best_threshold = None

    for threshold in np.arange(0.05, 0.951, 0.025):
        selected = scored.filter(pl.col("probability") >= float(threshold))

        grouped = (
            selected.group_by("s1_id")
            .agg(pl.col("rec_id"))
        )

        predictions = {
            row[0]: set(row[1])
            for row in grouped.iter_rows()
        }

        score = f05(predictions, truth)
        lines.append(f"threshold={threshold:.3f} macro_f0.5={score:.6f}")

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    lines.append(f"best_threshold: {best_threshold:.3f}")
    lines.append(f"best_macro_f0.5: {best_score:.6f}")
    lines.append(f"model_file: {MODEL_OUT}")

    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved {MODEL_OUT} and {REPORT_OUT}")


if __name__ == "__main__":
    main()
