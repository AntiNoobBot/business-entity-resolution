"""Deadline trial: score completed retrieval files, then assemble on Mac.

This is NOT the complete V11 recipe unless all V11 candidate channels are supplied.
It reuses existing feature functions/model without training or changing them.
"""
import argparse
import contextlib
import glob
import importlib.util
import io
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import lightgbm as lgb
import psutil
from common import atomic_json, digest, read_json, sha256

THRESHOLDS = {("India", "S2"): .6, ("India", "S3"): .65,
              ("US", "S2"): .675, ("US", "S3"): .65}


def module(project, name):
    spec = importlib.util.spec_from_file_location(name, project/name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feature_frame(df, v1, v2, training):
    with contextlib.redirect_stdout(io.StringIO()):
        df = v1.add_fuzzy_features(v1.add_pair_features(df))
        df = v2.add_fuzzy_features(v2.add_polars_features(df))
    return training.encode(df)


def score(args):
    import sys
    sys.path.insert(0, str(args.project))
    from fast_features import prepare_entity, feature_frame
    v1 = module(args.project, "10_build_dev_features.py")
    v2 = module(args.project, "18_build_features_v2.py")
    train = module(args.project, "19_train_lgbm_v2.py")
    choice = read_json(args.config) if args.config else None
    model_paths=sorted({p for cfg in choice['countries'].values() for p in cfg['models']}) if choice else [str(args.project/'lgbm_v2.txt')]
    models={p:lgb.Booster(model_file=p) for p in model_paths}
    model=next(iter(models.values()))
    features = train.FEATURES + ["country_code", "source_code"]
    if model.num_feature() != len(features) or model.feature_name() != features:
        raise RuntimeError("Model feature order differs from training script. Do not disable this check.")
    for model in models.values():
        if model.feature_name()!=features:raise RuntimeError('Feature order mismatch in model configuration')
    paths = sorted({Path(x).resolve() for pattern in args.inputs for x in glob.glob(pattern, recursive=True)
                    if x.endswith(".parquet") and ".partial." not in x})
    if not paths:
        raise SystemExit("No completed candidate files matched --inputs.")
    args.out.mkdir(parents=True, exist_ok=True)
    fingerprint = digest({name: sha256(args.project/name) for name in
                          ("norm.py", "10_build_dev_features.py", "18_build_features_v2.py",
                           "19_train_lgbm_v2.py", "lgbm_v2.txt")})
    if choice:fingerprint=digest({'base':fingerprint,'choice':choice,'models':{p:sha256(p) for p in model_paths}})
    table_cache, cached_slice = None, None
    for path in paths:
        file_hash = sha256(path)
        directory = args.out / file_hash[:20]
        directory.mkdir(exist_ok=True)
        metadata = {"input": str(path), "input_sha256": file_hash,
                    "feature_model_sha256": fingerprint, "batch_size": args.batch_size,
                    "thresholds": {f"{c}/{s}": t for (c,s),t in THRESHOLDS.items()},
                    "unseen_country_threshold": .65,
                    "script_sha256": sha256(Path(__file__))}
        metadata["fast_features_sha256"] = sha256(Path(__file__).parent/"fast_features.py")
        meta_path = directory/"input.json"
        if meta_path.exists():
            previous = read_json(meta_path)
            if {k:v for k,v in previous.items() if k != "input"} != {k:v for k,v in metadata.items() if k != "input"}:
                raise RuntimeError("Scoring configuration changed; use a different --out.")
            # An identical copied candidate file is the same input, not new work.
            metadata = previous
        atomic_json(meta_path, metadata)
        if (directory/"complete.json").exists():
            done = read_json(directory/"complete.json")
            for name, checksum in done["files"].items():
                if sha256(directory/name) != checksum:
                    raise RuntimeError("Changed scoring shard")
            print(f"Already scored: {path.name}", flush=True)
            continue
        outputs, input_rows = {}, 0
        start_time = time.perf_counter()
        for i, arrow in enumerate(pq.ParquetFile(path).iter_batches(batch_size=args.batch_size,
                                                               columns=["s1_id", "rec_id", "country", "source"])):
            candidates = pl.from_arrow(arrow)
            input_rows += candidates.height
            output = directory / f"selected_{i:06d}.parquet"
            receipt = output.with_suffix(".json")
            if output.exists() and receipt.exists() and read_json(receipt).get("sha256") == sha256(output):
                outputs[output.name] = sha256(output)
                continue
            parts = []
            for pair in candidates.select("country", "source").unique().iter_rows():
                country, source = pair
                if source not in {"S2", "S3"}:
                    raise RuntimeError(f"Invalid source {source}")
                if cached_slice != pair:
                    table_cache = None
                    import gc
                    gc.collect()
                    if psutil.virtual_memory().available < 3 * 2**30:
                        raise RuntimeError("Less than 3 GiB RAM available. Resume after the other Mac job finishes.")
                    left = (pl.scan_parquet(args.project/"parquet/test_s1.parquet")
                            .filter(pl.col("country") == country)
                            .select(pl.col("entity_id").alias("s1_id"),
                                    pl.col("business_name").alias("s_name"),
                                    pl.col("business_address").alias("s_addr")).collect())
                    print(f"Normalizing query entities once: {country}/{source}",flush=True)
                    left = prepare_entity(left,"s_",country,v2)
                    right = (pl.scan_parquet(args.project/f"parquet/test_{source.lower()}.parquet")
                             .filter(pl.col("country") == country)
                             .select(pl.col("entity_id").alias("rec_id"),
                                     pl.col("business_name").alias("r_name"),
                                     pl.col("business_address").alias("r_addr")).collect())
                    print(f"Normalizing indexed entities once: {country}/{source}",flush=True)
                    right = prepare_entity(right,"r_",country,v2)
                    table_cache, cached_slice = (left,right), pair
                subset = candidates.filter((pl.col("country") == country) & (pl.col("source") == source)).unique()
                joined = subset.join(table_cache[0], on="s1_id", how="inner", validate="m:1").join(
                    table_cache[1], on="rec_id", how="inner", validate="m:1")
                if joined.height != subset.height:
                    raise RuntimeError("Candidate has unknown test ID or wrong country/source")
                frame = feature_frame(joined, v1, v2, train)
                x = frame.select(features).to_numpy().astype(np.float32)
                cfg=choice['countries'].get(country,choice['countries']['France']) if choice else None
                active=cfg['models'] if cfg else model_paths
                prob = np.mean([models[p].predict(x,num_threads=args.threads) for p in active],axis=0)
                if not np.isfinite(prob).all():
                    raise RuntimeError("Non-finite predictions")
                scored = joined.select("s1_id", "rec_id", "country", "source").with_columns(
                    pl.Series("probability", prob))
                threshold=cfg['threshold'] if cfg else THRESHOLDS.get(pair,.65)
                parts.append(scored.filter(pl.col("probability") >= threshold))
            selected = pl.concat(parts)
            temp = output.with_suffix(".tmp")
            selected.write_parquet(temp, compression="zstd")
            os.replace(temp, output)
            checksum = sha256(output)
            atomic_json(receipt, {"sha256": checksum})
            outputs[output.name] = checksum
            print(f"{path.name}: scored {input_rows:,} rows; elapsed {(time.perf_counter()-start_time)/60:.1f} min", flush=True)
            if args.pilot_only:
                print("Pilot complete. Rerun without --pilot-only to resume.")
                return
        atomic_json(directory/"complete.json", {"files": outputs, "input_rows": input_rows})


def assemble(args):
    # Read only fully scored inputs. Never treat partial scoring as complete.
    manifests = sorted(args.scored.glob("*/complete.json"))
    if not manifests:
        raise SystemExit("No fully scored input files.")
    files, inputs, fingerprints = [], [], set()
    for path in manifests:
        meta = read_json(path.parent/"input.json")
        fingerprints.add(meta["feature_model_sha256"])
        inp = Path(meta["input"])
        if sha256(inp) != meta["input_sha256"]:
            raise RuntimeError("Candidate input changed after scoring")
        inputs.append(str(inp))
        for name, checksum in read_json(path)["files"].items():
            shard = path.parent/name
            if sha256(shard) != checksum:
                raise RuntimeError("Selected-pair shard changed")
            files.append(str(shard))
    if len(fingerprints) != 1:
        raise RuntimeError("Cannot mix different feature/model versions in one submission.")
    required = {f"address_char_{c}_{s}.parquet" for c in ("France","India","US") for s in ("S2","S3")}
    missing = required - {Path(x).name for x in inputs}
    if missing and not args.allow_incomplete_address:
        raise SystemExit(f"Not all address slices fully scored: {sorted(missing)}")
    if args.out.exists():
        raise SystemExit("Output folder exists. Choose a NEW --out to preserve prior submissions.")
    args.out.mkdir(parents=True)
    s1 = pl.read_parquet(args.project/"parquet/test_s1.parquet", columns=["entity_id"])
    if s1["entity_id"].n_unique() != s1.height:
        raise RuntimeError("Duplicate test S1 IDs")
    # Max score is deterministic for duplicate pairs; one owner per S2/S3 record,
    # not one match per S1. Keep all above-threshold records owned by each S1.
    scored_scan=pl.scan_parquet(files)
    selection=read_json(args.selection) if args.selection else None
    if selection:
        threshold=pl.lit(float(selection['countries']['France']['threshold']))
        for country,cfg in selection['countries'].items():
            threshold=pl.when(pl.col('country')==country).then(pl.lit(float(cfg['threshold']))).otherwise(threshold)
        scored_scan=scored_scan.filter(pl.col('probability')>=threshold)
    selected = (scored_scan.group_by("s1_id","rec_id").agg(pl.col("probability").max())
                .sort(["rec_id","probability","s1_id"], descending=[False,True,False])
                .unique(subset=["rec_id"], keep="first", maintain_order=True).collect(engine="streaming"))
    matches = selected.group_by("s1_id").agg(pl.col("rec_id").sort().str.join(",").alias("matched_entity_ids"))
    result = (s1.rename({"entity_id":"source1_entity_id"})
              .join(matches, left_on="source1_entity_id", right_on="s1_id", how="left", validate="1:1")
              .with_columns(pl.col("matched_entity_ids").fill_null("")))
    path = args.out/"matching_results.tsv"
    temp = args.out/"matching_results.tsv.partial"
    result.write_csv(temp, separator="\t", quote_style="never")
    os.replace(temp,path)
    selected.write_parquet(args.out/"selected_pairs.parquet", compression="zstd")
    atomic_json(args.out/"submission_manifest.json", {
        "candidate_inputs": inputs, "scored_directory": str(args.scored),
        "test_s1_rows": s1.height, "selected_pairs": selected.height,
        "matching_sha256": sha256(path), "missing_address_slices": sorted(missing),
        "recipe": "Validated retrieval/reranking upgrade: supplied candidate channels, configured model blend, thresholds and one owner.",
        "selection_config":selection,
        "score": "UNKNOWN: not the V11 oracle score or a measured leaderboard score.",
        "france": "Legacy non-India feature code and unvalidated 0.65 fallback threshold."})
    print(f"CREATED: {path}\nRows: {result.height:,}; selected pairs: {selected.height:,}")
    print("Next run the official validator, then upload matching_results.tsv only.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--config",type=Path)
    p.add_argument("--selection",type=Path)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score")
    s.add_argument("--inputs", nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--batch-size", type=int, default=25000)
    s.add_argument("--pilot-only", action="store_true")
    s.add_argument("--threads",type=int,default=4)
    a = sub.add_parser("assemble")
    a.add_argument("--scored", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--allow-incomplete-address", action="store_true",
                   help="Emergency-only: explicitly allow blank/uncovered queries from unfinished slices")
    args = p.parse_args()
    args.project = args.project.resolve()
    if args.command == "score":
        score(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
