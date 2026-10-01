"""Independent CUDA retrieval worker. Never trains or reads labels."""
import argparse
import gc
import os
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl
import psutil
import torch
import hnswlib
from sentence_transformers import SentenceTransformer
from common import atomic_json, digest, read_json, sha256, verify_inputs


def records(root, source, mode, query=False):
    df = pl.read_parquet(root / f"inputs/test_India_{source.lower()}.parquet")
    name = pl.col("business_name").fill_null("")
    if mode == "generic":
        df = df.filter(name.str.replace_all(" ", "").str.len_chars() >= 4)
        expr = name
    else:
        expr = pl.lit("business name: ") + name + pl.lit(" address: ") + pl.col("business_address").fill_null("")
    prefix = "query: " if query else "passage: "
    df = df.select("entity_id", (pl.lit(prefix) + expr).alias("text"))
    if mode == "finetuned":
        df = df.filter(pl.col("text").str.replace_all(" ", "").str.len_chars() >= 20)
    return df


def encode(model, texts, batch):
    # Chunk-local deduplication retains every original record and ID.
    unique = list(dict.fromkeys(texts))
    lookup = {s: i for i, s in enumerate(unique)}
    while True:
        try:
            with torch.inference_mode():
                result = model.encode(unique, batch_size=batch, show_progress_bar=False,
                                      normalize_embeddings=True, convert_to_numpy=True)
            result = np.asarray(result, dtype=np.float32)
            result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
            if not np.isfinite(result).all():
                raise RuntimeError("Non-finite embeddings; retry the job with --precision fp32.")
            return result[[lookup[s] for s in texts]], batch
        except RuntimeError as error:
            if 'out of memory' not in str(error).lower():raise
            if batch <= 4:
                raise
            batch //= 2
            gc.collect()
            if torch.backends.mps.is_available():torch.mps.empty_cache()
            elif torch.cuda.is_available():torch.cuda.empty_cache()
            print(f"CUDA memory backoff: batch={batch}", flush=True)


def benchmark(model, texts, precision, threads):
    texts = texts[:512]
    model.float()
    reference, _ = encode(model, texts[:64], 8)
    if precision == "fp16":
        model.half()
    trial, _ = encode(model, texts[:64], 8)
    drift = float(np.min(np.sum(reference * trial, axis=1)))
    if drift < 0.995:
        raise RuntimeError(f"Precision pilot failed: min cosine={drift:.6f}. Use --precision fp32.")
    timings = []
    for batch in (16, 32, 64, 96, 128):
        if torch.backends.mps.is_available():torch.mps.synchronize()
        start = time.perf_counter()
        _, actual = encode(model, texts, batch)
        if torch.backends.mps.is_available():torch.mps.synchronize()
        seconds = time.perf_counter() - start
        timings.append((len(texts) / seconds, actual))
        print(f"pilot batch={actual}: {len(texts)/seconds:.1f} texts/sec", flush=True)
        if actual != batch:
            break
    speed, batch = max(timings)
    return {"batch_size": batch, "pilot_texts_per_second": speed,
            "minimum_fp32_precision_cosine": drift,
            "warning": "Speed sample only; not an F0.5 score or whole-job ETA. HNSW uses CPU."}


def embed(model, df, directory, signature, batch):
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "progress.json"
    path = directory / "embeddings.npy"
    state = {"signature": signature, "rows": df.height, "dimension": 384, "done": 0}
    if meta_path.exists():
        previous = read_json(meta_path)
        if any(previous[k] != state[k] for k in ("signature", "rows", "dimension")):
            raise RuntimeError("Embedding cache configuration changed; use a new --work-name.")
        state = previous
        mmap = np.load(path, mmap_mode="r+")
        if mmap.shape != (df.height, 384):
            raise RuntimeError("Invalid embedding cache shape")
    else:
        mmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(df.height, 384))
        atomic_json(meta_path, state)
    start_time, start_row = time.perf_counter(), state["done"]
    for start in range(state["done"], df.height, 4096):
        end = min(start + 4096, df.height)
        vectors, batch = encode(model, df["text"].slice(start, end-start).to_list(), batch)
        mmap[start:end] = vectors
        mmap.flush()
        state["done"] = end
        atomic_json(meta_path, state)
        speed = (end-start_row) / max(time.perf_counter()-start_time, 1e-6)
        print(f"{directory.name}: encoded {end:,}/{df.height:,}; {speed:.1f}/sec; "
              f"encoding ETA {(df.height-end)/speed/60:.1f} min", flush=True)
    del mmap
    return np.load(path, mmap_mode="r"), batch


def retrieve(root, work, mode, model, query, query_vectors, source, signature, batch, threads):
    output = work / "results" / source
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "complete.json"
    if completed.exists():
        manifest = read_json(completed)
        if manifest["signature"] != signature:
            raise RuntimeError("Result signature mismatch")
        for name, value in manifest["files"].items():
            if sha256(output / name) != value:
                raise RuntimeError(f"Changed result shard: {name}")
        print(f"{source}: verified complete; skipping", flush=True)
        return
    right = records(root, source, mode)
    vectors, batch = embed(model, right, work / f"embeddings_{source}", signature, batch)
    print(f"{source}: building CPU HNSW index for {right.height:,} records; "
          f"available RAM={psutil.virtual_memory().available/2**30:.1f} GiB", flush=True)
    index = hnswlib.Index(space="cosine", dim=384)
    index.init_index(max_elements=right.height, ef_construction=200, M=16, random_seed=100)
    index.set_num_threads(threads)
    for start in range(0, right.height, 25000):
        end = min(start+25000, right.height)
        index.add_items(np.asarray(vectors[start:end]), np.arange(start, end))
        print(f"{source}: indexed {end:,}/{right.height:,}", flush=True)
    index.set_ef(100)
    ids = right["entity_id"].to_numpy()
    k = min(25, right.height)
    score = "finetuned_score" if mode == "finetuned" else "multilingual_name_score"
    rank = "finetuned_rank" if mode == "finetuned" else "multilingual_name_rank"
    files = {}
    for start in range(0, query.height, 4096):
        end = min(start+4096, query.height)
        path = output / f"part_{start:09d}_{end:09d}.parquet"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists() and read_json(receipt) == {"signature": signature, "sha256": sha256(path)}:
            files[path.name] = sha256(path)
            continue
        labels, distances = index.knn_query(np.asarray(query_vectors[start:end]), k=k, num_threads=threads)
        result = pl.DataFrame({
            "s1_id": np.repeat(query["entity_id"].slice(start, end-start).to_numpy(), k),
            "rec_id": ids[labels.reshape(-1)], "country": ["India"]*((end-start)*k),
            "source": [source]*((end-start)*k),
            score: (1-distances).reshape(-1).astype(np.float32),
            rank: np.tile(np.arange(1,k+1,dtype=np.int16), end-start),
        })
        temp = path.with_suffix(".tmp")
        result.write_parquet(temp, compression="zstd")
        os.replace(temp, path)
        checksum = sha256(path)
        atomic_json(receipt, {"signature": signature, "sha256": checksum})
        files[path.name] = checksum
        print(f"{source}: queried {end:,}/{query.height:,}", flush=True)
    atomic_json(completed, {"signature": signature, "files": files,
                           "queries": query.height, "rows": query.height*k, "top_k": k})
    del index, vectors, right
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--mode", choices=["finetuned", "generic"], default="finetuned")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--encode-only", action="store_true", help="Prepare GPU embeddings, no CPU HNSW yet")
    parser.add_argument("--work-name", default="run1")
    args = parser.parse_args()
    root = args.root.resolve()
    if Path(args.work_name).name != args.work_name:
        raise SystemExit("--work-name must be a simple folder name")
    manifest = verify_inputs(root)
    if not torch.backends.mps.is_available():
        raise SystemExit("Apple MPS unavailable; this worker requires the Mac GPU.")
    torch.set_num_threads(max(1, args.threads))
    torch.manual_seed(41)
    np.random.seed(41)
    model_dir = root / "models" / ("finetuned_e5" if args.mode == "finetuned" else "base_e5")
    model = SentenceTransformer(str(model_dir), device="mps", local_files_only=True)
    if args.mode == "finetuned":
        model.max_seq_length = 192
    if model.get_sentence_embedding_dimension() != 384:
        raise RuntimeError("Expected 384-dimensional model")
    work = root / "work" / args.mode / args.work_name
    work.mkdir(parents=True, exist_ok=True)
    query = records(root, "S1", args.mode, query=True)
    corpus_sample = records(root, "S2", args.mode).sample(n=256, seed=41)
    sample = query.sample(n=256, seed=41)["text"].to_list() + corpus_sample["text"].to_list()
    pilot = benchmark(model, sample, args.precision, args.threads)
    total = sum(manifest["counts"].values())
    pilot["rough_encoding_only_hours"] = total / pilot["pilot_texts_per_second"] / 3600
    pilot["model_max_seq_length"] = model.max_seq_length
    atomic_json(work / "pilot.json", pilot)
    print(pilot, flush=True)
    print("HNSW build, search, transfer, Mac scoring and validation are ADDITIONAL time.", flush=True)
    if args.pilot_only:
        return
    if shutil.disk_usage(root).free < 12 * 2**30:
        raise SystemExit("Need 12 GiB free for this retrieval mode; caches are retained for resume.")
    signature = digest({"handoff": sha256(root/"handoff_manifest.json"), "mode": args.mode,
                        "precision": args.precision, "max_seq_length": model.max_seq_length,
                        "torch": torch.__version__, "gpu": "Apple MPS",
                        "script": sha256(Path(__file__))})
    state_path = work / "run.json"
    state = {"signature": signature, "mode": args.mode, "precision": args.precision,
             "handoff_sha256": sha256(root/"handoff_manifest.json")}
    if state_path.exists() and read_json(state_path) != state:
        raise RuntimeError("Run configuration changed. Choose a new --work-name.")
    atomic_json(state_path, state)
    qvec, batch = embed(model, query, work / "embeddings_S1", signature, pilot["batch_size"])
    if args.encode_only:
        for source in ("S2", "S3"):
            right = records(root, source, args.mode)
            vectors, batch = embed(model, right, work/f"embeddings_{source}", signature, batch)
            del vectors, right
            gc.collect()
        print("EMBEDDINGS READY. Rerun without --encode-only for CPU index/search.")
        return
    for source in ("S2", "S3"):
        retrieve(root, work, args.mode, model, query, qvec, source, signature, batch, args.threads)
        # Export each completed source immediately; Mac need not wait for both.
        output = work / "results" / source
        files = {p.relative_to(work).as_posix(): sha256(p) for p in output.glob("*.parquet")}
        for p in (output / "complete.json", work / "pilot.json", state_path):
            files[p.relative_to(work).as_posix()] = sha256(p)
        return_manifest = dict(state, source=source, country="India", files=files)
        (root/"returns").mkdir(exist_ok=True)
        archive = root / "returns" / f"return_{args.mode}_{source}_{args.work_name}.zip"
        temp = archive.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            import json
            z.writestr("return_manifest.json", json.dumps(return_manifest, indent=2))
            for name in sorted(files):
                z.write(work/name, name)
        os.replace(temp, archive)
        print(f"SEND TO MAC NOW: {archive}", flush=True)


if __name__ == "__main__":
    main()
