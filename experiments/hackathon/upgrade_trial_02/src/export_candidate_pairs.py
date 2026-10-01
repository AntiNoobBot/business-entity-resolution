"""Export the exact union of fully scored candidate inputs with bounded memory.

Does not retrain, rescore, modify matching_results.tsv, or touch running scripts.
Disk-partitioning avoids materializing ~87 million pairs in memory at once.
"""
import argparse
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl
import pyarrow.parquet as pq
from common import atomic_json, read_json, sha256


def export(project, submission, work=None, buckets=64):
    manifest = read_json(submission/"submission_manifest.json")
    matching_path = submission/"matching_results.tsv"
    final = submission/"candidate_pairs.tsv"
    partial = submission/"candidate_pairs.tsv.partial"
    if final.exists() or partial.exists():
        raise RuntimeError("Candidate output or partial already exists. Preserve it and inspect before rerunning.")
    if sha256(matching_path) != manifest["matching_sha256"]:
        raise RuntimeError("Matching TSV changed after assembly")
    if manifest.get("missing_address_slices"):
        raise RuntimeError("This exporter requires the complete six-slice trial.")
    queries = pl.read_parquet(project/"parquet/test_s1.parquet",columns=["entity_id"]).rename({"entity_id":"s1_id"})
    matches = pl.read_csv(matching_path,separator="\t",infer_schema=False).rename({"source1_entity_id":"s1_id"})
    if queries["s1_id"].n_unique()!=queries.height or matches["s1_id"].n_unique()!=matches.height:
        raise RuntimeError("Duplicate S1 IDs in test data or matching output")
    if matches.height!=queries.height or matches.join(queries,on="s1_id",how="anti").height:
        raise RuntimeError("Matching output does not cover precisely the test S1 IDs")
    queries = queries.join(matches,on="s1_id",how="left",validate="1:1").with_columns(
        (pl.col("s1_id").hash(seed=41)%buckets).cast(pl.UInt16).alias("_bucket"))
    scored=Path(manifest["scored_directory"])
    evidence={}
    for complete in scored.glob("*/complete.json"):
        meta=read_json(complete.parent/"input.json")
        evidence[str(Path(meta["input"]).resolve())]=(meta,read_json(complete))
    inputs=[]
    for name in dict.fromkeys(manifest["candidate_inputs"]):
        path=Path(name).resolve()
        if str(path) not in evidence:
            raise RuntimeError(f"No completed scoring receipt for {path}")
        meta,done=evidence[str(path)]
        if sha256(path)!=meta["input_sha256"]:
            raise RuntimeError(f"Candidate input changed: {path}")
        rows=pq.ParquetFile(path).metadata.num_rows
        if rows!=done["input_rows"]:
            raise RuntimeError(f"Input row count differs from scoring receipt: {path}")
        inputs.append((path,rows,meta["input_sha256"]))
    if not inputs:
        raise RuntimeError("Submission manifest has no candidate inputs")
    raw_total=sum(rows for _,rows,_ in inputs)
    # Approximate scratch+output capacity guard, not an exact size prediction.
    needed=max(3*2**30,raw_total*55)
    if shutil.disk_usage(submission).free<needed:
        raise RuntimeError(f"Need approximately {needed/2**30:.1f} GiB free for export and scratch")
    work=work or submission/"candidate_export_work"
    work.mkdir(parents=True,exist_ok=False)
    atomic_json(work/"inputs.json",{"matching_sha256":manifest["matching_sha256"],
                                     "inputs":{str(p):h for p,_,h in inputs},"buckets":buckets})
    writers={}
    seen_rows=0
    started=time.perf_counter()
    try:
        for path,rows,_ in inputs:
            print(f"Partitioning {path.name}: {rows:,} pairs",flush=True)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=500000,columns=["s1_id","rec_id"]):
                frame=pl.from_arrow(batch)
                invalid=frame.filter(pl.col("s1_id").is_null() | pl.col("rec_id").is_null() |
                                     ~pl.col("s1_id").str.starts_with("S1-") |
                                     ~(pl.col("rec_id").str.starts_with("S2-") | pl.col("rec_id").str.starts_with("S3-")))
                if invalid.height:
                    raise RuntimeError("Null or invalid source-prefixed candidate IDs")
                frame=frame.with_columns((pl.col("s1_id").hash(seed=41)%buckets).cast(pl.UInt16).alias("_bucket"))
                for (bucket,),part in frame.partition_by("_bucket",as_dict=True,include_key=False).items():
                    table=part.to_arrow()
                    if bucket not in writers:
                        writers[bucket]=pq.ParquetWriter(work/f"bucket_{bucket:03d}.parquet",table.schema,compression="zstd",compression_level=1)
                    writers[bucket].write_table(table)
                seen_rows+=frame.height
            print(f"Partitioned {seen_rows:,}/{raw_total:,}; elapsed {(time.perf_counter()-started)/60:.1f} min",flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    if seen_rows!=raw_total:
        raise RuntimeError("Input pair count changed while exporting")
    output_rows,unique_pairs,selected_links=0,0,0
    with partial.open("wb") as output:
        for bucket in range(buckets):
            expected=queries.filter(pl.col("_bucket")==bucket).drop("_bucket")
            path=work/f"bucket_{bucket:03d}.parquet"
            if path.exists():
                pairs=pl.read_parquet(path)
                # This also checks S1 membership; partition hashing is identical.
                if pairs.select("s1_id").unique().join(expected.select("s1_id"),on="s1_id",how="anti").height:
                    raise RuntimeError("Candidate contains an unknown test S1 ID")
                grouped=pairs.group_by("s1_id").agg(pl.col("rec_id").unique().sort().alias("candidate_ids"))
            else:
                grouped=pl.DataFrame(schema={"s1_id":pl.String,"candidate_ids":pl.List(pl.String)})
            rows=(expected.join(grouped,on="s1_id",how="left",validate="1:1")
                  .with_columns(pl.col("candidate_ids").fill_null(pl.lit([],dtype=pl.List(pl.String))),
                                pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(
                                    pl.element().filter(pl.element()!="")).alias("matched_ids")))
            if rows.filter(pl.col("matched_ids").list.set_difference(pl.col("candidate_ids")).list.len()>0).height:
                raise RuntimeError("A final match is absent from the exact scored candidate union")
            unique_pairs+=rows["candidate_ids"].list.len().sum() or 0
            selected_links+=rows["matched_ids"].list.len().sum() or 0
            formatted=rows.select(pl.col("s1_id").alias("source1_entity_id"),
                                  pl.col("candidate_ids").list.join(",").alias("candidate_entity_ids"))
            formatted.write_csv(output,separator="\t",quote_style="never",include_header=(bucket==0))
            output_rows+=formatted.height
            print(f"Exported bucket {bucket+1}/{buckets}; {output_rows:,}/{queries.height:,} S1 rows",flush=True)
        output.flush()
        os.fsync(output.fileno())
    if output_rows!=queries.height:
        raise RuntimeError("Candidate export does not contain every test S1 exactly once")
    # The grouping/partition checks establish uniqueness without loading the TSV.
    os.replace(partial,final)
    report={"status":"PASS", "candidate_file":str(final),"matching_file":str(matching_path),
            "s1_rows":output_rows,"input_pair_rows":raw_total,"unique_candidate_pairs":unique_pairs,
            "selected_links_checked_as_subset":selected_links,"candidate_sha256":sha256(final),
            "matching_sha256":manifest["matching_sha256"],"candidate_input_sha256":{str(p):h for p,_,h in inputs},
            "validation":"Exporter checks: full S1 coverage, unique rows/lists, valid source prefixes, every match is a candidate. Test target-ID membership was checked for every pair by the completed scoring stage.",
            "scratch_directory":str(work),"elapsed_minutes":(time.perf_counter()-started)/60}
    atomic_json(submission/"candidate_export_report.json",report)
    print(f"CANDIDATE EXPORT CHECKS: PASS\n{final}\nS1 rows: {output_rows:,}; unique pairs: {unique_pairs:,}",flush=True)
    print("Both TSVs are now present. This exporter does not upload files.",flush=True)
    print(f"Scratch partitions retained for troubleshooting: {work}",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project",type=Path,required=True)
    p.add_argument("--submission",type=Path,required=True)
    p.add_argument("--work",type=Path)
    args=p.parse_args()
    export(args.project.resolve(),args.submission.resolve(),args.work)


if __name__=="__main__":
    main()
