"""Convert all TSVs to Parquet once. Usage: python 00_to_parquet.py path/to/student_resource"""
import sys, time
from pathlib import Path
import polars as pl

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "student_resource")
OUT = Path("parquet"); OUT.mkdir(exist_ok=True)

files = {
    "train_s1": ROOT / "dataset/train/train_source1.tsv",
    "train_s2": ROOT / "dataset/train/train_source2.tsv",
    "train_s3": ROOT / "dataset/train/train_source3.tsv",
    "train_gt": ROOT / "dataset/train/train_ground_truth.tsv",
    "test_s1": ROOT / "dataset/test/test_source1.tsv",
    "test_s2": ROOT / "dataset/test/test_source2.tsv",
    "test_s3": ROOT / "dataset/test/test_source3.tsv",
}

for name, path in files.items():
    t = time.time()
    # all columns as strings; quote_char=None so stray quotes in names/addresses don't break parsing
    df = pl.read_csv(path, separator="\t", infer_schema_length=0,
                     quote_char=None, truncate_ragged_lines=True)
    df.write_parquet(OUT / f"{name}.parquet", compression="zstd")
    print(f"{name:9s} rows={df.height:>10,}  cols={df.columns}  ({time.time()-t:.1f}s)")
