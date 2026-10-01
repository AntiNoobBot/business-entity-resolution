"""First look at the data. Run after 00_to_parquet.py. Writes eda_report.txt (paste it back to Claude)."""
from pathlib import Path
import polars as pl

P = Path("parquet")
rd = lambda n: pl.read_parquet(P / f"{n}.parquet")
s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")
t1, t2, t3 = rd("test_s1"), rd("test_s2"), rd("test_s3")

lines = []
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)

out("== SIZES ==")
for n, d in [("train_s1", s1), ("train_s2", s2), ("train_s3", s3), ("train_gt", gt),
             ("test_s1", t1), ("test_s2", t2), ("test_s3", t3)]:
    out(f"{n:9s} {d.height:>10,}")

out("\n== NULL / EMPTY COUNTS ==")
for n, d in [("train_s1", s1), ("train_s2", s2), ("train_s3", s3), ("test_s2", t2)]:
    out(n, d.null_count().to_dicts()[0],
        "empty_name:", (d["business_name"] == "").sum(),
        "empty_addr:", (d["business_address"] == "").sum())

out("\n== COUNTRY COUNTS ==")
for n, d in [("train_s1", s1), ("train_s2", s2), ("train_s3", s3),
             ("test_s1", t1), ("test_s2", t2), ("test_s3", t3)]:
    out(n, dict(d["country"].value_counts().sort("country").iter_rows()))

out("\n== EXACT DUPLICATES (same name+address+country) INSIDE A SOURCE ==")
for n, d in [("train_s1", s1), ("train_s2", s2), ("train_s3", s3)]:
    g = d.group_by(["business_name", "business_address", "country"]).len().filter(pl.col("len") > 1)
    out(n, "duplicated groups:", g.height)

# ---- ground truth structure ----
out("\n== GROUND TRUTH STRUCTURE ==")
long = (gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids")
          .rename({"matched_entity_ids": "rec_id"})
          .with_columns(pl.col("rec_id").str.strip_chars()))

n_with = long["source1_entity_id"].n_unique()
out("S1 entities in GT:", gt.height, "| with >=1 match:", n_with,
    "| singletons:", gt.height - n_with, f"({(gt.height - n_with) / gt.height:.1%})")

per = long.group_by("source1_entity_id").agg(
    pl.len().alias("n"),
    pl.col("rec_id").str.starts_with("S2-").sum().alias("n_s2"),
    pl.col("rec_id").str.starts_with("S3-").sum().alias("n_s3"))
out("matches per S1 (non-singletons):", dict(per["n"].describe().iter_rows()))
out("distribution of #matches:", dict(per.group_by("n").len().sort("n").head(12).iter_rows()))
out("avg S2 matches:", round(per["n_s2"].mean(), 3), "| avg S3 matches:", round(per["n_s3"].mean(), 3))

dup = long.group_by("rec_id").len().filter(pl.col("len") > 1)
out("S2/S3 records claimed by MORE THAN ONE S1 (want 0 -> one-owner constraint is valid):", dup.height)

ids23 = pl.concat([s2.select("entity_id"), s3.select("entity_id")])["entity_id"]
out("GT ids missing from S2/S3 files:", long.filter(~pl.col("rec_id").is_in(ids23)).height)
m2 = long.filter(pl.col("rec_id").str.starts_with("S2-"))["rec_id"].n_unique()
m3 = long.filter(pl.col("rec_id").str.starts_with("S3-"))["rec_id"].n_unique()
out(f"S2 records matched to some S1: {m2:,}/{s2.height:,} ({m2 / s2.height:.1%})",
    f"| S3: {m3:,}/{s3.height:,} ({m3 / s3.height:.1%})")

# ---- side-by-side pairs ----
s1r = s1.rename({"entity_id": "source1_entity_id", "business_name": "s1_name",
                 "business_address": "s1_addr", "country": "s1_country"})
rec = pl.concat([s2, s3]).rename({"entity_id": "rec_id", "business_name": "r_name",
                                  "business_address": "r_addr", "country": "r_country"})
pairs = long.join(s1r, on="source1_entity_id").join(rec, on="rec_id")
out("\nmatched pairs whose country labels DIFFER:", pairs.filter(pl.col("s1_country") != pl.col("r_country")).height,
    "of", pairs.height)

out("\n== TRUE PAIRS, side by side (per country) ==")
for c in sorted(pairs["s1_country"].unique().to_list()):
    sub = pairs.filter(pl.col("s1_country") == c)
    out(f"\n--- {c} ---")
    for r in sub.sample(min(10, sub.height), seed=7).iter_rows(named=True):
        out(f"S1 {r['s1_name']} | {r['s1_addr']}\n{r['rec_id'][:2]} {r['r_name']} | {r['r_addr']}\n")

out("\n== SOME SINGLETON S1 RECORDS ==")
single_ids = gt.filter(pl.col("matched_entity_ids").is_null() | (pl.col("matched_entity_ids") == ""))
sing = single_ids.join(s1r, on="source1_entity_id")
for r in sing.sample(min(10, sing.height), seed=7).iter_rows(named=True):
    out(f"{r['s1_country']} | {r['s1_name']} | {r['s1_addr']}")

out("\n== TEST: a few random rows per country (look at France!) ==")
for n, d in [("test_s1", t1), ("test_s2", t2), ("test_s3", t3)]:
    for c in sorted(d["country"].unique().to_list()):
        sub = d.filter(pl.col("country") == c)
        out(f"\n{n} / {c}")
        for r in sub.sample(min(5, sub.height), seed=3).iter_rows(named=True):
            out(f"  {r['business_name']} | {r['business_address']}")

Path("eda_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved eda_report.txt")
