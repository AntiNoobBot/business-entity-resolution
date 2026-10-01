"""Targeted EDA #2. Run after 01_eda.py. Writes eda2_report.txt (paste it back to Claude)."""
import random
from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz

pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(70)

P = Path("parquet")
rd = lambda n: pl.read_parquet(P / f"{n}.parquet")
s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")
t1, t2, t3 = rd("test_s1"), rd("test_s2"), rd("test_s3")

lines = []
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)

long = (gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids")
          .rename({"matched_entity_ids": "rec_id"})
          .with_columns(pl.col("rec_id").str.strip_chars()))
recs = pl.concat([s2, s3])

SUF = r"\b(pvt|private|ltd|limited|inc|incorporated|llc|llp|lp|corp|corporation|co|company|plc|pllc|pc|sarl|sas|sasu|eurl|sa|ets|m s)\b"
def norm_name(c):
    return (c.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ")
             .str.replace_all(SUF, " ").str.replace_all(r"\s+", " ").str.strip_chars())
def norm_addr(c):
    return c.str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", " ").str.strip_chars()

# ---------- A. script / format profile ----------
nm, ad = pl.col("business_name"), pl.col("business_address")
def prof(d):
    return d.group_by("country").agg(
        pl.len().alias("n"),
        nm.str.contains(r"[\x{0370}-\x{1DFF}]").mean().alias("name_nonlatin"),
        nm.str.contains(r"[\x{00C0}-\x{024F}]").mean().alias("name_accent"),
        nm.str.contains(r"(?i)\.(com|net|org|in)\b").mean().alias("name_domain"),
        (nm == nm.str.to_uppercase()).mean().alias("name_allcaps"),
        ad.is_null().mean().alias("addr_null"),
        ad.str.contains(r"[\x{0370}-\x{1DFF}]").mean().alias("addr_nonlatin"),
        (ad == ad.str.to_uppercase()).mean().alias("addr_allcaps"),
        ad.str.contains(r"\b\d{6}\b").mean().alias("addr_has_6digit"),
        ad.str.contains(r"\b\d{5}\b").mean().alias("addr_has_5digit"),
    ).sort("country")

out("== A. FORMAT PROFILE (fractions of rows) ==")
for n, d in [("train_s1", s1), ("train_s2", s2), ("train_s3", s3),
             ("test_s1", t1), ("test_s2", t2), ("test_s3", t3)]:
    out(f"\n{n}"); out(prof(d))

# ---------- B. decoys ----------
out("\n== B. UNMATCHED S2/S3 RECORDS (decoys) ==")
un = recs.join(long.select("rec_id"), left_on="entity_id", right_on="rec_id", how="anti")
out("unmatched S2+S3:", f"{un.height:,}", "of", f"{recs.height:,}")
s1n = s1.select(norm_name(pl.col("business_name")).alias("nn"), "country").unique()
un_n = un.with_columns(norm_name(pl.col("business_name")).alias("nn"))
hit = un_n.join(s1n, on=["nn", "country"], how="semi").height
out(f"decoys whose normalized name equals SOME S1 name (same country): {hit:,}/{un.height:,} ({hit / max(un.height, 1):.1%})")
for c in sorted(un["country"].unique().to_list()):
    sub = un.filter(pl.col("country") == c)
    out(f"--- decoy samples: {c} ---")
    for r in sub.sample(min(15, sub.height), seed=11).iter_rows(named=True):
        out(f"  {r['entity_id'][:2]} {r['business_name']} | {r['business_address']}")

# ---------- C. exact duplicate groups ----------
out("\n== C. EXACT-DUPLICATE GROUPS INSIDE S2 / S3: who owns them? ==")
owner = long.select(pl.col("rec_id").alias("entity_id"), "source1_entity_id")
for n, d in [("train_s2", s2), ("train_s3", s3)]:
    g = (d.join(owner, on="entity_id", how="left")
          .group_by(["business_name", "business_address", "country"])
          .agg(pl.len().alias("n"),
               pl.col("source1_entity_id").is_not_null().sum().alias("n_matched"),
               pl.col("source1_entity_id").drop_nulls().n_unique().alias("n_owners"))
          .filter(pl.col("n") > 1))
    kind = (pl.when(pl.col("n_matched") == 0).then(pl.lit("none matched"))
              .when((pl.col("n_matched") == pl.col("n")) & (pl.col("n_owners") == 1)).then(pl.lit("all matched, same S1"))
              .when(pl.col("n_matched") == pl.col("n")).then(pl.lit("all matched, DIFFERENT S1"))
              .otherwise(pl.lit("some matched, some not")))
    out(n, dict(g.with_columns(kind.alias("kind")).group_by("kind").len().iter_rows()))

out("\n== D. TRUE PAIRS: exact-after-normalization rates (sample) ==")
s1r = s1.rename({"entity_id": "source1_entity_id", "business_name": "s1_name",
                 "business_address": "s1_addr", "country": "s1_country"})
rec = recs.rename({"entity_id": "rec_id", "business_name": "r_name",
                   "business_address": "r_addr", "country": "r_country"})
samp = long.sample(min(400_000, long.height), seed=5)
pairs = (samp.join(s1r, on="source1_entity_id").join(rec, on="rec_id")
         .with_columns(norm_name(pl.col("s1_name")).alias("n1"), norm_name(pl.col("r_name")).alias("n2"),
                       norm_addr(pl.col("s1_addr")).alias("a1"), norm_addr(pl.col("r_addr")).alias("a2"),
                       pl.col("r_addr").is_null().alias("addr_missing"),
                       pl.col("rec_id").str.slice(0, 2).alias("src"))
         .with_columns((pl.col("n1") == pl.col("n2")).alias("name_eq"),
                       (pl.col("a1") == pl.col("a2")).fill_null(False).alias("addr_eq")))
out(pairs.group_by(["s1_country", "src"]).agg(
    pl.len().alias("n"),
    pl.col("name_eq").mean().alias("name_exact"),
    pl.col("addr_eq").mean().alias("addr_exact"),
    (pl.col("name_eq") & pl.col("addr_eq")).mean().alias("both_exact"),
    pl.col("addr_missing").mean().alias("addr_missing")).sort(["s1_country", "src"]))

out("\nfuzz.token_set_ratio on normalized names: TRUE pairs vs RANDOM same-country pairs")
for c in sorted(pairs["s1_country"].unique().to_list()):
    sub = pairs.filter(pl.col("s1_country") == c).head(5000)
    a, b = sub["n1"].to_list(), sub["n2"].to_list()
    br = b[:]; random.Random(1).shuffle(br)
    tr = [fuzz.token_set_ratio(x, y) for x, y in zip(a, b)]
    rn = [fuzz.token_set_ratio(x, y) for x, y in zip(a, br)]
    out(f"{c}: TRUE p5/p10/p25/p50 = {np.percentile(tr, [5, 10, 25, 50]).round(0).tolist()}"
        f" | RANDOM p50/p90/p99 = {np.percentile(rn, [50, 90, 99]).round(0).tolist()}")

out("\n== E. LAST COMMA-PART OF ADDRESS (state / city forms), top 12 ==")
def last_part(d):
    return d.select(ad.str.split(",").list.last().str.strip_chars().alias("lp"), "country")
todo = [("train_s1", s1), ("train_s2", s2), ("train_s3", s3)]
for n, d in todo:
    lp = last_part(d)
    for c in sorted(lp["country"].unique().to_list()):
        vc = lp.filter(pl.col("country") == c).group_by("lp").len().sort("len", descending=True)
        out(f"{n}/{c}: distinct={vc.height} top={dict(vc.head(12).iter_rows())}")
for n, d in [("test_s1", t1), ("test_s2", t2), ("test_s3", t3)]:
    lp = last_part(d.filter(pl.col("country") == "France"))
    vc = lp.group_by("lp").len().sort("len", descending=True)
    out(f"{n}/France: distinct={vc.height} top={dict(vc.head(12).iter_rows())}")

Path("eda2_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved eda2_report.txt")
