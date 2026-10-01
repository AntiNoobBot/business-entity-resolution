"""Normalization v1 check: learns a state/region map from TRAIN pairs, then measures how many true pairs
match exactly after normalization and prints failure examples. Writes norm_report.txt + state_map.json."""
import json
from pathlib import Path
import polars as pl
from norm import add_name_cols, norm_addr, plain, split_last, token_key

pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_rows(80); pl.Config.set_tbl_width_chars(200)
P = Path("parquet")
rd = lambda n: pl.read_parquet(P / f"{n}.parquet")
s1, s2, s3, gt = rd("train_s1"), rd("train_s2"), rd("train_s3"), rd("train_gt")

lines = []
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s); lines.append(s)

long = (gt.filter(pl.col("matched_entity_ids").is_not_null() & (pl.col("matched_entity_ids") != ""))
          .with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids").rename({"matched_entity_ids": "rec_id"})
          .with_columns(pl.col("rec_id").str.strip_chars()))
s1r = s1.rename({"entity_id": "source1_entity_id", "business_name": "s1_name",
                 "business_address": "s1_addr", "country": "country"})
rec = pl.concat([s2, s3]).rename({"entity_id": "rec_id", "business_name": "r_name",
                                  "business_address": "r_addr"}).drop("country")
N = 1_000_000
pairs = (long.sample(min(N, long.height), seed=5).join(s1r, on="source1_entity_id").join(rec, on="rec_id")
         .with_columns(pl.col("rec_id").str.slice(0, 2).alias("src")))
b1, l1 = split_last(pl.col("s1_addr")); b2, l2 = split_last(pl.col("r_addr"))
pairs = pairs.with_columns(b1.alias("b1"), l1.alias("l1"), b2.alias("b2"), l2.alias("l2"))
pairs = pairs.with_columns(plain(pl.col("l1")).alias("l1"), plain(pl.col("l2")).alias("l2"))
# split by S1 entity (hash) so no entity is in both parts
is_learn = (pl.col("source1_entity_id").hash(seed=1) % 10) < 7
learn, ev = pairs.filter(is_learn), pairs.filter(~is_learn)

# ---- learn common final-component aliases from the learn split ----
# A final component is often a state/region, but noisy records may put a city or
# street there.  This map is therefore a soft feature, never an address gate.
cnt = learn.filter(pl.col("l2").is_not_null()).group_by(["country", "l2", "l1"]).len()
tot = cnt.group_by(["country", "l2"]).agg(pl.col("len").sum().alias("tot"))
best = (cnt.sort("len", descending=True).group_by(["country", "l2"], maintain_order=True).first()
        .join(tot, on=["country", "l2"])
        .filter((pl.col("tot") >= 300) & (pl.col("len") / pl.col("tot") >= 0.6) & (pl.col("l2") != pl.col("l1")))
        .select("country", pl.col("l2").alias("raw"), pl.col("l1").alias("canon"), "tot")
        .sort(["country", "tot"], descending=[False, True]))
out("learned final-address-component alias entries:", best.height)
for c in best["country"].unique().sort().to_list():
    sub = best.filter(pl.col("country") == c)
    out(f"{c}: {sub.height} entries; top: " + ", ".join(f"{r[0]}->{r[1]}" for r in sub.select("raw", "canon").head(25).iter_rows()))
best.write_json("state_map.json")

def canon(df, col):
    m = best.select("country", pl.col("raw").alias(col), pl.col("canon").alias("_c"))
    return (df.join(m, on=["country", col], how="left")
              .with_columns(pl.coalesce(["_c", col]).alias(col)).drop("_c"))

ev = canon(canon(ev, "l1"), "l2")
ev = add_name_cols(ev, "s1_name", "n1_"); ev = add_name_cols(ev, "r_name", "n2_")
ev = ev.with_columns(norm_addr(pl.col("b1"), pl.col("country")).alias("nb1"),
                     norm_addr(pl.col("b2"), pl.col("country")).alias("nb2"))
ev = ev.with_columns(
    (pl.col("n1_core") == pl.col("n2_core")).alias("name_core_eq"),
    ((pl.col("n1_core") == pl.col("n2_core")) | (pl.col("n1_sorted") == pl.col("n2_sorted"))
     | (pl.col("n1_compact") == pl.col("n2_compact"))).alias("name_any_eq"),
    (pl.col("l1") == pl.col("l2")).fill_null(False).alias("tail_eq"),
    (pl.col("nb1") == pl.col("nb2")).fill_null(False).alias("body_eq"),
    (token_key(pl.col("nb1")) == token_key(pl.col("nb2"))).fill_null(False).alias("body_token_eq"),
    pl.col("r_addr").is_null().alias("addr_missing"))
ev = ev.with_columns(
    (pl.col("tail_eq") & pl.col("body_eq")).alias("addr_exact_with_tail"),
    (pl.col("body_eq") | pl.col("body_token_eq")).alias("addr_body_key"))

out("\n== NORMALIZATION v2 TRUE-PAIR KEY COVERAGE (held-out sample) ==")
out(ev.group_by(["country", "src"]).agg(
    pl.len().alias("n"),
    pl.col("name_core_eq").mean().alias("name_core"),
    pl.col("name_any_eq").mean().alias("name_any"),
    pl.col("tail_eq").mean().alias("tail_alias"),
    pl.col("body_eq").mean().alias("body"),
    pl.col("body_token_eq").mean().alias("body_token"),
    pl.col("addr_exact_with_tail").mean().alias("addr_exact_tail"),
    pl.col("addr_body_key").mean().alias("addr_any_key"),
    (pl.col("name_any_eq") & pl.col("addr_body_key")).mean().alias("name_and_addr"),
    (pl.col("name_any_eq") | pl.col("addr_body_key")).mean().alias("name_or_addr"),
    pl.col("addr_missing").mean().alias("addr_missing")).sort(["country", "src"]))

out("\n== ADDRESS-KEY FAILURES (true pairs, address present, no exact body key) ==")
bad = ev.filter(~pl.col("addr_body_key") & ~pl.col("addr_missing"))
for c in sorted(ev["country"].unique().to_list()):
    sub = bad.filter(pl.col("country") == c)
    out(f"--- {c} ---")
    for r in sub.sample(min(14, sub.height), seed=2).iter_rows(named=True):
        out(f"{r['src']} S1: {r['s1_addr']}\n   R : {r['r_addr']}\n   norm S1: [{r['nb1']}] [tail: {r['l1']}]\n   norm R : [{r['nb2']}] [tail: {r['l2']}]\n")

out("\n== NAME FAILURES (true pairs, no name key equal) ==")
bn = ev.filter(~pl.col("name_any_eq"))
for c in sorted(ev["country"].unique().to_list()):
    sub = bn.filter(pl.col("country") == c)
    out(f"--- {c} ---")
    for r in sub.sample(min(14, sub.height), seed=3).iter_rows(named=True):
        out(f"{r['src']} S1: {r['s1_name']}  ->[{r['n1_core']}]\n   R : {r['r_name']}  ->[{r['n2_core']}]\n")

Path("norm_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved norm_report.txt and state_map.json")
