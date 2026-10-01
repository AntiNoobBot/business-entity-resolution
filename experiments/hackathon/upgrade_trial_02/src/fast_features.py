"""Same v2 feature definitions; normalize each entity once, not each pair."""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from norm import add_name_cols, norm_addr, plain, split_last, token_key


def prepare_entity(df, p, country, v2):
    df = add_name_cols(df, p+"name", p)
    _, tail = split_last(pl.col(p+"addr"))
    df = df.with_columns(
        norm_addr(pl.col(p+"addr"),pl.lit(country)).alias(p+"addr_norm"),
        plain(tail).alias(p+"tail"),
        pl.col(p+"addr").is_null().alias(p+"addr_missing"),
        v2.legal_code(pl.col(p+"name")).alias(p+"legal_code"),
        pl.col(p+"core").str.len_chars().alias(p+"name_len"),
    ).with_columns(
        pl.col(p+"addr_norm").str.len_chars().fill_null(0).alias(p+"addr_len"),
        token_key(pl.col(p+"addr_norm")).alias(p+"address_key"),
    )
    return df.drop(p+"name",p+"addr")


def feature_frame(df, v1, v2, training):
    df = df.with_columns(
        pl.col("s_addr_norm").str.extract_all(r"\d+").alias("s_numbers"),
        pl.col("r_addr_norm").str.extract_all(r"\d+").alias("r_numbers"),
        pl.col("s_core").str.split(" ").alias("s_name_tokens"),
        pl.col("r_core").str.split(" ").alias("r_name_tokens"),
        pl.col("s_addr_norm").str.split(" ").alias("s_addr_tokens"),
        pl.col("r_addr_norm").str.split(" ").alias("r_addr_tokens"),
        (pl.col("s_core")==pl.col("r_core")).alias("name_core_exact"),
        (pl.col("s_sorted")==pl.col("r_sorted")).alias("name_sorted_exact"),
        (pl.col("s_compact")==pl.col("r_compact")).alias("name_compact_exact"),
        (pl.col("s_address_key")==pl.col("r_address_key")).fill_null(False).alias("address_token_exact"),
        (pl.col("s_tail")==pl.col("r_tail")).fill_null(False).alias("tail_exact"),
        (pl.col("s_core").str.slice(0,4)==pl.col("r_core").str.slice(0,4)).alias("name_prefix4_exact"),
        (pl.col("s_name_len").cast(pl.Float32)/pl.max_horizontal("s_name_len","r_name_len",pl.lit(1))).alias("name_length_ratio"),
        (pl.col("s_addr_len").cast(pl.Float32)/pl.max_horizontal("s_addr_len","r_addr_len",pl.lit(1))).alias("address_length_ratio"),
        ((pl.col("s_legal_code")>0)&(pl.col("r_legal_code")>0)&(pl.col("s_legal_code")==pl.col("r_legal_code"))).alias("legal_form_agrees"),
        ((pl.col("s_legal_code")>0)&(pl.col("r_legal_code")>0)&(pl.col("s_legal_code")!=pl.col("r_legal_code"))).alias("legal_form_conflicts"),
        ((pl.col("s_legal_code")>0)^(pl.col("r_legal_code")>0)).alias("legal_form_one_missing"),
    ).with_columns(
        pl.col("s_numbers").list.set_intersection(pl.col("r_numbers")).list.len().alias("shared_number_count"),
        pl.col("s_numbers").list.set_union(pl.col("r_numbers")).list.len().alias("number_union_count"),
        (pl.col("s_numbers").list.len()==pl.col("r_numbers").list.len()).alias("number_count_exact"),
        pl.col("s_name_tokens").list.set_intersection(pl.col("r_name_tokens")).list.len().alias("shared_name_tokens"),
        pl.col("s_name_tokens").list.set_union(pl.col("r_name_tokens")).list.len().alias("name_token_union"),
        pl.col("s_addr_tokens").list.set_intersection(pl.col("r_addr_tokens")).list.len().alias("shared_address_tokens"),
        pl.col("s_addr_tokens").list.set_union(pl.col("r_addr_tokens")).list.len().alias("address_token_union"),
    ).with_columns(
        pl.when(pl.col("number_union_count")>0).then(pl.col("shared_number_count")/pl.col("number_union_count")).otherwise(0.).alias("number_jaccard"),
        pl.when(pl.col("name_token_union")>0).then(pl.col("shared_name_tokens")/pl.col("name_token_union")).otherwise(0.).alias("name_token_jaccard"),
        pl.when(pl.col("address_token_union")>0).then(pl.col("shared_address_tokens")/pl.col("address_token_union")).otherwise(0.).alias("address_token_jaccard"),
    )
    columns=[]
    for kind, left, right, minimum in (("name","s_core","r_core",4),("address","s_addr_norm","r_addr_norm",12)):
        a=df[left].fill_null("").to_list(); b=df[right].fill_null("").to_list()
        for suffix, scorer in (("ratio",fuzz.ratio),("token_set",fuzz.token_set_ratio),("partial_ratio",fuzz.partial_ratio)):
            # Float64 scoring then /100 then Float32 matches the original code.
            values=(process.cpdist(a,b,scorer=scorer,dtype=np.float64,workers=2)/100.).astype(np.float32)
            columns.append(pl.Series(kind+"_"+suffix,values))
        values=np.fromiter((int(min(len(x),len(y))>=minimum and (x in y or y in x)) for x,y in zip(a,b)),dtype=np.int8,count=len(a))
        columns.append(pl.Series(kind+"_contains",values))
    return training.encode(df.with_columns(columns))
