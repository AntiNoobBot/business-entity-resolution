"""Recover omitted exact/rare-name/number routes, without expensive TF-IDF search."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import sys,argparse,time,gc
from pathlib import Path
import polars as pl,numpy as np,pyarrow.parquet as pq
from rapidfuzz import process,fuzz
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from mac_finish_fast import module
from common import atomic_json,sha256
V4=module(ROOT,'07_blocking_eval_v4.py')
OUT=ROOT/'upgrade_trial_02'


def keys(df,side,route):
    idcol='s1_id' if side=='s_' else 'rec_id'
    if route=='exact':return V4.exact_keys(df,idcol,idcol,side).select(idcol,'key')
    kind='name' if route=='name' else 'number'
    col=side+('core' if route=='name' else 'addr_norm')
    return V4.token_keys(df,idcol,idcol,col,kind).select(idcol,'key')


def run(split,country,source):
    path=OUT/f'candidates/{split}_rules_{country}_{source}.parquet'
    report=path.with_suffix('.json')
    if path.exists():
        print('Already exists',path,flush=True);return
    prefix='train' if split=='dev' else 'test'
    q=pl.scan_parquet(ROOT/f'parquet/{prefix}_s1.parquet').filter(pl.col('country')==country)
    if split=='dev':q=q.filter(pl.col('entity_id').hash(seed=41)%100==7)
    left=q.collect().rename({'entity_id':'s1_id','business_name':'s_name','business_address':'s_addr'})
    right=(pl.scan_parquet(ROOT/f'parquet/{prefix}_{source.lower()}.parquet').filter(pl.col('country')==country).collect()
           .rename({'entity_id':'rec_id','business_name':'r_name','business_address':'r_addr'}))
    started=time.perf_counter()
    print('Normalizing rule index',split,country,source,right.height,flush=True)
    left=V4.enrich(left,'s1_id','s_name','s_addr','s_').drop('s_name','s_addr')
    right=V4.enrich(right,'rec_id','r_name','r_addr','r_').drop('r_name','r_addr')
    indices={}
    for route,cap in [('exact',1000),('name',200),('number',100)]:
        z=keys(right,'r_',route)
        indices[route]=(z.group_by('key').agg(pl.col('rec_id').alias('rec_ids'))
                        .filter(pl.col('rec_ids').list.len()<=cap))
        print(route,'active keys',indices[route].height,flush=True)
        del z;gc.collect()
    writer=None;rows=0;raw_rows=0
    tmp=path.with_suffix('.partial.parquet')
    try:
        for start in range(0,left.height,10000):
            l=left.slice(start,10000)
            frames=[]
            for route,index in indices.items():
                z=keys(l,'s_',route).join(index,on='key',how='inner').select('s1_id','rec_ids')
                z=z.explode('rec_ids').rename({'rec_ids':'rec_id'})
                frames.append(z)
            pairs=pl.concat(frames).unique();raw_rows+=pairs.height
            if pairs.height>10000000:raise RuntimeError('Unexpected rule expansion; stop to protect memory')
            pairs=pairs.join(l.select('s1_id','s_core','s_addr_norm'),on='s1_id',validate='m:1').join(
                right.select('rec_id','r_core','r_addr_norm'),on='rec_id',validate='m:1')
            if pairs.height:
                ns=process.cpdist(pairs['s_core'].fill_null('').to_list(),pairs['r_core'].fill_null('').to_list(),scorer=fuzz.token_set_ratio,dtype=np.float32,workers=4)
                ads=process.cpdist(pairs['s_addr_norm'].fill_null('').to_list(),pairs['r_addr_norm'].fill_null('').to_list(),scorer=fuzz.token_set_ratio,dtype=np.float32,workers=4)
                # Broad, label-free prefilter. The final candidate TSV contains
                # only this retained set plus address candidates, as actually scored.
                keep=(ns>=85)|(ads>=80)|((ns>=55)&(ads>=35))
                pairs=pairs.filter(pl.Series(keep))
            pairs=pairs.select('s1_id','rec_id').with_columns(pl.lit(country).alias('country'),pl.lit(source).alias('source'))
            table=pairs.to_arrow()
            if writer is None:writer=pq.ParquetWriter(tmp,table.schema,compression='zstd')
            writer.write_table(table);rows+=pairs.height
            print(split,country,source,min(start+10000,left.height),'/',left.height,'retained pairs',rows,'minutes',round((time.perf_counter()-started)/60,2),flush=True)
    finally:
        if writer:writer.close()
    os.replace(tmp,path)
    atomic_json(report,{'split':split,'country':country,'source':source,'raw_pairs':raw_rows,'retained_pairs':rows,'queries':left.height,
                        'seconds':time.perf_counter()-started,'sha256':sha256(path),'code_sha256':sha256(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--country',required=True);p.add_argument('--source',choices=['S2','S3'],required=True);a=p.parse_args()
    run(a.split,a.country,a.source)
