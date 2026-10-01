"""Targeted, dev-checked fine-tuned retrieval for Indic-script reference names."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import sys,gc,time
from pathlib import Path
import numpy as np,polars as pl,pyarrow.parquet as pq,hnswlib,torch
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'upgrade_trial_02'
sys.path.insert(0,str(ROOT))
from common import atomic_json,sha256,digest
from neural_mac import embed


def records(split,source):
    prefix='train' if split=='dev' else 'test'
    q=pl.scan_parquet(ROOT/f'parquet/{prefix}_{source.lower()}.parquet').filter(pl.col('country')=='India')
    if source=='S1' and split=='dev':q=q.filter(pl.col('entity_id').hash(seed=41)%100==7)
    if source!='S1':q=q.filter(pl.col('business_name').fill_null('').str.contains(r'[\u0900-\u0DFF]'))
    pref='query: ' if source=='S1' else 'passage: '
    return q.select('entity_id',(pl.lit(pref+'business name: ')+pl.col('business_name').fill_null('')+pl.lit(' address: ')+pl.col('business_address').fill_null('')).alias('text')).collect()


def main():
    if not torch.backends.mps.is_available():raise RuntimeError('Run with Mac GPU access outside sandbox')
    torch.set_num_threads(2);torch.manual_seed(41)
    modeldir=ROOT/'models/biencoder_e5_v1'
    model=SentenceTransformer(str(modeldir),device='mps',local_files_only=True);model.max_seq_length=192;model.half()
    signature=digest({'model':sha256(modeldir/'model.safetensors'),'code':sha256(Path(__file__)),'precision':'fp16','max_length':192})
    for split in ['dev','test']:
        q=records(split,'S1');work=OUT/'indic_cache'/split;work.mkdir(parents=True,exist_ok=True)
        qvec,batch=embed(model,q,work/'S1',signature,64)
        for source in ['S2','S3']:
            output=OUT/f'candidates/{split}_indic_India_{source}.parquet'
            if output.exists():print('Completed',output,flush=True);continue
            r=records(split,source)
            print('INDIC RETRIEVAL',split,source,'index records',r.height,'query records',q.height,flush=True)
            rvec,batch=embed(model,r,work/source,signature,batch)
            index=hnswlib.Index(space='cosine',dim=384)
            index.init_index(max_elements=r.height,ef_construction=200,M=16,random_seed=100);index.set_num_threads(4)
            for start in range(0,r.height,25000):
                end=min(start+25000,r.height);index.add_items(np.asarray(rvec[start:end]),np.arange(start,end))
                print(split,source,'indexed',end,'/',r.height,flush=True)
            index.set_ef(100);ids=r['entity_id'].to_numpy();k=min(25,r.height)
            temp=output.with_suffix('.partial.parquet');writer=None;rows=0
            try:
                for start in range(0,q.height,4096):
                    end=min(start+4096,q.height)
                    labels,dist=index.knn_query(np.asarray(qvec[start:end]),k=k,num_threads=4)
                    pairs=pl.DataFrame({'s1_id':np.repeat(q['entity_id'].slice(start,end-start).to_numpy(),k),
                                        'rec_id':ids[labels.reshape(-1)],'finetuned_score':(1-dist).reshape(-1).astype(np.float32),
                                        'finetuned_rank':np.tile(np.arange(1,k+1,dtype=np.int16),end-start)}).with_columns(pl.lit('India').alias('country'),pl.lit(source).alias('source'))
                    table=pairs.to_arrow()
                    if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
                    writer.write_table(table);rows+=pairs.height
                    print(split,source,'queried',end,'/',q.height,flush=True)
            finally:
                if writer:writer.close()
            os.replace(temp,output)
            atomic_json(output.with_suffix('.json'),{'signature':signature,'rows':rows,'queries':q.height,'index_records':r.height,'sha256':sha256(output)})
            del index,rvec,r;gc.collect()
            torch.mps.empty_cache()
        del qvec,q;gc.collect()


if __name__=='__main__':main()
