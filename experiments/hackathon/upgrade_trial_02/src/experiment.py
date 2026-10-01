"""Deadline experiment; all outputs isolated under upgrade_trial_02."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','4')
import sys, argparse, gc, time
from pathlib import Path
import numpy as np
import polars as pl
import lightgbm as lgb
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from common import atomic_json,read_json,sha256
from mac_finish_fast import module
from fast_features import prepare_entity,feature_frame
V1=module(ROOT,'10_build_dev_features.py')
V2=module(ROOT,'18_build_features_v2.py')
TR=module(ROOT,'19_train_lgbm_v2.py')
FEATURES=TR.FEATURES+['country_code','source_code']
OUT=ROOT/'upgrade_trial_02'


def queries(split):
    q=pl.scan_parquet(ROOT/'parquet/train_s1.parquet')
    bucket=pl.col('entity_id').hash(seed=41)%100
    q=q.filter(bucket.is_in([0,1]) if split=='train' else bucket==7)
    return q.select(pl.col('entity_id').alias('s1_id'),'country').collect()


def truth(q):
    return (pl.scan_parquet(ROOT/'parquet/train_gt.parquet')
            .select(pl.col('source1_entity_id').alias('s1_id'),'matched_entity_ids')
            .join(q.lazy().select('s1_id'),on='s1_id',how='inner')
            .filter(pl.col('matched_entity_ids').is_not_null()&(pl.col('matched_entity_ids')!=''))
            .with_columns(pl.col('matched_entity_ids').str.split(','))
            .explode('matched_entity_ids',empty_as_null=True)
            .select('s1_id',pl.col('matched_entity_ids').str.strip_chars().alias('rec_id')).unique().collect())


def build(candidates, output, q):
    if output.exists():
        print('Already built:',output,flush=True);return
    gt=truth(q).with_columns(pl.lit(1,dtype=pl.Int8).alias('label'))
    tmp=output.with_suffix('.partial.parquet'); writer=None
    try:
        for country,source in candidates.select('country','source').unique().sort('country','source').iter_rows():
            c=candidates.filter((pl.col('country')==country)&(pl.col('source')==source)).select('s1_id','rec_id','country','source').unique()
            if not c.height:continue
            s=(pl.scan_parquet(ROOT/'parquet/train_s1.parquet')
               .filter(pl.col('country')==country).join(q.lazy().select(pl.col('s1_id').alias('entity_id')),on='entity_id',how='inner')
               .select(pl.col('entity_id').alias('s1_id'),pl.col('business_name').alias('s_name'),pl.col('business_address').alias('s_addr')).collect())
            r=(pl.scan_parquet(ROOT/f'parquet/train_{source.lower()}.parquet').filter(pl.col('country')==country)
               .select(pl.col('entity_id').alias('rec_id'),pl.col('business_name').alias('r_name'),pl.col('business_address').alias('r_addr')).collect())
            print('Preparing',country,source,'pairs',c.height,flush=True)
            s=prepare_entity(s,'s_',country,V2);r=prepare_entity(r,'r_',country,V2)
            for start in range(0,c.height,100000):
                part=c.slice(start,100000)
                joined=part.join(s,on='s1_id',validate='m:1').join(r,on='rec_id',validate='m:1')
                if joined.height!=part.height:raise RuntimeError('Missing feature join IDs')
                f=feature_frame(joined,V1,V2,TR).select('s1_id','rec_id','country','source',*FEATURES)
                f=f.join(gt,on=['s1_id','rec_id'],how='left').with_columns(pl.col('label').fill_null(0))
                table=f.to_arrow()
                if writer is None:writer=pq.ParquetWriter(tmp,table.schema,compression='zstd')
                writer.write_table(table)
                print(output.name,country,source,min(start+100000,c.height),'/',c.height,flush=True)
            del s,r,c;gc.collect()
    finally:
        if writer:writer.close()
    os.replace(tmp,output)


def prepare():
    tq,dq=queries('train'),queries('dev')
    assert tq.join(dq,on='s1_id').height==0
    base=pl.read_parquet(ROOT/'train_features_v2.parquet',columns=['s1_id','rec_id'])
    assert base.select('s1_id').unique().join(tq.select('s1_id'),on='s1_id',how='anti').height==0
    extra=(pl.read_parquet(ROOT/'train_retrieval_finetuned_india.parquet')
           .join(base,on=['s1_id','rec_id'],how='anti').select('s1_id','rec_id','country','source').unique())
    assert extra.select('s1_id').unique().join(tq.select('s1_id'),on='s1_id',how='anti').height==0
    del base;gc.collect()
    build(extra,OUT/'features/train_extra.parquet',tq)
    dev=pl.read_parquet(ROOT/'dev_candidates_v11_finetuned_india.parquet',columns=['s1_id','rec_id','country','source'])
    build(dev,OUT/'features/dev_v11.parquet',dq)
    atomic_json(OUT/'split_audit.json',{'train_s1':tq.height,'dev_s1':dq.height,'train_buckets':[0,1],'dev_bucket':7,'lockbox_bucket':8,'lockbox_used':False,'training_extra_pairs':extra.height})


def train():
    base=TR.encode(pl.read_parquet(ROOT/'train_features_v2.parquet'))
    extra=pl.read_parquet(OUT/'features/train_extra.parquet')
    x=np.concatenate([base.select(FEATURES).to_numpy().astype(np.float32),extra.select(FEATURES).to_numpy().astype(np.float32)])
    y=np.concatenate([base['label'].to_numpy(),extra['label'].to_numpy()])
    del base,extra;gc.collect()
    dev=pl.read_parquet(OUT/'features/dev_v11.parquet')
    dx=dev.select(FEATURES).to_numpy().astype(np.float32);dy=dev['label'].to_numpy()
    del dev;gc.collect()
    print('TRAIN START',x.shape,'positive',int(y.sum()),flush=True)
    # One bounded experiment; no class weighting because it shifts calibration.
    model=lgb.train({'objective':'binary','metric':'binary_logloss','learning_rate':.06,'num_leaves':63,
                     'min_data_in_leaf':80,'feature_fraction':.9,'bagging_fraction':.85,'bagging_freq':1,
                     'lambda_l2':2.,'num_threads':4,'verbosity':-1,'seed':41,'force_col_wise':True,'max_bin':127},
                    lgb.Dataset(x,label=y,feature_name=FEATURES,free_raw_data=True),num_boost_round=650,
                    valid_sets=[lgb.Dataset(dx,label=dy,reference=None,feature_name=FEATURES)],
                    callbacks=[lgb.early_stopping(50),lgb.log_evaluation(50)])
    model.save_model(str(OUT/'models/lgbm_hardneg.txt'))
    atomic_json(OUT/'models/training_report.json',{'rows':len(y),'positives':int(y.sum()),'iterations':model.best_iteration,'train_extra_sha256':sha256(OUT/'features/train_extra.parquet')})


def evaluate():
    dev=pl.read_parquet(OUT/'features/dev_v11.parquet')
    q=queries('dev'); gt=truth(q)
    totals=q.join(gt.group_by('s1_id').len().rename({'len':'true_count'}),on='s1_id',how='left').with_columns(pl.col('true_count').fill_null(0))
    x=dev.select(FEATURES).to_numpy().astype(np.float32)
    address=pl.read_parquet(ROOT/'dev_retrieval_address_char.parquet',columns=['s1_id','rec_id']).unique().with_columns(pl.lit(True).alias('address_member'))
    meta=(dev.select('s1_id','rec_id','country','source','label').with_row_index('_row')
          .join(address,on=['s1_id','rec_id'],how='left').sort('_row').drop('_row')
          .with_columns(pl.col('address_member').fill_null(False)))
    rule_paths=sorted((OUT/'candidates').glob('dev_rules_*.parquet'))
    if len(rule_paths)==4:
        rules=pl.concat([pl.read_parquet(p,columns=['s1_id','rec_id']) for p in rule_paths]).unique()
        if rules.join(meta.select('s1_id','rec_id'),on=['s1_id','rec_id'],how='anti').height:
            raise RuntimeError('Rule candidates outside existing V11 feature coverage')
        meta=(meta.with_row_index('_row').join(rules.with_columns(pl.lit(True).alias('rule_member')),on=['s1_id','rec_id'],how='left')
              .sort('_row').drop('_row').with_columns(pl.col('rule_member').fill_null(False)))
    results=[]
    probabilities={}
    for modelname,path in [('old',ROOT/'lgbm_v2.txt'),('hardneg',OUT/'models/lgbm_hardneg.txt')]:
        if not path.exists():continue
        probabilities[modelname]=lgb.Booster(model_file=str(path)).predict(x,num_threads=4)
    if len(probabilities)==2:probabilities['blend']=(probabilities['old']+probabilities['hardneg'])/2
    for modelname,prob in probabilities.items():
        scored=meta.with_columns(pl.Series('probability',prob))
        scored.write_parquet(OUT/f'dev/scored_{modelname}.parquet',compression='zstd')
        for recipe in ['address_only','v11']+(['address_rules'] if 'rule_member' in meta.columns else []):
            frame=scored.filter(pl.col('address_member')) if recipe=='address_only' else scored.filter(pl.col('address_member')|pl.col('rule_member')) if recipe=='address_rules' else scored
            for threshold in [.35,.45,.5,.55,.6,.65,.7,.75,.8,.85,.9]:
                s=(frame.filter(pl.col('probability')>=threshold).sort(['rec_id','probability','s1_id'],descending=[False,True,False])
                   .unique('rec_id',keep='first').group_by('s1_id').agg(pl.len().alias('pred_count'),pl.col('label').sum().alias('tp')))
                e=totals.join(s,on='s1_id',how='left').with_columns(pl.col('pred_count').fill_null(0),pl.col('tp').fill_null(0))
                e=e.with_columns(pl.when((pl.col('true_count')==0)&(pl.col('pred_count')==0)).then(1.)
                                 .when(pl.col('tp')==0).then(0.).otherwise(1.25*pl.col('tp')/(.25*pl.col('true_count')+pl.col('pred_count'))).alias('f05'))
                row={'model':modelname,'recipe':recipe,'threshold':threshold,'macro_f05':e['f05'].mean()}
                row['by_country']={c:float(v) for c,v in e.group_by('country').agg(pl.col('f05').mean()).iter_rows()}
                results.append(row); print(row,flush=True)
    atomic_json(OUT/'dev/evaluation.json',results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);a=p.parse_args()
    {'prepare':prepare,'train':train,'evaluate':evaluate}[a.stage]()
