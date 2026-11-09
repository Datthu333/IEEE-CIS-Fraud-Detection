"""Create version 2 stratified split and fit preprocessing on training-fit rows only.

Reads (never edits) the v1 merged CSVs under ../cleaned_data. Outputs are new
files under this versioned directory. No model is trained.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CHUNK = 5000
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.20
MAX_MEDIAN_FIT_ROWS = 100000

def read_chunks(path, **kwargs):
    return pd.read_csv(path, chunksize=CHUNK, low_memory=False, **kwargs)

def id_digest(values):
    h=hashlib.sha256()
    for x in sorted(map(int,values)):
        h.update(str(x).encode()); h.update(b"\n")
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default=str(Path(__file__).resolve().parent.parent/'cleaned_data'))
    ap.add_argument('--output-dir',default=str(Path(__file__).resolve().parent))
    args=ap.parse_args(); data=Path(args.data_dir); out=Path(args.output_dir); pp=out/'preprocessed'
    pp.mkdir(parents=True,exist_ok=True)
    outputs=[pp/'train_fit_encoded.csv',pp/'validation_encoded.csv',pp/'test_encoded.csv']
    if any(p.exists() and p.stat().st_size for p in outputs) or (out/'fitted_preprocessing.json').exists():
        raise FileExistsError('Versioned outputs already exist; choose a new output directory to avoid overwriting them.')
    train_path=data/'train_cleaned.csv'; test_path=data/'test_cleaned.csv'
    if not train_path.is_file() or not test_path.is_file(): raise FileNotFoundError('Expected v1 merged train/test CSVs')
    test_stat_before=(test_path.stat().st_size,test_path.stat().st_mtime_ns)

    # Stratify labels first, with a fixed seed; no feature/test data participates.
    labels=pd.read_csv(train_path,usecols=['TransactionID','isFraud'])
    if labels.TransactionID.isna().any() or labels.TransactionID.duplicated().any(): raise ValueError('Invalid training TransactionID keys')
    if labels.isFraud.isna().any() or not set(labels.isFraud.unique()).issubset({0,1}): raise ValueError('Invalid isFraud labels')
    rng=np.random.default_rng(RANDOM_STATE); val_ids=set(); split_counts={}
    for cls,group in labels.groupby('isFraud',sort=True):
        ids=group.TransactionID.astype('int64').to_numpy()
        shuffled=rng.permutation(ids); n_val=int(round(len(ids)*VALIDATION_FRACTION))
        val_ids.update(map(int,shuffled[:n_val]))
        split_counts[str(int(cls))]={'original':int(len(ids)),'validation':n_val,'train_fit':int(len(ids)-n_val)}
    fit_ids=set(map(int,labels.TransactionID))-val_ids
    fitmask=~labels.TransactionID.astype('int64').isin(val_ids)
    fit_y=labels.loc[fitmask,'isFraud']; val_y=labels.loc[~fitmask,'isFraud']
    split_manifest={'random_state':RANDOM_STATE,'splitter':'class-wise numpy Generator permutation; 20% from each isFraud class to validation',
       'validation_fraction':VALIDATION_FRACTION,'fit_rows':len(fit_ids),'validation_rows':len(val_ids),
       'class_counts':split_counts,'fit_fraud_rate':float(fit_y.mean()),'validation_fraud_rate':float(val_y.mean()),
       'original_fraud_rate':float(labels.isFraud.mean()),'fit_transaction_id_sha256':id_digest(fit_ids),
       'validation_transaction_id_sha256':id_digest(val_ids),
       'test_input':str(test_path.resolve()),'test_input_size_bytes':test_stat_before[0],
       'test_input_mtime_ns_before':test_stat_before[1]}
    (out/'split_manifest.json').write_text(json.dumps(split_manifest,indent=2),encoding='utf-8')
    del labels,fitmask,fit_y,val_y

    first=pd.read_csv(train_path,nrows=1000,low_memory=False)
    feature_order=[c for c in first.columns if c not in ('TransactionID','isFraud')]
    test_header=pd.read_csv(test_path,nrows=0).columns.tolist()
    if [c for c in first.columns if c!='isFraud']!=test_header: raise ValueError('V1 train/test feature schemas are not aligned')
    numeric=[c for c in feature_order if is_numeric_dtype(first[c].dtype)]
    categorical=[c for c in feature_order if c not in numeric]

    # Fit every data-dependent transformation from the new training-fit partition only.
    vocab={c:set() for c in categorical}; median_frames=[]; median_n=0; fit_count=0; val_count=0
    for chunk in read_chunks(train_path):
        valmask=chunk.TransactionID.astype('int64').isin(val_ids); fitting=chunk.loc[~valmask]
        fit_count+=len(fitting); val_count+=int(valmask.sum())
        for c in categorical:
            vocab[c].update(fitting[c].astype('string').dropna().tolist())
        if median_n<MAX_MEDIAN_FIT_ROWS:
            sample=(fitting.TransactionID.astype('int64')%997)<210
            take=fitting.loc[sample,numeric].iloc[:MAX_MEDIAN_FIT_ROWS-median_n].copy()
            for c in numeric: take[c]=pd.to_numeric(take[c],errors='coerce').replace([np.inf,-np.inf],np.nan)
            median_frames.append(take); median_n+=len(take)
    if fit_count!=len(fit_ids) or val_count!=len(val_ids): raise ValueError('Split counts changed while fitting preprocessing')
    medsample=pd.concat(median_frames,ignore_index=True)
    medians={c:(float(medsample[c].median()) if medsample[c].notna().any() else 0.0) for c in numeric}
    categories={c:{v:i+2 for i,v in enumerate(sorted(vals))} for c,vals in vocab.items()}
    fitted={'fit_rows':fit_count,'validation_rows':val_count,'random_state':RANDOM_STATE,
      'numeric_medians':medians,'category_codes':categories,'numeric_features':numeric,
      'categorical_features':categorical,'feature_order':feature_order,
      'categorical_code_meaning':{'0':'missing','1':'unseen in training-fit partition','2+':'category observed in training-fit partition'},
      'fit_scope':'new stratified training-fit partition only; validation and test transform-only',
      'median_sampling':'deterministic training-fit rows with TransactionID % 997 < 210, capped at 100000 rows',
      'median_fit_sample_rows':median_n}
    (out/'fitted_preprocessing.json').write_text(json.dumps(fitted,indent=2),encoding='utf-8')

    paths={'train_fit':pp/'train_fit_encoded.csv','validation':pp/'validation_encoded.csv','test':pp/'test_encoded.csv'}
    handles={k:open(v,'w',encoding='utf-8',newline='') for k,v in paths.items()}
    firstwrite={k:True for k in paths}; rows={k:0 for k in paths}; missing={k:0 for k in paths}
    try:
        for split,path in [('train',train_path),('test',test_path)]:
            for ix,chunk in enumerate(read_chunks(path),start=1):
                if split=='train':
                    valmask=chunk.TransactionID.astype('int64').isin(val_ids)
                    groups=[('train_fit',chunk.loc[~valmask].copy()),('validation',chunk.loc[valmask].copy())]
                else: groups=[('test',chunk.copy())]
                for key,part in groups:
                    if not len(part): continue
                    for c in numeric:
                        part[c]=pd.to_numeric(part[c],errors='coerce').replace([np.inf,-np.inf],np.nan).fillna(medians[c])
                    for c in categorical:
                        vals=part[c].astype('string'); codes=vals.map(categories[c]).fillna(1).astype('int32')
                        codes.loc[vals.isna()]=0; part[c]=codes
                    part.to_csv(handles[key],index=False,header=firstwrite[key]); firstwrite[key]=False
                    rows[key]+=len(part); missing[key]+=int(part[feature_order].isna().sum().sum())
                if ix%20==0: print(f'{split}: chunk {ix}, rows={rows}',flush=True)
    finally:
        for h in handles.values(): h.close()
    expected_test=json.loads((data/'original_schema.json').read_text(encoding='utf-8'))['test_transaction.csv']['rows']
    if rows!={'train_fit':fit_count,'validation':val_count,'test':expected_test}: raise ValueError(f'Output row counts failed: {rows}')
    if any(missing.values()): raise ValueError(f'Unexpected missing transformed features: {missing}')
    if (test_path.stat().st_size,test_path.stat().st_mtime_ns)!=test_stat_before: raise ValueError('Input test dataset was modified')
    print(json.dumps({'rows':rows,'feature_count':len(feature_order),'class_counts':split_counts,'missing_feature_cells':missing,'test_input_unchanged':True},indent=2))

if __name__=='__main__': main()
