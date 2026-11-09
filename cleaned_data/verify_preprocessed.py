import pandas as pd,json
from pathlib import Path
root=Path(__file__).resolve().parent; out=root/'preprocessed'
files={k:out/f'{n}.csv' for k,n in [('train_fit','train_fit_encoded'),('validation','validation_encoded'),('test','test_encoded')]}
counts={k:0 for k in files}; missing={k:0 for k in files}; columns={}
for k,p in files.items():
    cols=pd.read_csv(p,nrows=1).columns.tolist(); columns[k]=cols
    for ch in pd.read_csv(p,chunksize=30000,low_memory=False):
        counts[k]+=len(ch)
        feat=[c for c in ch.columns if c not in ('TransactionID','isFraud')]
        missing[k]+=int(ch[feat].isna().sum().sum())
expected=json.loads((root/'original_schema.json').read_text())
train_cols=[c for c in columns['train_fit'] if c!='isFraud']
assert train_cols==columns['test']
assert columns['validation']==columns['train_fit']
assert counts['train_fit']+counts['validation']==expected['train_transaction.csv']['rows']
assert counts['test']==expected['test_transaction.csv']['rows']
assert 'isFraud' not in columns['test']
assert missing=={'train_fit':0,'validation':0,'test':0}
pre=json.loads((out/'fitted_preprocessing.json').read_text())
result={'rows':counts,'feature_count':len(columns['test'])-1,'features_align':True,'test_target_absent':True,'post_transform_missing_feature_cells':missing,'fit_rows_in_artifact':pre['fit_rows'],'validation_rows_in_artifact':pre['validation_rows'],'median_fit_sample_rows':pre['median_fit_sample_rows'],'model_trained':False,'fit_scope':'training fit partition only; validation and test transform only'}
(out/'preprocessing_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

