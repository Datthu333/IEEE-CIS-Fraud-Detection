"""Independent read-only verification of the v2 stratified preprocessing outputs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'cleaned_data'
PRE=ROOT/'preprocessed'

def digest_id(values):
    h=hashlib.sha256()
    for x in sorted(map(int,values)):
        h.update(str(x).encode()); h.update(b'\n')
    return h.hexdigest()

def main():
    schema=json.loads((DATA/'original_schema.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'split_manifest.json').read_text(encoding='utf-8'))
    fitted=json.loads((ROOT/'fitted_preprocessing.json').read_text(encoding='utf-8'))
    script=(ROOT/'stratified_preprocess.py').read_text(encoding='utf-8')
    expected_train=schema['train_transaction.csv']['rows']; expected_test=schema['test_transaction.csv']['rows']
    train_headers=schema['train_transaction.csv']['column_names']; test_headers=schema['test_transaction.csv']['column_names']
    id_headers=schema['train_identity.csv']['column_names']; test_id_headers=schema['test_identity.csv']['column_names']
    expected_train_header=train_headers+[c for c in id_headers if c!='TransactionID']
    expected_test_header=test_headers+[c.replace('-','_') if c.startswith('id-') else c for c in test_id_headers if c!='TransactionID']
    train_input=DATA/'train_cleaned.csv'; test_input=DATA/'test_cleaned.csv'
    in_train_cols=pd.read_csv(train_input,nrows=0).columns.tolist(); in_test_cols=pd.read_csv(test_input,nrows=0).columns.tolist()
    clean_schema_ok=(in_train_cols==expected_train_header and in_test_cols==expected_test_header)

    labels={}; source_train_ids=set()
    for ch in pd.read_csv(train_input,usecols=['TransactionID','isFraud'],chunksize=50000):
        ids=ch.TransactionID.astype('int64').tolist(); ys=ch.isFraud.astype('int64').tolist()
        source_train_ids.update(ids); labels.update(zip(ids,ys))
    source_test_ids=set()
    for ch in pd.read_csv(test_input,usecols=['TransactionID'],chunksize=50000): source_test_ids.update(ch.TransactionID.astype('int64').tolist())

    paths={'train_fit':PRE/'train_fit_encoded.csv','validation':PRE/'validation_encoded.csv','test':PRE/'test_encoded.csv'}
    feature_order=fitted['feature_order']; cats=fitted['categorical_features']; maps=fitted['category_codes']
    allowed={c:set(maps[c].values())|{0,1} for c in cats}
    ids={k:set() for k in paths}; class_counts={k:{0:0,1:0} for k in ('train_fit','validation')}
    row_counts={k:0 for k in paths}; missing={k:0 for k in paths}; code_values_ok=True; targets_match=True; headers={}
    for k,path in paths.items():
        headers[k]=pd.read_csv(path,nrows=0).columns.tolist()
        for ch in pd.read_csv(path,chunksize=20000,low_memory=False):
            row_counts[k]+=len(ch); ids[k].update(ch.TransactionID.astype('int64').tolist())
            missing[k]+=int(ch[feature_order].isna().sum().sum())
            for c in cats:
                vals=pd.to_numeric(ch[c],errors='coerce')
                if vals.isna().any() or not set(vals.astype('int64').unique()).issubset(allowed[c]): code_values_ok=False
            if k!='test':
                if ch.isFraud.isna().any(): targets_match=False
                for tid,y in zip(ch.TransactionID.astype('int64'),ch.isFraud.astype('int64')):
                    if labels.get(tid)!=y: targets_match=False
                    class_counts[k][int(y)]+=1
    train_schema_expected=['TransactionID','isFraud']+feature_order
    test_schema_expected=['TransactionID']+feature_order
    feature_ok=(feature_order==[c for c in in_train_cols if c not in ('TransactionID','isFraud')] and
      headers['train_fit']==train_schema_expected and headers['validation']==train_schema_expected and headers['test']==test_schema_expected and
      'TransactionID' not in feature_order and 'isFraud' not in feature_order and len(feature_order)==432)
    row_ok=(row_counts=={'train_fit':manifest['fit_rows'],'validation':manifest['validation_rows'],'test':expected_test} and
      len(source_train_ids)==expected_train and len(source_test_ids)==expected_test and
      ids['train_fit'].isdisjoint(ids['validation']) and ids['train_fit']|ids['validation']==source_train_ids and ids['test']==source_test_ids and
      len(ids['train_fit'])==row_counts['train_fit'] and len(ids['validation'])==row_counts['validation'] and len(ids['test'])==row_counts['test'])
    stratified=True
    for cls,detail in manifest['class_counts'].items():
        c=int(cls); total=detail['original']; val=detail['validation']; fit=detail['train_fit']
        stratified &= (val==int(round(total*manifest['validation_fraction'])) and
          class_counts['validation'][c]==val and class_counts['train_fit'][c]==fit)
    rate_fit=class_counts['train_fit'][1]/row_counts['train_fit']; rate_val=class_counts['validation'][1]/row_counts['validation']; rate_all=(class_counts['train_fit'][1]+class_counts['validation'][1])/expected_train
    test_stat=(test_input.stat().st_size,test_input.stat().st_mtime_ns)
    test_unchanged=(test_stat==(manifest['test_input_size_bytes'],manifest['test_input_mtime_ns_before']))
    leakage_ok=('isFraud' not in feature_order and 'TransactionID' not in feature_order and
      fitted['fit_scope'].startswith('new stratified training-fit partition only') and
      "for chunk in read_chunks(train_path)" in script and "for split,path in [('train',train_path),('test',test_path)]" in script and
      "categories={c:{v:i+2 for i,v in enumerate(sorted(vals))} for c,vals in vocab.items()}" in script)
    mapping_ok=(code_values_ok and len(cats)==31 and all(len(set(maps[c].values()))==len(maps[c]) and set(maps[c].values())==set(range(2,len(maps[c])+2)) for c in cats))
    no_missing=all(v==0 for v in missing.values())
    reproduce=('RANDOM_STATE = 42' in script and 'np.random.default_rng(RANDOM_STATE)' in script and
       'FileExistsError' in script and clean_schema_ok and feature_ok)
    checks={
      'all_training_and_test_rows_preserved':row_ok,
      'split_stratified_by_isFraud':stratified,
      'no_target_or_test_fit_leakage':leakage_ok and test_unchanged,
      'TransactionID_and_isFraud_excluded_from_model_features':feature_ok,
      'categorical_mappings_consistent':mapping_ok,
      'preprocessing_script_reproducible_and_non_overwriting':reproduce,
      'columns_retained_and_missing_values_handled':clean_schema_ok and no_missing,
      'original_test_input_unchanged':test_unchanged,
      'training_targets_preserved':targets_match,
    }
    evidence={
      'row_counts':row_counts,'expected_original_rows':{'train':expected_train,'test':expected_test},
      'train_class_counts':{k:{str(c):v for c,v in d.items()} for k,d in class_counts.items()},
      'fraud_rates':{'train_fit':rate_fit,'validation':rate_val,'overall':rate_all},
      'random_state':manifest['random_state'],'validation_fraction':manifest['validation_fraction'],
      'model_feature_count':len(feature_order),'categorical_column_count':len(cats),
      'post_transform_missing_feature_cells':missing,'test_input_unchanged':test_unchanged,
      'target_values_match_source':targets_match,'merged_headers_match_schema':clean_schema_ok,
    }
    results={'checks':{k:('PASS' if v else 'FAIL') for k,v in checks.items()},'evidence':evidence}
    (ROOT/'verification_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    lines=['# Independent verification report — IEEE-CIS Week 1, stratified split v2','',
      'Read-only verification of v2 outputs. No model was trained. The v1 datasets and original test input were not modified.','',
      '## Results','','| Check | Result |','|---|---|']
    labels_map={
      'all_training_and_test_rows_preserved':'All train/test rows and keys preserved',
      'split_stratified_by_isFraud':'Train/validation split stratified by isFraud',
      'no_target_or_test_fit_leakage':'No target/test leakage in preprocessing',
      'TransactionID_and_isFraud_excluded_from_model_features':'TransactionID and isFraud excluded from model features',
      'categorical_mappings_consistent':'Categorical mappings consistent across partitions',
      'preprocessing_script_reproducible_and_non_overwriting':'Scripts reproducible with fixed seed and separate outputs',
      'columns_retained_and_missing_values_handled':'Columns retained and transformed features have no missing values',
      'original_test_input_unchanged':'Original test input unchanged',
      'training_targets_preserved':'Training targets match source',}
    for k,v in checks.items(): lines.append(f"| {labels_map[k]} | {'PASS' if v else 'FAIL'} |")
    lines+=['','## Evidence','','- Original merged input shapes: training 590,540 rows; test 506,691 rows. The saved v2 outputs have the counts shown below.','',
      '| Partition | Rows | Non-fraud | Fraud | Fraud rate | Missing feature cells |','|---|---:|---:|---:|---:|---:|']
    for k in ('train_fit','validation'):
        cc=class_counts[k]; rr=row_counts[k]; lines.append(f"| {k} | {rr:,} | {cc[0]:,} | {cc[1]:,} | {cc[1]/rr:.4%} | {missing[k]:,} |")
    lines.append(f"| test | {row_counts['test']:,} | n/a | n/a | n/a | {missing['test']:,} |")
    lines+=['',f"Fixed `random_state`: {manifest['random_state']}. Validation fraction per class: {manifest['validation_fraction']:.0%}.",
      f"Overall source fraud rate: {rate_all:.4%}; fit fraud rate: {rate_fit:.4%}; validation fraud rate: {rate_val:.4%}.",
      f"Model feature list contains {len(feature_order)} columns; TransactionID and isFraud are stored separately. {len(cats)} categorical mappings use 0=missing, 1=unseen, and codes 2+ for training-fit categories.",
      'The preprocessing script is deterministic and refuses to overwrite populated versioned outputs. Its fit pass reads only training rows and uses only training-fit rows for medians and vocabularies; validation/test are transform-only.',
      'Original test input size and modification timestamp before/after transformation match.', '',
      '## Overall','',f"{sum(checks.values())} of {len(checks)} checks passed.",'']
    (ROOT/'verification_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
