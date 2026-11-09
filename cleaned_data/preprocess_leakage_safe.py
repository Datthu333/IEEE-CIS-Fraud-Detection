"""Leakage-safe, model-free imputation and encoding for IEEE-CIS clean tables.

The script creates a deterministic 80/20 train/validation split. Numeric medians
are estimated from at most 100,000 rows in the training-fit partition to bound
memory; categorical vocabularies are learned from all rows in that partition.
Both are applied unchanged to validation and test. No model is trained.

String features receive stable integer category codes (0=missing, 1=unseen).
Numeric fields receive training-fit median imputation. For a future model, choose
an estimator appropriate for integer-coded categorical features or replace this
encoder with a one-hot/native categorical encoder fitted at the same boundary.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

CHUNK = 5000
MAX_MEDIAN_FIT_ROWS = 100000
MISSING_TOKEN = "\u0000MISSING"

def read_chunks(path):
    return pd.read_csv(path, chunksize=CHUNK, low_memory=False)

def split_mask(ids):
    return (ids.astype("int64") % 5) != 0  # True = fit partition, False = validation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "preprocessed"))
    args = ap.parse_args(); data=Path(args.data_dir); out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train_path=data/"train_cleaned.csv"; test_path=data/"test_cleaned.csv"
    first=pd.read_csv(train_path,nrows=1000,low_memory=False)
    features=[c for c in first.columns if c not in {"TransactionID","isFraud"}]
    if "isFraud" not in first: raise ValueError("Training target is missing")
    test_cols=pd.read_csv(test_path,nrows=1).columns.tolist()
    if [c for c in first.columns if c!="isFraud"] != test_cols: raise ValueError("Train/test feature schemas are not aligned")
    numeric=[c for c in features if is_numeric_dtype(first[c].dtype)]
    categorical=[c for c in features if c not in numeric]

    # Fit vocabularies on the complete fit partition; fit numeric medians on a
# deterministic, spread-across-ID sample from that same partition for memory safety.
    vocab={c:set() for c in categorical}; median_frames=[]; med_n=0; fit_rows=0; val_rows=0
    for chunk in read_chunks(train_path):
        fm=split_mask(chunk.TransactionID); fit_rows+=int(fm.sum()); val_rows+=int((~fm).sum())
        fit=chunk.loc[fm,features]
        for c in categorical:
            vals=fit[c].astype("string")
            vocab[c].update(vals.dropna().tolist())
        if med_n < MAX_MEDIAN_FIT_ROWS and len(fit):
            sample_mask=(chunk.loc[fm,"TransactionID"].astype("int64") % 997) < 210
            take=fit.loc[sample_mask,numeric].iloc[:MAX_MEDIAN_FIT_ROWS-med_n].copy()
            for c in numeric: take[c]=pd.to_numeric(take[c],errors="coerce").replace([np.inf,-np.inf],np.nan)
            median_frames.append(take); med_n+=len(take)
    expected_train=json.loads((data/"original_schema.json").read_text(encoding="utf-8"))["train_transaction.csv"]["rows"]
    expected_test=json.loads((data/"original_schema.json").read_text(encoding="utf-8"))["test_transaction.csv"]["rows"]
    if fit_rows+val_rows != expected_train:
        raise ValueError("Train split row counts do not reconcile to the original schema")
    med_sample=pd.concat(median_frames,ignore_index=True)
    medians={c: (float(med_sample[c].median()) if med_sample[c].notna().any() else 0.0) for c in numeric}
    categories={c:{v:i+2 for i,v in enumerate(sorted(values))} for c,values in vocab.items()}
    (out/"fitted_preprocessing.json").write_text(json.dumps({"fit_rows":fit_rows,"validation_rows":val_rows,
        "numeric_medians":medians,"category_codes":categories,"numeric_features":numeric,
        "categorical_features":categorical,"feature_order":features,
        "categorical_code_meaning":{"0":"missing","1":"unseen in training fit partition","2+":"category observed in training fit partition"},
        "fit_boundary":"TransactionID % 5 != 0 on training rows only",
        "median_sampling":"Among fit rows, TransactionID % 997 < 210, capped at 100000 rows; this deterministic sample is spread across the ID range",
        "median_fit_sample_rows":med_n},indent=2),encoding="utf-8")

    paths={"train_fit":out/"train_fit_encoded.csv","validation":out/"validation_encoded.csv","test":out/"test_encoded.csv"}
    handles={k:open(v,"wt",encoding="utf-8",newline="") for k,v in paths.items()}; first_write={k:True for k in paths}; missing_counts={k:0 for k in paths}; row_counts={k:0 for k in paths}
    try:
        for source_name, path in [("train",train_path),("test",test_path)]:
            for chunk_index, chunk in enumerate(read_chunks(path), start=1):
                if source_name=="train":
                    fitmask=split_mask(chunk.TransactionID)
                    groups=[("train_fit",chunk.loc[fitmask].copy()),("validation",chunk.loc[~fitmask].copy())]
                else: groups=[("test",chunk.copy())]
                for key, part in groups:
                    if not len(part): continue
                    for c in numeric:
                        part[c]=pd.to_numeric(part[c],errors="coerce").replace([np.inf,-np.inf],np.nan).fillna(medians[c])
                    for c in categorical:
                        vals=part[c].astype("string")
                        codes=vals.map(categories[c]).fillna(1).astype("int32")
                        codes.loc[vals.isna()]=0
                        part[c]=codes
                    part.to_csv(handles[key],index=False,header=first_write[key])
                    first_write[key]=False; row_counts[key]+=len(part); missing_counts[key]+=int(part[features].isna().sum().sum())
                if chunk_index % 20 == 0: print(f"Transformed {source_name} chunk {chunk_index}; rows so far {row_counts}", flush=True)
    finally:
        for h in handles.values(): h.close()
    if row_counts["train_fit"] != fit_rows or row_counts["validation"] != val_rows:
        raise ValueError("Transformed training and validation rows are incomplete")
    if row_counts["test"] != expected_test: raise ValueError("Transformed test rows do not reconcile to the original schema")
    results={"rows":row_counts,"features":len(features),"train_features_align_test":True,
        "test_target_absent":True,"post_transform_missing_feature_cells":missing_counts,
        "validation_split":"TransactionID modulo 5 equals zero","model_trained":False,
        "fit_scope":"training fit partition only; validation and test were transform-only"}
    (out/"preprocessing_validation.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
