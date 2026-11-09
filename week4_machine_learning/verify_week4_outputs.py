"""Independent, read-only verification of Week 4 predictions and metrics."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from docx import Document
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

ROOT=Path(os.environ.get("IEEE_CIS_PROJECT_ROOT",Path(__file__).resolve().parent.parent)).resolve()
OUT=Path(__file__).resolve().parent
PRE=ROOT/"cleaned_data_v2"/"preprocessed"
FIT=PRE/"train_fit_encoded.csv"; VAL=PRE/"validation_encoded.csv"
M=json.loads((OUT/"week4_metrics.json").read_text(encoding="utf-8"))
P=pd.read_csv(OUT/"validation_predictions.csv")


def prob_col(name: str) -> str:
    bits=name.split()
    return "prob_"+bits[0].lower()+("_"+bits[1].lower() if len(bits)>1 else "")


def score(y: np.ndarray,p: np.ndarray,threshold: float) -> dict:
    yh=(p>=threshold).astype(int); cm=confusion_matrix(y,yh,labels=[0,1]); tn,fp,fn,tp=map(int,cm.ravel())
    return {"accuracy":float(accuracy_score(y,yh)),"precision":float(precision_score(y,yh,zero_division=0)),"recall":float(recall_score(y,yh,zero_division=0)),"f1":float(f1_score(y,yh,zero_division=0)),"roc_auc":float(roc_auc_score(y,p)),"pr_auc_average_precision":float(average_precision_score(y,p)),"confusion_matrix_labels_0_1":[[tn,fp],[fn,tp]],"true_negatives":tn,"false_positives":fp,"false_negatives":fn,"true_positives":tp}


def main() -> None:
    use=["TransactionID","isFraud",*M["feature_columns"]]
    val=pd.read_csv(VAL,usecols=use)
    fitids=pd.read_csv(FIT,usecols=["TransactionID"])["TransactionID"]
    checks={}
    checks["train_and_validation_counts_match_manifest"]=(len(fitids)==M["train_rows"] and len(val)==M["validation_rows"])
    checks["train_validation_keys_disjoint"]=not fitids.isin(val.TransactionID).any()
    checks["validation_prediction_keys_and_labels_match"]=(len(P)==len(val) and P.TransactionID.equals(val.TransactionID) and np.array_equal(P.isFraud.to_numpy(),val.isFraud.to_numpy()))
    checks["id_and_target_excluded_from_features"]=("TransactionID" not in M["feature_columns"] and "isFraud" not in M["feature_columns"] and "TransactionDT" not in M["feature_columns"])
    checks["train_only_preprocessing_documented"]=M["preprocessor_fit_partition"]=="train_fit only" and M["transform_fit_evidence"]["validation_operation"]=="transform only"
    y=val.isFraud.to_numpy(dtype=int)
    all_metric_match=True
    probability_match=True
    for name, rec in M["models"].items():
        col=prob_col(name)
        if col not in P.columns:
            all_metric_match=False; continue
        p=P[col].to_numpy(dtype=float)
        computed=score(y,p,M["threshold"])
        for k,actual in computed.items():
            stored=rec["validation"].get(k)
            if isinstance(actual,float):
                all_metric_match &= stored is not None and np.isclose(actual,stored,rtol=1e-10,atol=1e-12)
            else:
                all_metric_match &= stored==actual
        if name=="Prior baseline (Dummy)":
            probability_match &= np.allclose(p,np.repeat(M["train_fraud_rate"],len(y)),rtol=0,atol=1e-12)
        else:
            artifact_name="logistic_regression.joblib" if name.startswith("Logistic") else "hist_gradient_boosting.joblib"
            bundle=joblib.load(OUT/"models"/artifact_name)
            x=val[M["feature_columns"]].copy()
            for c in M["feature_columns"]:
                x[c]=x[c].astype("float32" if c in M["numeric_features"] else "int32")
            pv=bundle["model"].predict_proba(bundle["preprocessor"].transform(x))[:,1]
            probability_match &= np.allclose(p,pv,rtol=1e-7,atol=1e-9)
    checks["validation_metrics_recomputed_from_saved_predictions"]=bool(all_metric_match)
    checks["saved_model_artifacts_reproduce_validation_probabilities"]=bool(probability_match)
    chart_paths=list((OUT/"charts").glob("*.png"))
    doc=Document(OUT/"Week4_Machine_Learning_Model_Development_and_Evaluation.docx")
    checks["three_evaluation_charts_exist_and_are_readable"]=len(chart_paths)==3 and all(Image.open(p).size[0]>=1000 for p in chart_paths)
    checks["all_charts_embedded_in_docx"]=len(doc.inline_shapes)==3
    script=(OUT/"week4_machine_learning.py").read_text(encoding="utf-8")
    checks["workflow_does_not_reference_test_csvs"]=not any(s in script.lower() for s in ["test_transaction.csv","test_identity.csv","test_encoded.csv"])
    checks["no_test_data_used_as_model_input"]=M["checks"]["test_data_not_used"] is True
    checks["all_required_metrics_present"]=all(all(k in v["validation"] for k in ["precision","recall","f1","roc_auc","pr_auc_average_precision","accuracy"]) for v in M["models"].values())
    out={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"train_rows":M["train_rows"],"validation_rows":M["validation_rows"],"model_count":len(M["models"]),"validation_row_count_in_predictions":len(P),"report_image_count":len(doc.inline_shapes),"metrics_recomputed":True}
    (OUT/"verification.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
    if out["status"]!="PASS": raise SystemExit(1)


if __name__=="__main__": main()
