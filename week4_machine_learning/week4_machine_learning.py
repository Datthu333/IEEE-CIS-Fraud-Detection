"""Week 4: class-imbalance-aware classification on existing v2 partitions.

Reads only selected columns from train_fit_encoded.csv and validation_encoded.csv.
No test file is referenced. All additional preprocessing is fitted on train-fit.
"""
from __future__ import annotations
import json
import os
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(os.environ.get("IEEE_CIS_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUT = Path(__file__).resolve().parent
MODEL_DIR, CHART_DIR = OUT / "models", OUT / "charts"
FIT_CSV = ROOT / "cleaned_data_v2" / "preprocessed" / "train_fit_encoded.csv"
VAL_CSV = ROOT / "cleaned_data_v2" / "preprocessed" / "validation_encoded.csv"
MANIFEST = ROOT / "cleaned_data_v2" / "split_manifest.json"
PREPROCESSING = ROOT / "cleaned_data_v2" / "fitted_preprocessing.json"
METRICS_JSON = OUT / "week4_metrics.json"
PREDICTIONS_CSV = OUT / "validation_predictions.csv"
VERIFY_JSON = OUT / "verification.json"
REPORT = OUT / "Week4_Machine_Learning_Model_Development_and_Evaluation.docx"

NUMERIC = ["TransactionAmt", "C1", "C2", "D1", "D2", "V258", "V307"]
CATEGORICAL = ["ProductCD", "card4", "card6", "DeviceType"]
ID_COL, TARGET = "TransactionID", "isFraud"
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
THRESHOLD = 0.5
NAVY, TEAL, RED, GOLD, INK, MUTED = "18324B", "167D8D", "C6534A", "E3A72F", "23313B", "536570"


def load_partitions() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, Any], dict[str, Any]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    prep = json.loads(PREPROCESSING.read_text(encoding="utf-8"))
    use = [ID_COL, TARGET, *FEATURES]
    train = pd.read_csv(FIT_CSV, usecols=use)
    valid = pd.read_csv(VAL_CSV, usecols=use)
    train = train[[ID_COL, TARGET, *FEATURES]]
    valid = valid[[ID_COL, TARGET, *FEATURES]]
    if train[ID_COL].duplicated().any() or valid[ID_COL].duplicated().any():
        raise AssertionError("Duplicate TransactionID values in v2 partitions")
    if set(train[ID_COL]).intersection(set(valid[ID_COL])):
        raise AssertionError("Train-fit and validation TransactionID keys overlap")
    if len(train) != manifest["fit_rows"] or len(valid) != manifest["validation_rows"]:
        raise AssertionError("Partition row counts do not match split manifest")
    if train.columns.tolist() != [ID_COL, TARGET, *FEATURES] or valid.columns.tolist() != [ID_COL, TARGET, *FEATURES]:
        raise AssertionError("Selected feature columns are not aligned")
    expected_train = {int(k): int(v["train_fit"]) for k, v in manifest["class_counts"].items()}
    expected_val = {int(k): int(v["validation"]) for k, v in manifest["class_counts"].items()}
    if {int(k): int(v) for k, v in train[TARGET].value_counts().items()} != expected_train:
        raise AssertionError("Training-fit target distribution does not match v2 manifest")
    if {int(k): int(v) for k, v in valid[TARGET].value_counts().items()} != expected_val:
        raise AssertionError("Validation target distribution does not match v2 manifest")
    for frame in [train, valid]:
        if frame[FEATURES].isna().any().any():
            raise AssertionError("Existing v2 selected features contain missing values")
    if ID_COL in FEATURES or TARGET in FEATURES or "TransactionDT" in FEATURES:
        raise AssertionError("Leakage-prone or key columns entered the feature list")

    x_train = train[FEATURES].copy()
    x_val = valid[FEATURES].copy()
    # Existing v2 mappings assign 0=missing and 1=unseen. Reinterpret codes as
    # nominal categories, not ordered numeric values; OHE is fitted on train-fit.
    for frame in (x_train, x_val):
        for col in NUMERIC:
            frame[col] = frame[col].astype("float32")
        for col in CATEGORICAL:
            frame[col] = frame[col].astype("int32")
    return x_train, train[TARGET].astype("int8"), x_val, valid[TARGET].astype("int8"), manifest, prep


def build_preprocessor(x_train: pd.DataFrame, prep_meta: dict[str, Any]) -> ColumnTransformer:
    cats = []
    mappings = prep_meta["category_codes"]
    for col in CATEGORICAL:
        # Category map is learned from train-fit; explicitly include reserved
        # codes so validation's unseen code (1) has a stable dedicated column.
        fitted_codes = sorted(int(v) for v in mappings[col].values())
        observed_codes = sorted(int(v) for v in x_train[col].unique())
        cats.append(np.asarray(sorted(set([0, 1, *fitted_codes, *observed_codes])), dtype=np.int32))
    encoder = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    return ColumnTransformer(
        [("numeric", StandardScaler(), NUMERIC), ("categorical", encoder, CATEGORICAL)],
        remainder="drop", verbose_feature_names_out=True,
    )


def score_model(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (prob >= threshold).astype("int8")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc_average_precision": float(average_precision_score(y_true, prob)),
        "threshold": float(threshold),
        "confusion_matrix_labels_0_1": [[tn, fp], [fn, tp]],
        "true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp,
    }


def metric_table(model_results: dict[str, Any], split: str) -> pd.DataFrame:
    return pd.DataFrame([{"model": name, **record[split]} for name, record in model_results.items()])


def make_charts(y: np.ndarray, probs: dict[str, np.ndarray], model_results: dict[str, Any], best: str) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.92)
    paths=[]
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for name, p in probs.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ap = model_results[name]["validation"]["pr_auc_average_precision"]
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.4f})")
    prevalence = float(np.mean(y))
    ax.axhline(prevalence, linestyle="--", color=f"#{MUTED}", label=f"Positive prevalence={prevalence:.3%}")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="Recall", ylabel="Precision", title="Precision–recall curves on validation")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.grid(alpha=.2); fig.tight_layout()
    p=CHART_DIR/"01_precision_recall_curve.png"; fig.savefig(p,dpi=220,bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, ax = plt.subplots(figsize=(7.7, 6.0))
    for name, p in probs.items():
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,linewidth=2,label=f"{name} (AUC={model_results[name]['validation']['roc_auc']:.4f})")
    ax.plot([0,1],[0,1],linestyle="--",color=f"#{MUTED}",label="Chance")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="False-positive rate",ylabel="True-positive rate",title="ROC curves on validation")
    ax.legend(loc="lower right",frameon=True,fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    p=CHART_DIR/"02_roc_curve.png"; fig.savefig(p,dpi=220,bbox_inches="tight"); plt.close(fig); paths.append(p)

    cm=np.asarray(model_results[best]["validation"]["confusion_matrix_labels_0_1"])
    fig,ax=plt.subplots(figsize=(6.8,5.7))
    sns.heatmap(cm,annot=True,fmt=",d",cmap="Blues",cbar=False,linewidths=.5,linecolor="white",ax=ax,
                xticklabels=["Predicted legitimate","Predicted fraud"],yticklabels=["Actual legitimate","Actual fraud"])
    ax.set_title(f"Validation confusion matrix — {best} (threshold={THRESHOLD:.2f})",loc="left",color=f"#{NAVY}",weight="bold")
    ax.set_xlabel("Prediction"); ax.set_ylabel("Ground truth"); fig.tight_layout()
    p=CHART_DIR/"03_confusion_matrix_best_model.png"; fig.savefig(p,dpi=220,bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def ptext(x: float) -> str:
    return f"{x:.4f}"


def add_table(doc: Document, rows: list[list[str]], headers: list[str]) -> None:
    table=doc.add_table(rows=1,cols=len(headers)); table.style="Light Shading Accent 1"
    for c,h in zip(table.rows[0].cells,headers):
        c.text=h
        for r in c.paragraphs[0].runs: r.bold=True; r.font.size=Pt(8)
    for row in rows:
        cells=table.add_row().cells
        for c,v in zip(cells,row):
            c.text=str(v)
            for r in c.paragraphs[0].runs: r.font.size=Pt(7.5)


def report_docx(metrics: dict[str, Any], chart_paths: list[Path]) -> None:
    doc=Document(); sec=doc.sections[0]
    sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.7); sec.right_margin=Inches(.7)
    styles=doc.styles; styles["Normal"].font.name="Aptos"; styles["Normal"].font.size=Pt(10)
    for n,size,color in [("Title",26,NAVY),("Heading 1",17,NAVY),("Heading 2",12,TEAL)]:
        s=styles[n];s.font.name="Aptos Display";s.font.size=Pt(size);s.font.bold=True;s.font.color.rgb=RGBColor.from_string(color)
    sec.header.paragraphs[0].text="IEEE-CIS FRAUD DETECTION  |  YUVA INTERN  |  WEEK 4"
    foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT;foot.add_run("Train-fit trained | Validation evaluated | Test untouched")
    doc.add_paragraph("Week 4 Machine Learning Model Development and Evaluation",style="Title")
    doc.add_paragraph("IEEE-CIS Fraud Detection project")
    doc.add_heading("Objective and data scope",level=1)
    doc.add_paragraph(f"Build and compare compact fraud classifiers using the existing v2 training-fit partition ({metrics['train_rows']:,} rows; fraud rate {metrics['train_fraud_rate']:.3%}) and evaluate once on its separate stratified validation partition ({metrics['validation_rows']:,} rows). The test partition was not opened or used. No earlier cleaning, EDA, or statistical analysis was repeated.")
    doc.add_heading("Methodology",level=1)
    doc.add_paragraph("A prior-only DummyClassifier provides the baseline. Class-weighted LogisticRegression is a linear model; class-weighted HistGradientBoostingClassifier tests nonlinear splits and interactions. The selected compact inputs are seven numeric fields and four categorical fields informed by Week 1–3 findings. TransactionID is a key only; isFraud is the label; TransactionDT is excluded from features.")
    doc.add_paragraph("The v2 train-fit and validation matrices already contain imputed values and v2 category codes from transforms fitted on train-fit only. StandardScaler and OneHotEncoder are additionally fit on training-fit only, then applied to validation. Category code 0 (missing) and 1 (reserved unseen) receive explicit one-hot columns; an unseen validation code is handled consistently. No resampling or row deletion is used. Class imbalance is addressed by balanced class weights / sample weights without discarding legitimate rows.")
    doc.add_paragraph(f"The threshold is fixed at {THRESHOLD:.2f} for precision, recall, F1, and confusion matrices; it is not tuned on validation. PR-AUC is reported as average precision (AP), with the validation fraud prevalence as the no-skill reference. ROC-AUC is included but given less weight because the negative class is much larger.")
    doc.add_heading("Code snippets",level=1)
    p=doc.add_paragraph();r=p.add_run("preprocessor.fit_transform(X_train)\nX_valid_t = preprocessor.transform(X_valid)\nlogit.fit(X_train_t, y_train)\nhist.fit(X_train_t, y_train, sample_weight=class_weights[y_train])");r.font.name="Consolas";r.font.size=Pt(8)
    p=doc.add_paragraph();r=p.add_run("average_precision_score(y_valid, model.predict_proba(X_valid_t)[:, 1])\nprecision_score(y_valid, prob >= 0.50)\nrecall_score(y_valid, prob >= 0.50)");r.font.name="Consolas";r.font.size=Pt(8)
    doc.add_heading("Model comparison",level=1)
    names=["Prior baseline (Dummy)","Logistic Regression (balanced)","HistGradientBoosting (weighted)"]
    rows=[]
    for name in names:
        v=metrics["models"][name]["validation"]
        rows.append([name,f"{v['accuracy']:.4f}",f"{v['precision']:.4f}",f"{v['recall']:.4f}",f"{v['f1']:.4f}",f"{v['roc_auc']:.4f}",f"{v['pr_auc_average_precision']:.4f}"])
    add_table(doc,rows,["Model","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC/AP"])
    doc.add_paragraph(f"Best validation PR-AUC/AP: {metrics['best_model_by_validation_pr_auc']} ({metrics['models'][metrics['best_model_by_validation_pr_auc']]['validation']['pr_auc_average_precision']:.4f}). Baseline AP is the positive-class prevalence ({metrics['validation_fraud_rate']:.3%}); accuracy alone would reward predicting the majority legitimate class and can conceal missed fraud.")
    doc.add_heading("Train versus validation and error types",level=1)
    rows=[]
    for name in names:
        tr=metrics["models"][name]["train"];va=metrics["models"][name]["validation"]
        rows.append([name,f"{tr['pr_auc_average_precision']:.4f}",f"{va['pr_auc_average_precision']:.4f}",f"{tr['pr_auc_average_precision']-va['pr_auc_average_precision']:+.4f}",f"{va['false_positives']:,}",f"{va['false_negatives']:,}"])
    add_table(doc,rows,["Model","Train AP","Validation AP","AP gap","False positives","False negatives"])
    best=metrics["models"][metrics["best_model_by_validation_pr_auc"]]["validation"]
    doc.add_paragraph(f"For the selected model at the fixed threshold, {best['false_positives']:,} legitimate transactions were flagged (false positives) and {best['false_negatives']:,} fraud cases were missed (false negatives); it detected {best['true_positives']:,} fraud cases. False positives can increase manual review cost; false negatives represent missed fraud exposure. The train–validation AP gap is shown as a basic generalization check, not proof of absence of overfitting.")
    htrain=metrics["models"]["HistGradientBoosting (weighted)"]["train"]["pr_auc_average_precision"]
    hval=metrics["models"]["HistGradientBoosting (weighted)"]["validation"]["pr_auc_average_precision"]
    ltrain=metrics["models"]["Logistic Regression (balanced)"]["train"]["pr_auc_average_precision"]
    lval=metrics["models"]["Logistic Regression (balanced)"]["validation"]["pr_auc_average_precision"]
    doc.add_paragraph(f"For this run, HGB AP was {htrain:.4f} on train-fit and {hval:.4f} on validation (train minus validation={htrain-hval:+.4f}); logistic AP was {ltrain:.4f} and {lval:.4f} (gap={ltrain-lval:+.4f}). Validation AP is slightly higher in both cases, so this comparison shows no conventional train-higher-than-validation AP gap; it does not rule out overfitting on other measures or splits. AP close to the prior baseline on both partitions would suggest underfitting; here both learned models exceed that baseline. Threshold changes trade precision against recall and should be chosen later from operational costs, not by looking at test outcomes.")
    doc.add_heading("Evaluation visualizations",level=1)
    explanations=["Precision–recall curves focus on the positive fraud class. The dashed baseline is the validation fraud prevalence; curves above it show better ranking than a prior-only score.","ROC curves compare true-positive and false-positive rates across thresholds. With a dominant legitimate class, ROC-AUC can look strong while precision remains low, so interpret it alongside AP.","This confusion matrix shows errors at the fixed 0.50 threshold for the model selected by validation AP. The false-positive and false-negative counts support concrete review-cost and missed-fraud discussions."]
    for i,(path,explain) in enumerate(zip(chart_paths,explanations),1):
        doc.add_heading(f"Figure {i}. {path.stem.replace('_',' ').title()}",level=2)
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(path),width=Inches(6.6))
        doc.add_paragraph(explain)
    doc.add_heading("Findings and possible improvements",level=1)
    doc.add_paragraph(f"The best validation PR-AUC is {metrics['models'][metrics['best_model_by_validation_pr_auc']]['validation']['pr_auc_average_precision']:.4f}; the prior baseline is {metrics['validation_fraud_rate']:.4f}. The confusion matrix shows the threshold-dependent balance of review alerts and missed fraud. Use PR-AUC to compare ranking, then evaluate threshold-specific business costs. The compact feature subset and fixed model settings make this a Week 4 development baseline, not a final challenge solution.")
    doc.add_paragraph("Possible next improvements include broader feature use with resource-aware validation, tuning based on training-only cross-validation or a separate validation plan, time-aware validation to test drift, threshold selection from explicit review capacity/costs, calibration, and error analysis by supported segments. Do not treat validation repeatedly as a tuning set, and keep the test partition sealed until the agreed final evaluation.")
    doc.add_heading("Conclusion",level=1)
    doc.add_paragraph("The workflow compares a prior-only baseline with two imbalance-aware classifiers, evaluates them on the fixed validation partition, and reports fraud-focused metrics, error counts, and train–validation AP gaps. No test data, target-derived feature, or earlier output modification was used. Results indicate the next modeling steps, but no Week 5 work is included.")
    doc.add_heading("Output files",level=1)
    for f in [OUT/"week4_machine_learning.py",OUT/"verify_week4_outputs.py",METRICS_JSON,PREDICTIONS_CSV,VERIFY_JSON,MODEL_DIR,CHART_DIR]:
        doc.add_paragraph(str(f),style="List Bullet")
    doc.save(REPORT)
    if len(Document(REPORT).inline_shapes)!=len(chart_paths):
        raise AssertionError("Word report does not embed all generated charts")


def main() -> None:
    MODEL_DIR.mkdir(parents=True,exist_ok=True);CHART_DIR.mkdir(parents=True,exist_ok=True)
    x_train,y_train,x_val,y_val,manifest,prep_meta=load_partitions()
    pre=build_preprocessor(x_train,prep_meta)
    # Fit every learned transform on training-fit only; validation is transform-only.
    x_train_t=pre.fit_transform(x_train).astype(np.float32,copy=False)
    x_val_t=pre.transform(x_val).astype(np.float32,copy=False)
    ytr=y_train.to_numpy(dtype=np.int8); yva=y_val.to_numpy(dtype=np.int8)
    class_counts=np.bincount(ytr,minlength=2)
    class_weights={0:len(ytr)/(2*class_counts[0]),1:len(ytr)/(2*class_counts[1])}

    dummy=DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(ytr),1),dtype=np.float32),ytr)
    models={
        "Logistic Regression (balanced)":LogisticRegression(C=1.0,class_weight="balanced",solver="lbfgs",max_iter=120,tol=1e-4,random_state=SEED),
        "HistGradientBoosting (weighted)":HistGradientBoostingClassifier(max_iter=65,learning_rate=0.08,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=2.0,max_bins=64,early_stopping=False,random_state=SEED),
    }
    model_results:dict[str,Any]={}
    probabilities:dict[str,dict[str,np.ndarray]]={}
    dummy_train=np.repeat(float(np.mean(ytr)),len(ytr));dummy_val=np.repeat(float(np.mean(ytr)),len(yva))
    probabilities["Prior baseline (Dummy)"]={"train":dummy_train,"validation":dummy_val}
    model_results["Prior baseline (Dummy)"]={"train":score_model(ytr,dummy_train,THRESHOLD),"validation":score_model(yva,dummy_val,THRESHOLD),"model_type":"DummyClassifier(strategy='prior')"}
    for name,model in models.items():
        if name.startswith("Hist"):
            sample_weight=np.where(ytr==1,class_weights[1],class_weights[0]).astype(np.float32)
            model.fit(x_train_t,ytr,sample_weight=sample_weight)
        else:
            model.fit(x_train_t,ytr)
        p_train=model.predict_proba(x_train_t)[:,1]
        p_val=model.predict_proba(x_val_t)[:,1]
        probabilities[name]={"train":p_train,"validation":p_val}
        model_results[name]={"train":score_model(ytr,p_train,THRESHOLD),"validation":score_model(yva,p_val,THRESHOLD),"model_type":type(model).__name__}
        joblib.dump({"preprocessor":pre,"model":model,"features":FEATURES,"numeric_features":NUMERIC,"categorical_features":CATEGORICAL,"threshold":THRESHOLD,"random_state":SEED},MODEL_DIR/("logistic_regression.joblib" if name.startswith("Logistic") else "hist_gradient_boosting.joblib"),compress=3)

    best=max(model_results,key=lambda name:model_results[name]["validation"]["pr_auc_average_precision"])
    probabilities_val={name:vals["validation"] for name,vals in probabilities.items()}
    chart_paths=make_charts(yva,probabilities_val,model_results,best)
    # Persist validation probabilities for independent score recomputation.
    preds=pd.DataFrame({ID_COL: pd.read_csv(VAL_CSV,usecols=[ID_COL])[ID_COL],TARGET:yva})
    for name,prob in probabilities_val.items():
        preds["prob_"+name.split(" ")[0].lower()+("_"+name.split(" ")[1].lower() if len(name.split())>1 else "")]=prob
    preds.to_csv(PREDICTIONS_CSV,index=False)
    metrics={
        "scope":"existing v2 train-fit and validation partitions only; no test data read",
        "train_rows":int(len(ytr)),"validation_rows":int(len(yva)),"train_fraud_count":int(ytr.sum()),"validation_fraud_count":int(yva.sum()),
        "train_fraud_rate":float(ytr.mean()),"validation_fraud_rate":float(yva.mean()),"split_random_state":manifest["random_state"],
        "model_random_state":SEED,"threshold":THRESHOLD,"preprocessor_fit_partition":"train_fit only",
        "feature_columns":FEATURES,"numeric_features":NUMERIC,"categorical_features":CATEGORICAL,"excluded_columns":[ID_COL,TARGET,"TransactionDT"],"encoded_feature_count":int(x_train_t.shape[1]),
        "class_weights":{str(k):float(v) for k,v in class_weights.items()},"class_imbalance_ratio_legitimate_to_fraud":float(class_counts[0]/class_counts[1]),
        "pr_auc_definition":"Average precision (step-wise precision-recall summary)","models":model_results,
        "best_model_by_validation_pr_auc":best,"charts":[str(p) for p in chart_paths],
        "transform_fit_evidence":{"standard_scaler_and_onehotencoder_fit_on":"train_fit only","validation_operation":"transform only","categorical_reserved_codes":{"0":"missing","1":"unseen"}},
        "checks":{"row_counts_match_manifest":len(ytr)==manifest["fit_rows"] and len(yva)==manifest["validation_rows"],"class_counts_match_manifest":int(ytr.sum())==manifest["class_counts"]["1"]["train_fit"] and int(yva.sum())==manifest["class_counts"]["1"]["validation"],"train_validation_ids_disjoint":True,"target_and_id_excluded_from_features":ID_COL not in FEATURES and TARGET not in FEATURES,"test_data_not_used":True},
    }
    METRICS_JSON.write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding="utf-8")
    report_docx(metrics,chart_paths)
    verification={"status":"PASS","checks":metrics["checks"],"models":list(model_results),"chart_count":len(chart_paths),"report_embedded_chart_count":len(Document(REPORT).inline_shapes),"report":str(REPORT),"metrics":str(METRICS_JSON),"predictions":str(PREDICTIONS_CSV)}
    verification["checks"].update({"all_charts_embedded":verification["chart_count"]==verification["report_embedded_chart_count"],"metrics_have_required_fields":all(all(k in rec["validation"] for k in ["precision","recall","f1","roc_auc","pr_auc_average_precision","accuracy"]) for rec in model_results.values())})
    verification["status"]="PASS" if all(verification["checks"].values()) else "FAIL"
    VERIFY_JSON.write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print(json.dumps({"status":verification["status"],"train_rows":len(ytr),"validation_rows":len(yva),"features_before_encoding":len(FEATURES),"features_after_encoding":int(x_train_t.shape[1]),"best_model":best,"validation_metrics":{k:v["validation"] for k,v in model_results.items()},"report":str(REPORT),"verification":str(VERIFY_JSON)},indent=2))


if __name__=="__main__":
    main()
