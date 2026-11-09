"""Independent verification of Week 3 statistical outputs (read-only)."""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from docx import Document

ROOT = Path(os.environ.get("IEEE_CIS_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUT = Path(__file__).resolve().parent
RESULTS = json.loads((OUT / "week3_results.json").read_text(encoding="utf-8"))
MANIFEST = json.loads((ROOT / "cleaned_data_v2" / "split_manifest.json").read_text(encoding="utf-8"))
FEATURES = ["TransactionID", "isFraud", "TransactionAmt", "ProductCD", "card6", "V258"]


def p_holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda v: v[1])
    out, running, m = {}, 0.0, len(ordered)
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, (m-i)*p))
        out[name] = running
    return out


def main() -> None:
    fit_keys = pd.read_csv(ROOT / "cleaned_data_v2" / "preprocessed" / "train_fit_encoded.csv", usecols=["TransactionID", "isFraud"])
    source = pd.read_csv(ROOT / "cleaned_data" / "train_cleaned.csv", usecols=FEATURES)
    assert fit_keys.TransactionID.is_unique and source.TransactionID.is_unique
    data = source.loc[source.TransactionID.isin(set(fit_keys.TransactionID))].merge(fit_keys, on="TransactionID", validate="one_to_one", suffixes=("_source", "_fit"))
    assert len(data) == MANIFEST["fit_rows"] == RESULTS["fit_rows"]
    assert np.array_equal(data.isFraud_source.to_numpy(), data.isFraud_fit.to_numpy())
    target=data.isFraud_fit.to_numpy(dtype=int)
    fraud=data.loc[target==1]; legit=data.loc[target==0]
    p_recomputed={}
    mw=stats.mannwhitneyu(fraud.TransactionAmt.to_numpy(),legit.TransactionAmt.to_numpy(),alternative="two-sided",method="asymptotic")
    p_recomputed["amount_mann_whitney"]=float(mw.pvalue)
    welch=stats.ttest_ind(np.log1p(fraud.TransactionAmt),np.log1p(legit.TransactionAmt),equal_var=False)
    p_recomputed["log_amount_welch"]=float(welch.pvalue)
    for col,key in [("ProductCD","productcd_chi_square"),("card6","card6_chi_square")]:
        vals=data[col].astype("string").fillna("Missing")
        table=pd.crosstab(vals,data.isFraud_fit).reindex(columns=[0,1],fill_value=0)
        chi=stats.chi2_contingency(table.to_numpy(),correction=False)
        if key=="card6_chi_square" and not (chi.expected_freq>=5).all():
            mc=stats.random_table.rvs(table.to_numpy().sum(axis=1),table.to_numpy().sum(axis=0),size=20000,random_state=np.random.default_rng(4321))
            sim=np.sum((mc-chi.expected_freq)**2/chi.expected_freq,axis=(1,2))
            p_recomputed[key]=float((np.count_nonzero(sim>=chi.statistic)+1)/20001)
        else:
            p_recomputed[key]=float(chi.pvalue)
    miss_table=pd.crosstab(data.isFraud_fit,data.V258.isna()).reindex(index=[0,1],columns=[False,True],fill_value=0)
    p_recomputed["v258_missingness_chi_square"]=float(stats.chi2_contingency(miss_table.to_numpy(),correction=False).pvalue)
    observed=data.loc[data.V258.notna()]
    vmw=stats.mannwhitneyu(observed.loc[observed.isFraud_fit.eq(1),"V258"],observed.loc[observed.isFraud_fit.eq(0),"V258"],alternative="two-sided",method="asymptotic")
    p_recomputed["v258_mann_whitney"]=float(vmw.pvalue)
    saved_tests=RESULTS["tests"]
    checks={
        "fit_row_and_class_counts_match_manifest": len(data)==MANIFEST["fit_rows"] and int(target.sum())==MANIFEST["class_counts"]["1"]["train_fit"] and int((target==0).sum())==MANIFEST["class_counts"]["0"]["train_fit"],
        "fit_ids_unique_and_target_matches": fit_keys.TransactionID.is_unique and np.array_equal(data.isFraud_source.to_numpy(),data.isFraud_fit.to_numpy()),
        "all_six_test_pvalues_recomputed": all(np.isclose(p_recomputed[k],saved_tests[k]["p_value"],rtol=1e-10,atol=1e-14) for k in p_recomputed),
        "holm_values_recomputed": all(np.isclose(v,saved_tests[k]["p_value_holm"],rtol=1e-10,atol=1e-14) for k,v in p_holm(p_recomputed).items()),
        "underflow_pvalues_are_labeled_in_results": all(t.get("p_value_display")=="<1e-300 (floating-point underflow)" for t in saved_tests.values() if t["p_value"]==0.0),
        "regular_chi_square_expected_counts_ge_5": all(saved_tests[k]["expected_all_at_least_5"] and saved_tests[k]["expected_min"]>=5 for k in ["productcd_chi_square","v258_missingness_chi_square"]),
        "card6_sparse_expected_counts_use_conditional_monte_carlo": saved_tests["card6_chi_square"]["expected_all_at_least_5"] or saved_tests["card6_chi_square"].get("monte_carlo",{}).get("replicates")==20000,
        "effect_sizes_and_intervals_valid": all(np.isfinite(t["effect"]["estimate"]) and len(t["effect"].get("ci_95",t["effect"].get("ci_95_approx",t["effect"].get("ci_95_bootstrap",[]))))==2 for t in saved_tests.values()),
        "group_summaries_reconcile_to_fit_rows": sum(int(r["n"]) for r in RESULTS["group_summaries"] if r["feature"]=="ProductCD")==len(data) and sum(int(r["n"]) for r in RESULTS["group_summaries"] if r["feature"]=="card6")==len(data),
        "four_pngs_exist_and_are_readable": len(list((OUT/"charts").glob("*.png")))==4 and all(Image.open(p).size[0]>=1000 for p in (OUT/"charts").glob("*.png")),
        "all_four_pngs_embedded_in_report": len(Document(OUT/"Week3_Statistical_Analysis_and_Hypothesis_Testing.docx").inline_shapes)==4,
        "no_test_source_access_in_workflow": "test_transaction.csv" not in (OUT/"week3_statistics.py").read_text(encoding="utf-8") and "test_identity.csv" not in (OUT/"week3_statistics.py").read_text(encoding="utf-8"),
    }
    output={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"fit_rows":len(data),"fraud_rows":int(target.sum()),"legitimate_rows":int((target==0).sum()),"recomputed_p_values":p_recomputed,"report_embedded_images":len(Document(OUT/"Week3_Statistical_Analysis_and_Hypothesis_Testing.docx").inline_shapes)}
    (OUT/"verification.json").write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(output,indent=2))
    if output["status"]!="PASS":
        raise SystemExit(1)


if __name__=="__main__":
    main()
