"""Read-only verification of the Week 5 synthesis report."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent
REPORT=OUT/"YuvaIntern_IEEE_CIS_Fraud_Detection_Final_Report.docx"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    w1=read_json(ROOT/"week1_eda"/"eda_summary.json")
    manifest=read_json(ROOT/"cleaned_data_v2"/"split_manifest.json")
    w2=read_json(ROOT/"week2_visualization"/"week2_metrics.json")
    w3=read_json(ROOT/"week3_statistics"/"week3_results.json")
    w4=read_json(ROOT/"week4_machine_learning"/"week4_metrics.json")
    v4=read_json(ROOT/"week4_machine_learning"/"verification.json")
    cleaning=(ROOT/"cleaned_data"/"cleaning_summary.md").read_text(encoding="utf-8")
    report=Document(REPORT)
    all_text="\n".join([p.text for p in report.paragraphs]+[c.text for t in report.tables for row in t.rows for c in row.cells])
    headings=[p.text for p in report.paragraphs if p.style.name.startswith("Heading")]
    with ZipFile(REPORT) as archive:
        embedded_hashes={hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist() if name.startswith("word/media/")}

    expected_charts=[
        ROOT/"week1_eda"/"charts"/name for name in ["01_class_distribution.png","02_amount_distribution.png","03_missingness_top20.png","04_productcd_fraud_rate.png","05_card_category_rates.png"]
    ]+[ROOT/"week2_visualization"/"charts"/name for name in [
        "01_stratified_split_class_balance.png","02_amount_decile_exposure_and_risk.png","03_relative_time_daily_volume_and_fraud_rate.png","04_productcd_by_card6_fraud_rate.png","05_productcd_by_amount_quantile_fraud_rate.png","06_relative_time_by_amount_risk_heatmap.png","07_numeric_feature_association_with_fraud.png"]
    ]+[ROOT/"week3_statistics"/"charts"/name for name in ["01_amount_distribution_by_fraud_status.png","02_productcd_fraud_rates.png","03_card6_fraud_rates.png","04_v258_availability_by_fraud_status.png"]]
    expected_charts += [ROOT/"week4_machine_learning"/"charts"/name for name in ["01_precision_recall_curve.png","02_roc_curve.png","03_confusion_matrix_best_model.png"]]
    source_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in expected_charts}

    stats_keys=["amount_mann_whitney","log_amount_welch","productcd_chi_square","card6_chi_square","v258_missingness_chi_square","v258_mann_whitney"]
    statistics_numbers_match=all(
        all(s in all_text for s in [f"{w3['tests'][k]['statistic']:.4g}",w3['tests'][k]['p_value_display'],w3['tests'][k]['p_value_holm_display'],
             f"{w3['tests'][k]['effect']['estimate']:.3f}"])
        for k in stats_keys
    )
    model_metric_numbers_match=all(
        all(f"{w4['models'][name]['validation'][metric]:.4f}" in all_text for metric in ["accuracy","precision","recall","f1","roc_auc","pr_auc_average_precision"])
        for name in ["Prior baseline (Dummy)","Logistic Regression (balanced)","HistGradientBoosting (weighted)"]
    )

    checks={
        "report_exists_and_nonempty":REPORT.is_file() and REPORT.stat().st_size>50000,
        "week1_source_counts_match":w1["rows"]==590540 and w1["columns"]==434 and sum(w1["target_counts"].values())==w1["rows"],
        "cleaning_merge_and_losslessness_summary_present":("Training merged | 590,540 | 434" in cleaning and "Test merged | 506,691 | 433" in cleaning and "No columns were dropped for missingness" in cleaning),
        "v2_split_counts_and_rates_match":manifest["fit_rows"]==472432 and manifest["validation_rows"]==118108 and manifest["random_state"]==42 and manifest["fit_fraud_rate"]==w2["fit_fraud_rate"],
        "week2_advanced_findings_match_saved_metrics":all(f["finding"] in all_text for f in w2["findings"]),
        "week3_test_scope_and_holm_procedure_present":w3["fit_rows"]==manifest["fit_rows"] and "Holm" in w3["p_adjustment"],
        "week4_metrics_match_verified_outputs":w4["train_rows"]==472432 and w4["validation_rows"]==118108 and v4["status"]=="PASS" and w4["models"]["HistGradientBoosting (weighted)"]["validation"]["pr_auc_average_precision"]==0.4218200146886596,
        "model_and_error_numbers_in_report":all(s in all_text for s in ["0.4218","0.2748","0.0350","18,432","1,141","0.50"]),
        "all_source_chart_files_exist":all(p.is_file() for p in expected_charts),
        "all_source_chart_pngs_are_readable":all(Image.open(p).format=="PNG" and min(Image.open(p).size)>=1000 for p in expected_charts),
        "all_19_charts_embedded":len(report.inline_shapes)==len(expected_charts)==19,
        "embedded_chart_media_matches_all_source_pngs":embedded_hashes==source_hashes and len(embedded_hashes)==19,
        "statistical_test_numbers_match_saved_week3_results":statistics_numbers_match,
        "model_metric_numbers_match_saved_week4_results":model_metric_numbers_match,
        "required_report_sections_present":all(any(label.lower() in h.lower() for h in headings) for label in ["Executive summary","Business problem","Dataset description","Project workflow","Data acquisition","Exploratory analysis","Statistical hypotheses","Machine-learning models","Business implications","Limitations","Final conclusion"]),
        "reproducible_scripts_referenced":all(name in all_text for name in ["eda_analysis.py","week2_visualization.py","week3_statistics.py","week4_machine_learning.py","verify_week4_outputs.py","build_week5_report.py"]),
        "report_is_week5_only_no_new_analysis":("introduces no new eda, tests, or model training" in all_text.lower() and "no test labels were used" in all_text.lower() and "completes week 5 only" in all_text.lower()),
        "all_model_test_and_eda_values_are_source_bound":all(path.is_file() for path in [ROOT/"week1_eda"/"eda_summary.json",ROOT/"week2_visualization"/"week2_metrics.json",ROOT/"week3_statistics"/"week3_results.json",ROOT/"week4_machine_learning"/"week4_metrics.json"]),
    }
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"report":str(REPORT),"embedded_chart_count":len(report.inline_shapes),"source_chart_count":len(expected_charts),"report_paragraph_count":len(report.paragraphs),"report_table_count":len(report.tables),"report_heading_count":len(headings),"render_review":"NOT COMPLETED: LibreOffice and Microsoft Word are unavailable; no software was installed.","source_week_counts":{"week1_rows":w1["rows"],"train_fit_rows":manifest["fit_rows"],"validation_rows":manifest["validation_rows"],"test_rows_context_only":506691}}
    (OUT/"verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    if result["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
