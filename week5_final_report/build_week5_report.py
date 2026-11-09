"""Build the Week 5 synthesis report from saved Week 1–4 outputs only.

This script formats existing metrics, findings, charts, and code references; it
does not rerun EDA/statistical tests or train models.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
W1 = ROOT / "week1_eda"
W2 = ROOT / "week2_visualization"
W3 = ROOT / "week3_statistics"
W4 = ROOT / "week4_machine_learning"
REPORT = OUT / "YuvaIntern_IEEE_CIS_Fraud_Detection_Final_Report.docx"
NAVY, TEAL, GOLD, RED, INK, MUTED, PALE = "18324B", "167D8D", "E3A72F", "C6534A", "23313B", "536570", "EAF3F5"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def shade(cell: Any, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def set_cell(cell: Any, text: str, *, bold: bool = False, color: str = INK, size: float = 8.3, fill: str | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(1)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Aptos"
    r.font.size = Pt(size)
    r.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    if fill:
        shade(cell, fill)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None, font_size: float = 8) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    table.autofit = True
    for c, h in zip(table.rows[0].cells, headers):
        set_cell(c, h, bold=True, color="FFFFFF", size=font_size, fill=NAVY)
    for row in rows:
        cells = table.add_row().cells
        for c, val in zip(cells, row):
            set_cell(c, val, size=font_size)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Inches(width)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def bullet(doc: Document, text: str, *, level: int = 0) -> None:
    p = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
    p.paragraph_format.space_after = Pt(3)
    p.add_run(text)


def code_block(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.18)
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(text)
    r.font.name = "Consolas"
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(INK)


def add_page_number(p: Any) -> None:
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("YuvaIntern | Week 5  •  ")
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor.from_string(MUTED)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    p._p.append(field)


def figure(doc: Document, week: str, title: str, path: Path, what: str, implication: str, *, width: float = 6.55) -> None:
    doc.add_page_break()
    doc.add_heading(f"{week} | {title}", level=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    shape = p.add_run().add_picture(str(path), width=Inches(width))
    shape._inline.docPr.set("descr", f"{title}. {what}")
    cap = doc.add_paragraph(f"Figure: {title} (source: {path.parent.parent.name}/{path.name}).")
    cap.style = "Caption"
    doc.add_heading("What it shows", level=3)
    doc.add_paragraph(what)
    doc.add_heading("Why it matters", level=3)
    doc.add_paragraph(implication)


def chart_catalog(w1: dict, w2: dict, w3: dict, w4: dict) -> list[tuple[str, str, Path, str, str]]:
    # All text below explains the existing chart; values are read from saved outputs.
    return [
        ("Week 1 • EDA", "Fraud class distribution", W1/"charts"/"01_class_distribution.png",
         f"The training data contains {w1['target_counts']['1']:,} frauds among {w1['rows']:,} transactions ({w1['fraud_rate']:.2%}).",
         "The rare fraud class makes accuracy alone a poor success measure; the model must find positives without creating unmanageable review volume."),
        ("Week 1 • EDA", "Transaction amount distribution", W1/"charts"/"02_amount_distribution.png",
         f"Amounts are right-skewed. The fraud median is ${w1['amount_summary']['by_class']['1']['median']:.2f}, versus ${w1['amount_summary']['by_class']['0']['median']:.2f} for legitimate transactions; the overall 99th percentile is ${w1['amount_summary']['overall']['p99']:,.2f}.",
         "Amount segments can inform review, but the Week 3 tests found no statistically supported overall amount-distribution difference after multiple-testing correction."),
        ("Week 1 • EDA", "Top missing-value patterns", W1/"charts"/"03_missingness_top20.png",
         f"Average missingness is {w1['missingness']['average_identity_feature_pct']:.1f}% across identity features and {w1['missingness']['average_transaction_feature_pct']:.1f}% across transaction features in the merged training table.",
         "Missing identity values are common and can represent absent identity linkage. Preserve these signals and monitor source coverage instead of deleting columns indiscriminately."),
        ("Week 1 • EDA", "Fraud rate by ProductCD", W1/"charts"/"04_productcd_fraud_rate.png",
         "ProductCD C has the highest observed fraud rate (11.69%, n=68,519); W is the largest category by volume.",
         "Rate and volume answer different questions. Category patterns are descriptive and should be checked on later data before operational rules are considered."),
        ("Week 1 • EDA", "Card category rates", W1/"charts"/"05_card_category_rates.png",
         "Among card4 categories with at least 1,000 transactions, discover has a 7.73% observed fraud rate; among card6 categories meeting that threshold, credit has a 6.68% rate.",
         "A segment with a higher rate may still have different volume and uncertainty; report support counts with rates."),
        ("Week 2 • Advanced visualization", "Stratified split and class imbalance", W2/"charts"/"01_stratified_split_class_balance.png",
         w2["chart_details"][0]["insight"],
         "The near-identical fraud shares in train-fit and validation support a fairer holdout comparison for the rare-event class."),
        ("Week 2 • Advanced visualization", "Amount exposure and fraud risk", W2/"charts"/"02_amount_decile_exposure_and_risk.png",
         w2["chart_details"][1]["insight"],
         "The largest share of transaction amount is not automatically the segment with the highest fraud frequency; keep exposure and rate separate."),
        ("Week 2 • Advanced visualization", "Daily volume and fraud rate over relative time", W2/"charts"/"03_relative_time_daily_volume_and_fraud_rate.png",
         w2["chart_details"][2]["insight"],
         "The visible variation motivates monitoring for drift, but TransactionDT is elapsed time, not a calendar clock; it cannot identify seasons or causes."),
        ("Week 2 • Advanced visualization", "ProductCD by card type", W2/"charts"/"04_productcd_by_card6_fraud_rate.png",
         w2["chart_details"][3]["insight"],
         "Two-way segments can reveal patterns hidden by broad averages. Cells with small support were not ranked as high-risk evidence."),
        ("Week 2 • Advanced visualization", "ProductCD by amount quantile", W2/"charts"/"05_productcd_by_amount_quantile_fraud_rate.png",
         w2["chart_details"][4]["insight"],
         "Amount behavior can differ within product groups. These descriptive cells need validation before being used to trigger reviews."),
        ("Week 2 • Advanced visualization", "Relative time by amount risk", W2/"charts"/"06_relative_time_by_amount_risk_heatmap.png",
         w2["chart_details"][5]["insight"],
         "Time-window and amount combinations can be monitored together, while avoiding calendar-time or causal claims."),
        ("Week 2 • Advanced visualization", "Selected numeric associations", W2/"charts"/"07_numeric_feature_association_with_fraud.png",
         w2["chart_details"][6]["insight"],
         "The point-biserial coefficient is a descriptive univariate association; it is not a causal effect or trained-model feature importance."),
        ("Week 3 • Statistical analysis", "Amount distributions by fraud status", W3/"charts"/"01_amount_distribution_by_fraud_status.png",
         "The plotted medians differ by $6.50, but the Mann–Whitney and log-amount Welch tests did not reject their null hypotheses after Holm adjustment.",
         "Visual differences should be interpreted alongside uncertainty, effect size, and the formal test rather than treated as proof of a meaningful difference."),
        ("Week 3 • Statistical analysis", "ProductCD fraud rates", W3/"charts"/"02_productcd_fraud_rates.png",
         "The group rates and Wilson intervals show differences across ProductCD categories; the omnibus association test was statistically significant with Cramer's V 0.1665.",
         "A category association can help prioritize monitoring, but does not explain the cause of fraud or justify automatic declines."),
        ("Week 3 • Statistical analysis", "card6 fraud rates", W3/"charts"/"03_card6_fraud_rates.png",
         "Observed rates vary across card6 types. Low-count categories are combined in this chart for display only; the statistical test used a conditional Monte Carlo procedure because some expected counts were sparse.",
         "Keep uncertainty and support visible; do not make policy decisions from tiny segments."),
        ("Week 3 • Statistical analysis", "V258 availability by fraud status", W3/"charts"/"04_v258_availability_by_fraud_status.png",
         "V258 is observed for 48.7% of fraud rows and 21.1% of legitimate rows in the fit partition.",
         "This is a data availability pattern. It can reflect collection processes and is not evidence that missingness causes fraud."),
        ("Week 4 • Model evaluation", "Precision–recall curves", W4/"charts"/"01_precision_recall_curve.png",
         f"The histogram gradient booster has validation AP {w4['models']['HistGradientBoosting (weighted)']['validation']['pr_auc_average_precision']:.4f}, above the no-skill prevalence baseline of {w4['validation_fraud_rate']:.3%}.",
         "Precision–recall performance focuses on the rare fraud class and is more informative than accuracy alone for comparing alert ranking."),
        ("Week 4 • Model evaluation", "ROC curves", W4/"charts"/"02_roc_curve.png",
         f"Validation ROC-AUC is {w4['models']['HistGradientBoosting (weighted)']['validation']['roc_auc']:.4f} for the histogram gradient booster and {w4['models']['Logistic Regression (balanced)']['validation']['roc_auc']:.4f} for logistic regression.",
         "ROC-AUC summarizes ranking across thresholds, but must be read with AP and precision because the legitimate class dominates."),
        ("Week 4 • Model evaluation", "Best-model confusion matrix", W4/"charts"/"03_confusion_matrix_best_model.png",
         f"At the fixed 0.50 threshold, the selected model has {w4['models']['HistGradientBoosting (weighted)']['validation']['true_positives']:,} true positives, {w4['models']['HistGradientBoosting (weighted)']['validation']['false_positives']:,} false positives, and {w4['models']['HistGradientBoosting (weighted)']['validation']['false_negatives']:,} false negatives.",
         "False positives consume review capacity; false negatives are missed fraud. Threshold selection must reflect explicit operating costs and capacity."),
    ]


def build() -> dict[str, Any]:
    cleaning_text = (ROOT/"cleaned_data"/"cleaning_summary.md").read_text(encoding="utf-8")
    w1 = load_json(W1/"eda_summary.json")
    manifest = load_json(ROOT/"cleaned_data_v2"/"split_manifest.json")
    w2 = load_json(W2/"week2_metrics.json")
    w3 = load_json(W3/"week3_results.json")
    w4 = load_json(W4/"week4_metrics.json")
    figs = chart_catalog(w1,w2,w3,w4)

    doc=Document()
    section=doc.sections[0]
    section.top_margin=Inches(.62);section.bottom_margin=Inches(.62);section.left_margin=Inches(.72);section.right_margin=Inches(.72)
    section.header_distance=Inches(.28);section.footer_distance=Inches(.28)
    styles=doc.styles
    styles["Normal"].font.name="Aptos";styles["Normal"].font.size=Pt(10);styles["Normal"].font.color.rgb=RGBColor.from_string(INK)
    styles["Normal"].paragraph_format.space_after=Pt(5)
    for name,size,color in [("Title",30,NAVY),("Heading 1",19,NAVY),("Heading 2",13,TEAL),("Heading 3",10,TEAL)]:
        st=styles[name];st.font.name="Aptos Display" if name!="Heading 3" else "Aptos";st.font.size=Pt(size);st.font.bold=True;st.font.color.rgb=RGBColor.from_string(color)
        st.paragraph_format.space_before=Pt(10);st.paragraph_format.space_after=Pt(5)
    styles["Caption"].font.name="Aptos";styles["Caption"].font.size=Pt(8);styles["Caption"].font.color.rgb=RGBColor.from_string(MUTED)
    section.header.paragraphs[0].text="IEEE-CIS FRAUD DETECTION  |  YUVA INTERN  |  FINAL PROJECT REPORT"
    section.header.paragraphs[0].style="Caption"
    add_page_number(section.footer.paragraphs[0])

    title=doc.add_paragraph(style="Title");title.add_run("IEEE-CIS Fraud Detection")
    sub=doc.add_paragraph();r=sub.add_run("Comprehensive Data Science Project Report  |  YuvaIntern Weeks 1–5");r.font.size=Pt(16);r.font.color.rgb=RGBColor.from_string(TEAL)
    doc.add_paragraph("Project synthesis for internship evaluation and technical interview discussion")
    doc.add_paragraph("Prepared from the existing Week 1–4 outputs. This report introduces no new EDA, tests, or model training.")
    doc.add_paragraph()
    add_table(doc,["Project at a glance","Result"],[["Merged cleaned training data",f"{w1['rows']:,} rows × {w1['columns']:,} columns"],["Fraud prevalence",f"{w1['fraud_rate']:.2%} in the full training data"],["Model development split",f"{manifest['fit_rows']:,} train-fit / {manifest['validation_rows']:,} validation; seed {manifest['random_state']}"],["Best validation PR-AUC / AP",f"{w4['models'][w4['best_model_by_validation_pr_auc']]['validation']['pr_auc_average_precision']:.4f}"],["Test data", "Held separate from model training and validation"]],font_size=9)

    doc.add_page_break()
    doc.add_heading("1. Executive summary",level=1)
    doc.add_paragraph(f"This project studies how to identify potentially fraudulent card-not-present transactions in the IEEE-CIS dataset. Fraud is rare: {w1['target_counts']['1']:,} of {w1['rows']:,} training records ({w1['fraud_rate']:.2%}). The work preserves the transaction rows and their linked identity information, documents substantial missingness, and uses a corrected stratified training/validation split.")
    doc.add_paragraph(f"The Week 4 comparison found the strongest validation ranking with weighted histogram gradient boosting: PR-AUC (average precision) {w4['models']['HistGradientBoosting (weighted)']['validation']['pr_auc_average_precision']:.4f}, compared with {w4['models']['Logistic Regression (balanced)']['validation']['pr_auc_average_precision']:.4f} for balanced logistic regression and {w4['validation_fraud_rate']:.4f} for the prior-only baseline. At the fixed 0.50 threshold, the selected model found {w4['models']['HistGradientBoosting (weighted)']['validation']['true_positives']:,} of {w4['validation_fraud_count']:,} validation frauds, while generating {w4['models']['HistGradientBoosting (weighted)']['validation']['false_positives']:,} false alerts and missing {w4['models']['HistGradientBoosting (weighted)']['validation']['false_negatives']:,} frauds.")
    doc.add_paragraph("The model is a useful development baseline, not a deployment-ready fraud control. Alert volume, missed fraud, data availability, threshold choice, and performance under time drift still require business-led validation. No test labels were used for model development or evaluation.")

    doc.add_heading("2. Business problem and project objectives",level=1)
    doc.add_paragraph("The business problem is to rank transactions so investigators or automated controls can focus attention on cases with higher fraud risk. A practical system must balance missed fraud against false alerts, review capacity, and customer friction. This dataset does not provide a review-cost schedule or operational capacity target, so the report evaluates statistical and classification metrics rather than claiming business savings.")
    for x in ["Preserve the original transaction records and fraud target while making the feature data usable.","Describe class imbalance, transaction amount, missingness, categories, time patterns, and selected feature relationships.","Test selected fraud-versus-legitimate differences with suitable statistical methods and effect sizes.","Compare a simple baseline and two imbalance-aware classifiers on the fixed validation partition.","Translate evidence into cautious, testable recommendations without making causal claims."]:
        bullet(doc,x)

    doc.add_heading("3. Dataset description and scope",level=1)
    doc.add_paragraph("The project started with five provided files: training transactions, training identity, test transactions, test identity, and sample submission. TransactionID is the join key; isFraud is present only in training. The training transaction and identity tables were left-joined, and the test tables were handled separately. The final cleaned training table has 590,540 rows and 434 columns; cleaned test has 506,691 rows and 433 columns. The two cleaned splits share 432 model-feature columns plus TransactionID; training also contains isFraud.")
    add_table(doc,["Stage / partition","Rows","Fraud","Legitimate","Fraud rate"],[["Full training data",f"{w1['rows']:,}",f"{w1['target_counts']['1']:,}",f"{w1['target_counts']['0']:,}",f"{w1['fraud_rate']:.3%}"],["v2 train-fit",f"{manifest['fit_rows']:,}",f"{manifest['class_counts']['1']['train_fit']:,}",f"{manifest['class_counts']['0']['train_fit']:,}",f"{manifest['fit_fraud_rate']:.3%}"],["v2 validation",f"{manifest['validation_rows']:,}",f"{manifest['class_counts']['1']['validation']:,}",f"{manifest['class_counts']['0']['validation']:,}",f"{manifest['validation_fraud_rate']:.3%}"],["Test (features only)","506,691","Not used","Not used","Not applicable"]],font_size=8)
    doc.add_paragraph("The v2 split assigns 20% of each fraud class to validation using random_state=42. Its train-fit and validation fraud rates are nearly identical. Test records were kept separate and were not used for Week 4 training, tuning, or evaluation.")

    doc.add_heading("4. Project workflow",level=1)
    add_table(doc,["1 · Acquire","2 · Clean / join","3 · Split / preprocess","4 · EDA / visualize","5 · Test hypotheses","6 · Train / evaluate","7 · Report"],[["Five supplied CSVs; source schemas recorded.","Identity left-join by TransactionID; originals retained.","Stratified fit/validation; transforms fit on fit only.","Weeks 1–2 describe overall and segmented patterns.","Week 3 tests selected distribution and category differences.","Week 4 compares dummy, logistic, and gradient boosting.","Week 5 consolidates saved evidence and recommendations."]],font_size=7.2)
    doc.add_paragraph("Reproducible workflow scripts: cleaned_data\\clean_and_audit.py; cleaned_data_v2\\stratified_preprocess.py and verify_v2.py; week1_eda\\eda_analysis.py; week2_visualization\\week2_visualization.py; week3_statistics\\week3_statistics.py and verify_week3_statistics.py; week4_machine_learning\\week4_machine_learning.py and verify_week4_outputs.py. The Week 5 report builder only reads existing summaries and charts.")

    doc.add_heading("5. Data acquisition, cleaning, and preprocessing",level=1)
    doc.add_paragraph("The supplied CSVs were inspected before transformation, with original shapes, ordered headers, and inferred types recorded. Training and test were merged separately on TransactionID. Exact duplicate rows and duplicate TransactionID keys were not found, so no rows were removed. Test identity fields named id-01…id-38 were normalized to id_01…id_38 to align schemas.")
    doc.add_paragraph("Cleaning kept numeric-looking features numeric, retained high-missingness columns, and preserved missing values in the auditable cleaned tables. No column was dropped solely for missingness. The cleaned training table contains 115,523,073 missing cells (45.07% of its cells); missing identity fields often reflect that a transaction has no linked identity row. The encoded preprocessing outputs use train-fit medians and category mappings; validation and test were transform-only, and the saved v2 verification found zero missing feature cells after preprocessing.")
    doc.add_paragraph("For modeling, Week 4 selected 11 fields supported by the prior project work: TransactionAmt, C1, C2, D1, D2, V258, V307, ProductCD, card4, card6, and DeviceType. TransactionID, isFraud, and TransactionDT were excluded from model features. The selected numeric fields were scaled and selected categorical codes one-hot encoded using transformations fit only on train-fit; the resulting model matrix had 30 features. This compact set reduced computation but does not represent all 432 available model features.")

    doc.add_heading("6. Exploratory analysis and advanced visual findings",level=1)
    doc.add_heading("Week 1 — Core EDA",level=2)
    doc.add_paragraph(f"The full training set is highly imbalanced ({w1['fraud_rate']:.2%} fraud). Transaction amounts are right-skewed: overall median ${w1['amount_summary']['overall']['median']:.2f}, mean ${w1['amount_summary']['overall']['mean']:.2f}, and 99th percentile ${w1['amount_summary']['overall']['p99']:,.2f}. ProductCD C has the highest observed fraud rate in the EDA categories; identity features are much more incomplete than transaction features. These are descriptive findings, not causal explanations.")
    doc.add_heading("Week 2 — Advanced visualization",level=2)
    for item in w2["findings"]:
        bullet(doc,item["finding"])
    doc.add_paragraph("Relative-time charts use TransactionDT as elapsed time from the dataset origin, not calendar dates. Heatmap rates with low support were suppressed or clearly marked. The selected numeric associations are descriptive, not model importance.")
    doc.add_heading("Week 1 and Week 2 figures",level=2)

    # The existing visualizations are embedded, not regenerated or edited.
    for info in chart_catalog(w1,w2,w3,w4)[:12]:
        figure(doc,*info)

    doc.add_heading("7. Statistical hypotheses, tests, and conclusions",level=1)
    doc.add_paragraph("Week 3 analyzed only the 472,432-row v2 train-fit partition. It used two-sided Mann–Whitney tests for skewed distributions, Welch’s t-test on log1p(amount) as a sensitivity analysis, and chi-square tests for categorical associations. Holm adjustment was applied across six reported tests. Effect sizes and 95% intervals are included; sparse card6 expected counts were handled with a fixed-margin conditional Monte Carlo test. Transactions are treated as independent observations, an approximation because the same cardholder or merchant can contribute multiple rows.")
    test_rows=[]
    test_info=[("Amount distribution","amount_mann_whitney","H₀: same distribution; H₁: distributions differ"),("Mean log1p amount","log_amount_welch","H₀: equal transformed means; H₁: means differ"),("ProductCD × fraud","productcd_chi_square","H₀: independent; H₁: associated"),("card6 × fraud","card6_chi_square","H₀: independent; H₁: associated"),("V258 missingness × fraud","v258_missingness_chi_square","H₀: independent; H₁: associated"),("Observed V258 values","v258_mann_whitney","H₀: same distribution; H₁: distributions differ")]
    for label,key,h0 in test_info:
        t=w3["tests"][key];ef=t["effect"]
        ci=ef.get("ci_95",ef.get("ci_95_approx",ef.get("ci_95_bootstrap")))
        decision="Reject H₀" if t["reject_at_0_05_holm"] else "Do not reject H₀"
        test_rows.append([label,h0,f"{t['statistic']:.4g}",t["p_value_display"],t["p_value_holm_display"],f"{ef['name']} {ef['estimate']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]",decision])
    add_table(doc,["Comparison","Hypotheses (H₀ / H₁)","Statistic","p","Holm p","Effect size (95% CI)","α=.05"],test_rows,font_size=6.8)
    doc.add_paragraph("The amount distribution result was not statistically significant after adjustment: Cliff’s delta is close to zero and its interval includes zero. ProductCD, card6, V258 missingness, and observed V258 values showed evidence of association/difference after correction. Statistical significance does not establish causation or practical value; the V258 value test uses observed values only, and its missingness pattern is itself strongly associated with target status.")
    doc.add_paragraph("For p-values below floating-point display range, saved reports label the value <1e-300 rather than treating a displayed zero as an exact probability. Cramér’s V intervals use bootstrap resampling from the observed contingency table; rank effect intervals are asymptotic.")

    doc.add_heading("Week 3 figures",level=2)
    for info in chart_catalog(w1,w2,w3,w4)[12:16]:
        figure(doc,*info)

    doc.add_heading("8. Machine-learning models and validation",level=1)
    doc.add_paragraph("The class distribution is about 27.6 legitimate transactions per fraud. To address this without discarding legitimate rows, Week 4 compared a prior-only DummyClassifier with balanced LogisticRegression and weighted HistGradientBoostingClassifier. The existing stratified split was preserved; all additional scaling and one-hot encoding were fit on train-fit and applied to validation. No test data or labels were used.")
    model_rows=[]
    names=["Prior baseline (Dummy)","Logistic Regression (balanced)","HistGradientBoosting (weighted)"]
    for name in names:
        m=w4["models"][name]["validation"]
        model_rows.append([name,f"{m['accuracy']:.4f}",f"{m['precision']:.4f}",f"{m['recall']:.4f}",f"{m['f1']:.4f}",f"{m['roc_auc']:.4f}",f"{m['pr_auc_average_precision']:.4f}"])
    add_table(doc,["Model","Accuracy","Precision","Recall","F1","ROC-AUC","PR-AUC / AP"],model_rows,font_size=7.2)
    doc.add_paragraph(f"The prior-only baseline predicts no fraud at the fixed 0.50 threshold: accuracy is {w4['models']['Prior baseline (Dummy)']['validation']['accuracy']:.2%}, but recall and F1 are zero and all {w4['validation_fraud_count']:,} validation fraud cases are missed. This demonstrates why accuracy is misleading when fraud is rare. AP uses the precision–recall curve and has a no-skill reference equal to the fraud prevalence ({w4['validation_fraud_rate']:.3%}).")
    hgb=w4["models"]["HistGradientBoosting (weighted)"]
    lr=w4["models"]["Logistic Regression (balanced)"]
    doc.add_paragraph(f"The weighted gradient booster has the best validation AP ({hgb['validation']['pr_auc_average_precision']:.4f}) and ROC-AUC ({hgb['validation']['roc_auc']:.4f}). At threshold 0.50 it identifies {hgb['validation']['true_positives']:,} frauds, flags {hgb['validation']['false_positives']:,} legitimate rows, and misses {hgb['validation']['false_negatives']:,} frauds. Logistic regression has lower AP ({lr['validation']['pr_auc_average_precision']:.4f}) and recall ({lr['validation']['recall']:.3f}), with more false alerts at this threshold. The current false-positive count is material for a human-review workflow; the threshold is not a business-optimal operating point.")
    doc.add_paragraph(f"AP train-fit versus validation is {hgb['train']['pr_auc_average_precision']:.4f} versus {hgb['validation']['pr_auc_average_precision']:.4f} for gradient boosting, and {lr['train']['pr_auc_average_precision']:.4f} versus {lr['validation']['pr_auc_average_precision']:.4f} for logistic regression. These comparisons show no train-higher-than-validation AP gap in this run, but do not rule out overfitting on other metrics or data splits. Scores are substantially above baseline, so the tested models are not simply reproducing the all-legitimate predictor.")
    doc.add_paragraph("False positives are legitimate transactions flagged for review; false negatives are fraudulent transactions missed at the chosen threshold. Raising the threshold often reduces alerts and can also reduce recall; lowering it often catches more fraud while increasing review workload. The appropriate balance depends on costs and review capacity, which were not provided here.")
    doc.add_heading("Week 4 evaluation figures",level=2)
    for info in chart_catalog(w1,w2,w3,w4)[16:]:
        figure(doc,*info)

    doc.add_heading("9. Business implications and practical recommendations",level=1)
    recommendations=[
        ("Use fraud-focused evaluation.","Track average precision, recall, precision, and false-alert volume alongside ROC-AUC. Do not select a model on accuracy alone; the baseline achieved high accuracy while missing every fraud case."),
        ("Set an operating threshold from explicit costs.",f"The best tested model currently generates {hgb['validation']['false_positives']:,} false positives and {hgb['validation']['false_negatives']:,} false negatives at 0.50. Define review capacity, missed-fraud costs, and customer-friction costs before choosing a threshold; assess the trade-off on validation and reserve test data for a later final evaluation."),
        ("Treat model output as a review-prioritization signal.","The compact model is a development baseline trained on 11 selected fields. Do not deploy automatic declines or claim savings from these results; broader features, calibration, temporal validation, and operational review are still needed."),
        ("Monitor feature availability.","Identity and V258 missingness vary substantially. Monitor feed coverage by source and segment, retain missingness indicators where appropriate, and check whether collection changes shift model behavior."),
        ("Validate segment patterns over time.","ProductCD, card6, amount, and relative-time patterns can suggest monitoring slices. Confirm their stability on later time periods and supported samples before turning them into policies."),
        ("Strengthen evaluation before production.","Use a time-aware holdout or rolling backtest to assess drift, calibrate probabilities, and compare against review capacity. Keep the supplied test partition sealed until the planned final evaluation."),
    ]
    for title_text,body in recommendations:
        p=doc.add_paragraph(style="List Number");p.paragraph_format.space_after=Pt(5);r=p.add_run(title_text+" ");r.bold=True;p.add_run(body)

    doc.add_heading("10. Limitations, risks, and future improvements",level=1)
    limitations=[
        "The Week 4 model used a compact 11-field subset rather than all 432 available features; validation results may change with broader feature engineering.",
        "Only one fixed stratified train/validation split was used. Because transaction patterns may drift, a time-based evaluation is needed before making claims about future performance.",
        "The tested classifiers used fixed settings and a fixed 0.50 decision threshold; there was no broad hyperparameter search, probability calibration, or threshold optimization against business costs.",
        "Rows may not be statistically independent because entities can contribute multiple transactions. Statistical tests are unadjusted group comparisons beyond the familywise p-value correction and do not establish causal effects.",
        "The card6 omnibus test used conditional Monte Carlo because sparse expected counts made the asymptotic approximation unsuitable. V258 value comparisons use complete cases; missingness is strongly nonrandom with respect to the target in this dataset.",
        "The high false-positive volume at threshold 0.50 may exceed practical review capacity. No actual operational cost, fraud loss amount, chargeback outcome, or investigator capacity data were supplied.",
        "No test-set model performance, deployment outcome, or business savings are reported. The current figures are validation results for a prototype, not evidence of production performance.",
    ]
    for x in limitations: bullet(doc,x)
    doc.add_heading("Future improvements",level=2)
    for x in ["Expand from the compact subset with resource-aware feature selection across the aligned 432 features.","Use time-aware validation and grouped checks where entity identifiers permit, then assess drift and stability.","Calibrate predicted risk and define decision thresholds using documented review capacity and error costs.","Analyze false positives and false negatives by supported product, card, amount, and data-availability segments.","Run the final sealed test evaluation only after choices are locked, and document the evaluation protocol."]:
        bullet(doc,x)

    doc.add_heading("11. Reproducible files and references",level=1)
    doc.add_paragraph("This synthesis uses the saved outputs below as its evidence. The existing weekly scripts are the reproducible sources for the analyses; this Week 5 builder only assembles those outputs into a report.")
    references=[
        ("Cleaning and schema/audit",ROOT/"cleaned_data"/"clean_and_audit.py"),("v2 split and preprocessing",ROOT/"cleaned_data_v2"/"stratified_preprocess.py"),("v2 independent verification",ROOT/"cleaned_data_v2"/"verify_v2.py"),
        ("Week 1 EDA script",W1/"eda_analysis.py"),("Week 1 findings",W1/"findings.md"),("Week 2 visualization script",W2/"week2_visualization.py"),("Week 2 findings",W2/"findings.md"),
        ("Week 3 statistics script",W3/"week3_statistics.py"),("Week 3 independent verification",W3/"verify_week3_statistics.py"),("Week 3 results",W3/"week3_results.json"),
        ("Week 4 machine-learning script",W4/"week4_machine_learning.py"),("Week 4 independent verification",W4/"verify_week4_outputs.py"),("Week 4 metrics",W4/"week4_metrics.json"),
        ("Week 5 report builder",OUT/"build_week5_report.py"),("Week 5 report verifier",OUT/"verify_week5_report.py"),
    ]
    for label,path in references:
        p=doc.add_paragraph(style="List Bullet");p.paragraph_format.space_after=Pt(2);p.add_run(label+": ");r=p.add_run(str(path));r.font.name="Consolas";r.font.size=Pt(8)
    doc.add_paragraph("Data provenance: the five source CSVs were provided with the project. No external data, web-sourced results, or external performance claims were added to this report.")

    doc.add_heading("12. Final conclusion",level=1)
    doc.add_paragraph(f"The project produced an auditable cleaned dataset, preserved a stratified validation split, documented fraud-related patterns, tested selected differences, and compared three classifiers without using test labels. Weighted histogram gradient boosting achieved the strongest validation ranking (AP {hgb['validation']['pr_auc_average_precision']:.4f}), well above the prior baseline ({w4['validation_fraud_rate']:.3%}), but at threshold 0.50 it still creates many false alerts and misses some fraud. The practical next step is not to claim deployment readiness: it is to improve validation, establish the cost and capacity trade-off, and evaluate the locked workflow on a later or sealed test set. This report completes Week 5 only.")

    doc.save(REPORT)
    return {"report":str(REPORT),"figure_count":len(figs),"source_figures":[str(x[2]) for x in figs],"source_data":{"week1_rows":w1["rows"],"week2_fit_rows":w2["fit_rows"],"week3_fit_rows":w3["fit_rows"],"week4_train_rows":w4["train_rows"],"week4_validation_rows":w4["validation_rows"]}}


if __name__=="__main__":
    print(json.dumps(build(),indent=2))
