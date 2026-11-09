# IEEE-CIS-Fraud-Detection
IEEE-CIS Fraud Detection – Data Cleaning, Exploratory Data Analysis and Machine Learning using Python.

## Week 1: Data Cleaning and Exploratory Data Analysis

This repository contains the completed five-week YuvaIntern Data Science project using the IEEE-CIS Fraud Detection dataset. The Week 1 cleaning and EDA files remain in place; this update adds the Week 2–5 reports, analysis artifacts, final report, and supporting documentation.

### Objective

Prepare an auditable view of the transaction and identity data, understand its quality and missingness, and describe initial patterns in the labeled training data. Week 1 does not train a machine-learning model and does not use test data for exploratory analysis.

### Dataset and cleaning

The project source data contains 590,540 labeled training transactions and 506,691 test transactions. The training target is `isFraud`. Transaction and identity tables are joined by `TransactionID`, separately for training and test. The original CSV files are not included in this repository.

The cleaning workflow records source schemas before transformation, validates keys and row counts, and checks duplicate records. No exact duplicates or duplicate `TransactionID` keys were found, so no rows were removed. Test identity feature names were normalized to match the training schema. Identity data is left-joined so transactions without an identity record remain present. High-missingness columns were retained, and the clean merged tables preserve missing values for later handling.

### Week 1 findings

- Training contains 20,663 fraudulent transactions (3.50%) and 569,877 legitimate transactions (96.50%), so the target is strongly imbalanced.
- Transaction amounts are right-skewed: the median is 68.77, the mean is 135.03, and the 99th percentile is 1,104.00.
- The observed fraud-rate median amount is 75.00, compared with 68.50 for legitimate transactions. This is descriptive and does not establish a cause.
- ProductCD category C has the highest observed fraud rate (11.69%; 68,519 transactions); ProductCD W has the largest volume (439,670 transactions).
- Average missingness is 84.5% across identity columns and 41.3% across transaction feature columns in the merged training table. Many transactions have no linked identity row.
- Among categories with at least 1,000 transactions, `discover` has the highest observed `card4` fraud rate (7.73%), and `credit` has the highest observed `card6` fraud rate (6.68%).

These are patterns in the supplied training sample. They should not be interpreted as causal relationships or guarantees about future transactions.

### Visualizations

Five PNG charts are in `week1_eda/charts/`: class distribution, transaction amount distribution, top-column missingness, fraud rate by ProductCD, and card-category fraud rates. The Word report and `findings.md` explain what each chart shows and why it matters.

### Repository structure

```text
cleaned_data/                  # Cleaning, preprocessing and verification scripts; summaries only
cleaned_data_v2/               # Stratified split, train-fit preprocessing and verification
docs/                          # Consolidated results and project verification
week1_eda/                     # Week 1 EDA scripts, report, findings, metrics and charts
week2_visualization/           # Advanced visualization scripts, report, findings and charts
week3_statistics/              # Statistical tests, results, report and charts
week4_machine_learning/        # Model development, metrics, evaluation report and charts
week5_final_report/            # Integrated final report and report verification
submission_descriptions/       # YuvaIntern portal descriptions for all five weeks
README.md
requirements.txt
.gitignore
```

### Week 2: Advanced visualization and storytelling

Seven charts examine class balance, amount deciles, relative-time windows, ProductCD/card categories, amount quantiles, and selected numerical relationships. Findings include a 5.14% fraud rate in the top amount decile, a 45.4% share of total transaction amount in that decile, and supported rate differences across category combinations. Relative time is measured from the dataset’s transaction sequence; it does not identify calendar dates. See `week2_visualization/findings.md` and its Word report for chart explanations and support counts.

### Week 3: Statistical analysis

The tests use the training-fit partition only. Amount comparisons did not show evidence of a meaningful group difference (Mann–Whitney p=0.4354, Cliff’s delta −0.0036; Welch test on log1p amount p=0.4673, Hedges’ g −0.0067). ProductCD, card6, and V258 availability/value showed associations with fraud status. These are observational associations, not causal conclusions. Test statistics, adjusted p-values, confidence intervals, effect sizes, assumptions, and charts are in `week3_statistics/`.

### Week 4: Model development and evaluation

The existing stratified training and validation partitions were used to compare a dummy baseline, weighted logistic regression, and weighted HistGradientBoosting. The compact model uses seven numerical fields and four categorical fields; `TransactionID`, `isFraud`, and `TransactionDT` are excluded from model features. Imputers and encoders are fit on training-fit rows only. Validation average precision (PR-AUC) was 0.0350 for the dummy baseline, 0.2748 for logistic regression, and 0.4218 for HistGradientBoosting. The gradient-boosting model also achieved recall 0.7239, F1 0.2341, and ROC-AUC 0.8569. At the reported threshold it produced 2,992 true positives, 18,432 false positives, 1,141 false negatives, and 95,543 true negatives. The dummy model’s 96.5% accuracy with zero fraud recall illustrates why accuracy alone is misleading for this imbalanced task. These are validation results, not a production guarantee.

### Week 5: Integrated reporting

`week5_final_report/YuvaIntern_IEEE_CIS_Fraud_Detection_Final_Report.docx` consolidates methods, findings, 19 charts from Weeks 1–4, statistical results, model metrics, practical recommendations, limitations, and references to the scripts. `docs/PROJECT_RESULTS.md` provides a compact results summary; `VERIFICATION_REPORT.md` records package checks.

### Reproducibility and limitations

The included weekly scripts and verification scripts document how each stage was produced. Transaction-level source and processed files are intentionally excluded, so a reproducible run requires the authorized original data plus the outputs generated by the preceding stages. No model was retrained while preparing this repository update. Evaluation uses one stratified validation split and a selected threshold; future work should check temporal and operational performance and calibrate the threshold against review capacity and fraud costs. High observed subgroup rates should be read with their sample sizes, and none of the reported associations establish causality.

### Reproducing Week 1

Use Python 3.12 or a compatible Python version and install dependencies from `requirements.txt`. Obtain the five original CSV files from their authorized source and keep them outside Git. To create the cleaned tables, place the files in a local data folder and run:

```powershell
python cleaned_data/clean_and_audit.py --source-dir path/to/original_csvs --output-dir cleaned_data
```

Then run the Week 1 training-only analysis and report builder from the repository root:

```powershell
python week1_eda/eda_analysis.py
python week1_eda/build_report.py
```

The EDA script reads `cleaned_data/train_cleaned.csv`. It does not open test data. Generated transaction tables and original datasets are excluded from this repository and should remain untracked locally.

### Data and privacy note

No original or processed row-level CSVs, validation predictions, credentials, or user-specific local paths are included. The project summaries and figures contain aggregate findings only. Review the dataset provider’s terms before obtaining or using the source data.
