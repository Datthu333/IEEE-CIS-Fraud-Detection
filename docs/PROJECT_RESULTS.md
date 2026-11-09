# Results and verification summary

## Data integrity

- Training: 590,540 transactions; test: 506,691 transactions.
- No training or test rows were removed for duplicates; transaction keys were checked before joining identity records.
- Stratified train-fit/validation split: 472,432 / 118,108 rows with 16,530 / 4,133 frauds respectively; both rates are approximately 3.499%.
- Test records remain separate; no `isFraud` target is present in test features.
- High-missingness features were preserved; fitted missing-value and encoding transformations are train-fit-only.

## Statistical evidence

- Amount comparisons: Mann–Whitney p=0.4354, Cliff’s delta −0.0036; Welch test on log1p amount p=0.4673, Hedges’ g −0.0067. No meaningful difference was detected by these comparisons.
- ProductCD association: chi-square=13,090.07, Cramér’s V=0.1665.
- card6 association: chi-square=4,726.84, Cramér’s V=0.1000.
- V258 availability association: chi-square=7,059.12, Cramér’s V=0.1222; observed V258 values also differed substantially between fraud groups (Cliff’s delta 0.4915).

Categorical and feature-level results describe associations, not causes.

## Validation model metrics

| Model | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|
| Dummy baseline | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0350 |
| Weighted logistic regression | 0.1014 | 0.6366 | 0.1750 | 0.7886 | 0.2748 |
| Weighted HistGradientBoosting | 0.1397 | 0.7239 | 0.2341 | 0.8569 | 0.4218 |

HistGradientBoosting confusion counts at the reported threshold: TP 2,992; FP 18,432; FN 1,141; TN 95,543. Accuracy is omitted here because the dummy baseline achieves 96.5% while detecting no fraud. These figures are from the existing validation outputs; no model was retrained to prepare this submission.
