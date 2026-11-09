# Independent verification report — IEEE-CIS Week 1, stratified split v2

Read-only verification of v2 outputs. No model was trained. The v1 datasets and original test input were not modified.

## Results

| Check | Result |
|---|---|
| All train/test rows and keys preserved | PASS |
| Train/validation split stratified by isFraud | PASS |
| No target/test leakage in preprocessing | PASS |
| TransactionID and isFraud excluded from model features | PASS |
| Categorical mappings consistent across partitions | PASS |
| Scripts reproducible with fixed seed and separate outputs | PASS |
| Columns retained and transformed features have no missing values | PASS |
| Original test input unchanged | PASS |
| Training targets match source | PASS |

## Evidence

- Original merged input shapes: training 590,540 rows; test 506,691 rows. The saved v2 outputs have the counts shown below.

| Partition | Rows | Non-fraud | Fraud | Fraud rate | Missing feature cells |
|---|---:|---:|---:|---:|---:|
| train_fit | 472,432 | 455,902 | 16,530 | 3.4989% | 0 |
| validation | 118,108 | 113,975 | 4,133 | 3.4993% | 0 |
| test | 506,691 | n/a | n/a | n/a | 0 |

Fixed `random_state`: 42. Validation fraction per class: 20%.
Overall source fraud rate: 3.4990%; fit fraud rate: 3.4989%; validation fraud rate: 3.4993%.
Model feature list contains 432 columns; TransactionID and isFraud are stored separately. 31 categorical mappings use 0=missing, 1=unseen, and codes 2+ for training-fit categories.
The preprocessing script is deterministic and refuses to overwrite populated versioned outputs. Its fit pass reads only training rows and uses only training-fit rows for medians and vocabularies; validation/test are transform-only.
Original test input size and modification timestamp before/after transformation match.

## Overall

9 of 9 checks passed.
