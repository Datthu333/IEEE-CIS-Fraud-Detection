# IEEE-CIS cleaning summary

## Cleaning decisions

- All five originals were read only. Their shapes, ordered headers, and inferred dtypes were recorded in `original_schema.json` before merges.
- No exact duplicate rows or duplicate `TransactionID` keys were found. No rows were removed.
- Test identity fields named `id-01` through `id-38` were renamed to `id_01` through `id_38` to match the training schema.
- Numeric-looking feature values were kept in their inferred numeric types rather than guessed to be nominal categories; string features are encoded only in the separate preprocessing outputs.
- Identity data was left-joined by `TransactionID`, separately for train and test. This retains every transaction and keeps missing identity fields for transactions without identity records.
- No columns were dropped for missingness. Clean merged tables preserve the original missing values so the auditable source stays lossless.
- `isFraud` remains in training only. Test features have matching names/order after excluding the target. `sample_submission.csv` is used only to validate test IDs and their order.
- Model-time preprocessing is separate: a deterministic 80/20 split of training rows is used; numeric median values are estimated from up to 100,000 deterministic fit-partition rows spread across the ID range, while categorical vocabularies are learned from all fit-partition rows. Validation and test are transform-only. No model was trained.

## Shapes and merge checks

| Output | Rows | Columns | Identity matches | Transactions without identity |
|---|---:|---:|---:|---:|
| Training merged | 590,540 | 434 | 144,233 | 446,307 |
| Test merged | 506,691 | 433 | 141,907 | 364,784 |

There are 432 shared model features, plus `TransactionID`; training additionally contains `isFraud`. Transaction IDs are unique within each source split, the train and test ID sets are disjoint, left merges preserved row counts, and sample submission IDs match test IDs and order.

## Missingness before and after merge

The full per-column records are in `missing_before.json`, `missing_after_train.csv`, and `missing_after_test.csv`.

| Dataset snapshot | Columns | Missing cells | Total cells | Missing percentage |
|---|---:|---:|---:|---:|
| Original train transaction | 394 | 95,566,686 | 232,672,760 | 41.07% |
| Original train identity | 41 | 2,104,107 | 5,913,553 | 35.58% |
| Original test transaction | 393 | 73,490,163 | 199,129,563 | 36.91% |
| Original test identity | 41 | 2,105,385 | 5,818,187 | 36.19% |
| Merged training table | 434 | 115,523,073 | 256,294,360 | 45.07% |
| Merged test table | 433 | 90,186,908 | 219,397,203 | 41.11% |

Merged missingness increases because most transactions have no corresponding identity row. These are expected absent identity features, not lost transaction rows. The separate encoded fit, validation, and test files have zero missing feature cells after imputation/encoding; details are in `preprocessed/preprocessing_validation.json`.

## Target and preprocessing outputs

Training target counts are 569,877 non-fraud and 20,663 fraud. The preprocessing outputs contain 472,432 fit rows, 118,108 validation rows, and 506,691 test rows. Test has no target. Preprocessing parameters are stored in `preprocessed/fitted_preprocessing.json`, and are fitted using training rows only.
