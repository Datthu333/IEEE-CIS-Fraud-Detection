# Week 2 evidence-based findings

Source: training-fit partition only (472,432 rows, 16,530 frauds). No test data or model was used.

## 1. The v2 split retained nearly identical fraud rates: 3.499% in 472,432 fit rows and 3.499% in 118,108 validation rows.

**Business implication:** The held-out validation partition preserves the rare-event mix, so later model evaluation can compare performance without a large class-rate shift.

**Evidence:** split_manifest.json: fit fraud=16,530; validation fraud=4,133; random_state=42.

## 2. Q10 contributes 45.4% of the total transaction-amount sum; its observed fraud rate is 5.14% across 47,244 fit rows.

**Business implication:** Amount exposure and fraud frequency are different views. Monitoring should retain both, rather than treating the largest-amount segment as automatically the highest-rate segment.

**Evidence:** Training-fit amount quantiles; amount sums and fraud rates are computed directly from the selected rows.

## 3. Across supported 30-day relative windows, the highest pooled rate is 4.35% in window 7 (193/4,434); the lowest is 2.51% in window 1 (2,698/107,361).

**Business implication:** The observed variation motivates time-based monitoring. The dataset uses elapsed time, so it cannot support calendar-season explanations by itself.

**Evidence:** 30-day bins begin at the minimum TransactionDT; periods below 1,000 rows are excluded from this comparison.

## 4. The highest supported ProductCD × card6 cell is ProductCD C with card6 credit: 16.68% fraud across 22,002 rows.

**Business implication:** Category combinations may be useful segments for further review, with counts and uncertainty checked before operational use.

**Evidence:** Two-way group aggregation; only cells with at least 200 rows are ranked.

## 5. The highest supported ProductCD × amount-quantile cell is ProductCD H / Q10, at 39.05% across 630 rows.

**Business implication:** Risk differences can vary within a product category by amount band; aggregate category rates can hide these pockets.

**Evidence:** Rates are computed from raw TransactionAmt quantiles and isFraud in the training-fit partition; support threshold=200.

## 6. The highest supported 30-day-time × amount-quantile cell is window 7 / Q6, at 8.59% across 454 rows.

**Business implication:** Amount-band monitoring may need to account for changes over the transaction sequence rather than relying on one global rate.

**Evidence:** Time windows are 30 elapsed days from the first TransactionDT; cells below 200 rows are not ranked.

## 7. Among the seven selected numeric fields, V258 has the largest absolute point-biserial association with isFraud (r=0.300, pairwise n=104,446).

**Business implication:** This is a screening signal for later feature review, not a claim of causation or evidence that a trained model would rank it as important.

**Evidence:** Pairwise Pearson correlation with the binary target, calculated on all nonmissing training-fit values for each selected field.
