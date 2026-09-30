# Module 3 - Anomaly Model Evaluation (Sprint 9)

Models are tested, not just run: the cleaned data has no ground-truth
anomaly label, so 100 known synthetic anomalies
(25 each of extreme_price, extreme_quantity, extreme_return, odd_combo, seed 7)
were appended to the 98918 cleaned rows and both models scored the
mixed frame transductively - exactly as in production. Flag thresholds use
the production contamination rate (0.01).

## Results

| model | ROC AUC | precision | recall | F1 | precision@K | flagged |
|---|---|---|---|---|---|---|
| isolation_forest | 0.9922 | 0.0761 | 0.7500 | 0.1381 | 0.7300 | 986 |
| lof | 0.8136 | 0.2807 | 0.1600 | 0.2038 | 0.1900 | 57 |
| either_model_flags | 0.9487 | 0.0815 | 0.8100 | 0.1481 | 0.1500 | 994 |
| both_models_agree | 0.9487 | 0.2041 | 0.1000 | 0.1342 | 0.1500 | 49 |

K = 100 (precision@K = share of injected rows in the top-K scores).

## Recall by injection type (at production flag thresholds)

| model | extreme_price | extreme_quantity | extreme_return | odd_combo |
|---|---|---|---|---|
| isolation_forest | 1.00 | 1.00 | 0.00 | 1.00 |
| lof | 0.00 | 0.16 | 0.24 | 0.24 |
| either_model_flags | 1.00 | 1.00 | 0.24 | 1.00 |
| both_models_agree | 0.00 | 0.16 | 0.00 | 0.24 |

Each column is one injected family (25 rows each): IF catches price,
quantity and joint extremes perfectly but is blind to far-out returns
(see below); LOF is partial everywhere at its strict combo-level cut.

## Confusion matrices (at production flag thresholds)

**isolation_forest** (`[[TN, FP], [FN, TP]]`):

```
[98007, 911]
[25, 75]
```

**lof** (`[[TN, FP], [FN, TP]]`):

```
[98877, 41]
[84, 16]
```

**either_model_flags** (`[[TN, FP], [FN, TP]]`):

```
[98005, 913]
[19, 81]
```

**both_models_agree** (`[[TN, FP], [FN, TP]]`):

```
[98879, 39]
[90, 10]
```

## ROC curves

![ROC curves](roc_curve.png)

## Interpretation

- IF (AUC 0.99) isolates far-out points globally - but misses the
  `extreme_return` family entirely (0/25). Each tree fits a 256-row
  subsample, and uniform splits only isolate a point when samples sit on
  both sides of it: beyond the sampled maximum (price side, where real
  AMAZONFEE extremes provide nearby company) isolation is fast, while down
  the bare negative-quantity tail the injected rows travel with the bulk
  and score normal. A documented model blind spot, not a bug.
- LOF (AUC 0.81) is partial at its strict combo-level cut: only ~57 rows
  flag, so recall is capped (0.16) while precision (0.28) beats IF's.
- Precision looks low everywhere because the 1% contamination cut flags
  ~989 rows against 100 injected: even perfect ranking caps precision at
  ~0.10 at full recall. Read recall/AUC as the detection story,
  precision as the triage-cost story.
- Defence in depth: the rule engine (`quantity_out_of_bounds`,
  `quantity_bulk`) catches every far-out return the forest misses - ML
  and rules cover each other's blind spots.
- The `either_model_flags` row is the production recall story; the
  `both_models_agree` row is the high-precision story behind the
  `anomaly_severity` in `validation_report.json`.

## Reproducibility

- Seeds: injection 7, IsolationForest 42 (fixed in
  `anomaly_detector.py`); LOF is deterministic.
- Rerun: `python module3_validation/evaluation.py --n-per-type 25 --seed 7`
- Generated: 2026-09-30T11:40:07+00:00

## Limitations

- Synthetic extremes are far-out by construction; subtle real anomalies
  near the plausibility boundary are not covered by this test.
- Injected rows reuse real rows for non-numeric columns, so only the
  Quantity/UnitPrice signal is synthetic.
