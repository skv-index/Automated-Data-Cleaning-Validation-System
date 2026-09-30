# Pipeline flow — how data moves through the system

## The chain (one execution, three handoffs)

```text
online_retail_II.xlsx ──(50k rows/sheet, 100,000 sampled)──▶ [1 PROFILING]
  profiling_report.json (flags: 13 total, 0 critical) + visuals/*.png
  │  gated by module2_cleaning/expected_schema.json (versioned contract)
  ▼
[2 CLEANING] normalize → impute (statistical lookup, then KNN) → dedupe →
  features (LineValue, IsCancellation/IsReturn/IsGiveaway, calendar parts) →
  z-score scaling → quality scoring
  cleaned_data.csv (98,918 rows × 21 cols) + cleaning_log.json
  quality 91.19 → 100.0 (1,082 exact duplicates removed, 408 descriptions
  imputed, 1,213 Country aliases mapped)
  ▼
[3 VALIDATION] 20 rules → IF+LOF anomalies → NLP column check →
  impossible/suspicious/drift findings → severity + health scoring
  validation_report.json — health 84.53, Watch
```

Row-count continuity (asserted by `tests/test_pipeline.py`): profiling
`n_rows` == cleaning `n_rows_in`; cleaning `n_rows_out` == validation
`n_rows`. The 1,082 dropped rows are exact full-row duplicates (keep first);
near-duplicate description variants are flagged, never merged.

## What validation actually found (real run)

- **Rules:** 9 of 20 fire, all review/info (bulk quantities, £1k+ prices,
  return-without-cancellation ×186, duplicate line keys ×2,002, rare
  micro-countries). Zero hard violations — cleaning held.
- **Anomalies:** IF flags 988, LOF 58, both agree on 53 (£13.5k AMAZONFEE
  fees, the −9,360 record return, bulk/adjustment lines).
- **Columns:** 10/10 match, 0 mismatch.
- **Drift:** ALERT 2009→2010 — UnitPrice spread exploded on the 2010 fee
  regime (max £1,998 → £13,541); medians flat, return/guest shares drifted
  14–19% (watch). 2011 excluded (9-day stub).
- **Scores:** error_severity 1.25, anomaly_severity 0.05 (0 = clean);
  health = 0.40·validity + 0.25·anomaly + 0.20·consistency + 0.15·stability.
  Stability floors at 0 on the drift spike → Watch, with the action
  "segment 2010 fee codes out of like-for-like revenue".

## Configuration surface

Everything is set in `module4_pipeline/config.yaml` (or overridden by
`pipeline.py --input/--output`, which writes the merged `run_config.yaml`
next to the results): input/schema paths, all seven output paths,
`nrows_per_sheet`, `heatmap_rows`, `imputation_method` (+ `knn_neighbors`,
`fuzzy_threshold`), `contamination`, `classifier_sample_n`, log
level/console, `stop_on_stage_failure`. The orchestrator passes these
explicitly into each module API — no hardcoded settings.

## Logs and provenance

`pipeline.log` records config, preflight, per-stage START/DONE + seconds,
key metrics (rows, quality delta, health/grade), and warnings (critical
flags, non-Healthy grades). Every report embeds `generated_at`, its source
paths, and parameters, so any output is traceable to the exact run.
