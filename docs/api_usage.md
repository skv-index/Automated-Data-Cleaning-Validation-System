# API usage — how to call each module

All snippets run from the repo root. Every file-route function also has a
DataFrame-route twin for notebook/script use; CLIs mirror the same options.

## Module 1 — Profiling (`module1_profiling.profiling_api`)

```python
from module1_profiling.profiling_api import profile_dataframe, run_profiling

report = profile_dataframe(df)          # DataFrame in -> report dict
report = run_profiling()                # xlsx in -> profiling_report.json
```

```bash
python module1_profiling/profiling_api.py \
  --input online_retail_II.xlsx --nrows-per-sheet 50000
```

Building blocks: `metadata_extractor.extract_metadata(df)` (dtypes,
semantics, mixed-type detail), `profiler.profile_dataset(df, ...)`
(missingness, dtype consistency, cardinality, numeric profile + 3 PNGs),
`run_rule_engine(df, metadata, statistics)` (suspicious / format / PII flags).

## Module 2 — Cleaning (`module2_cleaning`)

```python
from module2_cleaning.schema_inference import run_schema_inference
from module2_cleaning.cleaner import clean_dataframe, build_sample_diff
from module2_cleaning.transformer import (
    extract_features, scale_features, encode_features)
from module2_cleaning.cleaning_api import run_cleaning_pipeline, score_quality

schema = run_schema_inference()         # report -> expected_schema.json
cleaned, log = clean_dataframe(df, schema)   # normalize -> impute -> dedupe
featured, feat_log = extract_features(cleaned)   # LineValue, flags, dates
scaled, scaler = scale_features(featured)        # z-score *_scaled columns
cleaned, log = run_cleaning_pipeline()  # xlsx -> cleaned_data.csv + log
score_quality(df, schema)               # 0-100 completeness/consistency/
                                        # uniqueness/validity breakdown
```

```bash
python module2_cleaning/cleaning_api.py --cleaned cleaned_data.csv
```

## Module 3 — Validation (`module3_validation`)

```python
from module3_validation.rule_validator import validate_dataframe, run_validation
from module3_validation.anomaly_detector import detect_anomalies, run_detection
from module3_validation.column_classifier import (
    classify_columns, build_classification_report, run_classifier)
from module3_validation.error_detector import detect_errors
from module3_validation.validation_api import (
    build_validation_report, run_validation_report)
from module3_validation.evaluation import evaluate, run_evaluation

validate_dataframe(df)                  # 20 rules -> counts + row positions
detect_anomalies(df)                    # IF + LOF scores/flags per row
classify_columns(df)                    # content-based semantics per column
detect_errors(df)                       # impossible / suspicious / drift
build_validation_report(df)             # everything -> report dict (no I/O)
run_validation_report()                 # cleaned CSV -> validation_report.json
evaluate(df)                            # injected-anomaly P/R/AUC (no I/O)
run_evaluation()                        # -> evaluation_report.md + roc_curve.png
```

```bash
python module3_validation/rule_validator.py
python module3_validation/anomaly_detector.py --contamination 0.01 --top-n 10
python module3_validation/column_classifier.py --demo-mislabel
python module3_validation/validation_api.py
python module3_validation/evaluation.py --n-per-type 25 --seed 7
```

## Module 4 — Pipeline (`module4_pipeline` + root CLI)

```python
from module4_pipeline.orchestrator import run_pipeline
from module4_pipeline.interfaces import load_config, validate_config

summary = run_pipeline()                            # config.yaml, full run
summary = run_pipeline("path/to/config.yaml", dry_run=True)  # preflight only
```

```bash
python pipeline.py --input data.csv --output results/
python pipeline.py --input online_retail_II.xlsx --output results/
python module4_pipeline/orchestrator.py --config module4_pipeline/config.yaml
```

`pipeline.py` extras: `--dry-run`, CSV→workbook staging, all artifacts
redirected under `--output/`, and the merged `run_config.yaml` saved next
to the results. Config reference lives in `module4_pipeline/config.yaml`
(paths, `nrows_per_sheet`, `imputation_method`, `knn_neighbors`,
`fuzzy_threshold`, `contamination`, `classifier_sample_n`, logging level).
