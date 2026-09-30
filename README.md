# Automated Data Cleaning & Validation System

Batch pipeline that turns the raw Online Retail II workbook into a cleaned,
validated dataset — with a machine-readable report at every step and one
health score at the end.

```text
online_retail_II.xlsx (100,000 rows)
  → Module 1 Profiling  → profiling_report.json
  → Module 2 Cleaning   → cleaned_data.csv (98,918 rows, quality 100.0)
  → Module 3 Validation → validation_report.json (health 84.53, Watch)
```

## Run it

```bash
pip install -r requirements.txt
python pipeline.py --input online_retail_II.xlsx --output results/
# CSV works too:  python pipeline.py --input data.csv --output results/
```

Or via Docker (data in/out through mounts):

```bash
docker build -t cadetx-pipeline .
docker run --rm -v "%cd%/online_retail_II.xlsx:/app/data/in.xlsx:ro" \
  -v "%cd%/results:/app/results" \
  cadetx-pipeline --input /app/data/in.xlsx --output /app/results
```

Or stage by stage: `python module4_pipeline/orchestrator.py`
(preflight only with `--dry-run`). All settings live in
`module4_pipeline/config.yaml` — no hardcoded paths or thresholds.

## Folder structure

```text
pipeline.py                  # production CLI (--input/--output)
Dockerfile / requirements.txt / .dockerignore
module1_profiling/           # S1-3: metadata, statistics+PNGs, rule flags
module2_cleaning/            # S4-6: schema contract, cleaning, features, scoring
module3_validation/          # S7-9: rules, IF+LOF anomalies, NLP columns,
                             #       drift, health scoring, evaluation
module4_pipeline/            # S10: config, logger, orchestrator
docs/                        # architecture, api_usage, pipeline_flow + diagram
tests/                       # 39 tests (unit + integration + deploy)
online_retail_II.xlsx        # raw input (Git-LFS)
```

## Where each module's output lives

| Output | Path | What it is |
|---|---|---|
| Profiling report + visuals | `module1_profiling/profiling_report.json`, `visuals/` | types, stats, 13 flags (0 critical) |
| Cleaned data + log | `module2_cleaning/cleaned_data.csv`, `cleaning_log.json` | 98,918 × 21, quality 91.19 → 100.0 |
| Rule violations | `module3_validation/rule_violations_sample.json` | 9/20 rules fire (review/info only) |
| Anomalies | `module3_validation/anomalies_sample.json` | IF 988, LOF 58, agree 53 |
| Columns | `module3_validation/column_classification.json` | 10/10 match |
| Validation report | `module3_validation/validation_report.json` | health 84.53, Watch (2010 fee-regime drift) |
| Evaluation | `module3_validation/evaluation_report.md`, `roc_curve.png` | IF AUC 0.99, LOF AUC 0.81 on 100 injected anomalies |
| Pipeline log | `module4_pipeline/pipeline.log` | per-stage timings + warnings |

## Docs & tests

- `docs/architecture.md` (with `architecture_diagram.png`), `docs/api_usage.md`,
  `docs/pipeline_flow.md`
- `python -m pytest tests/ -q` — 39 passing: unit, cross-stage integration
  (row-count handoff chain), and deploy (real CLI run) tests.

## Notes for the unfamiliar

- Large files (`*.xlsx`, cleaned CSV, `*.png`) are Git-LFS tracked.
- CSV writes use `lineterminator="\n"`, so Windows, Linux, and container
  outputs are byte-identical (verified by SHA-256).
- Known limitation, documented in the evaluation: IsolationForest misses
  far-out negative-quantity returns (subsample geometry) — the rule engine
  covers that blind spot, which is the point of the layered design.
