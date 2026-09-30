# Architecture — Automated Data Cleaning & Validation System

![Architecture diagram](architecture_diagram.png)

## What the system is

A four-module batch pipeline that takes the raw Online Retail II workbook
and produces a cleaned, validated dataset with machine-readable reports at
every step. Each module has a single file-based API; the orchestrator chains
them by passing output files as inputs of the next stage. Nothing is passed
in memory between modules — every handoff is an inspectable artifact.

```text
online_retail_II.xlsx (100,000 rows sampled, 8 columns)
  │  Module 1 — Profiling (Sprints 1-3)
  ▼  profiling_report.json + visuals/*.png
  │  Module 2 — Cleaning (Sprints 4-6, gated by expected_schema.json)
  ▼  cleaned_data.csv (98,918 × 21) + cleaning_log.json
  │  Module 3 — Validation (Sprints 7-9)
  ▼  validation_report.json  (health 84.53, Watch)
```

Module 4 (Sprints 10-11) is the harness around all of it: config, logging,
orchestration, CLI, container, versioning.

## Module map

| Module | Files | Sprints | Responsibility |
|---|---|---|---|
| `module1_profiling` | `metadata_extractor.py`, `profiler.py`, `profiling_api.py` | 1-3 | Types + semantics, statistics + PNGs, rule-engine flags (suspicious / format / PII) |
| `module2_cleaning` | `schema_inference.py`, `cleaner.py`, `transformer.py`, `cleaning_api.py` | 4-6 | Expected-schema contract, normalize → impute → dedupe, features + scaling, 0-100 quality scoring |
| `module3_validation` | `rule_validator.py`, `anomaly_detector.py`, `column_classifier.py`, `error_detector.py`, `validation_api.py`, `evaluation.py` | 7-9 | 20 explainable rules, IF + LOF anomalies, NLP column check, impossible/suspicious/drift findings, severity + health scoring, injected-anomaly evaluation |
| `module4_pipeline` | `interfaces.py`, `logger.py`, `orchestrator.py`, `config.yaml` | 10 | Stage contracts + config validation, logging, ordered execution |

Plus the Sprint 11 shell: root `pipeline.py` (CLI), `Dockerfile` +
`requirements.txt` + `.dockerignore`, Git-LFS versioning (`.gitattributes`).

## Key design decisions (and why)

- **File-based handoffs, not in-memory.** Any stage can be re-run, audited,
  or replaced without touching the others. The integration test asserts the
  handoff chain (profiling rows == cleaning rows-in; cleaning rows-out ==
  validation rows).
- **Schema as a versioned contract.** `expected_schema.json` (Sprint 4) is an
  *input* to cleaning, not a per-run output. Cleaning enforces it; the
  quality scorer measures against it (91.19 → 100.0).
- **Rules before models.** The 20 deterministic rules (Sprint 7) catch what
  is explainable; the ML layer (Sprint 8) only hunts what rules cannot
  express. They cover each other's blind spots — evaluation proved IF is
  blind to far-out returns that `quantity_out_of_bounds` catches trivially.
- **Determinism is engineered, not assumed.** Fixed seeds (IF 42, injection
  7), deduplicated LOF fitting (distance-0 neighbours made raw scores
  explode past 1e8), and `lineterminator="\n"` on every CSV write so Windows
  and Linux outputs are byte-identical (verified by SHA-256).
- **Content beats headers.** The column classifier weights cell evidence
  above column-name cues, so mislabeled columns are caught by what they
  contain (`--demo-mislabel` proves all three decoys flag).
- **Honest scoring.** Hard rules that pass report zero examples rather than
  fabricated ones; precision looks low in evaluation because a 1% flag rate
  against 100 injected rows caps it at ~0.10 — reported as triage cost, not
  hidden.

## Failure handling

- `interfaces.preflight` fails fast on missing stage inputs before any heavy
  work. `validate_config` reports *all* config problems at once.
- The orchestrator stops at the first failed stage by default
  (`stop_on_stage_failure`), verifies each stage's contract outputs exist,
  and escalates critical profiling flags / non-Healthy grades to log
  warnings. Stage timings and statuses go to `pipeline.log` and the summary.
