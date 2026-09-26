# Module 2 — Schema Inference (`module2_cleaning`)

Defines what "correct" looks like for the Online Retail II dataset *before*
anything is cleaned, and packages it into the Module 2 contract:
**`expected_schema.json`**.

Built in Sprint 4:

| Sprint | File | Contents |
|---|---|---|
| 4 — Schema inference | `schema_inference.py` | Reads Sprint 3's `profiling_report.json`, derives expected type / ranges / formats per column → `expected_schema.json` |

## Calling the engine

`schema_inference.py` is the only entry point. Two functions:

```python
from module2_cleaning.schema_inference import (
    load_profiling_report, infer_schema, run_schema_inference,
    build_column_schema,
)

# Option A — you already hold the Module 1 report as a dict:
schema: dict = infer_schema(report)

# Option B — file path route (loads report, saves expected_schema.json):
schema: dict = run_schema_inference(
    report_path="module1_profiling/profiling_report.json",  # default
    output_path="module2_cleaning/expected_schema.json",    # default
)

# Single column (e.g. for debugging one builder):
entry: dict = build_column_schema(report, "StockCode")
```

CLI equivalent:

```bash
python module2_cleaning/schema_inference.py
python module2_cleaning/schema_inference.py --report module1_profiling/profiling_report.json
python module2_cleaning/schema_inference.py --report ... --output expected_schema.json
```

The run exits non-zero unless every self-validation check passes.

## How inference works

Each column's `semantic_type` (from Module 1) selects a rule builder, which
binds **report evidence** to **domain priors**:

| Role | Columns | What is derived |
|---|---|---|
| `key` (identifier) | InvoiceNo, StockCode, CustomerID | Value *families* whose observed row counts sum to the profiled row count (e.g. StockCode = 82,795 numeric + 16,754 suffixed + 22 `C`-prefixed + 429 service codes), combined into one regex + strict as-observed variant |
| `measure` | Quantity, UnitPrice | Hard bounds + IQR tiers (`typical` 1.5×, `plausible` 3×, `atypical` review, `invalid` reject), sign/zero/currency rules |
| `timestamp` | InvoiceDate | Fixed `%Y-%m-%d %H:%M:%S` format + business window derived by flooring/ceiling the observed min/max to calendar years (2009-01-01…2011-12-31) |
| `text` | Description | Trim/case/spacing rules + null policy (impute from StockCode, zero residual nulls) |
| `dimension` | Country | Closed 28-name reference set + alias map (`EIRE`→Ireland, `USA`→United States, …), title-case pattern |

Every rule carries `basis`: `observed` (measured in the report), `domain`
(business/physical invariant no profiler can discover, e.g. "a price is
never negative"), or `derived` (computed from observed evidence by a
documented rule, e.g. the CustomerID block 12000…19000). Unknown columns
fall back to a generic dtype-only schema.

## `expected_schema.json` structure

```jsonc
{
  "module": "module2_cleaning",
  "schema_version": "1.0.0",
  "inferred_from": { "report": "module1_profiling/profiling_report.json", "...": "provenance" },
  "dataset": {
    "grain": "one row per invoice line item",
    "primary_key": null,               // no single column is unique
    "key_candidates": [{ "columns": ["InvoiceNo", "StockCode", "InvoiceDate"],
                          "uniqueness_test": "required" }],
    "role_counts": { "key": 3, "measure": 2, "...": "..." }
  },
  "row_validity_rules": [              // the 5 legitimate-but-special row kinds
    { "rule_id": "rv-2-return-line", "observed_rows": 2263,
      "handling": "keep with sign (IsReturn); never take abs()" }
  ],
  "derived_columns": [                 // LineValue, IsCancellation, IsReturn, IsGiveaway
    { "column": "LineValue", "expression": "Quantity * UnitPrice", "...": "..." }
  ],
  "columns": {
    "<column>": {
      "expected_dtype": "float",       // logical type after cleaning
      "nullable": false,
      "storage": { "observed_storage_type": "...", "expected_storage_type": "..." },
      "value_range": { "min": 0.0, "max": 20000.0, "tiers": [ ... ] },
      "expected_format": { "pattern": "...", "pattern_observed": "...", "...": "..." },
      "allowed_values": { "...": "Country only — closed reference set + aliases" },
      "null_policy": { "...": "Description only — impute, then zero nulls" },
      "constraints": [ { "constraint_id": "unitprice-non-negative",
                         "severity": "hard" /* hard | review | info */,
                         "basis": "observed", "observed_violations": 0 } ],
      "observed": { "...": "report figures this entry rests on" },
      "expected_post_clean": { "n_negative_max": 0, "...": "acceptance targets" }
    }
  },
  "cleaning_plan": [                   // the 7 cleaning ops the schema mandates
    { "step": 6, "operation": "impute_description_from_stockcode", "...": "..." }
  ],
  "validation": { "all_passed": true, "checks": [ ... ] }
}
```

## Constraint catalogue (47 total: 28 hard, 16 review, 3 info)

| Column | Highlights |
|---|---|
| InvoiceNo | Text, `^(?:\d{5,7}|C\d{5,7})$` (observed strictly 6-digit); 2,077 `C`-rows preserved, never dropped |
| StockCode | 4 families covering 100% of rows; upper-case + trimmed after cleaning |
| Description | Stripped, single-spaced, upper-case; 0.41% gaps imputed from StockCode → 0 residual nulls |
| Quantity | Integer, ±10,000 hard; ≠ 0; negatives (2.26%) kept as returns; \|q\| > 1,000 reviewed |
| InvoiceDate | `%Y-%m-%d %H:%M:%S`, second resolution, tz-naive, within 2009…2011, never null |
| UnitPrice | 0.0…20,000.0, ≤ 2 decimals; 595 zero-price rows kept flagged; > 1,000.0 verified |
| CustomerID | Nullable Int64 (not float); block 12000…19000; 31.78% nulls are structural guest orders; PII — hash before sharing |
| Country | Title-case pattern + closed 28-name set with alias normalisation; quasi-identifier |

`severity` is `hard` (reject/fix — data error), `review` (flag for a human —
plausible but suspicious), or `info` (expected quirk the pipeline must
preserve, e.g. return quantities).

## Verification

- The engine's `validation` section re-checks itself at run time: all 8
  columns covered, identifier family counts sum to the profiled row counts,
  every column has a concrete value domain, no stub text.
- Independently verified against the raw 100k sample: every derived regex,
  bound, precision and alias rule matches with **zero violations**
  (InvoiceNo 0, StockCode 0, InvoiceDate 0 mismatches; Quantity integer;
  UnitPrice ≥ 0 with ≤ 2 dp; CustomerID whole and in-block; all 28
  countries normalise into the reference set).

## Notes for downstream modules

- Import from the package root:
  `from module2_cleaning.schema_inference import infer_schema` (works whether
  the working directory is the repo root or `module2_cleaning/`).
- The report path is the **only** input — inference is deterministic; the
  sole non-deterministic field is the `generated_at` timestamp.
- `expected_post_clean` per column is the acceptance test for cleaning:
  assert those counts after each cleaning step.
- `cleaning_plan` maps each mandated cleaning operation to the schema
  rule(s) requiring it — implement cleaning steps in that order.
