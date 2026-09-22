# Module 1 — Data Profiling (`module1_profiling`)

Measures the health and shape of the Online Retail II dataset and packages
everything into the official Module 1 output: **`profiling_report.json`**.

Built over three sprints:

| Sprint | File | Contents |
|---|---|---|
| 1 — Metadata | `metadata_extractor.py` | Logical dtype, semantic type, mixed-type flag per column → `metadata_sample.json` |
| 2 — Statistics + visuals | `profiler.py` | Missing-value matrix, dtype consistency, cardinality, correlation + distributions; three PNGs in `visuals/` |
| 3 — Rules + final report | `profiling_api.py` | Rule engine (suspicious / format / PII flags) + unified API → `profiling_report.json` |

## Calling Module 1 from another module

`profiling_api.py` is the only entry point you need. Two functions:

```python
from module1_profiling.profiling_api import profile_dataframe, run_profiling

# Option A — you already hold the raw dataset as a DataFrame:
report: dict = profile_dataframe(df)                     # returns final report dict
report: dict = profile_dataframe(df, visuals_dir="...")  # custom PNG location

# Option B — let Module 1 load the workbook and save the report file:
report: dict = run_profiling(
    input_path="online_retail_II.xlsx",   # default: repo-root workbook
    output_path="module1_profiling/profiling_report.json",
    nrows_per_sheet=50000,                # 0 = all rows (slow, needs lots of RAM)
)
```

CLI equivalents:

```bash
python module1_profiling/profiling_api.py
python module1_profiling/profiling_api.py --input online_retail_II.xlsx --nrows-per-sheet 50000
python module1_profiling/profiling_api.py --output module1_profiling/profiling_report.json --heatmap-rows 1000
```

## Function signatures

```python
# profiling_api.py
def profile_dataframe(df: pd.DataFrame, visuals_dir: str | Path = "module1_profiling/visuals",
                      heatmap_rows: int = 1000, source_file: str | None = None,
                      nrows_per_sheet: int | None = None) -> dict
def run_profiling(input_path: str | Path = <repo>/online_retail_II.xlsx,
                  output_path: str | Path = "module1_profiling/profiling_report.json",
                  nrows_per_sheet: int = 50000,
                  visuals_dir: str | Path = "module1_profiling/visuals",
                  heatmap_rows: int = 1000) -> dict
def run_rule_engine(df: pd.DataFrame, metadata: dict, statistics: dict) -> dict
def check_suspicious_columns(df, metadata, statistics) -> list[dict]
def check_inconsistent_formats(df, metadata, statistics) -> list[dict]
def check_pii_columns(df, metadata) -> list[dict]

# metadata_extractor.py (Sprint 1 — used internally, reusable standalone)
def load_dataset(path, nrows_per_sheet=50000) -> pd.DataFrame
def extract_metadata(df: pd.DataFrame) -> dict
def infer_logical_dtype(series: pd.Series) -> str        # int | float | string | date | boolean
def detect_mixed_type(series: pd.Series) -> dict
def infer_semantic(column: str, series: pd.Series, dtype: str) -> dict

# profiler.py (Sprint 2 — used internally, reusable standalone)
def profile_dataset(df, visuals_dir=..., heatmap_rows=1000) -> dict
def missing_value_matrix(df: pd.DataFrame) -> dict
def dtype_consistency(df: pd.DataFrame, inferred: dict | None = None) -> dict
def cardinality(df: pd.DataFrame, high_card_threshold: float = 0.5) -> dict
def numeric_summary(df: pd.DataFrame, columns: list | None = None) -> dict
```

## `profiling_report.json` schema

Top level:

```jsonc
{
  "module": "module1_profiling",   // producing module
  "version": "1.0.0",              // report schema version
  "source_file": "...xlsx",        // provenance: workbook profiled
  "nrows_per_sheet": 50000,        // provenance: sampling bound (0 = full scan)
  "n_rows": 100000,                // rows profiled (all sheets concatenated)
  "n_columns": 8,
  "columns": ["InvoiceNo", "StockCode", "Description", "Quantity",
              "InvoiceDate", "UnitPrice", "CustomerID", "Country"],
  "metadata":    { "...": "Sprint 1 — see below" },
  "statistics":  { "...": "Sprint 2 — see below" },
  "flags":       { "...": "Sprint 3 — see below" },
  "visuals":     { "...": "paths of the three PNGs" }
}
```

`metadata` (Sprint 1 — from `extract_metadata`):

```jsonc
{
  "n_rows_profiled": 100000,
  "n_columns": 8,
  "columns": ["InvoiceNo", "..."],
  "fields": {
    "<column>": {
      "column": "CustomerID",
      "inferred_dtype": "int",              // int | float | string | date | boolean
      "semantic_type": "customer_id",       // e.g. invoice_id, quantity, unit_price, country
      "semantic_description": "...",
      "is_mixed_type": false,               // numeric-looking column with text codes mixed in
      "mixed_type_detail": { "reason": "...", "n_numeric_like": 0, "n_text": 0,
                             "numeric_ratio": 0.0, "text_examples": [] },
      "n_rows": 100000, "n_null": 31778, "null_pct": 31.78,
      "n_unique": 1720,
      "sample_values": ["13085", "..."],
      "min": 12346.0, "max": 18287.0       // present for int/float/date columns only
    }
  }
}
```

`statistics` (Sprint 2):

```jsonc
{
  "missing_value_matrix": {
    "per_column": { "<col>": { "n_missing": 0, "missing_pct": 0.0 } },
    "per_row": { "rows_with_0_missing": 0, "rows_with_1_missing": 0,
                 "rows_with_2_or_more_missing": 0, "max_missing_per_row": 0 },
    "total_cells": 0, "total_missing_cells": 0, "total_missing_pct": 0.0
  },
  "dtype_consistency": {
    "<col>": { "inferred_dtype": "int", "n_checked": 0, "n_consistent": 0,
               "n_inconsistent": 0, "consistency_rate": 1.0,
               "inconsistent_examples": [] }
  },
  "cardinality": {
    "<col>": { "n_unique": 0, "cardinality_ratio": 0.0,   // n_unique / n_rows
               "cardinality_level": "low (categorical-like)",  // low | medium | high (identifier-like)
               "top_values": { "<value>": 0 } }              // top-5 frequent values
  },
  "numeric_profile": {
    "columns": ["Quantity", "UnitPrice"],
    "pearson_correlation": { "Quantity": { "Quantity": 1.0, "UnitPrice": 0.0 } },
    "distributions": {
      "<col>": { "count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                 "min": 0.0, "max": 0.0, "q1": 0.0, "q3": 0.0, "iqr": 0.0,
                 "iqr_lower": 0.0, "iqr_upper": 0.0,
                 "n_iqr_outliers": 0, "outlier_pct": 0.0,
                 "n_negative": 0, "n_zero": 0, "skew": 0.0, "kurtosis": 0.0 }
    }
  }
}
```

`flags` (Sprint 3 — rule engine; an empty list means that check passed):

```jsonc
{
  "suspicious_columns":  [ { "rule_id": "missing-high", "category": "suspicious",
                             "severity": "warning", "column": "CustomerID",
                             "message": "...", "evidence": { "missing_pct": 31.78 } } ],
  "inconsistent_formats": [ { "rule_id": "mixed-type", "category": "format", ... } ],
  "pii_columns":          [ { "rule_id": "pii-direct", "category": "pii", ... } ],
  "summary": { "n_flags": 0, "n_critical": 0, "n_warning": 0, "n_info": 0 }
}
```

`severity` is one of `info` (expected quirk, e.g. return quantities), `warning`
(needs handling downstream), `critical` (unusable/erroneous, e.g. ≥95% missing,
negative prices). `visuals` maps `missing_heatmap` / `outlier_distribution` /
`correlation_heatmap` to the PNG paths.

## Rule catalogue

| rule_id | Family | Severity | What it flags (thresholds in `profiling_api.py`) |
|---|---|---|---|
| `missing-critical` | suspicious | critical | Column ≥ 95% missing |
| `missing-high` | suspicious | warning | Column ≥ 20% missing (CustomerID ≈ 32%) |
| `empty-column` / `constant-column` | suspicious | critical / warning | No usable values / single repeated value |
| `negative-quantity` | suspicious | info | Returns/cancellations (expected in retail) |
| `bulk-quantity` | suspicious | info | \|Quantity\| > 1000 (bulk orders or entry outliers) |
| `zero-quantity` | suspicious | warning | Physically meaningless line items |
| `negative-price` | suspicious | critical | Impossible data error |
| `zero-price` | suspicious | warning | Adjustments/giveaways, not real prices |
| `extreme-price` | suspicious | info | UnitPrice > 1000, verify vs source system |
| `cancelled-invoices` | suspicious | info | `C`-prefixed InvoiceNo share (not distinct orders) |
| `date-out-of-range` | suspicious | warning | InvoiceDate outside 2009-01-01…2011-12-31 |
| `mixed-type` | format | warning | Numeric + text codes (InvoiceNo `C…`, StockCode `85123A`/`POST`) |
| `dtype-mismatch` | format | warning | Values not matching the inferred dtype |
| `date-as-text` | format | warning | Object-dtype date column with unparseable rows |
| `padded-text` | format | warning | Leading/trailing whitespace (strip in cleaning) |
| `case-variants` | format | info | Values differing only by case (normalise) |
| `pii-direct` | pii | warning | Direct identifiers (CustomerID → anonymise/hash) |
| `pii-quasi` | pii | info | Quasi-identifiers (Country → mind k-anonymity) |

## Visuals

Rendered with the matplotlib `Agg` backend (files only, never interactive)
into `module1_profiling/visuals/`:

- `missing_heatmap.png` — binary missingness map over a 1000-row sample; the CustomerID band (≈32% missing) is clearly visible.
- `outlier_distribution.png` — 99.5%-capped histograms + boxplots for Quantity/UnitPrice.
- `correlation_heatmap.png` — Pearson correlation over numeric columns.

## Notes for downstream modules

- Import from the package root: `from module1_profiling.profiling_api import …` (works whether
  the working directory is the repo root or `module1_profiling/`).
- `profile_dataframe` does **not** rename headers — pass DataFrames with canonical
  headers, or use `run_profiling` / `load_dataset`, which normalise raw headers
  (`Invoice` → `InvoiceNo`, `Price` → `UnitPrice`, `Customer ID` → `CustomerID`).
- The default `nrows_per_sheet=50000` profiles a 100k-row representative sample
  (the full workbook has >1M rows); pass `0` for a full scan.
- The raw `*.xlsx` is git-ignored; the JSON report and PNGs are the committed artefacts.
