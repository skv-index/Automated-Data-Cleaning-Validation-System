"""Sprint 1 - Metadata Extraction Engine.

Loads the Online Retail dataset and, for every column, infers:
  * logical data type  : one of {"int", "float", "string", "date", "boolean"}
  * semantic meaning   : e.g. invoice_id, product_id/SKU, product_description,
                         quantity, transaction_datetime, unit_price,
                         customer_id, country (plus generic fallbacks such as
                         email / phone / url / category for reuse on other data)
  * mixed-type flag    : True when a column looks numeric but contains text
                         values mixed in (e.g. StockCode "85123A", InvoiceNo
                         "C536379", service codes like "POST" / "DOT").

Usage:
    python metadata_extractor.py
    python metadata_extractor.py --input ..\\online_retail_II.xlsx --output metadata_sample.json
    python metadata_extractor.py --input ..\\online_retail_II.xlsx --nrows-per-sheet 50000

The script prints a JSON summary to stdout and saves it to
`metadata_sample.json` (next to this file by default).

The default reads up to ``DEFAULT_NROWS_PER_SHEET`` rows per sheet. The full
workbook has >1M rows, so a bounded representative sample keeps inference fast
while remaining accurate for type/semantic detection. Pass
``--nrows-per-sheet 0`` to scan every row (slow, needs lots of RAM).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical column names expected in the deliverable sample output.
EXPECTED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]

#: Raw header (lower-cased, stripped) -> canonical name. The workbook uses
#: "Invoice" (not "InvoiceNo"), "Price" (not "UnitPrice") and "Customer ID".
COLUMN_ALIASES = {
    "invoice": "InvoiceNo",
    "invoiceno": "InvoiceNo",
    "invoice_no": "InvoiceNo",
    "invoicenumber": "InvoiceNo",
    "stockcode": "StockCode",
    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "qty": "Quantity",
    "invoicedate": "InvoiceDate",
    "invoice_date": "InvoiceDate",
    "price": "UnitPrice",
    "unitprice": "UnitPrice",
    "unit_price": "UnitPrice",
    "customerid": "CustomerID",
    "customer_id": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "online_retail_II.xlsx"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "metadata_sample.json"
DEFAULT_NROWS_PER_SHEET = 50000

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_RE = re.compile(r"^(https?://|www\.)\S+$", re.IGNORECASE)
PHONE_RE = re.compile(r"^\+?[\d][\d\s().-]{6,}$")
PURE_INT_RE = re.compile(r"^[+-]?\d+$")
PURE_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")
CANCELLED_INVOICE_RE = re.compile(r"^C\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Loading / normalisation
# ---------------------------------------------------------------------------

def normalize_column_name(raw: str) -> str:
    """Map a raw workbook header to its canonical name (fall back to stripped raw)."""
    key = str(raw).strip().lower()
    return COLUMN_ALIASES.get(key, str(raw).strip())


def load_dataset(
    path: str | Path = DEFAULT_INPUT,
    nrows_per_sheet: int = DEFAULT_NROWS_PER_SHEET,
) -> pd.DataFrame:
    """Load all sheets of the workbook, normalise headers, concatenate rows.

    Args:
        path: Path to ``online_retail_II.xlsx``.
        nrows_per_sheet: Max data rows read per sheet (``0``/``None`` = all rows).

    Returns:
        DataFrame with canonical column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    nrows = None if nrows_per_sheet in (None, 0) else int(nrows_per_sheet)
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, nrows=nrows)
        df.columns = [normalize_column_name(c) for c in df.columns]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Keep canonical order first, then anything unexpected.
    ordered = [c for c in EXPECTED_COLUMNS if c in combined.columns]
    rest = [c for c in combined.columns if c not in ordered]
    return combined[ordered + rest]


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def _non_null_strings(series: pd.Series) -> list[str]:
    """Non-null values rendered as stripped strings (for pattern analysis)."""
    return [str(v).strip() for v in series.dropna().tolist() if str(v).strip() != ""]


def infer_logical_dtype(series: pd.Series) -> str:
    """Infer one of {"int", "float", "string", "date", "boolean"}.

    Strategy (NaN-robust):
      1. datetime64 dtype (or name suggesting date + mostly parseable) -> "date"
      2. boolean dtype / 0-1 / true-false only                 -> "boolean"
      3. all values integer-like (numeric or digit strings)    -> "int"
      4. all values numeric-like (incl. decimals)              -> "float"
      5. otherwise                                             -> "string"
    """
    s = series.dropna()
    if s.empty:
        return "string"
    if pd.api.types.is_bool_dtype(s.dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return "date"

    lowered = {str(v).strip().lower() for v in s.tolist()}
    if lowered <= {"true", "false"} or lowered <= {"0", "1"} or lowered <= {"0", "1", "true", "false"}:
        # Genuine boolean only if the column is not a numeric measure/id with
        # many distinct values — a bare {0,1} set is the signal.
        if s.nunique() <= 2:
            return "boolean"

    # Datetime attempt for object/string columns — but only when the values
    # actually look like dates. (Plain digit strings such as invoice numbers
    # would otherwise "parse" as nanosecond timestamps, e.g. 489434 ->
    # 1970-01-01 00:00:00.000489434, producing false "date" hits.)
    if s.dtype == object or str(s.dtype) == "str":
        texts_preview = [str(v).strip() for v in s.head(2000).tolist()]
        datey = sum(
            1
            for t in texts_preview
            if re.search(r"[-/:]", t) or re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", t, re.I)
        )
        if texts_preview and datey / len(texts_preview) >= 0.5:
            try:
                with pd.option_context("mode.chained_assignment", None):
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(s, errors="coerce", format="mixed")
                if float(parsed.notna().mean()) >= 0.9:
                    return "date"
            except Exception:
                pass

    # Numeric attempt.
    numeric = pd.to_numeric(s, errors="coerce")
    if bool((numeric.notna()).all()):
        # All values numeric: int if every value is whole (6.0 counts as int-like
        # for dtype purposes; the float/int call is about representation).
        try:
            if bool(((numeric % 1) == 0).all()):
                return "int"
        except Exception:
            pass
        return "float"

    # Pure digit strings (e.g. a column read as object but all "12345").
    texts = _non_null_strings(series)
    if texts and all(PURE_INT_RE.match(t) for t in texts):
        return "int"
    if texts and all(PURE_FLOAT_RE.match(t) for t in texts):
        # Distinguish  "12" (int) from "12.5" (float).
        return "float" if any("." in t or "e" in t.lower() for t in texts) else "int"

    return "string"


# ---------------------------------------------------------------------------
# Mixed-type detection
# ---------------------------------------------------------------------------

def detect_mixed_type(series: pd.Series, min_text_examples: int = 5) -> dict:
    """Detect columns that look numeric but contain text values.

    A column is "mixed-type" when a substantial majority of values are
    numeric-like but a non-trivial minority are not (or vice versa for
    mostly-text columns with embedded numeric codes, e.g. StockCode).

    Returns a dict with ``is_mixed_type`` plus supporting evidence.
    """
    texts = _non_null_strings(series)
    total = len(texts)
    if total == 0:
        return {
            "is_mixed_type": False,
            "reason": "empty column (no non-null values)",
            "n_numeric_like": 0,
            "n_text": 0,
            "numeric_ratio": 0.0,
            "text_examples": [],
        }

    numeric_like = sum(1 for t in texts if PURE_FLOAT_RE.match(t))
    n_text = total - numeric_like
    numeric_ratio = numeric_like / total

    # Pure-digit core with alpha-suffixed variants, e.g. 85123 vs 85123A.
    digit_core = sum(1 for t in texts if re.match(r"^[+-]?\d+[A-Za-z]+$", t))
    # Cancellation-style prefix codes, e.g. C536379.
    prefixed = sum(1 for t in texts if CANCELLED_INVOICE_RE.match(t))

    text_examples: list[str] = []
    for t in texts:
        if not PURE_FLOAT_RE.match(t) and t not in text_examples:
            text_examples.append(t)
        if len(text_examples) >= min_text_examples:
            break

    # Mixed when both families are present beyond noise level:
    # at least `min_text_examples`-ish text values AND at least 1% of rows
    # (avoids flagging a single typo in a huge column... but StockCode/InvoiceNo
    # genuinely carry thousands of such codes, so they still trigger).
    minority = min(numeric_like, n_text)
    minority_ratio = minority / total
    is_mixed = minority >= min(min_text_examples, 50) and minority_ratio >= 0.005

    # Extra trigger: digit-core + letter-suffix pattern is the textbook
    # mixed-type case even at lower ratios.
    if not is_mixed and digit_core >= min_text_examples and numeric_ratio > 0.5:
        is_mixed = True

    if is_mixed:
        if numeric_ratio >= 0.5:
            reason = (
                f"mostly numeric ({numeric_ratio:.1%}) with {n_text} text value(s) "
                f"mixed in (e.g. {', '.join(text_examples[:3])})"
            )
        else:
            reason = (
                f"mostly text ({1 - numeric_ratio:.1%}) with {numeric_like} "
                f"pure-numeric value(s) mixed in"
            )
        if prefixed:
            reason += f"; includes {prefixed} cancellation-style 'C'-prefixed code(s)"
    else:
        reason = (
            "uniform type (all numeric-like or all text)"
            if minority == 0
            else f"minority class too small to be structural ({minority}/{total})"
        )

    return {
        "is_mixed_type": bool(is_mixed),
        "reason": reason,
        "n_numeric_like": int(numeric_like),
        "n_text": int(n_text),
        "numeric_ratio": round(float(numeric_ratio), 4),
        "n_digit_core_with_letter_suffix": int(digit_core),
        "n_cancelled_style_prefix": int(prefixed),
        "text_examples": text_examples,
    }


# ---------------------------------------------------------------------------
# Semantic inference
# ---------------------------------------------------------------------------

_SEMANTICS_BY_COLUMN = {
    "InvoiceNo": (
        "invoice_id",
        "Transaction/invoice identifier; a leading 'C' prefix denotes a "
        "cancelled transaction rather than a distinct order.",
    ),
    "StockCode": (
        "product_id (SKU)",
        "Product stock-keeping code; alphanumeric - numeric codes with "
        "letter suffixes (e.g. '85123A') and service codes ('POST', 'DOT', "
        "'M', 'BANK CHARGES') appear alongside plain numbers.",
    ),
    "Description": (
        "product_description",
        "Free-text name/description of the product.",
    ),
    "Quantity": (
        "quantity",
        "Units sold (positive) or returned/cancelled (negative).",
    ),
    "InvoiceDate": (
        "transaction_datetime",
        "Date and time the invoice was raised.",
    ),
    "UnitPrice": (
        "unit_price",
        "Price per unit in local currency (0.0 occurs for adjustments/giveaways).",
    ),
    "CustomerID": (
        "customer_id",
        "Anonymised customer identifier (ID, not a measure).",
    ),
    "Country": (
        "country",
        "Customer/delivery country (low-cardinality geography).",
    ),
}


def infer_semantic(column: str, series: pd.Series, dtype: str) -> dict:
    """Infer the likely semantic meaning of a column.

    Known retail columns resolve via a lookup; unknown columns fall back to
    generic content detectors (email, url, phone, person-name-ish, boolean
    flag, datetime, id, category, measure, text).
    """
    if column in _SEMANTICS_BY_COLUMN:
        sem_type, description = _SEMANTICS_BY_COLUMN[column]
        return {"semantic_type": sem_type, "description": description}

    texts = _non_null_strings(series)
    sample = texts[:2000]
    name = column.lower()

    if dtype == "date" or "date" in name or "time" in name:
        return {"semantic_type": "datetime", "description": "Date/time field."}
    if dtype == "boolean":
        return {"semantic_type": "boolean_flag", "description": "True/false indicator."}
    if sample and all(EMAIL_RE.match(t) for t in sample):
        return {"semantic_type": "email", "description": "Email address."}
    if sample and sum(bool(URL_RE.match(t)) for t in sample) / len(sample) >= 0.9:
        return {"semantic_type": "url", "description": "URL."}
    if sample and sum(bool(PHONE_RE.match(t)) for t in sample) / len(sample) >= 0.9:
        return {"semantic_type": "phone", "description": "Phone number."}
    if "price" in name or "amount" in name or "cost" in name or "total" in name:
        return {"semantic_type": "price/amount", "description": "Monetary measure."}
    if "qty" in name or "quantity" in name or "count" in name:
        return {"semantic_type": "quantity", "description": "Count/quantity measure."}
    if re.search(r"(^|_)(id|no|code|key)$|^id$", name) or name in {"code", "sku"}:
        return {"semantic_type": "id/code", "description": "Identifier/code field."}
    if "name" in name:
        return {"semantic_type": "name", "description": "Name field."}
    if dtype == "string" and series.nunique(dropna=True) <= max(50, 0.05 * len(series)):
        return {"semantic_type": "category", "description": "Low-cardinality categorical field."}
    if dtype in ("int", "float"):
        return {"semantic_type": "numeric_measure", "description": "Numeric measure."}
    return {"semantic_type": "text", "description": "Free-text field."}


# ---------------------------------------------------------------------------
# Per-column + dataset profiling
# ---------------------------------------------------------------------------

def profile_column(column: str, series: pd.Series, n_samples: int = 5) -> dict:
    """Build the metadata record for a single column."""
    dtype = infer_logical_dtype(series)
    mixed = detect_mixed_type(series)
    semantic = infer_semantic(column, series, dtype)

    n_rows = int(len(series))
    n_null = int(series.isna().sum())
    non_null = series.dropna()

    def _fmt(v) -> str:
        # Render integral floats cleanly ("13085", not "13085.0") — NaNs force
        # float64 on ID columns, but the samples should read as the IDs are.
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)

    record: dict = {
        "column": column,
        "inferred_dtype": dtype,
        "semantic_type": semantic["semantic_type"],
        "semantic_description": semantic["description"],
        "is_mixed_type": mixed["is_mixed_type"],
        "mixed_type_detail": {
            k: v for k, v in mixed.items() if k != "is_mixed_type"
        },
        "n_rows": n_rows,
        "n_null": n_null,
        "null_pct": round(100.0 * n_null / n_rows, 2) if n_rows else 0.0,
        "n_unique": int(non_null.nunique()),
        "sample_values": [_fmt(v) for v in non_null.head(n_samples).tolist()],
    }
    # Numeric / datetime ranges help downstream cleaning stages.
    if dtype in ("int", "float"):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            record["min"] = float(numeric.min())
            record["max"] = float(numeric.max())
    if dtype == "date":
        try:
            dt = pd.to_datetime(non_null, errors="coerce").dropna()
            if not dt.empty:
                record["min"] = str(dt.min())
                record["max"] = str(dt.max())
        except Exception:
            pass
    return record


def extract_metadata(df: pd.DataFrame) -> dict:
    """Extract metadata for every column of ``df``.

    Returns:
        Dict with ``n_rows``, ``columns`` (ordered list matching EXPECTED_COLUMNS
        first) and ``fields`` (column name -> metadata record).
    """
    ordered = [c for c in EXPECTED_COLUMNS if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    fields = {col: profile_column(col, df[col]) for col in ordered}
    return {
        "n_rows_profiled": int(len(df)),
        "n_columns": int(len(ordered)),
        "columns": ordered,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 1: metadata extraction engine.")
    p.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to online_retail_II.xlsx")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save metadata JSON")
    p.add_argument(
        "--nrows-per-sheet",
        type=int,
        default=DEFAULT_NROWS_PER_SHEET,
        help="Max rows read per sheet (0 = all rows; default 50000 for speed).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    df = load_dataset(args.input, nrows_per_sheet=args.nrows_per_sheet)
    metadata = extract_metadata(df)
    metadata["source_file"] = str(args.input)
    metadata["nrows_per_sheet"] = args.nrows_per_sheet

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nSaved metadata for {metadata['n_columns']} columns -> {out}")
    return metadata


if __name__ == "__main__":
    main()
