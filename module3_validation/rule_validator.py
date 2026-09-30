"""Sprint 7 - Rule-Based Validation Engine (Module 3: Validation).

Runs explainable, deterministic checks over ``cleaned_data.csv`` (the
official Module 2 output). No statistics, no models - every violation cites
the exact rule and the exact row(s) that break it.

Rule categories (the brief's category logic, mapped onto the columns that
actually exist - this dataset has no email/phone/postcode columns, so the
"format" category is applied to InvoiceNo/StockCode instead):

  * ``numeric_range`` - physical hard bounds from Sprint 4's
    ``expected_schema.json`` (UnitPrice >= 0, Quantity != 0 and inside
    +/-10,000, UnitPrice <= 20,000). Severity ``hard``: any hit is a data
    error. On the cleaned data these all pass (cleaning worked) and the
    sample records that honestly with zero examples.
  * ``numeric_outlier`` - review/info tiers inside the hard bounds but
    outside the plausible band (|Quantity| > 1000, UnitPrice > 1000,
    |LineValue| > 10,000, Quantity outside the 3x-IQR plausible band
    [-20, 29], zero-price giveaway lines). These DO fire on real cleaned
    rows (bulk/wholesale orders, premium items, adjustments) - flagged for
    a human, never auto-dropped.
  * ``categorical`` - Country against the closed 28-name reference set
    (hard, passes) plus a rare-country micro-segment flag (<= 5 rows:
    k-anonymity watch, fires on Nigeria/Israel/Bahrain/United States) and
    the CustomerID ID-block check (hard, passes).
  * ``format`` - InvoiceNo ``^(C?\\d{5,7})$`` and StockCode 4-family
    pattern, non-blank Description, InvoiceDate inside the 2009-2011
    business window (all hard; all pass after cleaning).
  * ``cross_field`` - impossible/odd combinations: return line without a
    ``C`` cancellation prefix (fires: 186 real zero-price adjustment
    returns), cancellation prefix with non-negative Quantity (passes),
    duplicated (InvoiceNo, StockCode, InvoiceDate) line keys (fires:
    same product stamped twice on one invoice), guest (null CustomerID)
    order with |LineValue| > 1000 (fires: unattributed high-value lines).

Usage:
    from module3_validation.rule_validator import validate_dataframe, run_validation

    report = run_validation()  # defaults: cleaned_data.csv in, sample json out

CLI:
    python module3_validation/rule_validator.py
    python module3_validation/rule_validator.py --input module2_cleaning/cleaned_data.csv
    python module3_validation/rule_validator.py --input ... --output module3_validation/rule_violations_sample.json --max-examples 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on ``sys.path`` so imports work no matter the
    working directory (repo root, ``module3_validation/``, ...)."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

MODULE_NAME = "module3_validation"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "module2_cleaning" / "cleaned_data.csv"
DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "module2_cleaning" / "expected_schema.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "rule_violations_sample.json"

#: Max example rows stored per rule in the sample document.
MAX_EXAMPLES = 3

#: Columns snapshotted per example row (canonical 8 first, then derived
#: columns when present in the cleaned file).
SNAPSHOT_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country", "LineValue",
    "IsCancellation", "IsReturn", "IsGiveaway",
]

#: Fallback reference sets (used when expected_schema.json is absent; when
#: present the schema values win - see _load_reference_sets).
_FALLBACK_COUNTRIES = [
    "Australia", "Austria", "Bahrain", "Belgium", "Channel Islands",
    "Cyprus", "Denmark", "Finland", "France", "Germany", "Greece",
    "Iceland", "Ireland", "Israel", "Italy", "Japan", "Lithuania",
    "Netherlands", "Nigeria", "Norway", "Poland", "Portugal", "Spain",
    "Sweden", "Switzerland", "United Arab Emirates", "United Kingdom",
    "United States",
]
_FALLBACK_INVOICE_PATTERN = r"^(?:\d{5,7}|C\d{5,7})$"
_FALLBACK_STOCK_PATTERN = (
    r"^(?:(?:\d{4,6})|(?:\d{4,6}[A-Za-z]{1,2})|(?:C\d{4,6})"
    r"|(?:[A-Za-z][A-Za-z0-9 .'/&_()-]{0,30}))$"
)

# Hard numeric bounds (Sprint 4 contract).
QUANTITY_MIN, QUANTITY_MAX = -10000, 10000
UNITPRICE_MIN, UNITPRICE_MAX = 0.0, 20000.0
# Review thresholds (Sprint 4 "review" tiers).
QUANTITY_BULK = 1000
QUANTITY_PLAUSIBLE_MIN, QUANTITY_PLAUSIBLE_MAX = -20.0, 29.0
UNITPRICE_PREMIUM = 1000.0
LINEVALUE_EXTREME = 10000.0
CUSTOMERID_MIN, CUSTOMERID_MAX = 12000, 19000
INVOICE_WINDOW_MIN = pd.Timestamp("2009-01-01 00:00:00")
INVOICE_WINDOW_MAX = pd.Timestamp("2011-12-31 23:59:59")
RARE_COUNTRY_MAX_ROWS = 5
GUEST_HIGH_VALUE = 1000.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cleaned(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the official Module 2 output."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_csv(path)


def _load_reference_sets(schema_path: str | Path = DEFAULT_SCHEMA) -> dict:
    """Reference sets for validation: schema values when available, else
    the module fallbacks above (identical to the schema's observed values)."""
    ref: dict = {
        "countries": list(_FALLBACK_COUNTRIES),
        "invoice_pattern": _FALLBACK_INVOICE_PATTERN,
        "stock_pattern": _FALLBACK_STOCK_PATTERN,
    }
    try:
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        cols = schema.get("columns", {})
        countries = (cols.get("Country", {}).get("allowed_values", {}).get("values", []))
        if countries:
            ref["countries"] = [str(c) for c in countries]
        inv_pat = cols.get("InvoiceNo", {}).get("expected_format", {}).get("pattern")
        if inv_pat:
            ref["invoice_pattern"] = inv_pat
        stock_pat = cols.get("StockCode", {}).get("expected_format", {}).get("pattern")
        if stock_pat:
            ref["stock_pattern"] = stock_pat
        ref["schema_loaded"] = str(schema_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        ref["schema_loaded"] = None
    return ref


def _ensure_derived(df: pd.DataFrame) -> pd.DataFrame:
    """(Re)compute LineValue + line-type flags if the input lacks them so
    rules behave identically on slim (8-column) and full (21-column)
    cleaned files. Existing columns are never overwritten."""
    out = df.copy()
    if "LineValue" not in out.columns:
        out["LineValue"] = (
            pd.to_numeric(out["Quantity"], errors="coerce")
            * pd.to_numeric(out["UnitPrice"], errors="coerce")
        ).round(2)
    if "IsCancellation" not in out.columns:
        out["IsCancellation"] = out["InvoiceNo"].astype("string").str.startswith("C")
    if "IsReturn" not in out.columns:
        out["IsReturn"] = pd.to_numeric(out["Quantity"], errors="coerce") < 0
    if "IsGiveaway" not in out.columns:
        out["IsGiveaway"] = pd.to_numeric(out["UnitPrice"], errors="coerce") == 0
    return out


# ---------------------------------------------------------------------------
# Rules (each returns a boolean mask aligned with df)
# ---------------------------------------------------------------------------

def _rule_masks(df: pd.DataFrame, ref: dict) -> dict[str, pd.Series]:
    """Compute every rule mask. Pure vectorized pandas; NaN never matches
    a violation except where the rule is explicitly about missing data."""
    q = pd.to_numeric(df["Quantity"], errors="coerce")
    p = pd.to_numeric(df["UnitPrice"], errors="coerce")
    lv = pd.to_numeric(df["LineValue"], errors="coerce")
    inv = df["InvoiceNo"].astype("string")
    sc = df["StockCode"].astype("string")
    desc = df["Description"].astype("string")
    country = df["Country"].astype("string")
    cust = pd.to_numeric(df["CustomerID"], errors="coerce")
    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    masks: dict[str, pd.Series] = {}
    # -- numeric_range (hard) --
    masks["unitprice_negative"] = p.notna() & (p < UNITPRICE_MIN)
    masks["unitprice_above_cap"] = p.notna() & (p > UNITPRICE_MAX)
    masks["quantity_zero"] = q.notna() & (q == 0)
    masks["quantity_out_of_bounds"] = q.notna() & ((q < QUANTITY_MIN) | (q > QUANTITY_MAX))
    # -- numeric_outlier (review/info; inside hard bounds, outside plausible) --
    masks["quantity_bulk"] = q.notna() & (q.abs() > QUANTITY_BULK)
    masks["quantity_atypical"] = (
        q.notna() & ((q < QUANTITY_PLAUSIBLE_MIN) | (q > QUANTITY_PLAUSIBLE_MAX))
    )
    masks["unitprice_premium"] = p.notna() & (p > UNITPRICE_PREMIUM)
    masks["linevalue_extreme"] = lv.notna() & (lv.abs() > LINEVALUE_EXTREME)
    masks["unitprice_giveaway"] = p.notna() & (p == 0)
    # -- categorical --
    masks["country_unknown"] = df["Country"].notna() & (~country.isin(ref["countries"]))
    counts = df["Country"].astype("string").value_counts(dropna=True)
    rare_values = set(counts[counts <= RARE_COUNTRY_MAX_ROWS].index.astype(str))
    masks["country_rare"] = country.isin(rare_values)
    masks["customerid_out_of_block"] = (
        df["CustomerID"].notna()
        & (cust.notna() & ((cust % 1 != 0) | (cust < CUSTOMERID_MIN) | (cust > CUSTOMERID_MAX)))
    )
    # -- format (InvoiceNo/StockCode stand in for the email/phone/postcode
    # -- category: this dataset has no such columns) --
    masks["invoiceno_format"] = ~inv.str.match(ref["invoice_pattern"], na=False)
    masks["stockcode_format"] = ~sc.str.match(ref["stock_pattern"], na=False)
    masks["description_missing_or_blank"] = df["Description"].isna() | (
        desc.notna() & (desc.str.strip() == "")
    )
    masks["invoicedate_out_of_window"] = (
        dt.isna() | (dt < INVOICE_WINDOW_MIN) | (dt > INVOICE_WINDOW_MAX)
    )
    # -- cross_field (impossible/odd combinations) --
    is_canc = inv.str.startswith("C", na=False)
    masks["return_without_cancellation"] = q.notna() & (q < 0) & (~is_canc)
    masks["cancellation_without_return"] = is_canc & ~(q.notna() & (q < 0))
    masks["duplicate_line_key"] = df.duplicated(
        subset=["InvoiceNo", "StockCode", "InvoiceDate"], keep=False
    )
    masks["guest_high_value"] = (
        df["CustomerID"].isna() & lv.notna() & (lv.abs() > GUEST_HIGH_VALUE)
    )
    return masks


#: Rule catalogue: stable id, category, severity, human description.
RULES: list[dict] = [
    {"rule_id": "unitprice_negative", "category": "numeric_range", "severity": "hard",
     "description": "UnitPrice < 0: a negative price is physically impossible (data error)."},
    {"rule_id": "unitprice_above_cap", "category": "numeric_range", "severity": "hard",
     "description": "UnitPrice > 20000: above the Sprint 4 hard cap (data error)."},
    {"rule_id": "quantity_zero", "category": "numeric_range", "severity": "hard",
     "description": "Quantity == 0: a zero-unit line item is physically meaningless."},
    {"rule_id": "quantity_out_of_bounds", "category": "numeric_range", "severity": "hard",
     "description": "Quantity outside [-10000, 10000]: outside the Sprint 4 hard bounds."},
    {"rule_id": "quantity_bulk", "category": "numeric_outlier", "severity": "review",
     "description": "|Quantity| > 1000: plausible wholesale/bulk order, must be reviewed not dropped."},
    {"rule_id": "quantity_atypical", "category": "numeric_outlier", "severity": "info",
     "description": "Quantity outside the plausible 3x-IQR band [-20, 29]: ordinary retail spread tail."},
    {"rule_id": "unitprice_premium", "category": "numeric_outlier", "severity": "review",
     "description": "UnitPrice > 1000: large-ticket price, verify against the source system."},
    {"rule_id": "linevalue_extreme", "category": "numeric_outlier", "severity": "review",
     "description": "|LineValue| > 10000: extreme signed line total, verify before trusting revenue."},
    {"rule_id": "unitprice_giveaway", "category": "numeric_outlier", "severity": "info",
     "description": "UnitPrice == 0: adjustment/giveaway line, keep flagged and exclude from revenue."},
    {"rule_id": "country_unknown", "category": "categorical", "severity": "hard",
     "description": "Country not in the closed 28-name reference set (unknown value)."},
    {"rule_id": "country_rare", "category": "categorical", "severity": "review",
     "description": "Country with <= 5 rows in the file: k-anonymity micro-segment, watch small cells."},
    {"rule_id": "customerid_out_of_block", "category": "categorical", "severity": "hard",
     "description": "Non-null CustomerID outside the anonymised block [12000, 19000] or fractional."},
    {"rule_id": "invoiceno_format", "category": "format", "severity": "hard",
     "description": "InvoiceNo must match ^(\\d{5,7}|C\\d{5,7})$ (C prefix = cancellation event)."},
    {"rule_id": "stockcode_format", "category": "format", "severity": "hard",
     "description": "StockCode must match one of the 4 known families (numeric / suffixed / C-prefixed / service code)."},
    {"rule_id": "description_missing_or_blank", "category": "format", "severity": "review",
     "description": "Description null or blank: product name not captured (should be 0 after cleaning)."},
    {"rule_id": "invoicedate_out_of_window", "category": "format", "severity": "hard",
     "description": "InvoiceDate unparseable or outside the 2009-01-01..2011-12-31 business window."},
    {"rule_id": "return_without_cancellation", "category": "cross_field", "severity": "review",
     "description": "Quantity < 0 but InvoiceNo lacks the C prefix: return/credit without a cancellation event."},
    {"rule_id": "cancellation_without_return", "category": "cross_field", "severity": "hard",
     "description": "C-prefixed InvoiceNo but Quantity >= 0: a cancellation must carry a negative quantity."},
    {"rule_id": "duplicate_line_key", "category": "cross_field", "severity": "review",
     "description": "Duplicated (InvoiceNo, StockCode, InvoiceDate): same product stamped twice on one invoice."},
    {"rule_id": "guest_high_value", "category": "cross_field", "severity": "review",
     "description": "Guest order (null CustomerID) with |LineValue| > 1000: unattributed high-value line."},
]


# ---------------------------------------------------------------------------
# Validation + sample building
# ---------------------------------------------------------------------------

def _jsonable(value):
    """Render one cell JSON-safe (Timestamp -> canonical string, NaN -> None)."""
    try:
        if value is None:
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            if pd.isna(value):
                return None
            return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, float) and (math.isnan(value) or pd.isna(value)):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            v = float(value)
            return None if math.isnan(v) else v
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if pd.isna(value):
            return None
        return value
    except (TypeError, ValueError):
        return str(value)


def _snapshot_row(df: pd.DataFrame, pos: int) -> dict:
    """Full-row snapshot for one positional row (real data, not synthetic)."""
    row = df.iloc[pos]
    snap = {}
    for col in SNAPSHOT_COLUMNS:
        snap[col] = _jsonable(row[col]) if col in df.columns else None
    return snap


def validate_dataframe(df: pd.DataFrame, ref: dict | None = None) -> dict:
    """Run every rule over ``df``. Returns per-rule masks, counts and rates
    (no file I/O - see ``run_validation`` for persistence)."""
    ref = ref or {"countries": list(_FALLBACK_COUNTRIES),
                  "invoice_pattern": _FALLBACK_INVOICE_PATTERN,
                  "stock_pattern": _FALLBACK_STOCK_PATTERN,
                  "schema_loaded": None}
    frame = _ensure_derived(df)
    masks = _rule_masks(frame, ref)
    n = int(len(frame))
    results = {}
    for rule in RULES:
        rid = rule["rule_id"]
        mask = masks[rid]
        positions = frame.index[mask].tolist() if hasattr(frame.index, "tolist") else []
        # Positional ids into the cleaned CSV's data-row order: the cleaned
        # file is written with a fresh RangeIndex, so label == position.
        # For robustness resolve labels -> positions explicitly.
        pos_list = [int(frame.index.get_loc(label)) if label in frame.index else -1
                    for label in positions]
        pos_list = [p for p in pos_list if p >= 0]
        results[rid] = {
            "n_violations": int(mask.sum()),
            "violation_pct": round(100.0 * float(mask.sum()) / n, 4) if n else 0.0,
            "row_positions": pos_list,
        }
    return {"n_rows": n, "results": results, "frame": frame}


def build_violations_sample(df: pd.DataFrame, ref: dict | None = None,
                            max_examples: int = MAX_EXAMPLES) -> dict:
    """Validate ``df`` and package a JSON-serialisable sample document.

    Every rule that fires carries up to ``max_examples`` REAL rows from the
    dataset (positional ``row_id`` + ``csv_line`` + full snapshots). Rules
    with zero hits record an empty example list and ``status: pass`` - an
    honest signal that cleaning held, not a fabricated example.
    """
    ref = ref or _load_reference_sets()
    validated = validate_dataframe(df, ref)
    frame, n = validated["frame"], validated["n_rows"]
    rule_entries = []
    for rule in RULES:
        rid = rule["rule_id"]
        res = validated["results"][rid]
        examples = []
        for pos in res["row_positions"][:max_examples]:
            examples.append({
                "row_id": int(pos),
                "csv_line": int(pos) + 2,  # +1 header, +1 1-based lines
                "values": _snapshot_row(frame, pos),
            })
        rule_entries.append({
            "rule_id": rid,
            "category": rule["category"],
            "severity": rule["severity"],
            "description": rule["description"],
            "n_violations": res["n_violations"],
            "violation_pct": res["violation_pct"],
            "status": "fail" if res["n_violations"] else "pass",
            "n_examples_shown": len(examples),
            "examples": examples,
        })
    n_failed = sum(1 for r in rule_entries if r["status"] == "fail")
    flagged_positions = set()
    for rule in RULES:
        flagged_positions.update(validated["results"][rule["rule_id"]]["row_positions"])
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for r in rule_entries:
        by_category[r["category"]] = by_category.get(r["category"], 0) + r["n_violations"]
        by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + r["n_violations"]
    return {
        "module": MODULE_NAME,
        "artifact": "rule_violations_sample",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "input": str(DEFAULT_INPUT),
            "schema": ref.get("schema_loaded"),
            "n_rows": n,
            "max_examples_per_rule": max_examples,
        },
        "rules_defined": len(RULES),
        "rules_failed": n_failed,
        "rules_passed": len(RULES) - n_failed,
        "n_flagged_rows_distinct": len(flagged_positions),
        "violations_by_category": by_category,
        "violations_by_severity": by_severity,
        "note": ("Hard rules with status=pass held on the cleaned data "
                 "(cleaning worked); review/info rules with status=fail show "
                 "REAL cleaned rows for human triage - never synthetic. "
                 "row_id is the 0-based position in cleaned_data.csv; "
                 "csv_line is the 1-based file line (header = line 1)."),
        "rules": rule_entries,
    }


# ---------------------------------------------------------------------------
# File-level API + CLI
# ---------------------------------------------------------------------------

def run_validation(
    input_path: str | Path = DEFAULT_INPUT,
    schema_path: str | Path = DEFAULT_SCHEMA,
    output_path: str | Path = DEFAULT_OUTPUT,
    max_examples: int = MAX_EXAMPLES,
) -> dict:
    """Load the cleaned data + reference sets, build the violations sample,
    save it to ``output_path`` and return the document."""
    df = load_cleaned(input_path)
    ref = _load_reference_sets(schema_path)
    doc = build_violations_sample(df, ref, max_examples=max_examples)
    doc["source"]["input"] = str(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 7: Module 3 rule-based validation - explainable "
                    "checks over cleaned_data.csv -> rule_violations_sample.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to cleaned_data.csv")
    p.add_argument("--schema", default=str(DEFAULT_SCHEMA),
                   help="Path to Sprint 4's expected_schema.json (reference sets)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Where to save rule_violations_sample.json")
    p.add_argument("--max-examples", type=int, default=MAX_EXAMPLES,
                   help="Max example rows stored per rule.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    doc = run_validation(args.input, args.schema, args.output, args.max_examples)
    summary = {
        "module": doc["module"],
        "n_rows": doc["source"]["n_rows"],
        "rules_defined": doc["rules_defined"],
        "rules_failed": doc["rules_failed"],
        "rules_passed": doc["rules_passed"],
        "n_flagged_rows_distinct": doc["n_flagged_rows_distinct"],
        "violations_by_category": doc["violations_by_category"],
        "violations_by_severity": doc["violations_by_severity"],
        "per_rule": {r["rule_id"]: r["n_violations"] for r in doc["rules"]},
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nRule violations sample -> {args.output}")
    return doc


if __name__ == "__main__":
    main()
