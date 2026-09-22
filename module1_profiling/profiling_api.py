"""Sprint 3 - Module 1 Profiling API: rule engine + final report.

Closes out Module 1 by packaging Sprint 1 (metadata), Sprint 2 (statistics +
visuals) and Sprint 3 (rule-engine flags) into the official Module 1 output::

    profiling_report.json

Rule engine flags three families of issues:

  * suspicious columns - near-100% missing, high missingness, impossible
    values (negative prices, zero quantities, out-of-range dates), extreme
    outliers, cancelled-invoice share, empty/constant columns.
  * inconsistent formats - mixed-type columns (InvoiceNo ``C536379``,
    StockCode ``85123A``/``POST``), dates stored as text in some rows,
    leading/trailing whitespace or case variants in text columns.
  * potential PII - direct identifiers (CustomerID) and quasi-identifiers
    (Country, ...), found by name-pattern + cardinality heuristics.

Public API (see README.md for details)::

    from module1_profiling.profiling_api import profile_dataframe, run_profiling

    report = profile_dataframe(df)          # raw DataFrame in -> full report dict
    report = run_profiling()                # loads xlsx, saves profiling_report.json

Usage:
    python profiling_api.py
    python profiling_api.py --input ..\\online_retail_II.xlsx --nrows-per-sheet 50000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

try:
    from metadata_extractor import EXPECTED_COLUMNS, extract_metadata, load_dataset
except ImportError:  # allow `python module1_profiling/profiling_api.py` from repo root
    from module1_profiling.metadata_extractor import (
        EXPECTED_COLUMNS,
        extract_metadata,
        load_dataset,
    )

try:
    from metadata_extractor import DEFAULT_INPUT as _DEFAULT_INPUT
    from metadata_extractor import DEFAULT_NROWS_PER_SHEET as _DEFAULT_NROWS
except ImportError:
    try:
        from module1_profiling.metadata_extractor import DEFAULT_INPUT as _DEFAULT_INPUT
        from module1_profiling.metadata_extractor import DEFAULT_NROWS_PER_SHEET as _DEFAULT_NROWS
    except Exception:  # pragma: no cover
        _DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "online_retail_II.xlsx"
        _DEFAULT_NROWS = 50000

try:
    from profiler import profile_dataset
except ImportError:
    from module1_profiling.profiler import profile_dataset

MODULE_NAME = "module1_profiling"
MODULE_VERSION = "1.0.0"
DEFAULT_REPORT = Path(__file__).resolve().parent / "profiling_report.json"
DEFAULT_VISUALS_DIR = Path(__file__).resolve().parent / "visuals"

# ---------------------------------------------------------------------------
# Thresholds (documented in README.md)
# ---------------------------------------------------------------------------

MISSING_CRITICAL_PCT = 95.0   # near-100% missing  -> critical
MISSING_HIGH_PCT = 20.0       # high missingness   -> warning
BULK_QUANTITY = 1000          # |Quantity| above this -> bulk/return outlier (info)
EXTREME_PRICE = 1000.0        # UnitPrice above this -> extreme price (info)
EXPECTED_DATE_MIN = pd.Timestamp("2009-01-01")
EXPECTED_DATE_MAX = pd.Timestamp("2011-12-31")

#: Column-name patterns that suggest identifying information.
PII_PATTERNS = {
    "customerid": ("direct", "customer identifier - singles out an individual"),
    "customer": ("direct", "customer identifier - singles out an individual"),
    "email": ("direct", "email address - directly identifies an individual"),
    "phone": ("direct", "phone number - directly identifies an individual"),
    "address": ("direct", "postal address - directly identifies a household"),
    "name": ("quasi", "name-like field - identifying in combination"),
    "country": ("quasi", "geography - quasi-identifier in combination"),
    "city": ("quasi", "geography - quasi-identifier in combination"),
    "postcode": ("quasi", "geography - quasi-identifier in combination"),
    "zip": ("quasi", "geography - quasi-identifier in combination"),
    "dob": ("quasi", "date of birth - quasi-identifier in combination"),
    "birth": ("quasi", "date of birth - quasi-identifier in combination"),
}


def _flag(rule_id: str, category: str, severity: str, message: str,
          column: str | None = None, evidence: dict | None = None) -> dict:
    """Build a single rule-engine flag record."""
    return {
        "rule_id": rule_id,
        "category": category,
        "severity": severity,  # one of {"info", "warning", "critical"}
        "column": column,
        "message": message,
        "evidence": evidence or {},
    }


# ---------------------------------------------------------------------------
# Rule family 1: suspicious columns
# ---------------------------------------------------------------------------

def check_suspicious_columns(df: pd.DataFrame, metadata: dict, statistics: dict) -> list[dict]:
    """Flag impossible values, extreme missingness and degenerate columns."""
    flags: list[dict] = []
    n_rows = len(df)
    missing = statistics.get("missing_value_matrix", {}).get("per_column", {})
    numeric = statistics.get("numeric_profile", {}).get("distributions", {})

    for col in df.columns:
        col_missing = missing.get(col, {}).get("missing_pct", 0.0)
        if col_missing >= MISSING_CRITICAL_PCT:
            flags.append(_flag("missing-critical", "suspicious", "critical",
                               f"'{col}' is {col_missing:.1f}% missing - effectively unusable.",
                               column=col, evidence={"missing_pct": col_missing}))
        elif col_missing >= MISSING_HIGH_PCT:
            flags.append(_flag("missing-high", "suspicious", "warning",
                               f"'{col}' is {col_missing:.1f}% missing - downstream joins/aggregations must handle gaps.",
                               column=col, evidence={"missing_pct": col_missing}))

        series = df[col].dropna()
        if series.empty:
            flags.append(_flag("empty-column", "suspicious", "critical",
                               f"'{col}' has no usable values.", column=col))
            continue
        if int(series.nunique()) == 1:
            flags.append(_flag("constant-column", "suspicious", "warning",
                               f"'{col}' holds a single repeated value - carries no signal.",
                               column=col, evidence={"value": str(series.iloc[0])}))

    qty = numeric.get("Quantity", {})
    if qty:
        if qty.get("n_negative", 0):
            flags.append(_flag("negative-quantity", "suspicious", "info",
                               "Negative Quantity values present - expected for returns/cancellations; "
                               "cleaning stage must decide whether to keep them.",
                               column="Quantity",
                               evidence={"n_negative": qty["n_negative"],
                                         "min": qty.get("min"),
                                         "pct": round(100.0 * qty["n_negative"] / max(qty.get("count", 1), 1), 2)}))
        n_bulk = int(((pd.to_numeric(df["Quantity"], errors="coerce") > BULK_QUANTITY)
                      | (pd.to_numeric(df["Quantity"], errors="coerce") < -BULK_QUANTITY)).sum()) \
            if "Quantity" in df.columns else 0
        if n_bulk:
            flags.append(_flag("bulk-quantity", "suspicious", "info",
                               f"{n_bulk} rows with |Quantity| > {BULK_QUANTITY} - bulk orders or data-entry outliers.",
                               column="Quantity",
                               evidence={"n_bulk": n_bulk, "min": qty.get("min"), "max": qty.get("max")}))
        if qty.get("n_zero", 0):
            flags.append(_flag("zero-quantity", "suspicious", "warning",
                               "Zero Quantity values present - physically meaningless for a line item.",
                               column="Quantity", evidence={"n_zero": qty["n_zero"]}))

    price = numeric.get("UnitPrice", {})
    if price:
        if price.get("n_negative", 0):
            flags.append(_flag("negative-price", "suspicious", "critical",
                               "Negative UnitPrice is impossible - data error.",
                               column="UnitPrice", evidence={"n_negative": price["n_negative"]}))
        if price.get("n_zero", 0):
            flags.append(_flag("zero-price", "suspicious", "warning",
                               "Zero UnitPrice values present - likely adjustments/giveaways, not real prices.",
                               column="UnitPrice",
                               evidence={"n_zero": price["n_zero"],
                                         "pct": round(100.0 * price["n_zero"] / max(price.get("count", 1), 1), 2)}))
        n_extreme = int((pd.to_numeric(df["UnitPrice"], errors="coerce") > EXTREME_PRICE).sum()) \
            if "UnitPrice" in df.columns else 0
        if n_extreme:
            flags.append(_flag("extreme-price", "suspicious", "info",
                               f"{n_extreme} rows with UnitPrice > {EXTREME_PRICE} - verify against source system.",
                               column="UnitPrice",
                               evidence={"n_extreme": n_extreme, "max": price.get("max")}))

    if "InvoiceNo" in df.columns:
        cancelled = df["InvoiceNo"].astype(str).str.match(r"^C\d+$", na=False)
        n_cancelled = int(cancelled.sum())
        if n_cancelled:
            flags.append(_flag("cancelled-invoices", "suspicious", "info",
                               f"{n_cancelled} ({100.0 * n_cancelled / max(n_rows, 1):.2f}%) rows are cancelled "
                               "transactions ('C' prefix) - not distinct orders.",
                               column="InvoiceNo", evidence={"n_cancelled": n_cancelled}))

    if "InvoiceDate" in df.columns:
        try:
            dates = pd.to_datetime(df["InvoiceDate"], errors="coerce").dropna()
            if not dates.empty and (dates.min() < EXPECTED_DATE_MIN or dates.max() > EXPECTED_DATE_MAX):
                flags.append(_flag("date-out-of-range", "suspicious", "warning",
                                   "InvoiceDate values fall outside the expected 2009-2011 window.",
                                   column="InvoiceDate",
                                   evidence={"min": str(dates.min()), "max": str(dates.max())}))
        except Exception:
            pass

    return flags


# ---------------------------------------------------------------------------
# Rule family 2: inconsistent formats
# ---------------------------------------------------------------------------

def check_inconsistent_formats(df: pd.DataFrame, metadata: dict, statistics: dict) -> list[dict]:
    """Flag mixed types, text-stored dates and whitespace/case inconsistencies."""
    flags: list[dict] = []
    fields = metadata.get("fields", {})
    consistency = statistics.get("dtype_consistency", {})

    for col, field in fields.items():
        if field.get("is_mixed_type"):
            detail = field.get("mixed_type_detail", {})
            flags.append(_flag("mixed-type", "format", "warning",
                               f"'{col}' mixes numeric and text codes "
                               f"({detail.get('reason', 'see evidence')}). Standardise before typing.",
                               column=col,
                               evidence={"text_examples": detail.get("text_examples", [])[:5],
                                         "numeric_ratio": detail.get("numeric_ratio")}))

    for col, check in consistency.items():
        if check.get("n_inconsistent", 0):
            flags.append(_flag("dtype-mismatch", "format", "warning",
                               f"{check['n_inconsistent']} value(s) in '{col}' do not match "
                               f"inferred dtype '{check['inferred_dtype']}'.",
                               column=col,
                               evidence={"inconsistent_examples": check.get("inconsistent_examples", [])}))

    # Dates stored as text in (some) rows: object-dtype column, inferred date,
    # with a mix of parseable / unparseable entries.
    for col, check in consistency.items():
        if check.get("inferred_dtype") == "date" and col in df.columns:
            series = df[col].dropna()
            if not series.empty and not pd.api.types.is_datetime64_any_dtype(df[col].dtype):
                parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                fail_rate = float(parsed.isna().mean())
                if fail_rate > 0:
                    flags.append(_flag("date-as-text", "format", "warning",
                                       f"'{col}' is stored as text and {fail_rate:.1%} of values "
                                       "do not parse as dates.",
                                       column=col, evidence={"unparseable_rate": round(fail_rate, 4)}))

    # Leading/trailing whitespace and case variants in text columns.
    for col in df.columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty or not (series.dtype == object or str(series.dtype) == "str"):
            continue
        texts = [str(v) for v in series.tolist()]
        n_padded = sum(1 for t in texts if t != t.strip())
        if n_padded:
            examples = list(dict.fromkeys(t for t in texts if t != t.strip()))[:3]
            flags.append(_flag("padded-text", "format", "warning",
                               f"'{col}' has {n_padded} value(s) with leading/trailing whitespace - "
                               "strip during cleaning.",
                               column=col, evidence={"n_padded": n_padded, "examples": examples}))
        lowered: dict[str, set[str]] = {}
        for t in set(texts):
            lowered.setdefault(t.strip().lower(), set()).add(t)
        variants = {k: sorted(v) for k, v in lowered.items() if len(v) > 1}
        if variants:
            flags.append(_flag("case-variants", "format", "info",
                               f"'{col}' has {len(variants)} value group(s) differing only by case - "
                               "normalise during cleaning.",
                               column=col,
                               evidence={"example_groups": list(variants.values())[:3]}))

    return flags


# ---------------------------------------------------------------------------
# Rule family 3: potential PII
# ---------------------------------------------------------------------------

def check_pii_columns(df: pd.DataFrame, metadata: dict) -> list[dict]:
    """Flag direct and quasi-identifiers by name pattern (+ ID-like cardinality)."""
    flags: list[dict] = []
    fields = metadata.get("fields", {})
    for col in df.columns:
        name = str(col).strip().lower()
        for pattern, (pii_type, reason) in PII_PATTERNS.items():
            if pattern in name:
                semantic = fields.get(col, {}).get("semantic_type", "")
                recommendation = ("anonymise/hash before sharing; never use as a model feature raw."
                                  if pii_type == "direct"
                                  else "safe to keep for grouping, but watch k-anonymity in small cells.")
                flags.append(_flag(f"pii-{pii_type}", "pii",
                                   "warning" if pii_type == "direct" else "info",
                                   f"'{col}' looks like {reason} (matched '{pattern}'). {recommendation}",
                                   column=col,
                                   evidence={"pii_type": pii_type, "matched_pattern": pattern,
                                             "semantic_type": semantic}))
                break
    return flags


def run_rule_engine(df: pd.DataFrame, metadata: dict, statistics: dict) -> dict:
    """Run all rule families. Returns ``{suspicious_columns, inconsistent_formats,
    pii_columns, summary}`` - empty lists mean the checks passed."""
    suspicious = check_suspicious_columns(df, metadata, statistics)
    formats = check_inconsistent_formats(df, metadata, statistics)
    pii = check_pii_columns(df, metadata)
    all_flags = suspicious + formats + pii
    return {
        "suspicious_columns": suspicious,
        "inconsistent_formats": formats,
        "pii_columns": pii,
        "summary": {
            "n_flags": len(all_flags),
            "n_critical": sum(1 for f in all_flags if f["severity"] == "critical"),
            "n_warning": sum(1 for f in all_flags if f["severity"] == "warning"),
            "n_info": sum(1 for f in all_flags if f["severity"] == "info"),
        },
    }


# ---------------------------------------------------------------------------
# Unified Module 1 API
# ---------------------------------------------------------------------------

def _canonical_order(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [c for c in EXPECTED_COLUMNS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def profile_dataframe(
    df: pd.DataFrame,
    visuals_dir: str | Path = DEFAULT_VISUALS_DIR,
    heatmap_rows: int = 1000,
    source_file: str | None = None,
    nrows_per_sheet: int | None = None,
) -> dict:
    """Take the raw dataset and return the final Module 1 profiling report.

    Args:
        df: Raw dataset as a DataFrame (canonical headers preferred; raw
            headers are NOT renamed here - use ``load_dataset`` first or
            :func:`run_profiling` for the file path route).
        visuals_dir: Directory the three PNGs are rendered into.
        heatmap_rows: Rows sampled in ``missing_heatmap.png``.
        source_file: Provenance label recorded in the report.
        nrows_per_sheet: Provenance label recorded in the report.

    Returns:
        Final report dict with ``metadata`` (Sprint 1), ``statistics``
        (Sprint 2), ``flags`` (Sprint 3) and ``visuals`` sections.
    """
    df = _canonical_order(df)
    metadata = extract_metadata(df)                       # Sprint 1
    sprint2 = profile_dataset(df, visuals_dir=Path(visuals_dir),
                              heatmap_rows=heatmap_rows)  # Sprint 2
    statistics = {
        "missing_value_matrix": sprint2["missing_value_matrix"],
        "dtype_consistency": sprint2["dtype_consistency"],
        "cardinality": sprint2["cardinality"],
        "numeric_profile": sprint2["numeric_profile"],
    }
    flags = run_rule_engine(df, metadata, statistics)     # Sprint 3
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "source_file": str(source_file) if source_file else None,
        "nrows_per_sheet": nrows_per_sheet,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "metadata": metadata,
        "statistics": statistics,
        "flags": flags,
        "visuals": sprint2["visuals"],
    }


def run_profiling(
    input_path: str | Path = _DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_REPORT,
    nrows_per_sheet: int = _DEFAULT_NROWS,
    visuals_dir: str | Path = DEFAULT_VISUALS_DIR,
    heatmap_rows: int = 1000,
) -> dict:
    """Load the workbook, profile it, save and return the final report.

    Args:
        input_path: Path to ``online_retail_II.xlsx``.
        output_path: Where ``profiling_report.json`` is written.
        nrows_per_sheet: Max rows read per sheet (``0`` = all rows).
        visuals_dir: Where the three PNGs are written.
        heatmap_rows: Rows sampled in ``missing_heatmap.png``.

    Returns:
        The final report dict (also written to ``output_path``).
    """
    df = load_dataset(input_path, nrows_per_sheet=nrows_per_sheet)
    report = profile_dataframe(df, visuals_dir=visuals_dir, heatmap_rows=heatmap_rows,
                               source_file=str(input_path), nrows_per_sheet=nrows_per_sheet)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 3: Module 1 profiling API - final report.")
    p.add_argument("--input", default=str(_DEFAULT_INPUT), help="Path to online_retail_II.xlsx")
    p.add_argument("--output", default=str(DEFAULT_REPORT), help="Where to save profiling_report.json")
    p.add_argument("--nrows-per-sheet", type=int, default=_DEFAULT_NROWS,
                   help="Max rows read per sheet (0 = all rows).")
    p.add_argument("--visuals-dir", default=str(DEFAULT_VISUALS_DIR), help="Where to save PNGs")
    p.add_argument("--heatmap-rows", type=int, default=1000, help="Rows sampled in missing heatmap")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run_profiling(args.input, args.output, args.nrows_per_sheet,
                           args.visuals_dir, args.heatmap_rows)
    summary = {
        "module": report["module"],
        "version": report["version"],
        "n_rows": report["n_rows"],
        "columns": report["columns"],
        "flags_summary": report["flags"]["summary"],
        "visuals": report["visuals"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nModule 1 complete - final report -> {args.output}")
    return report


if __name__ == "__main__":
    main()
