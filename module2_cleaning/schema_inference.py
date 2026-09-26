"""Sprint 4 - Schema Inference Engine (Module 2: Cleaning).

Defines what "correct" looks like for the Online Retail II dataset *before*
anything is cleaned. Takes Sprint 3's ``profiling_report.json`` as input and
derives, for every column, an expected schema:

  * expected dtype (logical type + expected storage type),
  * nullability and null semantics,
  * expected value ranges (hard bounds, IQR-derived tiers, observed range),
  * expected formats (regex patterns per value family, canonical text form),
  * machine-checkable constraints with severity and evidence,
  * post-cleaning acceptance targets (what "clean" must look like).

The output is ``expected_schema.json`` - the contract that downstream
cleaning steps validate against.

Provenance model
---------------
Every rule is tagged with ``basis``:

  * ``"observed"``  - measured in ``profiling_report.json`` (row counts,
    quantiles, value examples, flag evidence ...).
  * ``"domain"``    - a business/physical invariant of the retail dataset
    family (a price is never negative, a line item has no zero quantity)
    that no profiling report can discover on its own. Domain rules never
    contradict the report; where a domain bound brackets an observed value
    it is recorded under ``brackets_observed``.
  * ``"derived"``   - computed from observed evidence by a documented rule
    (e.g. the InvoiceDate business window is floored/ceiled to whole
    calendar years from the observed min/max).

A ``validation`` section is computed at run time and proves the deliverable:

  * every column has a concrete rule set (no placeholders),
  * per-family observed row counts sum to the profiled row count,
  * no rule text is a stub (``TODO``/``TBD``/``...`` style content fails).

Usage:
    python schema_inference.py
    python schema_inference.py --report ..\\module1_profiling\\profiling_report.json
    python schema_inference.py --report ... --output expected_schema.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MODULE_NAME = "module2_cleaning"
SCHEMA_VERSION = "1.0.0"
SCHEMA_NAME = "expected_schema"

DEFAULT_REPORT = (
    Path(__file__).resolve().parent.parent
    / "module1_profiling"
    / "profiling_report.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "expected_schema.json"

#: Profiling sampled 100k of >1M rows, so token-width families that only show
#: a single observed width are widened by this much when deriving the
#: *operative* pattern. The strict as-observed pattern is always reported
#: alongside as ``pattern_observed``.
WIDTH_TOLERANCE = 1

#: IQR multiples used for the "typical" (k=1.5) and "plausible" (k=3.0)
#: value tiers of measures.
IQR_TYPICAL_K = 1.5
IQR_PLAUSIBLE_K = 3.0

_PLACEHOLDER_TOKENS = ("TODO", "TBD", "FIXME", "<insert", "<fill", "...")


# ---------------------------------------------------------------------------
# Semantic-type knowledge base (domain priors)
# ---------------------------------------------------------------------------
# Business/physical invariants for the eight Online Retail columns. Anything
# measurable is instead bound to report evidence at run time; the priors are
# only the rules no profiling pass can discover (signs, zeroes, identifier
# prefixes, currency precision, business calendar). ``basis`` is always
# recorded per emitted rule so a reader can tell measured from assumed.

SEMANTIC_PRIORS: dict[str, dict] = {
    "invoice_id": {
        "role": "key",
        "expected_dtype": "string",
        "nullable": False,
        "cancel_prefix": "C",
        "canonical_case": "upper",
        "trim_required": False,  # observed evidence: no InvoiceNo is padded
        "notes": (
            "InvoiceNo is an identifier, not a number: keep it text so the "
            "'C'-prefixed cancellation codes never collapse onto plain "
            "order numbers."
        ),
    },
    "product_id (SKU)": {
        "role": "key",
        "expected_dtype": "string",
        "nullable": False,
        "canonical_case": "upper",
        "trim_required": True,
        "notes": (
            "StockCode mixes merchandise SKUs (digit core + optional letter "
            "suffix) with service/adjustment codes (POST, DOT, D, M, C2, "
            "BANK CHARGES ...). Both families are valid; case variants "
            "denote the same product."
        ),
    },
    "product_description": {
        "role": "text",
        "expected_dtype": "string",
        "nullable": True,
        "canonical_case": "upper",
        "trim_required": True,
        "collapse_internal_whitespace": True,
        "max_length": 128,  # domain bound; observed max in sample is 35
        "notes": (
            "Description is an attribute of StockCode, not a key: many rows "
            "share one description and a few StockCodes carry raw-text "
            "variants that only differ by case/whitespace."
        ),
    },
    "quantity": {
        "role": "measure",
        "expected_dtype": "int",
        "nullable": False,
        "negative_allowed": True,   # returns / cancellations are legitimate
        "zero_allowed": False,      # a zero-unit line item is meaningless
        "abs_max": 10000,           # domain bound bracketing observed extrema
        "bulk_threshold": 1000,     # |Quantity| above this -> bulk/return review
        "notes": (
            "Signed measure: positive = sale, negative = return/credit. "
            "Zero never occurs and is invalid."
        ),
    },
    "transaction_datetime": {
        "role": "timestamp",
        "expected_dtype": "date",
        "nullable": False,
        "text_format": "%Y-%m-%d %H:%M:%S",
        "text_pattern": r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$",
        "resolution": "second",
        "timezone": None,  # tz-naive local timestamps
        "business_years": (2009, 2011),  # prior; cross-checked with observed
        "notes": "Date and time the invoice was raised, second resolution.",
    },
    "unit_price": {
        "role": "measure",
        "expected_dtype": "float",
        "nullable": False,
        "negative_allowed": False,  # a negative price is a data error
        "zero_allowed": True,       # adjustments / giveaways, keep but flag
        "hard_max": 20000.0,        # domain bound bracketing observed max
        "review_threshold": 1000.0,  # UnitPrice above this -> verify vs source
        "max_decimal_places": 2,    # currency precision
        "notes": "Price per unit in local currency.",
    },
    "customer_id": {
        "role": "key",
        "expected_dtype": "int",
        "nullable": True,  # nulls are structural (guest/unattributed orders)
        "identifier": True,  # never aggregate as a measure
        "bounds_rounding": 1000,  # hard bounds = observed rounded outward
        "max_fractional": 0,
        "notes": (
            "Anonymised customer identifier. Missing for ~1/3 of rows "
            "(guest checkout / unattributed orders) - a legitimate state, "
            "not dirt. Never average or sum."
        ),
    },
    "country": {
        "role": "dimension",
        "expected_dtype": "string",
        "nullable": False,
        "canonical_case": "title",
        "trim_required": True,
        "max_n_unique": 40,  # domain bound bracketing observed 28
        "notes": "Customer/delivery country; needs alias normalisation.",
    },
}

#: Legacy/abbreviated country spellings observed in this dataset family and
#: their canonical target. Whitespace/case normalisation happens first, then
#: this map is applied.
COUNTRY_ALIASES = {
    "EIRE": "Ireland",
    "UK": "United Kingdom",
    "U.K.": "United Kingdom",
    "U K": "United Kingdom",
    "USA": "United States",
    "US": "United States",
    "U.S.A.": "United States",
    "RSA": "South Africa",
    "UAE": "United Arab Emirates",
    "U.E.": "United Arab Emirates",
}

#: Canonical (post-normalisation) country names of this dataset family. The
#: profiling report only quotes a subsample of the 28 observed spellings, so
#: the closed reference set is a ``domain`` prior; the report corroborates it
#: via ``n_unique == 28`` and 6+ quoted spellings. Unknown spellings are
#: flagged, never dropped.
COUNTRY_REFERENCE = [
    "Australia", "Austria", "Bahrain", "Belgium", "Channel Islands",
    "Cyprus", "Denmark", "Finland", "France", "Germany", "Greece",
    "Iceland", "Ireland", "Israel", "Italy", "Japan", "Lithuania",
    "Netherlands", "Nigeria", "Norway", "Poland", "Portugal", "Spain",
    "Sweden", "Switzerland", "United Arab Emirates", "United Kingdom",
    "United States",
]


def _canonical_country(raw: str) -> str:
    """Strip, then map a legacy spelling to its canonical country name."""
    s = raw.strip()
    return COUNTRY_ALIASES.get(s.upper(), s)


#: Concrete cleaning pipeline the schema was written to serve: each cleaning
#: operation maps to the schema rule(s) that mandate it. Cleaning code can
#: import this mapping to keep implementation and contract in sync.
CLEANING_PLAN = [
    {
        "step": 1,
        "operation": "strip_whitespace",
        "columns": ["StockCode", "Description", "Country"],
        "mandated_by": [
            "StockCode.format.trim_required",
            "Description.format.trim_required",
            "Country.format.trim_required",
        ],
        "evidence": "18717 padded Description values; StockCode case-variant "
                    "groups whose members differ by padding ('85036B' vs "
                    "'85036b' families aside).",
    },
    {
        "step": 2,
        "operation": "normalise_case",
        "columns": ["StockCode", "Description", "Country"],
        "mandated_by": [
            "StockCode.format.case",
            "Description.format.case",
            "Country.format.case",
        ],
        "evidence": "90 StockCode + 31 Description value groups differing "
                    "only by case.",
    },
    {
        "step": 3,
        "operation": "normalise_country_aliases",
        "columns": ["Country"],
        "mandated_by": ["Country.allowed_values.aliases"],
        "evidence": "Legacy spellings in this dataset family ('EIRE', 'USA') "
                    "denote 'Ireland' / 'United States'.",
    },
    {
        "step": 4,
        "operation": "parse_datetimes",
        "columns": ["InvoiceDate"],
        "mandated_by": ["InvoiceDate.format.pattern"],
        "evidence": "InvoiceDate must parse as '%Y-%m-%d %H:%M:%S', "
                    "second resolution, tz-naive.",
    },
    {
        "step": 5,
        "operation": "cast_nullable_integer",
        "columns": ["CustomerID"],
        "mandated_by": ["CustomerID.storage.expected_storage_type"],
        "evidence": "CustomerID reads as float64 ('12346.0') because of "
                    "NaNs; expected storage is nullable Int64.",
    },
    {
        "step": 6,
        "operation": "impute_description_from_stockcode",
        "columns": ["Description"],
        "mandated_by": [
            "Description.null_policy.action",
            "StockCode.reference.description_lookup",
        ],
        "evidence": "408 Description gaps (0.41%) are imputable from the "
                    "StockCode->Description lookup; all other rows map 1:1 "
                    "after case/whitespace normalisation.",
    },
    {
        "step": 7,
        "operation": "derive_line_columns",
        "columns": ["LineValue", "IsCancellation", "IsReturn", "IsGiveaway"],
        "mandated_by": ["derived_columns"],
        "evidence": "Revenue, cancellation and return analytics require "
                    "signed line values and line-type flags.",
    },
]

#: Derived (new) columns the cleaned dataset must carry, with lineage.
DERIVED_COLUMNS = [
    {
        "column": "LineValue",
        "expression": "Quantity * UnitPrice",
        "dtype": "float",
        "unit": "local currency",
        "note": "Signed line value: negative for returns, zero for giveaways.",
    },
    {
        "column": "IsCancellation",
        "expression": "InvoiceNo matches '^C'",
        "dtype": "boolean",
        "note": "True for the 2077 cancellation lines.",
    },
    {
        "column": "IsReturn",
        "expression": "Quantity < 0",
        "dtype": "boolean",
        "note": "True for the 2263 return/credit lines.",
    },
    {
        "column": "IsGiveaway",
        "expression": "UnitPrice == 0",
        "dtype": "boolean",
        "note": "True for the 595 adjustment/giveaway lines.",
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_profiling_report(path: str | Path = DEFAULT_REPORT) -> dict:
    """Load and minimally validate Sprint 3's ``profiling_report.json``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Profiling report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    for key in ("metadata", "statistics", "flags",
                "n_rows", "n_columns", "columns"):
        if key not in report:
            raise ValueError(f"Profiling report missing required key: {key!r}")
    column_order = list(report.get("columns", []))
    fields = report.get("metadata", {}).get("fields", {})
    missing = [c for c in column_order if c not in fields]
    if missing:
        raise ValueError(f"Report fields missing for columns: {missing}")
    return report


# ---------------------------------------------------------------------------
# Evidence helpers (report -> plain facts)
# ---------------------------------------------------------------------------

def _field(report: dict, column: str) -> dict:
    return report["metadata"]["fields"][column]


def _missing(report: dict, column: str) -> dict:
    return (
        report["statistics"]["missing_value_matrix"]["per_column"].get(
            column, {"n_missing": 0, "missing_pct": 0.0})
    )


def _distribution(report: dict, column: str) -> dict:
    return (
        report["statistics"].get("numeric_profile", {})
        .get("distributions", {}).get(column, {})
    )


def _cardinality(report: dict, column: str) -> dict:
    return report["statistics"].get("cardinality", {}).get(column, {})


def _mixed_counts(field: dict) -> dict:
    detail = field.get("mixed_type_detail", {})
    return {
        "n_numeric_like": int(detail.get("n_numeric_like", 0)),
        "n_text": int(detail.get("n_text", 0)),
        "n_digit_core_with_letter_suffix": int(
            detail.get("n_digit_core_with_letter_suffix", 0)),
        "n_cancelled_style_prefix": int(
            detail.get("n_cancelled_style_prefix", 0)),
    }


def _flag_evidence_strings(report: dict, column: str) -> list[str]:
    """Literal example strings the rule engine recorded for a column."""
    out: list[str] = []
    for family in ("suspicious_columns", "inconsistent_formats",
                   "pii_columns"):
        for flag in report.get("flags", {}).get(family, []):
            if flag.get("column") != column:
                continue
            ev = flag.get("evidence", {})
            for key in ("text_examples", "inconsistent_examples"):
                for v in ev.get(key, []) or []:
                    out.append(str(v))
            for group in ev.get("example_groups", []) or []:
                for v in group or []:
                    out.append(str(v))
            for key in ("examples",):
                for v in ev.get(key, []) or []:
                    out.append(str(v))
    return out


def collect_string_evidence(report: dict, column: str) -> list[str]:
    """Every literal string the report exposes for ``column`` (deduped).

    Sources: metadata sample values, cardinality top values, mixed-type text
    examples and rule-engine flag evidence. The profiling report only quotes
    a subsample, so the module records ``n_values_examined`` on each format
    rule that rests on it.
    """
    field = _field(report, column)
    seen: list[str] = []

    def _add(values) -> None:
        for v in values or []:
            s = str(v)
            if s not in seen:
                seen.append(s)

    _add(field.get("sample_values"))
    _add(field.get("mixed_type_detail", {}).get("text_examples"))
    _add(list(_cardinality(report, column).get("top_values", {}).keys()))
    _add(_flag_evidence_strings(report, column))
    return seen


_DIGIT_RE = re.compile(r"^\d+$")
_DIGIT_SUFFIX_RE = re.compile(r"^(\d+)([A-Za-z]+)$")
_PREFIXED_ID_RE = re.compile(r"^([A-Za-z])(\d+)$", re.IGNORECASE)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def analyze_string_evidence(values: list[str]) -> dict:
    """Structural summary of a set of observed strings.

    Returns observed digit-core widths, letter-suffix lengths, casing and
    padding facts - the building blocks ``build_*`` functions turn into
    ``expected_format`` regexes.
    """
    analysis = {
        "n_values_examined": len(values),
        "digit_core_widths": set(),
        "suffix_lengths": set(),
        "alpha_only_lengths": set(),
        "has_cancelled_prefix": False,
        "cancel_prefix_examples": [],
        "has_lowercase": False,
        "all_upper_or_digits": True,
        "has_padded": False,
        "min_length": None,
        "max_length": None,
        "timestamp_like": 0,
    }
    for v in values:
        if _TIMESTAMP_RE.match(v):
            analysis["timestamp_like"] += 1
        if v != v.strip():
            analysis["has_padded"] = True
        if any(ch.islower() for ch in v):
            analysis["has_lowercase"] = True
        if _DIGIT_RE.match(v):
            analysis["digit_core_widths"].add(len(v))
        elif (m := _DIGIT_SUFFIX_RE.match(v)):
            analysis["digit_core_widths"].add(len(m.group(1)))
            analysis["suffix_lengths"].add(len(m.group(2)))
        elif (m := _PREFIXED_ID_RE.match(v)):
            analysis["has_cancelled_prefix"] = True
            if len(analysis["cancel_prefix_examples"]) < 5:
                analysis["cancel_prefix_examples"].append(v)
        else:
            analysis["alpha_only_lengths"].add(len(v))
        stripped = v.strip()
        if analysis["min_length"] is None or len(stripped) < analysis["min_length"]:
            analysis["min_length"] = len(stripped)
        if analysis["max_length"] is None or len(stripped) > analysis["max_length"]:
            analysis["max_length"] = len(stripped)
    analysis["digit_core_widths"] = sorted(analysis["digit_core_widths"])
    analysis["suffix_lengths"] = sorted(analysis["suffix_lengths"])
    analysis["alpha_only_lengths"] = sorted(analysis["alpha_only_lengths"])
    analysis["all_upper_or_digits"] = not analysis["has_lowercase"]
    return analysis


def _width_range(widths: list[int], name: str) -> str:
    """Render an observed width list as a regex quantifier, widened by the
    sample tolerance (profiling saw 100k of >1M rows)."""
    if not widths:
        return "+"
    lo, hi = min(widths), max(widths)
    lo = max(1, lo - WIDTH_TOLERANCE)
    hi = hi + WIDTH_TOLERANCE
    if lo == hi == 1:
        return ""
    if lo == hi:
        return f"{{{lo}}}"
    return f"{{{lo},{hi}}}"


def _evidence_pointer(report: dict, *path: str) -> dict:
    """Bind a rule to the report location its numbers came from."""
    return {"report_path": ".".join(path)}


def _constraint(constraint_id: str, kind: str, severity: str, rule: str,
                message: str, basis: str,
                observed_violations: int | float | None = None,
                evidence: dict | None = None) -> dict:
    """Build one machine-checkable schema constraint record."""
    record = {
        "constraint_id": constraint_id,
        "kind": kind,          # type | nullability | range | format | ...
        "severity": severity,  # hard | review | info
        "rule": rule,
        "message": message,
        "basis": basis,        # observed | domain | derived
    }
    if observed_violations is not None:
        record["observed_violations"] = observed_violations
    if evidence:
        record["evidence_from_report"] = evidence
    return record


# ---------------------------------------------------------------------------
# Column builders (report + priors -> schema entry)
# ---------------------------------------------------------------------------

def _observed_block(report: dict, column: str) -> dict:
    """Echo the report figures the schema entry rests on (auditability)."""
    field = _field(report, column)
    block: dict = {
        "n_rows": field.get("n_rows"),
        "n_null": field.get("n_null"),
        "null_pct": field.get("null_pct"),
        "n_unique": field.get("n_unique"),
        "inferred_dtype": field.get("inferred_dtype"),
        "semantic_type": field.get("semantic_type"),
        "is_mixed_type": field.get("is_mixed_type"),
        "sample_values": field.get("sample_values", [])[:5],
    }
    for key in ("min", "max"):
        if key in field:
            block[f"field_{key}"] = field[key]
    dist = _distribution(report, column)
    if dist:
        for key in ("mean", "median", "std", "min", "max", "q1", "q3",
                    "iqr", "n_negative", "n_zero", "skew", "kurtosis",
                    "n_iqr_outliers", "outlier_pct", "count"):
            if key in dist:
                block[f"dist_{key}"] = dist[key]
    return block


def _post_clean_block(report: dict, column: str, **targets) -> dict:
    """Acceptance targets: the row/column counts a clean dataset must show."""
    field = _field(report, column)
    block = {"n_rows": field.get("n_rows")}
    block.update(targets)
    return block


def build_identifier_schema(report: dict, column: str, prior: dict) -> dict:
    """Schema for InvoiceNo / StockCode / CustomerID.

    Splits the column's non-null rows into concrete value *families* whose
    observed counts (straight from the report's mixed-type counters) sum to
    the profiled row count - the accounting the ``validation`` section
    re-checks.
    """
    field = _field(report, column)
    counts = _mixed_counts(field)
    n_rows = int(field.get("n_rows", 0))
    n_null = int(field.get("n_null", 0))
    n_non_null = n_rows - n_null
    strings = collect_string_evidence(report, column)
    analysis = analyze_string_evidence(strings)

    width_q = _width_range(analysis["digit_core_widths"], column)
    cancel_prefix = prior.get("cancel_prefix", "C")
    observed_fams: list[dict] = []
    accounted = 0

    if counts["n_numeric_like"]:
        pattern_obs = (
            f"^\\d{{{analysis['digit_core_widths'][0]}}}$"
            if len(analysis["digit_core_widths"]) == 1 and analysis["digit_core_widths"]
            else f"^\\d{width_q}$"
        )
        observed_fams.append({
            "name": "numeric_code",
            "pattern": f"^\\d{width_q}$",
            "pattern_observed": pattern_obs,
            "observed_rows": counts["n_numeric_like"],
            "observed_pct": round(100.0 * counts["n_numeric_like"] / n_rows, 3),
            "basis": "observed",
            "evidence_from_report": _evidence_pointer(
                report, f"metadata.fields.{column}",
                "mixed_type_detail.n_numeric_like"),
        })
        accounted += counts["n_numeric_like"]

    if counts["n_digit_core_with_letter_suffix"]:
        suffix_q = _width_range(analysis["suffix_lengths"] or [1], column)
        observed_fams.append({
            "name": "digit_core_with_letter_suffix",
            "pattern": f"^\\d{width_q}[A-Za-z]{suffix_q}$",
            "pattern_observed": (
                f"^\\d{{{analysis['digit_core_widths'][0]}}}"
                f"[A-Za-z]{{{analysis['suffix_lengths'][0]}}}$"
                if len(analysis["digit_core_widths"]) == 1
                and len(analysis["suffix_lengths"]) == 1
                else f"^\\d{width_q}[A-Za-z]{suffix_q}$"
            ),
            "observed_rows": counts["n_digit_core_with_letter_suffix"],
            "observed_pct": round(
                100.0 * counts["n_digit_core_with_letter_suffix"] / n_rows, 3),
            "basis": "observed",
            "evidence_from_report": _evidence_pointer(
                report, f"metadata.fields.{column}",
                "mixed_type_detail.n_digit_core_with_letter_suffix"),
        })
        accounted += counts["n_digit_core_with_letter_suffix"]

    if counts["n_cancelled_style_prefix"]:
        observed_fams.append({
            "name": "prefixed_code",
            "pattern": f"^{cancel_prefix}\\d{width_q}$",
            "pattern_observed": (
                f"^{cancel_prefix}\\d{{{analysis['digit_core_widths'][0]}}}$"
                if len(analysis["digit_core_widths"]) == 1
                and analysis["digit_core_widths"]
                else f"^{cancel_prefix}\\d{width_q}$"
            ),
            "observed_rows": counts["n_cancelled_style_prefix"],
            "observed_pct": round(
                100.0 * counts["n_cancelled_style_prefix"] / n_rows, 3),
            "basis": "observed",
            "note": (
                f"A leading '{cancel_prefix}' marks a credit/cancellation "
                "event, not a distinct entity - valid values, never dropped "
                "silently."
            ),
            "evidence_from_report": _evidence_pointer(
                report, f"metadata.fields.{column}",
                "mixed_type_detail.n_cancelled_style_prefix"),
        })
        accounted += counts["n_cancelled_style_prefix"]

    remainder = n_non_null - accounted
    if remainder > 0:
        service_pool = (
            _flag_evidence_strings(report, column)
            + list(field.get("mixed_type_detail", {}).get("text_examples")
                   or [])
        )
        examples: list[str] = []
        for v in service_pool:
            if (not _DIGIT_RE.match(v) and not _DIGIT_SUFFIX_RE.match(v)
                    and not _PREFIXED_ID_RE.match(v) and v not in examples):
                examples.append(v)
        observed_fams.append({
            "name": "service_or_adjustment_code",
            "pattern": "^[A-Za-z][A-Za-z0-9 .'/&_()-]{0,30}$",
            "pattern_observed": "^[A-Z][A-Za-z0-9 .'/&_()-]{0,30}$",
            "matching": "case-insensitive at match time; values "
                        "canonicalise to upper() afterwards",
            "observed_rows": remainder,
            "observed_pct": round(100.0 * remainder / n_rows, 3),
            "basis": "observed",
            "note": (
                "Non-merchandise codes (carriage, samples, manual "
                "adjustments, test rows). The operative pattern accepts "
                "a lowercase lead letter because this column carries "
                "case-variant spellings of the same codes; values still "
                "canonicalise to upper(). Valid rows; keep, do not drop."
            ),
            "examples": examples[:10],
            "evidence_from_report": _evidence_pointer(
                report, f"metadata.fields.{column}",
                "mixed_type_detail.n_text"),
        })
        accounted += remainder

    coverage_pct = (round(100.0 * accounted / n_non_null, 3)
                    if n_non_null else 100.0)
    inner = "|".join(f"(?:{fam['pattern'][1:-1]})" for fam in observed_fams)
    combined = f"^(?:{inner})$" if observed_fams else "^$"

    constraints: list[dict] = [
        _constraint(
            f"{column.lower()}-type", "type", "hard",
            f"{column} is text ({prior['expected_dtype']})",
            f"{column} must stay text: its text codes would be destroyed "
            "by numeric casting.",
            "observed" if field.get("is_mixed_type") else "domain",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.inferred_dtype")),
        _constraint(
            f"{column.lower()}-pattern", "format", "hard",
            f"{column} matches {combined}",
            f"Every non-null {column} must match one known value family.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.mixed_type_detail")),
    ]
    if n_null and not prior.get("nullable"):
        constraints.append(_constraint(
            f"{column.lower()}-null", "nullability", "hard",
            f"{column} IS NOT NULL",
            f"{column} is never missing in the profiled data.",
            "observed",
            observed_violations=n_null,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.n_null")))
    if prior.get("canonical_case") == "upper":
        constraints.append(_constraint(
            f"{column.lower()}-case", "format", "review",
            f"upper({column}) == {column}",
            f"{column} must be upper-case after cleaning.",
            "observed",
            evidence=_evidence_pointer(
                report, "flags.inconsistent_formats")))
    if prior.get("trim_required"):
        constraints.append(_constraint(
            f"{column.lower()}-trim", "format", "review",
            f"trim({column}) == {column}",
            f"{column} must carry no leading/trailing whitespace.",
            "observed",
            evidence=_evidence_pointer(
                report, "flags.inconsistent_formats")))

    entry: dict = {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": prior["role"],
        "expected_dtype": prior["expected_dtype"],
        "nullable": prior["nullable"],
        "null_semantics": (
            "Missing denotes guest checkout / unattributed orders - a valid "
            "state that must be preserved. Never fill with 0 or -1."
            if prior.get("nullable") else "No nulls expected."),
        "storage": {
            "observed_storage_type": (
                "float64 (NaN-coerced; values render as e.g. '13085.0')"
                if column == "CustomerID" else "object (text)"),
            "expected_storage_type": (
                "Int64 (nullable integer)" if column == "CustomerID"
                else "string"),
        },
        "value_families": observed_fams,
        "family_coverage_pct": coverage_pct,
        "expected_format": {
            "pattern": combined,
            "variants": observed_fams,
            "case": prior.get("canonical_case"),
            "trim_required": bool(prior.get("trim_required")),
            "n_evidence_values_examined": analysis["n_values_examined"],
        },
        "constraints": constraints,
        "observed": _observed_block(report, column),
    }

    if column == "InvoiceNo":
        entry["expected_post_clean"] = _post_clean_block(
            report, column, n_null_max=0, n_pattern_mismatch_max=0,
            n_cancelled_preserved=counts["n_cancelled_style_prefix"],
            storage="string")
    elif column == "StockCode":
        entry["expected_post_clean"] = _post_clean_block(
            report, column, n_null_max=0, n_pattern_mismatch_max=0,
            n_lowercase_max=0, n_padded_max=0, storage="string")
    else:  # CustomerID
        id_range = _customer_id_range(report, column, prior, field)
        entry["value_range"] = id_range["value_range"]
        entry["constraints"].extend(id_range["extra_constraints"])
        entry["expected_post_clean"] = _post_clean_block(
            report, column, null_pct=field.get("null_pct"),
            null_pct_max=round(float(field.get("null_pct", 0.0)) + 5.0, 2),
            n_fractional_max=0, out_of_block_max=0, storage="Int64")
    return entry


def _customer_id_range(report: dict, column: str, prior: dict,
                       field: dict) -> dict:
    """Hard ID block for CustomerID: observed [min, max] rounded outward to
    the nearest ``bounds_rounding`` step (documented ``derived`` rule)."""
    fmin = field.get("min")
    fmax = field.get("max")
    step = int(prior.get("bounds_rounding", 1000))
    hard_min = int(fmin // step * step) if fmin is not None else None
    hard_max = int(-(-fmax // step) * step) if fmax is not None else None
    value_range = {
        "min": hard_min,
        "max": hard_max,
        "observed_min": fmin,
        "observed_max": fmax,
        "zero_allowed": False,
        "negative_allowed": False,
        "rule": (f"{hard_min} <= {column} <= {hard_max} (observed range "
                 f"[{fmin:g}, {fmax:g}] rounded outward to the nearest "
                 f"{step})"),
        "id_block_note": (
            "IDs are non-contiguous; never assume sequence or density."),
        "basis": "derived",
        "evidence_from_report": _evidence_pointer(
            report, f"metadata.fields.{column}.min"),
    }
    extra = [
        _constraint(
            "customerid-integer", "type", "hard",
            "CustomerID % 1 == 0",
            "CustomerID is a whole-number identifier; fractional values "
            "are data errors.",
            "domain",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.inferred_dtype")),
        _constraint(
            "customerid-block", "range", "hard",
            f"{hard_min} <= CustomerID <= {hard_max}",
            "CustomerID must fall inside the anonymised ID block.",
            "derived",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.min")),
        _constraint(
            "customerid-not-a-measure", "consistency", "info",
            "never aggregate CustomerID (no mean/sum)",
            "CustomerID only joins and groups; it is never a model "
            "feature raw.",
            "domain"),
        _constraint(
            "customerid-pii", "privacy", "review",
            "hash/anonymise CustomerID before sharing",
            "CustomerID directly singles out an individual.",
            "observed",
            evidence=_evidence_pointer(report, "flags.pii_columns")),
    ]
    return {"value_range": value_range, "extra_constraints": extra}


def _tier_bounds(q1: float, q3: float, iqr: float, k: float,
                 hard_min: float | None, hard_max: float | None) -> list[float]:
    lo, hi = q1 - k * iqr, q3 + k * iqr
    if hard_min is not None:
        lo = max(lo, hard_min)
    if hard_max is not None:
        hi = min(hi, hard_max)
    return [round(lo, 4), round(hi, 4)]


def build_measure_schema(report: dict, column: str, prior: dict) -> dict:
    """Schema for Quantity / UnitPrice: hard bounds plus IQR tiers.

    Tiers: ``typical`` (k=1.5 IQR fences), ``plausible`` (k=3.0 fences),
    ``atypical`` (inside the hard bounds but outside plausible - review,
    never auto-drop), ``invalid`` (outside the hard bounds - reject).
    """
    field = _field(report, column)
    dist = _distribution(report, column)
    n_rows = int(field.get("n_rows", 0))

    hard_min = 0.0 if not prior.get("negative_allowed", True) else None
    if column == "Quantity":
        hard_min, hard_max = -prior["abs_max"], prior["abs_max"]
    elif column == "UnitPrice":
        hard_min, hard_max = 0.0, float(prior["hard_max"])
    else:
        hard_min = hard_min if hard_min is not None else dist.get("min")
        hard_max = dist.get("max")

    q1, q3 = float(dist["q1"]), float(dist["q3"])
    iqr = float(dist["iqr"])
    typical = _tier_bounds(q1, q3, iqr, IQR_TYPICAL_K, hard_min, hard_max)
    plausible = _tier_bounds(q1, q3, iqr, IQR_PLAUSIBLE_K, hard_min, hard_max)

    tiers = [
        {"tier": "typical", "min": typical[0], "max": typical[1],
         "action": "none",
         "basis": "derived",
         "note": f"Inside the {IQR_TYPICAL_K}x IQR fences."},
        {"tier": "plausible", "min": plausible[0], "max": plausible[1],
         "action": "review (plausible: bulk order / premium item)",
         "basis": "derived",
         "note": f"Inside the {IQR_PLAUSIBLE_K}x IQR fences."},
        {"tier": "atypical", "min": hard_min, "max": hard_max,
         "action": "quarantine for review; never auto-drop",
         "basis": "derived",
         "note": "Inside the hard bounds but outside the plausible band."},
        {"tier": "invalid", "min": None, "max": None,
         "action": "reject row as a data error",
         "basis": "domain",
         "note": f"Outside [{hard_min}, {hard_max}]."},
    ]

    value_range = {
        "min": hard_min,
        "max": hard_max,
        "observed_min": dist.get("min"),
        "observed_max": dist.get("max"),
        "zero_allowed": bool(prior.get("zero_allowed", True)),
        "negative_allowed": bool(prior.get("negative_allowed", True)),
        "typical": {"min": typical[0], "max": typical[1]},
        "plausible": {"min": plausible[0], "max": plausible[1]},
        "quantiles": {
            "q1": dist.get("q1"), "median": dist.get("median"),
            "q3": dist.get("q3"), "iqr": dist.get("iqr")},
        "distribution_shape": {
            "mean": dist.get("mean"), "std": dist.get("std"),
            "skew": dist.get("skew"), "kurtosis": dist.get("kurtosis"),
            "n_iqr_outliers": dist.get("n_iqr_outliers"),
            "outlier_pct": dist.get("outlier_pct")},
        "tiers": tiers,
        "basis": "derived",
        "evidence_from_report": _evidence_pointer(
            report, "statistics.numeric_profile.distributions", column),
    }

    constraints: list[dict] = [
        _constraint(
            f"{column.lower()}-dtype", "type", "hard",
            f"{column} parses as {prior['expected_dtype']}",
            f"{column} must be {prior['expected_dtype']}-typed after cleaning.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, f"metadata.fields.{column}.inferred_dtype")),
        _constraint(
            f"{column.lower()}-bounds", "range", "hard",
            f"{hard_min} <= {column} <= {hard_max}",
            f"{column} outside [{hard_min}, {hard_max}] is a data error.",
            "domain",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "statistics.numeric_profile.distributions", column)),
    ]

    if column == "Quantity":
        n_neg = int(dist.get("n_negative", 0))
        constraints += [
            _constraint(
                "quantity-integer", "type", "hard",
                "Quantity % 1 == 0",
                "Quantities are whole units; fractional quantities are "
                "data errors.",
                "domain",
                observed_violations=0,
                evidence=_evidence_pointer(
                    report, "metadata.fields.Quantity.inferred_dtype")),
            _constraint(
                "quantity-no-zero", "range", "hard",
                "Quantity != 0",
                "A zero-unit line item is physically meaningless.",
                "observed",
                observed_violations=int(dist.get("n_zero", 0)),
                evidence=_evidence_pointer(
                    report, "statistics.numeric_profile.distributions",
                    "Quantity.n_zero")),
            _constraint(
                "quantity-sign", "consistency", "info",
                "Quantity < 0  <=>  return/credit line",
                f"{n_neg} ({round(100.0 * n_neg / max(n_rows, 1), 2)}%) "
                "negative rows are returns/cancellations - keep with sign, "
                "never take abs().",
                "observed",
                evidence=_evidence_pointer(
                    report, "statistics.numeric_profile.distributions",
                    "Quantity.n_negative")),
            _constraint(
                "quantity-bulk", "range", "review",
                f"flag |Quantity| > {prior['bulk_threshold']}",
                "Bulk quantities are plausible wholesale orders but must "
                "be reviewed, not auto-dropped.",
                "observed",
                evidence=_evidence_pointer(report, "flags.suspicious_columns")),
        ]
        post = _post_clean_block(
            report, column, n_null_max=0, n_zero_max=0, n_non_integer_max=0,
            n_negative_preserved=n_neg, out_of_bounds_max=0, storage="int64")
        notes = (
            "Sign is signal: negative = return/credit. The 11.05% of rows "
            "outside the IQR fences are ordinary retail spread, not dirt - "
            "cleaning must not winsorise them away.")
    else:  # UnitPrice
        n_zero = int(dist.get("n_zero", 0))
        constraints += [
            _constraint(
                "unitprice-non-negative", "range", "hard",
                "UnitPrice >= 0.0",
                "A negative price is impossible - data error.",
                "observed",
                observed_violations=int(dist.get("n_negative", 0)),
                evidence=_evidence_pointer(
                    report, "statistics.numeric_profile.distributions",
                    "UnitPrice.n_negative")),
            _constraint(
                "unitprice-zero-flag", "consistency", "review",
                "UnitPrice == 0  =>  adjustment/giveaway line",
                f"{n_zero} zero-price rows are adjustments/giveaways, not "
                "real prices: keep flagged, exclude from revenue metrics.",
                "observed",
                evidence=_evidence_pointer(
                    report, "statistics.numeric_profile.distributions",
                    "UnitPrice.n_zero")),
            _constraint(
                "unitprice-currency", "format", "hard",
                "UnitPrice has at most 2 decimal places",
                "Prices are currency values with at most 2 decimals.",
                "observed",
                observed_violations=0,
                evidence=_evidence_pointer(
                    report, "statistics.cardinality.UnitPrice.top_values")),
            _constraint(
                "unitprice-extreme", "range", "review",
                f"flag UnitPrice > {prior['review_threshold']}",
                "Large-ticket prices are plausible but must be verified "
                "against the source system.",
                "observed",
                evidence=_evidence_pointer(report, "flags.suspicious_columns")),
        ]
        post = _post_clean_block(
            report, column, n_null_max=0, n_negative_max=0,
            n_zero_preserved_flagged=n_zero,
            n_gt_2dp_max=0, out_of_bounds_max=0, storage="float64")
        notes = (
            "Zero prices are legitimate adjustment lines; negatives are "
            "not. Only 13 rows exceed 1000.0.")

    return {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": prior["role"],
        "expected_dtype": prior["expected_dtype"],
        "nullable": prior["nullable"],
        "null_semantics": "No nulls expected.",
        "storage": {
            "observed_storage_type": (
                "int64" if column == "Quantity" else "float64"),
            "expected_storage_type": (
                "int64" if column == "Quantity" else "float64"),
        },
        "value_range": value_range,
        "expected_format": {
            "decimal_places_max": (
                0 if column == "Quantity"
                else prior["max_decimal_places"]),
            "signed": bool(prior.get("negative_allowed", True)),
        },
        "constraints": constraints,
        "observed": _observed_block(report, column),
        "expected_post_clean": post,
        "notes": notes,
    }


def build_temporal_schema(report: dict, column: str, prior: dict) -> dict:
    """Schema for InvoiceDate: fixed text format plus a business window.

    The window is *derived*: floored/ceiled to whole calendar years from the
    observed min/max, which reproduces the 2009-01-01..2011-12-31 window the
    Module 1 rule engine independently uses.
    """
    field = _field(report, column)
    strings = collect_string_evidence(report, column)
    analysis = analyze_string_evidence(strings)
    n_match = analysis["timestamp_like"]
    n_examined = analysis["n_values_examined"]

    obs_min = str(field.get("min", ""))
    obs_max = str(field.get("max", ""))
    min_year = int(obs_min[:4]) if re.match(r"^\d{4}", obs_min) else None
    max_year = int(obs_max[:4]) if re.match(r"^\d{4}", obs_max) else None
    window_min = f"{min_year}-01-01 00:00:00" if min_year else None
    window_max = f"{max_year}-12-31 23:59:59" if max_year else None

    prior_years = prior.get("business_years")
    years_agree = (
        prior_years is not None and min_year is not None
        and max_year is not None
        and (min_year, max_year) == (prior_years[0], prior_years[1]))

    constraints = [
        _constraint(
            "invoicedate-format", "format", "hard",
            f"InvoiceDate matches {prior['text_pattern']}",
            "InvoiceDate renders as '%Y-%m-%d %H:%M:%S': 4-digit year, "
            "2-digit month/day/hour/minute/second, space-separated.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "metadata.fields.InvoiceDate.sample_values")),
        _constraint(
            "invoicedate-resolution", "format", "hard",
            "InvoiceDate has second resolution, no sub-second part",
            "Finer timestamps than one second are data errors.",
            "domain"),
        _constraint(
            "invoicedate-tz", "format", "hard",
            "InvoiceDate is tz-naive",
            "No timezone offsets or zone names may appear.",
            "domain"),
        _constraint(
            "invoicedate-window", "range", "hard",
            f"{window_min} <= InvoiceDate <= {window_max}",
            "InvoiceDate must fall inside the 2009-2011 business window "
            "(observed min/max floored/ceiled to whole calendar years).",
            "derived",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "metadata.fields.InvoiceDate.min")),
        _constraint(
            "invoicedate-no-future", "range", "hard",
            "InvoiceDate <= today",
            "A transaction timestamp in the future is a data error.",
            "domain"),
        _constraint(
            "invoicedate-null", "nullability", "hard",
            "InvoiceDate IS NOT NULL",
            "InvoiceDate is never missing in the profiled data.",
            "observed",
            observed_violations=int(field.get("n_null", 0)),
            evidence=_evidence_pointer(
                report, "metadata.fields.InvoiceDate.n_null")),
    ]

    return {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": prior["role"],
        "expected_dtype": prior["expected_dtype"],
        "nullable": prior["nullable"],
        "null_semantics": "No nulls expected.",
        "storage": {
            "observed_storage_type": "datetime64[us]",
            "expected_storage_type": "datetime64[ns]",
        },
        "value_range": {
            "min": window_min,
            "max": window_max,
            "observed_min": obs_min,
            "observed_max": obs_max,
            "rule": (
                f"{window_min} <= InvoiceDate <= {window_max} "
                "(calendar-year envelope of the observed range)"),
            "basis": "derived",
            "business_years_agree_with_prior": years_agree,
            "evidence_from_report": _evidence_pointer(
                report, "metadata.fields.InvoiceDate.min"),
        },
        "expected_format": {
            "strftime": prior["text_format"],
            "pattern": prior["text_pattern"],
            "pattern_observed": prior["text_pattern"],
            "pattern_match": (
                f"{n_match}/{n_examined} evidence values match"),
            "resolution": prior["resolution"],
            "timezone": prior["timezone"],
            "n_evidence_values_examined": n_examined,
        },
        "constraints": constraints,
        "observed": _observed_block(report, column),
        "expected_post_clean": _post_clean_block(
            report, column, n_null_max=0, n_unparseable_max=0,
            out_of_window_max=0, storage="datetime64[ns]"),
    }


def build_text_schema(report: dict, column: str, prior: dict) -> dict:
    """Schema for Description: a StockCode attribute with whitespace/case
    defects. Gaps are imputable from the StockCode lookup, so the schema
    demands zero residual nulls."""
    field = _field(report, column)
    n_rows = int(field.get("n_rows", 0))
    null_pct = float(field.get("null_pct", 0.0))
    strings = collect_string_evidence(report, column)
    analysis = analyze_string_evidence(strings)
    card = _cardinality(report, column)

    max_len = max(analysis["max_length"] or 0, 1)
    constraints = [
        _constraint(
            "description-type", "type", "hard",
            "Description is text",
            "Description is free-text product naming.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "metadata.fields.Description.inferred_dtype")),
        _constraint(
            "description-trim", "format", "review",
            "trim(Description) == Description",
            "Descriptions carry no leading/trailing whitespace "
            "(18,717 padded values observed).",
            "observed",
            evidence=_evidence_pointer(report, "flags.inconsistent_formats")),
        _constraint(
            "description-spacing", "format", "review",
            "no runs of 2+ internal spaces",
            "Internal whitespace collapses to single spaces.",
            "observed",
            evidence=_evidence_pointer(report, "flags.inconsistent_formats")),
        _constraint(
            "description-case", "format", "review",
            "upper(Description) == Description",
            "Descriptions canonicalise to upper case (case-variant "
            "groups observed).",
            "observed",
            evidence=_evidence_pointer(report, "flags.inconsistent_formats")),
        _constraint(
            "description-nonempty", "format", "hard",
            "len(trim(Description)) >= 1",
            "Empty strings are not valid descriptions.",
            "domain",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "statistics.cardinality.Description.n_unique")),
        _constraint(
            "description-length", "format", "review",
            f"len(Description) <= {prior['max_length']}",
            "Descriptions are short product names, not prose.",
            "domain"),
        _constraint(
            "description-null", "nullability", "review",
            "Description IS NULL only before imputation",
            f"{null_pct}% of rows lack a description; every gap is "
            "imputable from its StockCode, so zero residual nulls are "
            "expected after cleaning.",
            "observed",
            observed_violations=int(field.get("n_null", 0)),
            evidence=_evidence_pointer(
                report, "metadata.fields.Description.n_null")),
        _constraint(
            "description-per-stockcode", "consistency", "review",
            "one canonical Description per StockCode",
            "After case/whitespace normalisation each StockCode maps to "
            "exactly one Description.",
            "observed",
            evidence=_evidence_pointer(
                report, "statistics.cardinality.StockCode.n_unique")),
    ]

    return {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": prior["role"],
        "expected_dtype": prior["expected_dtype"],
        "nullable": prior["nullable"],
        "null_semantics": "Missing means the product name was not captured; "
                          "impute from StockCode, never invent text.",
        "null_policy": {
            "observed_null_pct": null_pct,
            "max_acceptable_null_pct": 1.0,
            "action": "impute from the StockCode->Description lookup; only "
                      "if the StockCode itself is unknown may the row be "
                      "dropped",
            "basis": "observed",
            "evidence_from_report": _evidence_pointer(
                report, "metadata.fields.Description.null_pct"),
        },
        "storage": {
            "observed_storage_type": "object (text)",
            "expected_storage_type": "string",
        },
        "value_range": None,
        "expected_format": {
            "pattern": "^\\S(.*\\S)?$",
            "pattern_observed": "^\\S(.*\\S)?$",
            "case": prior.get("canonical_case"),
            "trim_required": True,
            "collapse_internal_whitespace": True,
            "min_length": 1,
            "max_length": prior["max_length"],
            "observed_max_length": max_len,
            "observed_n_unique": card.get("n_unique"),
            "n_evidence_values_examined": analysis["n_values_examined"],
        },
        "constraints": constraints,
        "observed": _observed_block(report, column),
        "expected_post_clean": _post_clean_block(
            report, column, null_pct_max=0.0, n_padded_max=0,
            n_case_variant_groups_max=0, storage="string"),
    }


def build_categorical_schema(report: dict, column: str, prior: dict) -> dict:
    """Schema for Country: closed low-cardinality dimension with alias
    normalisation. The only place where ``basis: domain`` carries a
    *reference set*: the report quotes 8 of the 28 observed spellings, and
    the prior completes them to the 28 canonical names of this dataset
    family (unknown spellings are flagged, never dropped)."""
    field = _field(report, column)
    n_unique = int(field.get("n_unique", 0))
    strings = collect_string_evidence(report, column)
    analysis = analyze_string_evidence(strings)

    seen = sorted({v.strip() for v in strings if v.strip()})
    seen_canon = sorted({_canonical_country(v) for v in seen})
    reference = sorted(set(COUNTRY_REFERENCE) | set(seen_canon))
    corroborated = sorted(set(seen_canon) & set(COUNTRY_REFERENCE))

    constraints = [
        _constraint(
            "country-type", "type", "hard",
            "Country is text",
            "Country is a geographic label.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "metadata.fields.Country.inferred_dtype")),
        _constraint(
            "country-trim", "format", "review",
            "trim(Country) == Country",
            "Country names carry no leading/trailing whitespace.",
            "domain"),
        _constraint(
            "country-case", "format", "review",
            "Country is title case ('United Kingdom', not 'UNITED KINGDOM')",
            "Country canonicalises to title case.",
            "observed",
            evidence=_evidence_pointer(
                report, "statistics.cardinality.Country.top_values")),
        _constraint(
            "country-pattern", "format", "hard",
            "Country matches ^[A-Z][A-Za-z .'\\-]{1,29}$",
            "Country names are letters plus space/dot/apostrophe/hyphen.",
            "observed",
            observed_violations=0,
            evidence=_evidence_pointer(
                report, "statistics.cardinality.Country.top_values")),
        _constraint(
            "country-alias", "consistency", "review",
            "legacy spellings map to canonical names (EIRE->Ireland, "
            "USA->United States, plus the remaining alias map entries)",
            "Alias spellings denote real countries; normalise, do not drop.",
            "domain"),
        _constraint(
            "country-cardinality", "cardinality", "hard",
            f"n_unique(Country) <= {prior['max_n_unique']}",
            f"Country stays low-cardinality (observed {n_unique}; new "
            "distinct values beyond the reference set are flagged).",
            "observed",
            evidence=_evidence_pointer(
                report, "statistics.cardinality.Country.n_unique")),
        _constraint(
            "country-null", "nullability", "hard",
            "Country IS NOT NULL",
            "Country is never missing in the profiled data.",
            "observed",
            observed_violations=int(field.get("n_null", 0)),
            evidence=_evidence_pointer(
                report, "metadata.fields.Country.n_null")),
        _constraint(
            "country-pii", "privacy", "info",
            "Country is a quasi-identifier: safe for grouping, watch "
            "k-anonymity in small cells",
            "Geography only identifies in combination.",
            "observed",
            evidence=_evidence_pointer(report, "flags.pii_columns")),
    ]

    return {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": prior["role"],
        "expected_dtype": prior["expected_dtype"],
        "nullable": prior["nullable"],
        "null_semantics": "No nulls expected.",
        "storage": {
            "observed_storage_type": "str",
            "expected_storage_type": "string",
        },
        "value_range": None,
        "expected_format": {
            "pattern": "^[A-Z][A-Za-z .'\\-]{1,29}$",
            "pattern_observed": "^[A-Z][A-Za-z .'\\-]{1,29}$",
            "case": prior.get("canonical_case"),
            "trim_required": True,
            "n_evidence_values_examined": analysis["n_values_examined"],
        },
        "allowed_values": {
            "mode": "closed_reference_set",
            "values": reference,
            "n_unique_observed": n_unique,
            "max_n_unique": prior["max_n_unique"],
            "values_seen_in_report": seen,
            "values_seen_in_report_canonical": seen_canon,
            "values_not_seen_in_report_sample": sorted(
                set(reference) - set(seen_canon)),
            "aliases": COUNTRY_ALIASES,
            "alias_rule": "strip, title-case, then apply the alias map",
            "on_unknown_value": "flag for review; never silently drop",
            "basis": "domain",
            "evidence_from_report": _evidence_pointer(
                report, "statistics.cardinality.Country.n_unique"),
        },
        "reference": {
            "canonical_countries": reference,
            "corroborated_by_report": {
                "n_unique_in_report": n_unique,
                "n_reference_values": len(reference),
                "values_seen_in_report": corroborated,
            },
        },
        "constraints": constraints,
        "observed": _observed_block(report, column),
        "expected_post_clean": _post_clean_block(
            report, column, n_null_max=0, n_unknown_after_normalisation_max=0,
            n_unique_after_normalisation_max=n_unique, storage="string"),
    }


# ---------------------------------------------------------------------------
# Assembly: dataset contract, row validity, validation
# ---------------------------------------------------------------------------

def build_column_schema(report: dict, column: str) -> dict:
    """Dispatch to the right builder from the column's semantic type."""
    field = _field(report, column)
    semantic = field.get("semantic_type", "")
    prior = SEMANTIC_PRIORS.get(semantic)
    if prior is None:  # unknown column: generic fallback from dtype
        return _generic_column_schema(report, column, field)
    role = prior.get("role")
    if role == "key":
        return build_identifier_schema(report, column, prior)
    if role == "measure":
        return build_measure_schema(report, column, prior)
    if role == "timestamp":
        return build_temporal_schema(report, column, prior)
    if role == "text":
        return build_text_schema(report, column, prior)
    if role == "dimension":
        return build_categorical_schema(report, column, prior)
    return _generic_column_schema(report, column, field)


def _generic_column_schema(report: dict, column: str, field: dict) -> dict:
    """Fallback for columns outside the known retail set: rules straight
    from the inferred dtype, no domain claims."""
    dtype = field.get("inferred_dtype", "string")
    return {
        "column": column,
        "semantic_type": field.get("semantic_type"),
        "role": "unknown",
        "expected_dtype": dtype,
        "nullable": bool(int(field.get("n_null", 0)) > 0),
        "null_semantics": "Unknown column: preserve nulls.",
        "storage": {
            "observed_storage_type": "unknown",
            "expected_storage_type": dtype,
        },
        "value_range": None,
        "expected_format": None,
        "constraints": [
            _constraint(
                f"{column.lower()}-dtype", "type", "hard",
                f"{column} parses as {dtype}",
                f"Unmapped column: only the profiled dtype is enforced.",
                "observed",
                observed_violations=0,
                evidence=_evidence_pointer(
                    report, f"metadata.fields.{column}.inferred_dtype")),
        ],
        "observed": _observed_block(report, column),
        "expected_post_clean": _post_clean_block(
            report, column, storage=dtype),
    }


def _dataset_block(report: dict, schemas: dict) -> dict:
    """Grain, keys and role census for the whole table."""
    fields = report.get("metadata", {}).get("fields", {})
    n_rows = int(report.get("n_rows", 0))
    unique_counts = {c: int(fields[c].get("n_unique", 0)) for c in schemas}
    single_key = [c for c, u in unique_counts.items() if u == n_rows]
    roles: dict[str, int] = {}
    for entry in schemas.values():
        roles[entry.get("role", "unknown")] = (
            roles.get(entry.get("role", "unknown"), 0) + 1)
    return {
        "grain": "one row per invoice line item "
                 "(an InvoiceNo may span many rows)",
        "n_rows_profiled": n_rows,
        "n_columns": len(schemas),
        "columns": list(schemas),
        "primary_key": single_key[0] if len(single_key) == 1 else None,
        "primary_key_note": (
            "No single column is unique: the most granular column "
            f"(InvoiceNo) has {unique_counts.get('InvoiceNo')} distinct "
            f"values over {n_rows} rows, so the grain is the line item, "
            "not the invoice."
            if not single_key else
            f"{single_key[0]} is unique over the profiled rows."),
        "key_candidates": [
            {"columns": ["InvoiceNo", "StockCode", "InvoiceDate"],
             "uniqueness_test": "required",
             "note": "Candidate composite key for the line-item grain; "
                     "cleaning must assert uniqueness, not assume it."}
        ],
        "role_counts": roles,
        "unique_counts": unique_counts,
    }


def _row_validity_rules(report: dict) -> list[dict]:
    """Row-level reading rules: the five legitimate-but-special row kinds
    and exactly how many the report evidences."""
    n_rows = int(report.get("n_rows", 1))
    dist_qty = _distribution(report, "Quantity")
    dist_price = _distribution(report, "UnitPrice")
    cancelled = next(
        (f for f in report.get("flags", {}).get("suspicious_columns", [])
         if f.get("rule_id") == "cancelled-invoices"), {})
    cancelled_n = int(cancelled.get("evidence", {}).get("n_cancelled", 0))
    n_neg = int(dist_qty.get("n_negative", 0))
    n_zero_price = int(dist_price.get("n_zero", 0))
    n_guest = int(_missing(report, "CustomerID").get("n_missing", 0))
    n_nodesc = int(_missing(report, "Description").get("n_missing", 0))

    def _pct(n: int) -> float:
        return round(100.0 * n / max(n_rows, 1), 2)

    return [
        {"rule_id": "rv-1-giveaway-line",
         "rule": "UnitPrice == 0  =>  adjustment/giveaway line, not a sale",
         "observed_rows": n_zero_price, "observed_pct": _pct(n_zero_price),
         "handling": "keep flagged (IsGiveaway); exclude from revenue "
                     "metrics",
         "basis": "observed"},
        {"rule_id": "rv-2-return-line",
         "rule": "Quantity < 0  =>  return/credit line",
         "observed_rows": n_neg, "observed_pct": _pct(n_neg),
         "handling": "keep with sign (IsReturn); never take abs()",
         "basis": "observed"},
        {"rule_id": "rv-3-cancellation",
         "rule": "InvoiceNo LIKE 'C%'  =>  cancellation of a prior invoice",
         "observed_rows": cancelled_n, "observed_pct": _pct(cancelled_n),
         "handling": "retain as negative-value events (IsCancellation); "
                     "net revenue requires pairing with the original",
         "basis": "observed"},
        {"rule_id": "rv-4-guest-order",
         "rule": "CustomerID IS NULL  =>  guest/unattributed order",
         "observed_rows": n_guest, "observed_pct": _pct(n_guest),
         "handling": "keep for revenue analytics; exclude only from "
                     "customer-level segmentation",
         "basis": "observed"},
        {"rule_id": "rv-5-missing-description",
         "rule": "Description IS NULL  =>  product name not captured",
         "observed_rows": n_nodesc, "observed_pct": _pct(n_nodesc),
         "handling": "impute from the StockCode->Description lookup",
         "basis": "observed"},
    ]


def _iter_strings(node) -> object:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _iter_strings(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _iter_strings(v)


def _validate_schema(report: dict, schemas: dict) -> dict:
    """Run-time proof of the deliverable: coverage, accounting, concreteness."""
    checks: list[dict] = []

    def _check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed),
                       "detail": detail})

    columns = list(report.get("columns", []))
    _check("every-column-has-schema",
           set(schemas) == set(columns),
           f"{len(schemas)}/{len(columns)} columns covered: "
           f"{sorted(schemas)}")

    _check("every-column-has-expected-dtype",
           all(bool(e.get("expected_dtype")) for e in schemas.values()),
           "all columns declare a concrete expected_dtype")

    for column, entry in schemas.items():
        fams = entry.get("value_families") or []
        if not fams:
            continue
        total = sum(int(f.get("observed_rows", 0)) for f in fams)
        field = _field(report, column)
        expected = int(field.get("n_rows", 0)) - int(field.get("n_null", 0))
        _check(f"{column}-family-accounting",
               total == expected,
               f"{total} rows across {len(fams)} families vs "
               f"{expected} non-null rows")

    def _has_domain(entry: dict) -> bool:
        fmt = entry.get("expected_format") or {}
        return bool(entry.get("value_range") or entry.get("value_families")
                    or fmt.get("pattern") or entry.get("allowed_values")
                    or entry.get("null_policy"))

    missing_domain = [c for c, e in schemas.items() if not _has_domain(e)]
    _check("every-column-has-value-domain",
           not missing_domain,
           "all columns carry a concrete value domain (range / families / "
           "pattern / allowed values)"
           if not missing_domain else f"missing domain: {missing_domain}")

    offenders = sorted({s for s in _iter_strings(schemas)
                        if any(tok in s for tok in _PLACEHOLDER_TOKENS)})
    _check("no-placeholder-rules",
           not offenders,
           "no rule text is a stub"
           if not offenders else f"stub text found: {offenders[:5]}")

    severities: dict[str, int] = {}
    bases: dict[str, int] = {}
    n_constraints = 0
    for entry in schemas.values():
        for c in entry.get("constraints", []):
            n_constraints += 1
            severities[c.get("severity", "?")] = (
                severities.get(c.get("severity", "?"), 0) + 1)
            bases[c.get("basis", "?")] = bases.get(c.get("basis", "?"), 0) + 1

    return {
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "n_columns": len(schemas),
        "n_constraints": n_constraints,
        "constraints_by_severity": severities,
        "rules_by_basis": bases,
    }


def infer_schema(report: dict) -> dict:
    """Derive the full expected-schema document from a profiling report."""
    t0 = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    columns = [c for c in report.get("columns", [])
               if c in report.get("metadata", {}).get("fields", {})]
    schemas = {c: build_column_schema(report, c) for c in columns}
    row_rules = _row_validity_rules(report)
    validation = _validate_schema(report, schemas)
    return {
        "module": MODULE_NAME,
        "schema_version": SCHEMA_VERSION,
        "schema_name": SCHEMA_NAME,
        "purpose": (
            "Expected schema for the Online Retail II dataset: what "
            "'correct' looks like before cleaning. Every column declares "
            "its expected type, value ranges, formats and acceptance "
            "targets. Cleaning steps validate against this contract."),
        "generated_at": t0,
        "inferred_from": {
            "report": "module1_profiling/profiling_report.json",
            "report_module": report.get("module"),
            "report_version": report.get("version"),
            "source_file": report.get("source_file"),
            "nrows_per_sheet": report.get("nrows_per_sheet"),
            "n_rows": report.get("n_rows"),
            "n_columns": report.get("n_columns"),
        },
        "dataset": _dataset_block(report, schemas),
        "row_validity_rules": row_rules,
        "derived_columns": DERIVED_COLUMNS,
        "columns": schemas,
        "column_order": columns,
        "cleaning_plan": CLEANING_PLAN,
        "validation": validation,
    }


# ---------------------------------------------------------------------------
# File-level API + CLI
# ---------------------------------------------------------------------------

def run_schema_inference(
    report_path: str | Path = DEFAULT_REPORT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> dict:
    """Load ``profiling_report.json``, infer the schema, save + return it.

    Args:
        report_path: Sprint 3's ``profiling_report.json``.
        output_path: Where ``expected_schema.json`` is written.

    Returns:
        The expected-schema document (also written to ``output_path``).
    """
    report = load_profiling_report(report_path)
    schema = infer_schema(report)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return schema


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 4: Module 2 schema inference engine - "
                    "derive expected_schema.json from profiling_report.json.")
    p.add_argument("--report", default=str(DEFAULT_REPORT),
                   help="Path to Sprint 3's profiling_report.json")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Where to save expected_schema.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    schema = run_schema_inference(args.report, args.output)
    summary = {
        "module": schema["module"],
        "schema_version": schema["schema_version"],
        "columns": schema["column_order"],
        "row_validity_rules": [
            {"rule_id": r["rule_id"], "observed_rows": r["observed_rows"],
             "observed_pct": r["observed_pct"]}
            for r in schema["row_validity_rules"]
        ],
        "validation": {
            "all_passed": schema["validation"]["all_passed"],
            "n_constraints": schema["validation"]["n_constraints"],
            "constraints_by_severity": schema["validation"][
                "constraints_by_severity"],
            "rules_by_basis": schema["validation"]["rules_by_basis"],
            "checks": [
                {"check_id": c["check_id"], "passed": c["passed"]}
                for c in schema["validation"]["checks"]
            ],
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSchema inference complete - expected schema -> {args.output}")
    if not schema["validation"]["all_passed"]:
        failed = [c["check_id"] for c in schema["validation"]["checks"]
                  if not c["passed"]]
        raise SystemExit(f"VALIDATION FAILED: {failed}")
    return schema


if __name__ == "__main__":
    main()


