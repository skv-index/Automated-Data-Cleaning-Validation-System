"""Sprint 8 - NLP Column Classifier (Module 3: Validation).

Independently re-classifies what each column semantically represents and
flags columns whose classified meaning does not match their label
(mislabeled or semantically inconsistent columns).

Two layers, kept independent so they can corroborate or disagree:

  * Baseline (Sprint 1 reuse): ``infer_semantic`` + ``infer_logical_dtype``
    from ``module1_profiling.metadata_extractor`` - header lookup with
    generic content-detector fallback (email / url / phone / ...).
  * NLP layer (new, extends Sprint 1): per-column text evidence -
    word-token statistics plus a ``sklearn`` TF-IDF vocabulary fit over a
    deterministic sample of cell strings (vocabulary size + top terms),
    combined with value-shape probes (invoice/SKU/country-gazetteer hit
    rates, datetime parse rate, numeric/decimal structure, cardinality and
    null rate). Transparent weighted evidence per candidate class; the
    argmax wins. Content evidence outweighs the header cue on purpose, so
    a renamed column still classifies by what it CONTAINS.

On the cleaned data all 8 columns classify as labelled (run with
``--demo-mislabel`` to see the flag path fire: Country/CustomerID/
Description are renamed to Email/Phone/Code headers and all three are
caught as mismatches by content).

Usage:
    from module3_validation.column_classifier import run_classifier

    report = run_classifier()  # cleaned_data.csv in, column_classification.json out

CLI:
    python module3_validation/column_classifier.py
    python module3_validation/column_classifier.py --demo-mislabel
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on ``sys.path`` so imports work no matter the
    working directory (repo root, ``module3_validation/``, ...)."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

try:
    from metadata_extractor import (  # noqa: F401
        EMAIL_RE,
        PHONE_RE,
        URL_RE,
        infer_logical_dtype,
        infer_semantic,
    )
except ImportError:  # allow running from the repo root
    from module1_profiling.metadata_extractor import (  # noqa: F401
        EMAIL_RE,
        PHONE_RE,
        URL_RE,
        infer_logical_dtype,
        infer_semantic,
    )

from sklearn.feature_extraction.text import TfidfVectorizer

MODULE_NAME = "module3_validation"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "module2_cleaning" / "cleaned_data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "column_classification.json"

#: Deterministic content sample per column (head slice, no randomness).
SAMPLE_N = 5000

#: Candidate semantic classes the NLP layer chooses from.
CANDIDATE_CLASSES = [
    "invoice_id", "product_id", "product_description", "quantity",
    "transaction_datetime", "unit_price", "customer_id", "country",
    "email", "phone", "url", "boolean_flag", "category", "text",
]

#: Expected semantic per canonical header (Sprint 1 semantics, normalised).
EXPECTED_BY_COLUMN = {
    "InvoiceNo": "invoice_id",
    "StockCode": "product_id",
    "Description": "product_description",
    "Quantity": "quantity",
    "InvoiceDate": "transaction_datetime",
    "UnitPrice": "unit_price",
    "CustomerID": "customer_id",
    "Country": "country",
}

#: Closed country reference set (Sprint 4 contract) + observed aliases.
COUNTRIES = [
    "Australia", "Austria", "Bahrain", "Belgium", "Channel Islands",
    "Cyprus", "Denmark", "Finland", "France", "Germany", "Greece",
    "Iceland", "Ireland", "Israel", "Italy", "Japan", "Lithuania",
    "Netherlands", "Nigeria", "Norway", "Poland", "Portugal", "Spain",
    "Sweden", "Switzerland", "United Arab Emirates", "United Kingdom",
    "United States",
]
_COUNTRY_SET = {c.lower() for c in COUNTRIES}
_COUNTRY_ALIASES = {"eire": "ireland", "usa": "united states",
                    "uk": "united kingdom", "u.k.": "united kingdom"}

INVOICE_RE = re.compile(r"^C?\d{5,7}$")
DIGIT_SUFFIX_RE = re.compile(r"^\d{4,6}[A-Za-z]{1,2}$")
CANCELLED_RE = re.compile(r"^C\d{4,6}$")
SERVICE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 .'/&_()-]{0,30}$")
PURE_INT_RE = re.compile(r"^[+-]?\d+$")
PRICE_RE = re.compile(r"^\d+(\.\d{1,2})?$")
WORD_RE = re.compile(r"\b\w+\b")

#: Mislabel demo: header -> decoy name (content must win over the header).
DEMO_MISLABELS = {
    "Country": "Contact_Email",
    "CustomerID": "PhoneNumber",
    "Description": "Item_Code",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_cleaned(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the official Module 2 output."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Header cues (name evidence; deliberately weaker than content evidence)
# ---------------------------------------------------------------------------

def header_cues(name: str) -> dict[str, float]:
    """Per-class 0/1 header evidence from the column name."""
    n = re.sub(r"[\s_-]+", "", str(name).lower())
    return {
        "invoice_id": 1.0 if ("invoice" in n and "date" not in n) else 0.0,
        "product_id": 1.0 if ("stockcode" in n or "sku" in n or n == "code"
                              or n.endswith("code")) else 0.0,
        "product_description": 1.0 if ("descript" in n or "productname" in n
                                       or n == "name") else 0.0,
        "quantity": 1.0 if ("quantity" in n or n == "qty") else 0.0,
        "transaction_datetime": 1.0 if ("date" in n or "time" in n) else 0.0,
        "unit_price": 1.0 if ("price" in n or "amount" in n or "cost" in n) else 0.0,
        "customer_id": 1.0 if ("customer" in n or "cust" in n) else 0.0,
        "country": 1.0 if ("country" in n or "nation" in n) else 0.0,
        "email": 1.0 if "email" in n or "e-mail" in str(name).lower() else 0.0,
        "phone": 1.0 if ("phone" in n or "tel" in n or "mobile" in n) else 0.0,
        "url": 1.0 if ("url" in n or "website" in n or "link" in n) else 0.0,
    }


def expected_from_header(name: str) -> str | None:
    """Header-implied semantic (the label's claim); None if unmapped.

    Calendar-part names (InvoiceYear, InvoiceMonth, ...) are out of scope:
    the cue fragment ("Invoice") would over-claim them as invoice ids, so
    they report ``unmapped`` instead of a false mismatch.
    """
    if name in EXPECTED_BY_COLUMN:
        return EXPECTED_BY_COLUMN[name]
    norm = re.sub(r"[\s_-]+", "", str(name).lower())
    if any(part in norm for part in
           ("year", "month", "day", "hour", "minute", "second",
            "week", "quarter", "weekday")):
        return None
    cues = header_cues(name)
    best = max(cues, key=lambda k: cues[k])
    return best if cues[best] > 0 else None


# ---------------------------------------------------------------------------
# NLP + value-shape signals
# ---------------------------------------------------------------------------

def _norm_country(text: str) -> str:
    t = text.strip().lower()
    return _COUNTRY_ALIASES.get(t, t)


def column_signals(series: pd.Series, sample_n: int = SAMPLE_N) -> dict:
    """Text/NLP + shape evidence for one column.

    Deterministic head sample of non-null cell strings; TF-IDF word
    vocabulary fit supplies ``vocab_size`` and ``top_terms``.
    """
    non_null = series.dropna()
    null_rate = float(series.isna().mean()) if len(series) else 0.0
    texts = [str(v).strip() for v in non_null.head(sample_n).tolist()]
    n = len(texts)
    if n == 0:
        return {"n_sampled": 0, "null_rate": round(null_rate, 4)}

    lowered = [t.lower() for t in texts]
    words = [WORD_RE.findall(t) for t in lowered]
    n_tokens = [len(w) for w in words]
    avg_tokens = float(sum(n_tokens) / n)
    alpha_chars = sum(1 for t in lowered for c in t if c.isalpha())
    all_chars = sum(len(t) for t in lowered) or 1
    digit_chars = sum(1 for t in lowered for c in t if c.isdigit())

    # Canonical numeric rendering: cleaned ID columns arrive as floats
    # ("13085.0"); integer-valued strings collapse to int form so the int /
    # ID-block / invoice probes see the value, not the storage cast.
    canon = []
    for t in texts:
        try:
            f = float(t)
            if math.isfinite(f) and f.is_integer():
                canon.append(str(int(f)))
            else:
                canon.append(t)
        except ValueError:
            canon.append(t)

    vectorizer = TfidfVectorizer(analyzer="word",
                                 token_pattern=r"(?u)\b\w+\b",
                                 max_features=10000)
    tfidf = vectorizer.fit_transform(lowered)
    terms = vectorizer.get_feature_names_out()
    mean_scores = tfidf.mean(axis=0).A1
    top_idx = mean_scores.argsort()[::-1][:5]
    top_terms = [str(terms[i]) for i in top_idx]

    numeric = pd.to_numeric(non_null.head(sample_n), errors="coerce")
    numeric_rate = float(numeric.notna().mean())
    num = numeric.dropna()
    has_negative = bool((num < 0).any()) if len(num) else False
    has_decimals = bool(((num % 1) != 0).any()) if len(num) else False
    median_abs = float(num.abs().median()) if len(num) else 0.0

    dt = pd.to_datetime(non_null.head(2000), errors="coerce", format="mixed")
    datetime_rate = float(dt.notna().mean())
    # Sprint 1 guard (tightened): plain digit strings (IDs, quantities)
    # "parse" as nanosecond timestamps, and scaled z-scores ("-0.027")
    # parse as offsets - so only trust the parse rate when values carry
    # genuine date structure (slashes/colons, month names, or a full
    # date pattern). A bare hyphen (negative numbers) must NOT count.
    datey = sum(1 for t in texts[:2000]
                if "/" in t or ":" in t
                or re.search(r"\d{4}-\d{1,2}-\d{1,2}", t)
                or re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                             t, re.I))
    if not texts[:2000] or datey / len(texts[:2000]) < 0.5:
        datetime_rate = 0.0

    sig = {
        "n_sampled": n,
        "null_rate": round(null_rate, 4),
        "n_unique_full": int(non_null.nunique()),
        "cardinality_ratio": round(float(non_null.nunique() / len(series)), 6) if len(series) else 0.0,
        # NLP signals
        "vocab_size": int(len(terms)),
        "avg_tokens": round(avg_tokens, 3),
        "alpha_ratio": round(alpha_chars / all_chars, 4),
        "digit_ratio": round(digit_chars / all_chars, 4),
        "top_terms": top_terms,
        # value-shape probes
        "numeric_rate": round(numeric_rate, 4),
        "has_negative": has_negative,
        "has_decimals": has_decimals,
        "median_abs": round(median_abs, 4),
        "datetime_rate": round(datetime_rate, 4),
        "email_rate": round(float(sum(1 for t in texts if EMAIL_RE.match(t)) / n), 4),
        "url_rate": round(float(sum(1 for t in texts if URL_RE.match(t)) / n), 4),
        "phone_rate": round(float(sum(1 for t in canon if PHONE_RE.match(t)) / n), 4),
        "bool_rate": round(float(sum(1 for t in lowered
                                     if t in {"true", "false", "0", "1", "yes", "no",
                                              "y", "n", "t", "f"}) / n), 4),
        "invoice_pattern_rate": round(float(sum(1 for t in canon if INVOICE_RE.match(t)) / n), 4),
        "service_code_rate": round(float(sum(1 for t in texts if SERVICE_RE.match(t)
                                             and not PURE_INT_RE.match(t)) / n), 4),
        "digit_suffix_rate": round(float(sum(1 for t in canon if DIGIT_SUFFIX_RE.match(t)) / n), 4),
        "cancelled_prefix_rate": round(float(sum(1 for t in canon if CANCELLED_RE.match(t)
                                                 or t.upper().startswith("C") and INVOICE_RE.match(t)) / n), 4),
        "price_pattern_rate": round(float(sum(1 for t in texts if PRICE_RE.match(t)) / n), 4),
        "country_hit_rate": round(float(sum(1 for t in texts if _norm_country(t) in _COUNTRY_SET) / n), 4),
    }
    # Numeric ID-block probe (5-digit ints inside the anonymised block).
    id_hits = 0
    for t in canon:
        if PURE_INT_RE.match(t):
            try:
                v = int(t)
                if 12000 <= v <= 19000:
                    id_hits += 1
            except ValueError:
                pass
    sig["id_block_rate"] = round(id_hits / n, 4)
    sig["int_rate"] = round(float(sum(1 for t in canon if PURE_INT_RE.match(t)) / n), 4)
    # SKU family rate: union of the Sprint 4 StockCode value families.
    sig["sku_family_rate"] = round(float(sum(
        1 for t in canon
        if PURE_INT_RE.match(t) and 4 <= len(t.lstrip("+-")) <= 6
        or DIGIT_SUFFIX_RE.match(t) or CANCELLED_RE.match(t)
        or (SERVICE_RE.match(t) and not PURE_INT_RE.match(t))
    ) / n), 4)
    return sig


# ---------------------------------------------------------------------------
# Transparent evidence scoring
# ---------------------------------------------------------------------------

def score_semantics(sig: dict, name: str) -> dict[str, float]:
    """Weighted content evidence per class + weak header cue (content wins
    ties on purpose: a mislabeled column must still classify by content).

    Weight rationale: exact membership in the closed 28-name country
    gazetteer (2.5) outranks the permissive SKU service-code pattern (2.0),
    which short country names also match - the closed set is the stronger
    claim. ID-block membership (2.0) plus structural guest nulls identify
    CustomerID over the generic invoice/int patterns it also matches."""
    cues = header_cues(name)
    get = lambda k, d=0.0: float(sig.get(k, d))
    # SKU codes are (at most) short codes: sustained multi-token prose is
    # not a product code even when short phrases match the service-code
    # pattern ("BANK CHARGES" excepted - the <= 2-token band stays whole).
    avg_tokens = get("avg_tokens")
    sku_token_factor = 1.0 if avg_tokens <= 2.0 else max(0.2, 2.0 / avg_tokens)
    scores = {
        "invoice_id": 2.0 * get("invoice_pattern_rate") + 0.5 * get("cancelled_prefix_rate")
                      + 1.0 * cues["invoice_id"],
        "product_id": 2.0 * get("sku_family_rate") * sku_token_factor
                      + 0.5 * get("digit_suffix_rate")
                      + 1.0 * cues["product_id"],
        "product_description": (0.6 * min(get("vocab_size") / 1000.0, 1.0)
                                + 0.8 * min(get("avg_tokens") / 3.0, 1.0)
                                + 0.6 * get("alpha_ratio")
                                + 1.0 * cues["product_description"]),
        "quantity": (1.5 * get("int_rate")
                     + 0.5 * (1.0 if sig.get("has_negative") else 0.0)
                     + 0.5 * (1.0 - (1.0 if sig.get("has_decimals") else 0.0))
                     + 1.0 * cues["quantity"]),
        "transaction_datetime": 2.0 * get("datetime_rate") + 1.0 * cues["transaction_datetime"],
        "unit_price": (1.5 * get("price_pattern_rate")
                       + 0.8 * (1.0 if sig.get("has_decimals") else 0.0)
                       + 1.0 * cues["unit_price"]),
        "customer_id": (2.0 * get("id_block_rate") + 0.5 * get("null_rate")
                        + 0.5 * get("int_rate")
                        + 1.0 * cues["customer_id"]),
        "country": (2.5 * get("country_hit_rate")
                    + 0.5 * (1.0 - min(get("cardinality_ratio") * 20.0, 1.0))
                    + 1.0 * cues["country"]),
        "email": 2.0 * get("email_rate") + 1.0 * cues["email"],
        "phone": 2.0 * get("phone_rate") + 1.0 * cues["phone"],
        "url": 2.0 * get("url_rate") + 1.0 * cues["url"],
        "boolean_flag": 2.5 * get("bool_rate"),
        "category": 0.5 * (1.0 - min(get("cardinality_ratio") * 20.0, 1.0)),
        "text": 0.3 * get("alpha_ratio") + 0.2 * min(get("avg_tokens") / 3.0, 1.0),
    }
    return {k: round(float(v), 4) for k, v in scores.items()}


def _normalise_sprint1(semantic_type: str) -> str:
    """Map Sprint 1's semantic strings onto the classifier's label set."""
    t = str(semantic_type).lower()
    t = re.sub(r"\s*\(.*?\)\s*", "", t).strip().replace(" ", "_").replace("/", "_")
    mapping = {"product_id": "product_id", "customer_id": "customer_id",
               "unit_price": "unit_price", "transaction_datetime": "transaction_datetime",
               "product_description": "product_description", "invoice_id": "invoice_id",
               "quantity": "quantity", "country": "country"}
    return mapping.get(t, t)


def classify_columns(df: pd.DataFrame, sample_n: int = SAMPLE_N) -> dict:
    """Re-classify every column by content; flag label/content mismatches."""
    columns = {}
    for col in df.columns:
        series = df[col]
        sig = column_signals(series, sample_n=sample_n)
        scores = score_semantics(sig, col)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        predicted, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = round((top - second) / top, 3) if top > 0 else 0.0
        if top < 0.3:
            predicted, confidence = "uncertain", 0.0
        try:
            dtype = infer_logical_dtype(series)
            baseline_raw = infer_semantic(col, series, dtype)["semantic_type"]
        except Exception:
            dtype, baseline_raw = "unknown", "unknown"
        baseline = _normalise_sprint1(baseline_raw)
        expected = expected_from_header(col)
        if expected is None:
            # Unmapped header (e.g. engineered columns): no claim to check.
            status = "unmapped"
        else:
            status = "match" if predicted == expected else "mismatch"
        columns[col] = {
            "expected": expected,
            "predicted": predicted,
            "confidence": confidence,
            "status": status,
            "inferred_dtype": dtype,
            "sprint1_baseline": baseline,
            "sprint1_raw": str(baseline_raw),
            "agrees_with_sprint1": bool(baseline == predicted),
            "class_scores": scores,
            "signals": sig,
        }
    return columns


def build_classification_report(df: pd.DataFrame,
                                sample_n: int = SAMPLE_N,
                                label: str = "cleaned_data") -> dict:
    """Full report document over ``df`` (JSON-serialisable)."""
    columns = classify_columns(df, sample_n=sample_n)
    checked = {c: v for c, v in columns.items() if v["status"] != "unmapped"}
    mismatches = sorted([c for c, v in checked.items() if v["status"] == "mismatch"])
    return {
        "module": MODULE_NAME,
        "artifact": "column_classification",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {"input": str(DEFAULT_INPUT), "label": label,
                   "n_rows": int(len(df)), "sample_n_per_column": sample_n},
        "candidate_classes": list(CANDIDATE_CLASSES),
        "n_columns": len(columns),
        "n_checked": len(checked),
        "n_match": sum(1 for v in checked.values() if v["status"] == "match"),
        "n_mismatch": len(mismatches),
        "mismatched_columns": mismatches,
        "note": ("predicted = argmax over transparent content evidence "
                 "(TF-IDF vocabulary + token/shape probes); content outweighs "
                 "the header cue so mislabeled columns are caught. status="
                 "mismatch means the label's claim disagrees with the content."),
        "columns": columns,
    }


# ---------------------------------------------------------------------------
# File-level API + CLI
# ---------------------------------------------------------------------------

def run_classifier(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    sample_n: int = SAMPLE_N,
    demo_mislabel: bool = False,
) -> dict:
    """Classify columns of the cleaned data, save the report, return it.

    With ``demo_mislabel=True`` three headers are renamed to decoy labels
    first (Country->Contact_Email etc.) to prove the mismatch flag fires on
    real content; the report label records the demo.
    """
    df = load_cleaned(input_path)
    label = "cleaned_data"
    if demo_mislabel:
        df = df.rename(columns={k: v for k, v in DEMO_MISLABELS.items()
                                if k in df.columns})
        label = "demo_mislabel"
    report = build_classification_report(df, sample_n=sample_n, label=label)
    report["source"]["input"] = str(input_path)
    if demo_mislabel:
        report["decoy_renames"] = dict(DEMO_MISLABELS)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 8: NLP column classifier - re-classify column "
                    "semantics by content -> column_classification.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to cleaned_data.csv")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Where to save column_classification.json")
    p.add_argument("--sample-n", type=int, default=SAMPLE_N,
                   help="Cell strings sampled per column.")
    p.add_argument("--demo-mislabel", action="store_true",
                   help="Rename 3 headers to decoys to prove mismatch flags fire.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run_classifier(args.input, args.output, args.sample_n,
                            args.demo_mislabel)
    summary = {c: {"expected": v["expected"], "predicted": v["predicted"],
                   "confidence": v["confidence"], "status": v["status"]}
               for c, v in report["columns"].items()
               if c in EXPECTED_BY_COLUMN or args.demo_mislabel}
    print(json.dumps({
        "module": report["module"],
        "n_columns": report["n_columns"],
        "n_match": report["n_match"],
        "n_mismatch": report["n_mismatch"],
        "mismatched_columns": report["mismatched_columns"],
        "columns": summary,
    }, indent=2, ensure_ascii=False))
    print(f"\nColumn classification -> {args.output}")
    return report


if __name__ == "__main__":
    main()
