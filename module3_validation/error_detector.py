"""Sprint 9 - Error Detection Layer (Module 3: Validation).

Three finding families over ``cleaned_data.csv``:

  * ``impossible`` - hard-severity rule hits (negative prices, zero/out-of-
    bound quantities, ...). Any hit is a data error. Reuses Sprint 7's
    ``rule_validator`` masks, so impossible/suspicious never disagree with
    the rule engine.
  * ``suspicious`` - review/info-severity rule hits (bulk quantities,
    premium prices, return-without-cancellation, duplicate line keys,
    ...). Plausible but human-triage worthy.
  * ``drift`` - year-over-year comparison of the two full years in the
    dataset (2009 vs 2010) across measures, event shares and the country
    mix. 2011 is a 9-day stub (7,482 rows) and is excluded by design, which
    the report states explicitly.

Usage:
    from module3_validation.error_detector import detect_errors

    errors = detect_errors(df)  # {"impossible": [...], "suspicious": [...], "drift": {...}}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

try:
    from rule_validator import RULES, _ensure_derived, _load_reference_sets, validate_dataframe
except ImportError:  # allow running from the repo root
    from module3_validation.rule_validator import (
        RULES,
        _ensure_derived,
        _load_reference_sets,
        validate_dataframe,
    )

#: Rows stored per finding; full detail lives in rule_violations_sample.json.
SAMPLE_POSITIONS = 25

#: Drift bands on absolute relative change: watch >= 0.10, alert >= 0.25.
DRIFT_WATCH = 0.10
DRIFT_ALERT = 0.25

#: Years compared for drift; 2011 is a stub (see _year_of).
DRIFT_YEAR_A = 2009
DRIFT_YEAR_B = 2010
DRIFT_STUB_YEAR = 2011


# ---------------------------------------------------------------------------
# Rule-based findings (impossible + suspicious)
# ---------------------------------------------------------------------------

def detect_rule_findings(df: pd.DataFrame, ref: dict | None = None) -> dict:
    """Map Sprint 7 rule results onto impossible/suspicious findings.

    Hard hits -> ``impossible`` (severity ``critical``); review hits ->
    ``suspicious`` (severity ``high``); info hits -> ``suspicious``
    (severity ``medium``). Zero-hit rules are omitted - a finding exists
    only when rows actually violate.
    """
    ref = ref or _load_reference_sets()
    validated = validate_dataframe(df, ref)
    by_id = {r["rule_id"]: r for r in RULES}
    impossible, suspicious = [], []
    for rid, res in validated["results"].items():
        if not res["n_violations"]:
            continue
        meta = by_id[rid]
        finding = {
            "finding_id": f"{meta['category']}:{rid}",
            "rule_id": rid,
            "category": meta["category"],
            "description": meta["description"],
            "n_rows": res["n_violations"],
            "violation_pct": res["violation_pct"],
            "row_positions_sample": res["row_positions"][:SAMPLE_POSITIONS],
        }
        if meta["severity"] == "hard":
            finding["severity"] = "critical"
            impossible.append(finding)
        elif meta["severity"] == "review":
            finding["severity"] = "high"
            suspicious.append(finding)
        else:
            finding["severity"] = "medium"
            suspicious.append(finding)
    return {"impossible": impossible, "suspicious": suspicious,
            "n_rows": validated["n_rows"]}


# ---------------------------------------------------------------------------
# Drift (2009 vs 2010)
# ---------------------------------------------------------------------------

def _year_of(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["InvoiceDate"], errors="coerce").dt.year


def _rel(a: float, b: float) -> float | None:
    if a is None or b is None or (isinstance(a, float) and np.isnan(a)):
        return None
    if a == 0:
        return None if b == 0 else float("inf")
    return float((b - a) / abs(a))


def _band(rel_change: float | None) -> str:
    if rel_change is None or (isinstance(rel_change, float) and np.isnan(rel_change)):
        return "n/a"
    magnitude = abs(rel_change)
    if magnitude >= DRIFT_ALERT:
        return "alert"
    if magnitude >= DRIFT_WATCH:
        return "watch"
    return "none"


def detect_drift(df: pd.DataFrame,
                 year_a: int = DRIFT_YEAR_A,
                 year_b: int = DRIFT_YEAR_B) -> dict:
    """Compare year_a vs year_b on measures, event shares and country mix.

    Returns per-metric {a, b, rel_change, band} plus a verdict (worst band)
    and the stub-year exclusion note.
    """
    frame = _ensure_derived(df)
    years = _year_of(frame)
    counts = years.value_counts(dropna=True)
    fa = frame[years == year_a]
    fb = frame[years == year_b]
    n_a, n_b = int(len(fa)), int(len(fb))

    def share(mask: pd.Series) -> float:
        return float(mask.mean()) if len(mask) else 0.0

    metrics: list[dict] = []
    for col in ("Quantity", "UnitPrice", "LineValue"):
        sa = pd.to_numeric(fa[col], errors="coerce").dropna()
        sb = pd.to_numeric(fb[col], errors="coerce").dropna()
        for stat, fn in (("median", lambda s: float(s.median())),
                         ("mean", lambda s: float(s.mean())),
                         ("std", lambda s: float(s.std()))):
            a, b = (fn(sa) if len(sa) else None), (fn(sb) if len(sb) else None)
            rel = _rel(a, b) if a is not None else None
            metrics.append({"metric": f"{col}.{stat}", "a": a, "b": b,
                            "rel_change": rel, "band": _band(rel)})
    qa = pd.to_numeric(fa["Quantity"], errors="coerce")
    qb = pd.to_numeric(fb["Quantity"], errors="coerce")
    pa = pd.to_numeric(fa["UnitPrice"], errors="coerce")
    pb = pd.to_numeric(fb["UnitPrice"], errors="coerce")
    shares = {
        "share_return_quantity_lt_0": (share(qa < 0), share(qb < 0)),
        "share_giveaway_unitprice_eq_0": (share(pa == 0), share(pb == 0)),
        "share_guest_customerid_null": (share(fa["CustomerID"].isna()),
                                        share(fb["CustomerID"].isna())),
        "share_cancellation": (share(fa["InvoiceNo"].astype("string").str.startswith("C")),
                               share(fb["InvoiceNo"].astype("string").str.startswith("C"))),
        "share_uk": (share(fa["Country"].astype("string") == "United Kingdom"),
                     share(fb["Country"].astype("string") == "United Kingdom")),
    }
    for name, (a, b) in shares.items():
        rel = _rel(a, b)
        metrics.append({"metric": name, "a": round(a, 4), "b": round(b, 4),
                        "rel_change": None if rel is None else round(rel, 4),
                        "band": _band(rel)})
    metrics.append({"metric": "n_rows", "a": n_a, "b": n_b,
                    "rel_change": _rel(n_a, n_b), "band": _band(_rel(n_a, n_b))})

    order = {"alert": 2, "watch": 1, "none": 0, "n/a": -1}
    verdict = max((m["band"] for m in metrics), key=lambda b: order[b])
    finite = [abs(m["rel_change"]) for m in metrics
              if isinstance(m["rel_change"], (int, float)) and np.isfinite(m["rel_change"])]
    return {
        "year_a": year_a,
        "year_b": year_b,
        "n_a": n_a,
        "n_b": n_b,
        "excluded": {str(DRIFT_STUB_YEAR): int(counts.get(DRIFT_STUB_YEAR, 0)),
                     "reason": "stub year (first 9 days of January only) - "
                               "not comparable to full years"},
        "year_counts": {str(k): int(v) for k, v in counts.items()},
        "thresholds": {"watch_gte": DRIFT_WATCH, "alert_gte": DRIFT_ALERT},
        "metrics": metrics,
        "max_abs_rel_change": round(max(finite), 4) if finite else 0.0,
        "verdict": verdict,
    }


def detect_errors(df: pd.DataFrame, ref: dict | None = None) -> dict:
    """All three finding families in one document section."""
    ref = ref or _load_reference_sets()
    rule_findings = detect_rule_findings(df, ref)
    drift = detect_drift(df)
    return {
        "n_rows": rule_findings["n_rows"],
        "impossible": rule_findings["impossible"],
        "suspicious": rule_findings["suspicious"],
        "n_impossible": len(rule_findings["impossible"]),
        "n_suspicious": len(rule_findings["suspicious"]),
        "drift": drift,
    }
