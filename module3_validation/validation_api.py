"""Sprint 9 - Validation API (Module 3: Validation).

Official Module 3 entry point: ``cleaned_data.csv`` in,
``validation_report.json`` out. Assembles every layer built so far:

  * Sprint 7 rules (``rule_validator``) -> per-rule counts + severity groups
  * Sprint 9 errors (``error_detector``) -> impossible / suspicious findings
    + 2009-vs-2010 drift verdict
  * Sprint 8 anomalies (``anomaly_detector``) -> IF/LOF flag counts,
    agreement, top rows
  * Sprint 8 columns (``column_classifier``) -> match / mismatch summary

Scoring (all deterministic, 0-100):

  * ``error_severity`` (0 = clean, higher = worse): 100 * (H + 0.3*R +
    0.1*I) / n, where H/R/I are DISTINCT rows hitting any hard / review /
    info rule. Hard violations count full weight; review and info are
    discounted because they flag plausible-but-checkable rows, not errors.
  * ``anomaly_severity`` (0 = clean): 100 * both_flag_rate - rows flagged
    by BOTH models (high-confidence anomalies). IF/LOF single-model rates
    are reported alongside for context.
  * ``health_score`` (100 = perfect): weighted composite mirroring Module
    2's ``score_quality`` precedent -
    validity 0.40 (1 - hard row rate) + anomaly 0.25 (1 - both-flag rate)
    + consistency 0.20 (1 - review row rate) + stability 0.15
    (1 - min(1, max_drift_rel / 0.5)). Grade: >= 95 Healthy, >= 80 Watch,
    else Review.

Usage:
    from module3_validation.validation_api import run_validation_report

    report = run_validation_report()  # cleaned_data.csv -> validation_report.json

CLI:
    python module3_validation/validation_api.py
    python module3_validation/validation_api.py --input ... --report ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

try:
    from anomaly_detector import DEFAULT_CONTAMINATION as _DEFAULT_CONTAM
    from anomaly_detector import build_anomalies_sample
    from column_classifier import SAMPLE_N as _DEFAULT_SAMPLE_N
    from column_classifier import build_classification_report
    from error_detector import detect_errors
    from rule_validator import RULES, _load_reference_sets, validate_dataframe
except ImportError:  # allow running from the repo root
    from module3_validation.anomaly_detector import (
        DEFAULT_CONTAMINATION as _DEFAULT_CONTAM,
    )
    from module3_validation.anomaly_detector import build_anomalies_sample
    from module3_validation.column_classifier import SAMPLE_N as _DEFAULT_SAMPLE_N
    from module3_validation.column_classifier import build_classification_report
    from module3_validation.error_detector import detect_errors
    from module3_validation.rule_validator import (
        RULES,
        _load_reference_sets,
        validate_dataframe,
    )

MODULE_NAME = "module3_validation"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "module2_cleaning" / "cleaned_data.csv"
DEFAULT_SCHEMA = Path(__file__).resolve().parent.parent / "module2_cleaning" / "expected_schema.json"
DEFAULT_REPORT = Path(__file__).resolve().parent / "validation_report.json"

#: Severity-group weights for error_severity (hard counts fully).
SEVERITY_WEIGHTS = {"hard": 1.0, "review": 0.3, "info": 0.1}

#: Health-score component weights (sum to 1.0).
HEALTH_WEIGHTS = {"validity": 0.40, "anomaly": 0.25,
                  "consistency": 0.20, "stability": 0.15}

#: Drift magnitude that fully zeroes the stability component.
DRIFT_FULL_PENALTY = 0.50

#: Rows kept per top-anomaly list inside the report (full detail lives in
#: anomalies_sample.json).
TOP_ANOMALIES_KEPT = 5


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _distinct_rows_by_severity(validated: dict) -> dict[str, set[int]]:
    """Distinct row positions per rule-severity group."""
    sev_of = {r["rule_id"]: r["severity"] for r in RULES}
    groups: dict[str, set[int]] = {"hard": set(), "review": set(), "info": set()}
    for rid, res in validated["results"].items():
        groups[sev_of[rid]].update(res["row_positions"])
    return groups


def score_error_severity(n_rows: int, groups: dict[str, set[int]]) -> dict:
    """0-100 penalty scale (0 = clean). Weighted distinct-row rate."""
    if n_rows == 0:
        raise ValueError("cannot score an empty frame")
    weighted = sum(len(groups[s]) * SEVERITY_WEIGHTS[s] for s in groups)
    score = round(min(100.0, 100.0 * weighted / n_rows), 2)
    return {
        "error_severity": score,
        "scale": "0-100 penalty (0 = clean)",
        "weights": dict(SEVERITY_WEIGHTS),
        "hard_rows": len(groups["hard"]),
        "review_rows": len(groups["review"]),
        "info_rows": len(groups["info"]),
    }


def score_anomaly_severity(n_rows: int, n_if: int, n_lof: int, n_both: int) -> dict:
    """0-100 penalty scale (0 = clean). High-confidence (both-model) rate."""
    if n_rows == 0:
        raise ValueError("cannot score an empty frame")
    return {
        "anomaly_severity": round(100.0 * n_both / n_rows, 4),
        "scale": "0-100 penalty (0 = clean)",
        "if_flag_rate": round(n_if / n_rows, 4),
        "lof_flag_rate": round(n_lof / n_rows, 4),
        "both_flag_rate": round(n_both / n_rows, 4),
        "n_if_flagged": n_if,
        "n_lof_flagged": n_lof,
        "n_both_flagged": n_both,
    }


def score_health(n_rows: int, groups: dict[str, set[int]],
                 n_both: int, max_drift_rel: float) -> dict:
    """0-100 quality scale (100 = perfect). Weighted component composite."""
    if n_rows == 0:
        raise ValueError("cannot score an empty frame")
    components = {
        "validity": 100.0 * (1 - len(groups["hard"]) / n_rows),
        "anomaly": 100.0 * (1 - n_both / n_rows),
        "consistency": 100.0 * (1 - len(groups["review"]) / n_rows),
        "stability": 100.0 * (1 - min(1.0, max_drift_rel / DRIFT_FULL_PENALTY)),
    }
    total = round(sum(components[k] * HEALTH_WEIGHTS[k] for k in components), 2)
    grade = "Healthy" if total >= 95 else ("Watch" if total >= 80 else "Review")
    return {
        "health_score": total,
        "scale": "0-100 quality (100 = perfect)",
        "weights": dict(HEALTH_WEIGHTS),
        "components": {k: round(v, 2) for k, v in components.items()},
        "grade": grade,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _recommended_actions(errors: dict, anomaly: dict, columns: dict,
                         health: dict) -> list[str]:
    actions = []
    if errors["n_impossible"]:
        actions.append(
            f"Resolve {errors['n_impossible']} impossible-value finding(s) "
            f"at source - hard violations are data errors, not triage.")
    review_hits = sum(f["n_rows"] for f in errors["suspicious"])
    if review_hits:
        actions.append(
            f"Triage {review_hits} suspicious-row events "
            f"({errors['n_suspicious']} patterns) before trusting revenue cuts.")
    if errors["drift"]["verdict"] == "alert":
        actions.append(
            "Drift ALERT 2009->2010 (UnitPrice/LineValue spread exploded on "
            "AMAZONFEE lines): segment 2010 fee/adjustment codes out of "
            "like-for-like revenue before modelling.")
    elif errors["drift"]["verdict"] == "watch":
        actions.append(
            "Drift WATCH 2009->2010: re-check guest/return/cancellation "
            "shares when refreshing the data.")
    if anomaly["n_both_flagged"]:
        actions.append(
            f"Review {anomaly['n_both_flagged']} high-confidence anomalies "
            f"(both models agree) - see anomalies_sample.json.")
    if columns["n_mismatch"]:
        actions.append(
            f"Investigate {columns['n_mismatch']} column mismatch(es): "
            f"{', '.join(columns['mismatched_columns'])}.")
    if health["grade"] == "Healthy" and not actions:
        actions.append("No action required - dataset is healthy; re-validate "
                       "on refresh.")
    return actions


def build_validation_report(df, contamination: float = _DEFAULT_CONTAM,
                            sample_n: int = _DEFAULT_SAMPLE_N) -> dict:
    """Run every Module 3 layer over ``df`` and assemble the report (no I/O)."""
    n = int(len(df))
    if n == 0:
        raise ValueError("cannot validate an empty frame")

    ref = _load_reference_sets()
    validated = validate_dataframe(df, ref)
    groups = _distinct_rows_by_severity(validated)
    by_sev = {r["rule_id"]: r["severity"] for r in RULES}
    by_cat = {r["rule_id"]: r["category"] for r in RULES}

    errors = detect_errors(df, ref)

    anomalies_doc = build_anomalies_sample(df, contamination=contamination,
                                           top_n=TOP_ANOMALIES_KEPT)
    top_rows = [{"row_id": a["row_id"], "csv_line": a["csv_line"],
                 "top_by": a["top_by"], "if_score": a["if_score"],
                 "lof_score": a["lof_score"], "values": a["values"]}
                for a in anomalies_doc["anomalies"][:TOP_ANOMALIES_KEPT]]

    columns = build_classification_report(df, sample_n=sample_n)

    error = score_error_severity(n, groups)
    anomaly = score_anomaly_severity(
        n, anomalies_doc["models"]["isolation_forest"]["n_flagged"],
        anomalies_doc["models"]["lof"]["n_flagged"],
        anomalies_doc["models"]["n_both_flagged"])
    health = score_health(n, groups, anomaly["n_both_flagged"],
                          errors["drift"]["max_abs_rel_change"])

    rules_section = {
        "defined": len(RULES),
        "failed": sum(1 for r in validated["results"].values()
                      if r["n_violations"]),
        "per_rule": {rid: {"severity": by_sev[rid], "category": by_cat[rid],
                           "n_violations": res["n_violations"],
                           "violation_pct": res["violation_pct"]}
                     for rid, res in validated["results"].items()},
        "detail_file": "module3_validation/rule_violations_sample.json",
    }
    rules_section["passed"] = rules_section["defined"] - rules_section["failed"]

    return {
        "module": MODULE_NAME,
        "artifact": "validation_report",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "source": {"input": str(DEFAULT_INPUT), "n_rows": n,
                   "contamination": contamination,
                   "classifier_sample_n": sample_n},
        "rules": rules_section,
        "errors": {
            "n_impossible": errors["n_impossible"],
            "n_suspicious": errors["n_suspicious"],
            "impossible": errors["impossible"],
            "suspicious": errors["suspicious"],
        },
        "drift": errors["drift"],
        "anomalies": {
            "models": anomalies_doc["models"],
            "scaler": anomalies_doc["scaler"],
            "top_rows": top_rows,
            "detail_file": "module3_validation/anomalies_sample.json",
        },
        "columns": {
            "n_checked": columns["n_checked"],
            "n_match": columns["n_match"],
            "n_mismatch": columns["n_mismatch"],
            "mismatched_columns": columns["mismatched_columns"],
            "detail_file": "module3_validation/column_classification.json",
        },
        "scores": {"error": error, "anomaly": anomaly, "health": health},
        "verdict": {
            "grade": health["grade"],
            "health_score": health["health_score"],
            "recommended_actions": _recommended_actions(
                errors, anomaly, columns, health),
        },
    }


def load_cleaned(path: str | Path = DEFAULT_INPUT):
    """Load the official Module 2 output (local import avoids cycles)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_csv(path)


def run_validation_report(
    input_path: str | Path = DEFAULT_INPUT,
    report_path: str | Path = DEFAULT_REPORT,
    contamination: float = _DEFAULT_CONTAM,
    sample_n: int = _DEFAULT_SAMPLE_N,
) -> dict:
    """Full pipeline: cleaned CSV in, validation_report.json out."""
    df = load_cleaned(input_path)
    report = build_validation_report(df, contamination=contamination,
                                     sample_n=sample_n)
    report["source"]["input"] = str(input_path)
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 9: Module 3 validation API - cleaned_data.csv "
                    "-> validation_report.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to cleaned_data.csv")
    p.add_argument("--report", default=str(DEFAULT_REPORT),
                   help="Where to save validation_report.json")
    p.add_argument("--contamination", type=float, default=_DEFAULT_CONTAM,
                   help="Anomaly flag rate per model.")
    p.add_argument("--sample-n", type=int, default=_DEFAULT_SAMPLE_N,
                   help="Cell strings sampled per column (classifier).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = run_validation_report(args.input, args.report,
                                   args.contamination, args.sample_n)
    print(json.dumps({
        "module": report["module"],
        "n_rows": report["source"]["n_rows"],
        "rules_failed": report["rules"]["failed"],
        "n_impossible": report["errors"]["n_impossible"],
        "n_suspicious": report["errors"]["n_suspicious"],
        "drift_verdict": report["drift"]["verdict"],
        "n_both_flagged": report["scores"]["anomaly"]["n_both_flagged"],
        "error_severity": report["scores"]["error"]["error_severity"],
        "anomaly_severity": report["scores"]["anomaly"]["anomaly_severity"],
        "health_score": report["scores"]["health"]["health_score"],
        "grade": report["verdict"]["grade"],
    }, indent=2, ensure_ascii=False))
    print(f"\nValidation report -> {args.report}")
    return report


if __name__ == "__main__":
    main()
