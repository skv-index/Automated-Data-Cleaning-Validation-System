"""Sprint 8-9 - tests for the Module 3 validation layer (synthetic, fast)."""

import numpy as np
import pandas as pd
import pytest

from module3_validation import anomaly_detector, column_classifier
from module3_validation import error_detector, evaluation, validation_api


def _toy_transactions(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Mostly-normal retail lines with three injected extremes."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "InvoiceNo": ["489434"] * n,
        "StockCode": ["85048"] * n,
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"] * n,
        "Quantity": rng.integers(1, 13, size=n).astype(float),
        "InvoiceDate": ["2009-12-01 07:45:00"] * n,
        "UnitPrice": np.round(rng.uniform(1.0, 10.0, size=n), 2),
        "CustomerID": [13085.0] * n,
        "Country": ["United Kingdom"] * n,
    })
    df.loc[0, ["Quantity", "UnitPrice"]] = [2500.0, 0.25]    # bulk qty
    df.loc[1, ["Quantity", "UnitPrice"]] = [-1.0, 4999.99]   # premium price
    df.loc[2, ["Quantity", "UnitPrice"]] = [-9360.0, 0.03]   # huge return
    return df


def _toy_catalogue(n: int = 40) -> pd.DataFrame:
    """One realistic column of each semantic (content must win)."""
    return pd.DataFrame({
        "InvoiceNo": ["489434", "489435", "C489449", "537434"] * (n // 4),
        "StockCode": (["85048", "79323P", "POST", "22111"] * (n // 4)),
        "Description": (["WHITE HANGING HEART T-LIGHT HOLDER",
                         "PINK CHERRY LIGHTS",
                         "RECORD FRAME 7 INCH SINGLE SIZE",
                         "STRAWBERRY CERAMIC TRINKET BOX"] * (n // 4)),
        "Quantity": ([12, 6, -2, 3] * (n // 4)),
        "InvoiceDate": (["2009-12-01 07:45:00", "2010-06-03 12:00:00",
                         "2011-01-09 15:13:00", "2010-12-06 16:57:00"] * (n // 4)),
        "UnitPrice": ([6.95, 1.25, 3.75, 2.10] * (n // 4)),
        "CustomerID": ([13085.0, None, 17841.0, 12636.0] * (n // 4)),
        "Country": (["United Kingdom", "France", "Germany", "Ireland"] * (n // 4)),
    })


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def test_both_models_score_every_shown_row():
    doc = anomaly_detector.build_anomalies_sample(
        _toy_transactions(), contamination=0.02, top_n=3)
    assert doc["n_anomalies_shown"] > 0
    for row in doc["anomalies"]:
        assert isinstance(row["if_score"], float)
        assert isinstance(row["lof_score"], float)
        assert set(row["values"]) >= {"Quantity", "UnitPrice", "InvoiceNo"}


def test_injected_extremes_surface_in_sample():
    doc = anomaly_detector.build_anomalies_sample(
        _toy_transactions(), contamination=0.02, top_n=3)
    shown = {a["row_id"] for a in doc["anomalies"]}
    # The premium-price line (-1 x 4999.99) is the global extreme here.
    assert 1 in shown


def test_detector_features_reject_missing_columns():
    with_lof = _toy_transactions().drop(columns=["UnitPrice"])
    try:
        anomaly_detector.build_features(with_lof)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing UnitPrice")


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def test_classifier_matches_all_canonical_labels():
    report = column_classifier.build_classification_report(
        _toy_catalogue(), sample_n=50)
    assert report["n_mismatch"] == 0
    for col, expected in column_classifier.EXPECTED_BY_COLUMN.items():
        assert report["columns"][col]["predicted"] == expected


def test_classifier_flags_mislabelled_column_by_content():
    df = _toy_catalogue().rename(columns={"Country": "Contact_Email"})
    report = column_classifier.build_classification_report(df, sample_n=50)
    assert "Contact_Email" in report["mismatched_columns"]
    assert report["columns"]["Contact_Email"]["predicted"] == "country"


# ---------------------------------------------------------------------------
# Sprint 9: error detection + drift
# ---------------------------------------------------------------------------

def _two_year_frame(n: int = 120, seed: int = 1) -> pd.DataFrame:
    """Clean toy frame spanning both full years, unique line keys."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "InvoiceNo": [str(489434 + i) for i in range(n)],
        "StockCode": ["85048"] * n,
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"] * n,
        "Quantity": rng.integers(1, 13, size=n),
        "InvoiceDate": (["2009-12-01 07:45:00"] * (n // 2)
                        + ["2010-06-03 12:00:00"] * (n - n // 2)),
        "UnitPrice": np.round(rng.uniform(1.0, 10.0, size=n), 2),
        "CustomerID": [13085.0 if i % 3 else None for i in range(n)],
        "Country": ["United Kingdom" if i % 4 else "France"
                    for i in range(n)],
    })


def test_impossible_finding_cites_real_row():
    df = _two_year_frame()
    df.loc[7, "UnitPrice"] = -5.0  # physically impossible price
    errors = error_detector.detect_errors(df)
    assert errors["n_impossible"] >= 1
    hit = next(f for f in errors["impossible"]
               if f["rule_id"] == "unitprice_negative")
    assert hit["severity"] == "critical"
    assert 7 in hit["row_positions_sample"]


def test_drift_compares_full_years_and_notes_stub():
    errors = error_detector.detect_errors(_two_year_frame())
    drift = errors["drift"]
    assert (drift["year_a"], drift["year_b"]) == (2009, 2010)
    assert drift["n_a"] > 0 and drift["n_b"] > 0
    assert drift["verdict"] in {"none", "watch", "alert"}
    assert "2011" in drift["excluded"]  # stub-year exclusion is explicit


# ---------------------------------------------------------------------------
# Sprint 9: scoring + validation API
# ---------------------------------------------------------------------------

def test_health_bounds_and_monotonicity():
    clean = _two_year_frame()
    dirty = clean.copy()
    dirty.loc[3, "Quantity"] = -30000  # hard out-of-bounds
    dirty.loc[5, "UnitPrice"] = -3.0   # hard negative price
    rep_clean = validation_api.build_validation_report(
        clean, contamination=0.05, sample_n=60)
    rep_dirty = validation_api.build_validation_report(
        dirty, contamination=0.05, sample_n=60)
    for rep in (rep_clean, rep_dirty):
        assert 0 <= rep["scores"]["health"]["health_score"] <= 100
        assert rep["scores"]["error"]["error_severity"] >= 0
        assert rep["scores"]["anomaly"]["anomaly_severity"] >= 0
        assert rep["verdict"]["grade"] in {"Healthy", "Watch", "Review"}
    assert (rep_dirty["scores"]["health"]["health_score"]
            < rep_clean["scores"]["health"]["health_score"])
    assert (rep_dirty["scores"]["error"]["error_severity"]
            > rep_clean["scores"]["error"]["error_severity"])


def test_report_sections_and_empty_frame_rejected():
    rep = validation_api.build_validation_report(
        _two_year_frame(), contamination=0.05, sample_n=60)
    assert set(rep) >= {"rules", "errors", "drift", "anomalies",
                        "columns", "scores", "verdict"}
    assert rep["rules"]["defined"] == 20
    assert len(rep["verdict"]["recommended_actions"]) >= 1
    with pytest.raises(ValueError):
        validation_api.build_validation_report(
            _two_year_frame().iloc[0:0])


def test_run_validation_report_writes_json(tmp_path):
    out = tmp_path / "validation_report.json"
    rep = validation_api.run_validation_report(
        validation_api.DEFAULT_INPUT, out,
        contamination=0.05, sample_n=200)
    assert out.exists()
    assert rep["scores"]["health"]["grade"] == rep["verdict"]["grade"]


# ---------------------------------------------------------------------------
# Sprint 9: evaluation on injected ground truth
# ---------------------------------------------------------------------------

def test_evaluation_metrics_are_sane_and_deterministic():
    base = _toy_transactions(400)
    first = evaluation.evaluate(base, n_per_type=5, seed=7,
                                contamination=0.05)
    second = evaluation.evaluate(base, n_per_type=5, seed=7,
                                 contamination=0.05)
    assert first["injection"]["n_injected"] == 20
    names = [m["model"] for m in first["models"]]
    assert names == ["isolation_forest", "lof", "either_model_flags",
                     "both_models_agree"]
    for m in first["models"]:
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert 0.0 <= m["precision"] <= 1.0
        assert 0.0 <= m["recall"] <= 1.0
        cm = m["confusion_matrix"]["tn_fp_fn_tp"]
        assert sum(sum(row) for row in cm) == 420
    assert set(first["recall_by_type"]) == set(names)
    assert (first["models"][0]["roc_auc"]
            == second["models"][0]["roc_auc"])  # seeded, deterministic


def test_evaluation_report_files(tmp_path):
    base = _toy_transactions(200)
    results = evaluation.evaluate(base, n_per_type=3, seed=7,
                                  contamination=0.05)
    md = evaluation.write_evaluation_report(
        results, tmp_path / "evaluation_report.md",
        tmp_path / "roc_curve.png")
    assert md.exists() and (tmp_path / "roc_curve.png").exists()
    text = md.read_text(encoding="utf-8")
    assert "ROC AUC" in text and "Confusion" in text
