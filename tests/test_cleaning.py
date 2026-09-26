"""Sprint 6 - tests for Module 2 cleaning (unit + integration).

Unit tests use small synthetic frames (fast, deterministic); the two
integration tests run the real pipeline on a 400-row workbook slice and on
the produced ``cleaned_data.csv``.
"""

from pathlib import Path

import pandas as pd
import pytest

from module2_cleaning import cleaner, cleaning_api, transformer

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "module2_cleaning" / "expected_schema.json"
XLSX_PATH = ROOT / "online_retail_II.xlsx"


@pytest.fixture(scope="module")
def schema():
    return cleaner.load_schema(SCHEMA_PATH)


def _perfect_frame() -> pd.DataFrame:
    """Fully canonical frame: quality must score exactly 100."""
    return pd.DataFrame({
        "InvoiceNo": ["489434", "489435", "C489449"],
        "StockCode": ["85048", "79323P", "22111"],
        "Description": ["WIDGET", "CHERRY LIGHTS", "GADGET"],
        "Quantity": [12, 6, -2],
        "InvoiceDate": ["2009-12-01 07:45:00", "2010-06-03 12:00:00",
                        "2011-01-09 15:13:00"],
        "UnitPrice": [6.95, 1.25, 3.75],
        "CustomerID": [13085.0, None, 17841.0],
        "Country": ["United Kingdom", "France", "Germany"],
    })


def _dirty_frame() -> pd.DataFrame:
    """One of every defect: exact dup, padding/case/alias, gaps."""
    return pd.DataFrame({
        "InvoiceNo": ["489434", "489434", "489435", "489436"],
        "StockCode": ["85048", "85048", "79323p", "85048"],
        "Description": ["WIDGET", "WIDGET", "  cherry lights ", None],
        "Quantity": [12, 12, 6, 1],
        "InvoiceDate": ["2009-12-01 07:45:00"] * 4,
        "UnitPrice": [6.95, 6.95, 1.25, 2.5],
        "CustomerID": [13085.0, 13085.0, None, 17841.0],
        "Country": ["United Kingdom", "United Kingdom", "EIRE", "USA"],
    })


# ---------------------------------------------------------------------------
# Unit: normalization
# ---------------------------------------------------------------------------

def test_normalize_text_fixes_casing_whitespace_aliases(schema):
    df = pd.DataFrame({
        "InvoiceNo": ["489434"],
        "StockCode": ["72349b"],
        "Description": ["  white  cherry   lights "],
        "Quantity": [2],
        "InvoiceDate": ["2009-12-01 07:45:00"],
        "UnitPrice": [4.30],
        "CustomerID": [None],
        "Country": ["EIRE"],
    })
    out, log = cleaner.normalize_text(df, schema)
    assert out.loc[0, "StockCode"] == "72349B"
    assert out.loc[0, "Description"] == "WHITE CHERRY LIGHTS"
    assert out.loc[0, "Country"] == "Ireland"
    assert str(out.loc[0, "InvoiceDate"]) == "2009-12-01 07:45:00"
    assert log["columns"]["Country"]["aliases_applied"] == ["Eire->Ireland"]
    # Nulls are never filled by normalization.
    assert pd.isna(out.loc[0, "CustomerID"])


# ---------------------------------------------------------------------------
# Unit: imputation (statistical + ML)
# ---------------------------------------------------------------------------

def test_impute_statistical_lookup_then_global_mode(schema):
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2", "3"],
        "StockCode": ["A1", "A1", "ZZZ"],
        "Description": ["WIDGET", None, None],
        "Quantity": [1, 2, 3],
        "InvoiceDate": ["2010-01-01 10:00:00"] * 3,
        "UnitPrice": [1.0, 2.0, 3.0],
        "CustomerID": [111.0, None, 222.0],
        "Country": ["France"] * 3,
    })
    norm, _ = cleaner.normalize_text(df, schema)
    out, log = cleaner.impute_missing(norm, schema, strategy="statistical")
    assert out.loc[1, "Description"] == "WIDGET"  # StockCode lookup
    assert out.loc[2, "Description"] == "WIDGET"  # global-mode fallback
    assert log["columns"]["Description"]["n_imputed_via_stockcode_lookup"] == 1
    assert log["columns"]["Description"]["n_imputed_via_global_mode"] == 1
    # Structural guest null is preserved, never fabricated.
    assert pd.isna(out.loc[1, "CustomerID"])


def test_impute_knn_fills_numeric_never_customerid(schema):
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2", "3", "4", "5", "6"],
        "StockCode": ["A"] * 6,
        "Description": ["W"] * 6,
        "Quantity": [1.0, 2.0, None, 4.0, 5.0, 6.0],
        "InvoiceDate": ["2010-01-01 10:00:00"] * 6,
        "UnitPrice": [1.0, 2.0, 3.0, None, 5.0, 6.0],
        "CustomerID": [111.0, None, 333.0, 444.0, None, 666.0],
        "Country": ["France"] * 6,
    })
    out, log = cleaner.impute_missing(df, schema, strategy="knn",
                                      knn_neighbors=2)
    assert out[["Quantity", "UnitPrice"]].notna().all().all()
    assert out["CustomerID"].isna().sum() == 2  # untouched
    assert "CustomerID" in log["excluded_columns"]


def test_verify_knn_imputer_reports_reconstruction(schema):
    df = pd.DataFrame({
        "Quantity": [1.0, 2.0, 3.0, 4.0, 50.0, 6.0, 7.0, 8.0] * 10,
        "UnitPrice": [1.5, 2.5, 1.0, 4.0, 9.0, 0.5, 2.0, 3.0] * 10,
    })
    report = cleaner.verify_knn_imputer(df, mask_frac=0.1, random_state=7)
    assert report["n_masked"] > 0
    assert report["mae_masked_quantity"] >= 0
    assert 0.0 <= report["exact_match_rate"] <= 1.0
    assert report["features"] == ["Quantity", "UnitPrice"]


# ---------------------------------------------------------------------------
# Unit: duplicates (exact + fuzzy)
# ---------------------------------------------------------------------------

def test_exact_duplicates_detected_and_removed(schema):
    df = _dirty_frame()
    norm, _ = cleaner.normalize_text(df, schema)
    mask, log = cleaner.detect_exact_duplicates(norm)
    assert int(mask.sum()) == 1  # rows 0+1 are identical once normalized
    assert log["policy"] == "removed (keep first)"
    cleaned, _ = cleaner.clean_dataframe(df, schema)
    assert len(cleaned) == len(df) - 1


def test_near_duplicates_flag_typo_pair_not_merged(schema):
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2", "3"],
        "StockCode": ["16012"] * 3,
        "Description": ["FOOD/DRINK SPONGE STICKERS",
                        "FOOD/DRINK SPUNGE STICKERS",
                        "FOOD/DRINK SPONGE STICKERS"],
        "Quantity": [1, 2, 1],
        "InvoiceDate": ["2010-01-01 10:00:00"] * 3,
        "UnitPrice": [0.21] * 3,
        "CustomerID": [1.0, 2.0, 3.0],
        "Country": ["France"] * 3,
    })
    norm, _ = cleaner.normalize_text(df, schema)
    groups = cleaner.detect_near_duplicates(norm)
    assert len(groups) == 1
    assert groups[0]["max_similarity"] >= 0.90
    assert set(groups[0]["descriptions"]) == {
        "FOOD/DRINK SPONGE STICKERS", "FOOD/DRINK SPUNGE STICKERS"}
    assert groups[0]["recommendation"] == "review - never auto-merge"
    # Detection never mutates the frame.
    assert len(norm) == 3


def test_near_duplicates_no_false_positive(schema):
    df = pd.DataFrame({
        "InvoiceNo": ["1", "2"],
        "StockCode": ["20615"] * 2,
        "Description": ["BLUE POLKADOT PASSPORT COVER", "PINK CHERRY LIGHTS"],
        "Quantity": [1, 2],
        "InvoiceDate": ["2010-01-01 10:00:00"] * 2,
        "UnitPrice": [1.0, 2.0],
        "CustomerID": [1.0, 2.0],
        "Country": ["France"] * 2,
    })
    norm, _ = cleaner.normalize_text(df, schema)
    assert cleaner.detect_near_duplicates(norm) == []


# ---------------------------------------------------------------------------
# Unit: transformer
# ---------------------------------------------------------------------------

def test_extract_features_math_and_flags():
    df = pd.DataFrame({
        "InvoiceNo": ["489434", "C489449"],
        "StockCode": ["85048", "22111"],
        "Description": ["W", "G"],
        "Quantity": [12, -2],
        "InvoiceDate": pd.to_datetime(["2009-12-01 07:45:00",
                                       "2011-03-15 14:05:00"]),
        "UnitPrice": [6.95, 0.0],
        "CustomerID": [13085.0, None],
        "Country": ["United Kingdom", "France"],
    })
    out, log = transformer.extract_features(df)
    assert out.loc[0, "LineValue"] == pytest.approx(12 * 6.95)
    assert out.loc[1, "LineValue"] == pytest.approx(0.0)
    assert out.loc[0, "IsCancellation"] == False
    assert out.loc[1, "IsCancellation"] == True
    assert out.loc[1, "IsReturn"] == True
    assert out.loc[1, "IsGiveaway"] == True
    assert out.loc[0, "InvoiceMonth"] == 12
    assert out.loc[1, "InvoiceYear"] == 2011
    assert out.loc[0, "InvoiceWeekday"] == 1  # 2009-12-01 was a Tuesday
    assert log["n_cancellations"] == 1


def test_scale_features_zero_mean_unit_variance():
    df = pd.DataFrame({
        "Quantity": [1.0, 2.0, 3.0, 4.0, 5.0],
        "UnitPrice": [2.0, 4.0, 1.0, 8.0, 5.0],
        "LineValue": [2.0, 8.0, 3.0, 32.0, 25.0],
    })
    out, params = transformer.scale_features(df)
    for col in ("Quantity", "UnitPrice", "LineValue"):
        assert out[f"{col}_scaled"].mean() == pytest.approx(0.0, abs=1e-9)
        assert out[f"{col}_scaled"].var(ddof=0) == pytest.approx(1.0)
        assert col in out.columns  # raw values preserved
    assert set(params["mean_"]) == {"Quantity", "UnitPrice", "LineValue"}


def test_encode_features_onehot_country():
    df = pd.DataFrame({
        "Quantity": [1.0, 2.0],
        "UnitPrice": [1.0, 2.0],
        "LineValue": [1.0, 4.0],
        "IsCancellation": [False, True],
        "IsReturn": [False, False],
        "IsGiveaway": [False, False],
        "Country": ["France", "United Kingdom"],
    })
    encoded, params = transformer.encode_features(df)
    country_cols = [c for c in encoded.columns if c.startswith("Country_")]
    assert params["country_values"] == ["France", "United Kingdom"]
    assert encoded[country_cols].sum(axis=1).tolist() == [1, 1]
    assert encoded.loc[1, "Country_United Kingdom"] == 1


# ---------------------------------------------------------------------------
# Unit: quality scoring
# ---------------------------------------------------------------------------

def test_score_quality_perfect_frame_is_100(schema):
    score = cleaning_api.score_quality(_perfect_frame(), schema)
    assert score["dimensions"] == {"completeness": 100.0,
                                   "consistency": 100.0,
                                   "uniqueness": 100.0,
                                   "validity": 100.0}
    assert score["total"] == 100.0


def test_score_quality_detects_dirt_then_confirms_fix(schema):
    dirty = _dirty_frame()
    before = cleaning_api.score_quality(dirty, schema)
    assert before["total"] < 100.0
    assert before["dimensions"]["uniqueness"] < 100.0
    assert before["dimensions"]["consistency"] < 100.0
    assert before["dimensions"]["completeness"] < 100.0
    cleaned, _ = cleaner.clean_dataframe(dirty, schema)
    after = cleaning_api.score_quality(cleaned, schema)
    assert after["total"] == 100.0
    assert after["total"] - before["total"] > 0


# ---------------------------------------------------------------------------
# Integration: real workbook slice -> official outputs
# ---------------------------------------------------------------------------

def test_pipeline_small_slice_produces_official_outputs(tmp_path, schema):
    cleaned_path = tmp_path / "cleaned_data.csv"
    log_path = tmp_path / "cleaning_log.json"
    cleaned, log = cleaning_api.run_cleaning_pipeline(
        input_path=XLSX_PATH, schema_path=SCHEMA_PATH,
        cleaned_path=cleaned_path, log_path=log_path,
        nrows_per_sheet=200)
    assert cleaned_path.exists() and log_path.exists()
    required = [c for c in schema["column_order"] if c != "CustomerID"]
    assert cleaned[required].notna().all().all()  # no nulls remain
    assert int(cleaned[required].duplicated().sum()) == 0  # no exact dups
    assert log["quality_after"]["total"] == 100.0
    assert log["quality_delta"] >= 0
    assert log["fixes"]["exact_duplicates_removed"] >= 0
    assert "LineValue" in cleaned.columns  # features present


def test_cleaned_csv_conforms_to_expected_schema(tmp_path, schema):
    cleaned_path = tmp_path / "cleaned_data.csv"
    log_path = tmp_path / "cleaning_log.json"
    cleaning_api.run_cleaning_pipeline(
        input_path=XLSX_PATH, schema_path=SCHEMA_PATH,
        cleaned_path=cleaned_path, log_path=log_path,
        nrows_per_sheet=200)
    out = pd.read_csv(cleaned_path, dtype=str)
    inv_pat = schema["columns"]["InvoiceNo"]["expected_format"]["pattern"]
    sc_pat = schema["columns"]["StockCode"]["expected_format"]["pattern"]
    assert out["InvoiceNo"].str.match(inv_pat, na=False).all()
    assert out["StockCode"].str.match(sc_pat, na=False).all()
    assert (out["Description"].str.strip() == out["Description"]).all()
