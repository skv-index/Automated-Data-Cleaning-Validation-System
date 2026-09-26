"""Sprint 6 - Cleaning API (Module 2: Cleaning).

The official Module 2 entry point: raw dataset in, two official outputs
out - ``cleaned_data.csv`` and ``cleaning_log.json``.

Pipeline: score (before) -> clean (``cleaner``) -> transform
(``transformer``) -> score (after) -> persist. The log records every fix
made plus the before/after quality scores and their delta.

Quality score (0-100, higher is better), four weighted dimensions:

  * ``completeness`` (0.25) - % rows with no missing values in required
    columns (every canonical column except CustomerID, whose nulls are
    structural guest orders, schema rv-4).
  * ``consistency`` (0.35) - % rows already canonical (no trim / case /
    alias / date-parse changes pending). Heaviest weight: this dataset's
    dirt lives here.
  * ``uniqueness`` (0.20) - % non-duplicate rows (exact full-row dups).
  * ``validity`` (0.20) - % rows passing every *hard* schema rule
    (identifier patterns, Quantity/UnitPrice bounds, date window,
    CustomerID block, Country reference set).

Usage:
    from module2_cleaning.cleaning_api import run_cleaning_pipeline, score_quality

    cleaned, log = run_cleaning_pipeline()  # defaults: xlsx in, csv+log out

CLI:
    python cleaning_api.py
    python cleaning_api.py --input ..\\online_retail_II.xlsx --nrows-per-sheet 50000
    python cleaning_api.py --cleaned cleaned_data.csv --log cleaning_log.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

from module2_cleaning.cleaner import (  # noqa: E402
    detect_exact_duplicates,
    load_dataset,
    load_schema,
    normalize_text,
)
from module2_cleaning.cleaner import clean_dataframe
from module2_cleaning.transformer import extract_features, scale_features

MODULE_NAME = "module2_cleaning"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "online_retail_II.xlsx"
DEFAULT_EXPECTED_SCHEMA = Path(__file__).resolve().parent / "expected_schema.json"
DEFAULT_CLEANED = Path(__file__).resolve().parent / "cleaned_data.csv"
DEFAULT_LOG = Path(__file__).resolve().parent / "cleaning_log.json"
DEFAULT_NROWS_PER_SHEET = 50000
DEFAULT_KNN_NEIGHBORS = 5
DEFAULT_FUZZY_THRESHOLD = 0.90

#: Dimension weights - consistency dominates: casing/whitespace/aliases
#: are this dataset's actual dirt; hard validity was never violated.
QUALITY_WEIGHTS = {
    "completeness": 0.25,
    "consistency": 0.35,
    "uniqueness": 0.20,
    "validity": 0.20,
}


def _canonical_columns(df: pd.DataFrame, schema: dict) -> list[str]:
    order = schema.get("column_order", list(df.columns))
    return [c for c in order if c in df.columns]


def _validity_mask(df: pd.DataFrame, schema: dict) -> pd.Series:
    """Per-row pass/fail against every hard schema rule (vectorized)."""
    cols = schema.get("columns", {})
    ok = pd.Series(True, index=df.index)

    inv_pat = cols["InvoiceNo"]["expected_format"]["pattern"]
    ok &= df["InvoiceNo"].astype("string").str.match(inv_pat, na=False)
    sc_pat = cols["StockCode"]["expected_format"]["pattern"]
    ok &= df["StockCode"].astype("string").str.match(sc_pat, na=False)

    qrange = cols["Quantity"]["value_range"]
    q = pd.to_numeric(df["Quantity"], errors="coerce")
    ok &= (q.notna() & (q % 1 == 0) & (q != 0)
           & (q >= qrange["min"]) & (q <= qrange["max"]))

    prange = cols["UnitPrice"]["value_range"]
    p = pd.to_numeric(df["UnitPrice"], errors="coerce")
    ok &= (p.notna() & (p >= prange["min"]) & (p <= prange["max"])
           & (p.round(2) == p))

    wrange = cols["InvoiceDate"]["value_range"]
    dt = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    ok &= (dt.notna() & (dt >= pd.Timestamp(wrange["min"]))
           & (dt <= pd.Timestamp(wrange["max"])))

    crange = cols["CustomerID"]["value_range"]
    c = pd.to_numeric(df["CustomerID"], errors="coerce")
    ok &= (df["CustomerID"].isna()
           | ((c % 1 == 0) & (c >= crange["min"]) & (c <= crange["max"])))

    allowed = cols["Country"]["allowed_values"]
    alias_map = {str(k).strip().title(): str(v)
                 for k, v in allowed["aliases"].items()}
    co = (df["Country"].astype("string").str.strip().str.title()
          .replace(alias_map))
    ok &= co.isin(list(allowed["values"]))

    d = df["Description"]
    ok &= d.isna() | (d.astype("string").str.strip() != "")
    return ok


def score_quality(df: pd.DataFrame, schema: dict) -> dict:
    """Score dataset quality 0-100 across the four weighted dimensions.

    Works on raw frames and on cleaned/transformed frames alike (always
    scored over the 8 canonical columns, so before/after are comparable).
    """
    canon = _canonical_columns(df, schema)
    frame = df[canon]
    n = int(len(frame))
    if n == 0:
        raise ValueError("cannot score an empty frame")

    required = [c for c in canon if c != "CustomerID"]
    complete_rows = int(frame[required].notna().all(axis=1).sum())
    completeness = round(100.0 * complete_rows / n, 2)

    normalized, _ = normalize_text(frame, schema)
    same = pd.Series(True, index=frame.index)
    for col in ("StockCode", "Description", "Country", "InvoiceDate"):
        a = frame[col].astype("string").fillna("")
        b = normalized[col].astype("string").fillna("")
        if col == "InvoiceDate":
            a = pd.to_datetime(frame[col], errors="coerce").astype("string")
            b = pd.to_datetime(normalized[col], errors="coerce").astype("string")
        same &= (a == b)
    consistency = round(100.0 * int(same.sum()) / n, 2)

    dup_mask, _ = detect_exact_duplicates(frame)
    uniqueness = round(100.0 * (n - int(dup_mask.sum())) / n, 2)

    validity = round(100.0 * int(_validity_mask(frame, schema).sum()) / n, 2)

    dimensions = {
        "completeness": completeness,
        "consistency": consistency,
        "uniqueness": uniqueness,
        "validity": validity,
    }
    total = round(sum(dimensions[k] * QUALITY_WEIGHTS[k]
                      for k in dimensions), 2)
    return {
        "dimensions": dimensions,
        "weights": dict(QUALITY_WEIGHTS),
        "total": total,
        "n_rows": n,
    }


# ---------------------------------------------------------------------------
# Official pipeline: raw dataset -> cleaned_data.csv + cleaning_log.json
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    input_path: str | Path = DEFAULT_INPUT,
    schema_path: str | Path = DEFAULT_EXPECTED_SCHEMA,
    cleaned_path: str | Path = DEFAULT_CLEANED,
    log_path: str | Path = DEFAULT_LOG,
    nrows_per_sheet: int = DEFAULT_NROWS_PER_SHEET,
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> tuple[pd.DataFrame, dict]:
    """Run the full Module 2 pipeline and persist both official outputs.

    Returns the cleaned+transformed frame and the log document (also
    written to ``log_path``; frame written to ``cleaned_path``).
    """
    df = load_dataset(input_path, nrows_per_sheet=nrows_per_sheet)
    schema = load_schema(schema_path)

    quality_before = score_quality(df, schema)

    cleaned, clean_log = clean_dataframe(
        df, schema, knn_neighbors=knn_neighbors,
        fuzzy_threshold=fuzzy_threshold)
    featured, feat_log = extract_features(cleaned)
    scaled, scaler_params = scale_features(featured)

    quality_after = score_quality(scaled, schema)
    delta = round(quality_after["total"] - quality_before["total"], 2)

    cleaned_path = Path(cleaned_path)
    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(cleaned_path, index=False)

    impute_stage = next(
        (s for s in clean_log["stages"]
         if str(s.get("stage", "")).startswith("impute_missing[statistical]")),
        {},
    )
    log = {
        "module": MODULE_NAME,
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "source": {
            "input": str(input_path),
            "schema": str(schema_path),
            "nrows_per_sheet": nrows_per_sheet,
            "n_rows_in": int(len(df)),
            "n_rows_out": int(len(scaled)),
        },
        "quality_before": quality_before,
        "quality_after": quality_after,
        "quality_delta": delta,
        "fixes": {
            "normalization": next(
                (s for s in clean_log["stages"]
                 if s.get("stage") == "normalize_text"), {}),
            "imputation_statistical": impute_stage,
            "exact_duplicates_removed": clean_log["n_exact_duplicates_removed"],
            "near_duplicate_groups": clean_log["near_duplicate_groups"],
            "n_near_duplicate_groups": clean_log["n_near_duplicate_groups"],
        },
        "transformation": {
            "features": feat_log,
            "scaling": scaler_params,
            "cleaned_columns": list(scaled.columns),
        },
    }
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False,
                                   default=str), encoding="utf-8")
    return scaled, log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 6: Module 2 cleaning API - raw dataset to "
                    "cleaned_data.csv + cleaning_log.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to online_retail_II.xlsx")
    p.add_argument("--schema", default=str(DEFAULT_EXPECTED_SCHEMA),
                   help="Path to Sprint 4's expected_schema.json")
    p.add_argument("--cleaned", default=str(DEFAULT_CLEANED),
                   help="Where to save cleaned_data.csv")
    p.add_argument("--log", default=str(DEFAULT_LOG),
                   help="Where to save cleaning_log.json")
    p.add_argument("--nrows-per-sheet", type=int,
                   default=DEFAULT_NROWS_PER_SHEET,
                   help="Max rows read per sheet (0 = all rows).")
    p.add_argument("--knn-neighbors", type=int, default=DEFAULT_KNN_NEIGHBORS,
                   help="KNNImputer neighbourhood size.")
    p.add_argument("--fuzzy-threshold", type=float,
                   default=DEFAULT_FUZZY_THRESHOLD,
                   help="Min difflib ratio for near-duplicate pairs.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    args = parse_args(argv)
    cleaned, log = run_cleaning_pipeline(
        args.input, args.schema, args.cleaned, args.log,
        args.nrows_per_sheet, args.knn_neighbors, args.fuzzy_threshold)
    print(json.dumps({
        "module": log["module"],
        "n_rows_in": log["source"]["n_rows_in"],
        "n_rows_out": log["source"]["n_rows_out"],
        "quality_before": log["quality_before"]["total"],
        "quality_after": log["quality_after"]["total"],
        "quality_delta": log["quality_delta"],
        "dimensions_before": log["quality_before"]["dimensions"],
        "dimensions_after": log["quality_after"]["dimensions"],
    }, indent=2, ensure_ascii=False))
    print(f"\nModule 2 complete - cleaned data -> {args.cleaned}")
    print(f"Module 2 complete - cleaning log  -> {args.log}")
    return cleaned, log


if __name__ == "__main__":
    main()
