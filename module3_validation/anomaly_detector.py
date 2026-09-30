"""Sprint 8 - ML Anomaly Detection (Module 3: Validation).

Machine-learning layer on top of the Sprint 7 rule engine. Two unsupervised
models score every row of ``cleaned_data.csv`` on the numeric fields
(Quantity, UnitPrice) after ``StandardScaler`` standardisation (the same
scaling semantics as Module 2's ``scale_features``):

  * ``IsolationForest`` - global isolation: how few random splits isolate
    the point (catches extreme prices / bulk quantities).
  * ``LocalOutlierFactor`` - local density: how isolated the point is from
    its ``n_neighbors`` neighbourhood (catches points odd for their region,
    e.g. bulk quantities at unusual prices). LOF is fit on DEDUPLICATED
    feature combinations (see ``run_lof``): the raw frame packs up to 2,165
    rows onto one exact (Quantity, UnitPrice) point, and distance-0
    neighbours make reachability density undefined (scores explode to 1e8).
    Exact ties carry no local-density information, so collapsing them
    before density estimation is required for sane scores; per-row scores
    map back through the duplicate groups deterministically.

Scores (higher = more anomalous, both models):
  * ``if_score``  = ``-decision_function(X)`` (0 ~= boundary, > 0 anomaly).
  * ``lof_score`` = ``-negative_outlier_factor_`` (1 ~= normal, >> 1 anomaly).

Binary flags come from the ``contamination`` rate per model; the sample
lists the top-IF and top-LOF rows (union) so both detectors' strongest
findings appear, each row carrying BOTH scores plus both flags.

Autoencoder: deliberately skipped (see ``AUTOENCODER_NOTE``) - a deep
reconstruction model buys nothing on a 2-D numeric frame and would add a
torch/TF dependency; IF (global) + LOF (local) already cover the two
complementary anomaly geometries.

Usage:
    from module3_validation.anomaly_detector import run_detection

    doc = run_detection()  # defaults: cleaned_data.csv in, anomalies_sample.json out

CLI:
    python module3_validation/anomaly_detector.py
    python module3_validation/anomaly_detector.py --contamination 0.01 --top-n 10
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on ``sys.path`` so imports work no matter the
    working directory (repo root, ``module3_validation/``, ...)."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

MODULE_NAME = "module3_validation"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "module2_cleaning" / "cleaned_data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "anomalies_sample.json"

#: Numeric fields both models score (Sprint 8 brief).
FEATURE_COLUMNS = ["Quantity", "UnitPrice"]

#: Columns snapshotted per anomalous row (canonical 8 first, then derived).
SNAPSHOT_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
    "UnitPrice", "CustomerID", "Country", "LineValue",
    "IsCancellation", "IsReturn", "IsGiveaway",
]

IF_N_ESTIMATORS = 200
IF_RANDOM_STATE = 42
LOF_N_NEIGHBORS = 20
DEFAULT_CONTAMINATION = 0.01
#: Per-model rows shown in the sample (top-N by IF + top-N by LOF, union).
DEFAULT_TOP_N = 10

AUTOENCODER_NOTE = (
    "Autoencoder skipped by design: on a 2-column numeric frame a deep "
    "reconstruction model adds no detection geometry beyond IsolationForest "
    "(global isolation) + LOF (local density), while adding a heavy "
    "torch/TensorFlow dependency and non-determinism. Revisit only if "
    "high-dimensional engineered features need joint reconstruction."
)


# ---------------------------------------------------------------------------
# Loading + features
# ---------------------------------------------------------------------------

def load_cleaned(path: str | Path = DEFAULT_INPUT) -> pd.DataFrame:
    """Load the official Module 2 output."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned data not found: {path}")
    return pd.read_csv(path)


def build_features(df: pd.DataFrame,
                   columns: list[str] | None = None) -> tuple[np.ndarray, dict]:
    """Numeric matrix + fitted ``StandardScaler`` params.

    Rows with non-numeric Quantity/UnitPrice are dropped from scoring (the
    cleaned file has none; the mask is returned so row ids stay aligned).
    """
    columns = list(columns or FEATURE_COLUMNS)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing: {missing}")
    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.notna().all(axis=1).to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(numeric.loc[valid_mask].to_numpy(dtype=float))
    params = {
        "columns": columns,
        "mean_": {c: float(m) for c, m in zip(columns, scaler.mean_)},
        "scale_": {c: float(s) for c, s in zip(columns, scaler.scale_)},
        "n_rows_total": int(len(df)),
        "n_rows_scored": int(valid_mask.sum()),
        "n_rows_dropped_non_numeric": int((~valid_mask).sum()),
    }
    return X, params


# ---------------------------------------------------------------------------
# Models (one function each, per the brief's "at least two models")
# ---------------------------------------------------------------------------

def run_isolation_forest(X: np.ndarray,
                         contamination: float = DEFAULT_CONTAMINATION,
                         n_estimators: int = IF_N_ESTIMATORS,
                         random_state: int = IF_RANDOM_STATE) -> dict:
    """Global-isolation anomalies. Returns per-row ``if_score`` (higher =
    more anomalous) and the ``if_flag`` contamination cut."""
    model = IsolationForest(n_estimators=n_estimators,
                            contamination=contamination,
                            random_state=random_state)
    flags = model.fit_predict(X)
    scores = -model.decision_function(X)  # > 0 past the boundary
    return {
        "method": "sklearn.ensemble.IsolationForest",
        "params": {"n_estimators": n_estimators,
                   "contamination": contamination,
                   "random_state": random_state},
        "scores": np.asarray(scores, dtype=float),
        "flags": np.asarray(flags == -1),
    }


def run_lof(X: np.ndarray,
            contamination: float = DEFAULT_CONTAMINATION,
            n_neighbors: int = LOF_N_NEIGHBORS) -> dict:
    """Local-density anomalies. Returns per-row ``lof_score`` (higher =
    more anomalous; 1 ~= normal) and the ``lof_flag`` contamination cut.

    Fit on deduplicated feature combinations: up to thousands of rows share
    one exact (Quantity, UnitPrice) point, and distance-0 neighbours make
    LOF's reachability density undefined (raw-fit scores explode past 1e8
    for ordinary rows). Ties carry no density information, so the model
    fits the unique-combination grid and scores/flags map back to every row
    sharing the combination. Deterministic (``np.unique`` sorts; LOF itself
    has no random state). The ``contamination`` cut applies at the
    combination level - rare combos hold few rows, so fewer rows flag than
    under IsolationForest; both counts are reported, not forced to agree.
    """
    combos, inverse = np.unique(X, axis=0, return_inverse=True)
    model = LocalOutlierFactor(n_neighbors=n_neighbors,
                               contamination=contamination)
    combo_flags = model.fit_predict(combos) == -1
    combo_scores = -model.negative_outlier_factor_
    return {
        "method": "sklearn.neighbors.LocalOutlierFactor",
        "params": {"n_neighbors": n_neighbors,
                   "contamination": contamination,
                   "fit": "deduplicated feature combinations "
                          f"({len(combos)} unique of {len(X)} rows)"},
        "scores": np.asarray(combo_scores[inverse], dtype=float),
        "flags": np.asarray(combo_flags[inverse]),
        "n_unique_combinations": int(len(combos)),
    }


def detect_anomalies(df: pd.DataFrame,
                     contamination: float = DEFAULT_CONTAMINATION,
                     n_estimators: int = IF_N_ESTIMATORS,
                     n_neighbors: int = LOF_N_NEIGHBORS,
                     random_state: int = IF_RANDOM_STATE) -> dict:
    """Score ``df`` with both models. Returns aligned per-row scores/flags
    over the scored (all-numeric) rows plus model configs and scaler params."""
    X, scaler_params = build_features(df)
    scored_index = df.index[pd.to_numeric(df["Quantity"], errors="coerce").notna()
                            & pd.to_numeric(df["UnitPrice"], errors="coerce").notna()]
    isolation = run_isolation_forest(X, contamination, n_estimators, random_state)
    lof = run_lof(X, contamination, n_neighbors)
    if_scores = isolation["scores"]
    lof_scores = lof["scores"]
    combined_rank = (pd.Series(if_scores).rank(pct=True)
                     + pd.Series(lof_scores).rank(pct=True)) / 2.0
    return {
        "n_rows_total": int(len(df)),
        "scored_positions": [int(df.index.get_loc(i)) for i in scored_index],
        "feature_columns": list(FEATURE_COLUMNS),
        "scaler": scaler_params,
        "if": {"scores": if_scores, "flags": isolation["flags"],
               "method": isolation["method"], "params": isolation["params"]},
        "lof": {"scores": lof_scores, "flags": lof["flags"],
                "method": lof["method"], "params": lof["params"]},
        "combined_rank": combined_rank.to_numpy(dtype=float),
        "n_if_flagged": int(isolation["flags"].sum()),
        "n_lof_flagged": int(lof["flags"].sum()),
        "n_both_flagged": int((isolation["flags"] & lof["flags"]).sum()),
    }


# ---------------------------------------------------------------------------
# Sample building
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
    return {c: _jsonable(row[c]) for c in SNAPSHOT_COLUMNS if c in df.columns}


def build_anomalies_sample(df: pd.DataFrame,
                           contamination: float = DEFAULT_CONTAMINATION,
                           top_n: int = DEFAULT_TOP_N) -> dict:
    """Detect anomalies and package the sample document.

    The sample is the union of the top-``top_n`` IsolationForest rows and
    the top-``top_n`` LOF rows (ranked by each model's own score), so both
    detectors' strongest findings appear. EVERY listed row carries the
    numeric score of EACH model (``if_score`` + ``lof_score``) plus both
    binary flags - the Sprint 8 definition of done.
    """
    detected = detect_anomalies(df, contamination=contamination)
    scored_pos = detected["scored_positions"]
    if_scores = detected["if"]["scores"]
    lof_scores = detected["lof"]["scores"]
    top_if = set(np.argsort(-if_scores)[:top_n].tolist())
    top_lof = set(np.argsort(-lof_scores)[:top_n].tolist())
    picked = sorted(top_if | top_lof,
                    key=lambda i: -(if_scores[i] + lof_scores[i]))

    anomalies = []
    for i in picked:
        pos = scored_pos[i]
        source = ("both" if (i in top_if and i in top_lof)
                  else "isolation_forest" if i in top_if else "lof")
        anomalies.append({
            "row_id": int(pos),
            "csv_line": int(pos) + 2,  # +1 header, +1 1-based lines
            "top_by": source,
            "if_score": round(float(if_scores[i]), 4),
            "lof_score": round(float(lof_scores[i]), 4),
            "combined_rank": round(float(detected["combined_rank"][i]), 4),
            "if_flag": bool(detected["if"]["flags"][i]),
            "lof_flag": bool(detected["lof"]["flags"][i]),
            "values": _snapshot_row(df, pos),
        })
    return {
        "module": MODULE_NAME,
        "artifact": "anomalies_sample",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "input": str(DEFAULT_INPUT),
            "n_rows": int(len(df)),
            "feature_columns": list(FEATURE_COLUMNS),
            "contamination": contamination,
            "top_n_per_model": top_n,
        },
        "models": {
            "isolation_forest": {
                **detected["if"]["params"],
                "method": detected["if"]["method"],
                "n_flagged": detected["n_if_flagged"],
                "score": "if_score = -decision_function (higher = more anomalous)",
            },
            "lof": {
                **detected["lof"]["params"],
                "method": detected["lof"]["method"],
                "n_flagged": detected["n_lof_flagged"],
                "score": "lof_score = -negative_outlier_factor_ (higher = more anomalous)",
            },
            "n_both_flagged": detected["n_both_flagged"],
            "autoencoder": AUTOENCODER_NOTE,
        },
        "scaler": detected["scaler"],
        "n_anomalies_shown": len(anomalies),
        "note": ("Union of the top-N rows of EACH model, ordered by summed "
                 "scores. Every row carries the numeric score of EACH model "
                 "(if_score + lof_score) plus both flags. row_id is the "
                 "0-based position in cleaned_data.csv; csv_line is the "
                 "1-based file line (header = line 1)."),
        "anomalies": anomalies,
    }


# ---------------------------------------------------------------------------
# File-level API + CLI
# ---------------------------------------------------------------------------

def run_detection(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    contamination: float = DEFAULT_CONTAMINATION,
    top_n: int = DEFAULT_TOP_N,
) -> dict:
    """Load cleaned data, detect anomalies with both models, save the sample
    document to ``output_path`` and return it."""
    df = load_cleaned(input_path)
    doc = build_anomalies_sample(df, contamination=contamination, top_n=top_n)
    doc["source"]["input"] = str(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 8: ML anomaly detection (IsolationForest + LOF) "
                    "over cleaned_data.csv -> anomalies_sample.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to cleaned_data.csv")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT),
                   help="Where to save anomalies_sample.json")
    p.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION,
                   help="Expected anomaly rate per model (flag threshold).")
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                   help="Top rows shown per model (union in the sample).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    doc = run_detection(args.input, args.output, args.contamination, args.top_n)
    print(json.dumps({
        "module": doc["module"],
        "n_rows": doc["source"]["n_rows"],
        "n_if_flagged": doc["models"]["isolation_forest"]["n_flagged"],
        "n_lof_flagged": doc["models"]["lof"]["n_flagged"],
        "n_both_flagged": doc["models"]["n_both_flagged"],
        "n_anomalies_shown": doc["n_anomalies_shown"],
    }, indent=2, ensure_ascii=False))
    print(f"\nAnomalies sample -> {args.output}")
    return doc


if __name__ == "__main__":
    main()
