"""Sprint 6 - Transformation Engine (Module 2: Cleaning).

Feature engineering, scaling and encoding for the *cleaned* dataset
(output of ``cleaner.clean_dataframe``). Three transforms:

  1. ``extract_features`` - interpretable new columns, straight from the
     Sprint 4 ``derived_columns`` contract plus calendar parts:
     ``LineValue`` (= Quantity * UnitPrice), ``IsCancellation`` /
     ``IsReturn`` / ``IsGiveaway`` flags, and ``InvoiceYear`` /
     ``InvoiceMonth`` / ``InvoiceDay`` / ``InvoiceHour`` /
     ``InvoiceWeekday`` / ``InvoiceQuarter`` for later time analysis.
  2. ``scale_features`` - z-score scaling (``StandardScaler``) of the
     numeric measures into ``Quantity_scaled`` / ``UnitPrice_scaled`` /
     ``LineValue_scaled``. Raw values are kept alongside; the fitted
     mean/std are returned so train/serve stay consistent.
  3. ``encode_features`` - one-hot encoding of ``Country`` (plus the three
     flags as ints) for modelling consumers. Not persisted to
     ``cleaned_data.csv`` (28 sparse columns would bloat the official
     output); available as a function, covered by tests.

Usage:
    from module2_cleaning.transformer import (
        extract_features, scale_features, encode_features,
    )

    df, log = extract_features(cleaned)
    df, scaler = scale_features(df)       # adds *_scaled columns
    model_frame, enc = encode_features(df)  # one-hot Country for modelling
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

#: Numeric measures scaled by ``scale_features`` (in this order).
SCALE_COLUMNS = ["Quantity", "UnitPrice", "LineValue"]

#: Calendar parts extracted from InvoiceDate (in this order).
DATE_PARTS = ["InvoiceYear", "InvoiceMonth", "InvoiceDay", "InvoiceHour",
              "InvoiceWeekday", "InvoiceQuarter"]

#: Line-type flags (Sprint 4 ``derived_columns`` contract).
FLAG_COLUMNS = ["IsCancellation", "IsReturn", "IsGiveaway"]


def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Add interpretable engineered columns (restates the raw ones).

    * ``LineValue`` = Quantity * UnitPrice, rounded to 2 dp (signed:
      negative for returns, zero for giveaways).
    * ``IsCancellation`` = InvoiceNo starts with ``'C'``;
      ``IsReturn`` = Quantity < 0; ``IsGiveaway`` = UnitPrice == 0.
    * Calendar parts of InvoiceDate (requires datetime64 - the cleaner
      guarantees this; unparseable values become NaT -> NaN parts).
    """
    out = df.copy()
    out["LineValue"] = (pd.to_numeric(out["Quantity"], errors="coerce")
                        * pd.to_numeric(out["UnitPrice"], errors="coerce")
                        ).round(2)
    out["IsCancellation"] = (out["InvoiceNo"].astype("string")
                             .str.startswith("C"))
    out["IsReturn"] = pd.to_numeric(out["Quantity"], errors="coerce") < 0
    out["IsGiveaway"] = pd.to_numeric(out["UnitPrice"], errors="coerce") == 0
    dt = pd.to_datetime(out["InvoiceDate"], errors="coerce")
    out["InvoiceYear"] = dt.dt.year
    out["InvoiceMonth"] = dt.dt.month
    out["InvoiceDay"] = dt.dt.day
    out["InvoiceHour"] = dt.dt.hour
    out["InvoiceWeekday"] = dt.dt.weekday
    out["InvoiceQuarter"] = dt.dt.quarter
    log = {
        "stage": "extract_features",
        "added": (["LineValue"] + FLAG_COLUMNS + DATE_PARTS),
        "n_rows": int(len(out)),
        "linevalue_total": round(float(out["LineValue"].sum()), 2),
        "n_cancellations": int(out["IsCancellation"].sum()),
        "n_returns": int(out["IsReturn"].sum()),
        "n_giveaways": int(out["IsGiveaway"].sum()),
    }
    return out, log


def scale_features(df: pd.DataFrame,
                   columns: list[str] | None = None,
                   method: str = "standard"
                   ) -> tuple[pd.DataFrame, dict]:
    """Z-score scale numeric measures into ``<col>_scaled`` columns.

    Raw columns are preserved. Only ``method="standard"``
    (``StandardScaler``: zero mean, unit variance) is supported - stated
    explicitly so a future method cannot silently change semantics.
    Returns the frame plus the fitted ``{mean, scale}`` per column.
    """
    if method != "standard":
        raise ValueError(f"unsupported scaling method: {method!r}")
    columns = list(columns or SCALE_COLUMNS)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"scaling columns missing: {missing}")
    out = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        out[columns].apply(pd.to_numeric, errors="coerce"))
    params = {
        "method": method,
        "columns": columns,
        "mean_": {c: float(m) for c, m in zip(columns, scaler.mean_)},
        "scale_": {c: float(s) for c, s in zip(columns, scaler.scale_)},
    }
    for pos, col in enumerate(columns):
        out[f"{col}_scaled"] = scaled[:, pos]
    return out, params


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One-hot encode Country (+ flags as ints) for modelling consumers.

    Returns the encoded frame (numeric feature columns only) plus the
    ``country_values`` ordering, so a serving path can rebuild identical
    columns. Not written to ``cleaned_data.csv`` - see module docstring.
    """
    country_values = sorted(set(df["Country"].dropna().astype(str)))
    dummies = pd.get_dummies(df["Country"].astype("string"),
                             prefix="Country", dtype=int)
    dummies = dummies.reindex(
        columns=[f"Country_{v}" for v in country_values], fill_value=0)
    encoded = pd.concat(
        [df[[c for c in SCALE_COLUMNS if c in df.columns]].apply(
            pd.to_numeric, errors="coerce"),
         df[FLAG_COLUMNS].astype(int) if all(
             c in df.columns for c in FLAG_COLUMNS) else pd.DataFrame(),
         dummies],
        axis=1,
    )
    params = {
        "country_values": country_values,
        "n_country_columns": int(dummies.shape[1]),
        "flag_columns": [c for c in FLAG_COLUMNS if c in df.columns],
        "n_rows": int(len(encoded)),
    }
    return encoded, params
