"""Sprint 2 - Profiling Engine + Visualizations.

Computes data health/shape metrics for the Online Retail dataset and saves
backend-rendered (non-interactive) PNG visualizations.

Metrics:
  * missing-value matrix  : per-column counts/pct + per-row gap distribution.
  * dtype consistency     : do values in each column match its inferred type?
  * cardinality           : unique counts + cardinality ratio per column.
  * correlation + distribution summary for numeric columns (Quantity, UnitPrice).

Visuals (saved under ``module1_profiling/visuals/``):
  * missing_heatmap.png        - binary missingness map (sample of rows).
  * outlier_distribution.png   - histograms + boxplots for Quantity/UnitPrice.
  * correlation_heatmap.png    - Pearson correlation heatmap of numerics.

Usage:
    python profiler.py
    python profiler.py --input ..\\online_retail_II.xlsx --nrows-per-sheet 50000
    python profiler.py --heatmap-rows 1000 --no-show
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend-rendered PNGs, never interactive.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from metadata_extractor import EXPECTED_COLUMNS, infer_logical_dtype, load_dataset
except ImportError:  # allow `python module1_profiling/profiler.py` from repo root
    from module1_profiling.metadata_extractor import (
        EXPECTED_COLUMNS,
        infer_logical_dtype,
        load_dataset,
    )

try:
    from metadata_extractor import DEFAULT_INPUT as _DEFAULT_INPUT
    from metadata_extractor import DEFAULT_NROWS_PER_SHEET as _DEFAULT_NROWS
except ImportError:
    try:
        from module1_profiling.metadata_extractor import DEFAULT_INPUT as _DEFAULT_INPUT
        from module1_profiling.metadata_extractor import DEFAULT_NROWS_PER_SHEET as _DEFAULT_NROWS
    except Exception:  # pragma: no cover - fallback if constants move
        _DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "online_retail_II.xlsx"
        _DEFAULT_NROWS = 50000

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "visuals"
DEFAULT_REPORT = Path(__file__).resolve().parent / "profiling_report.json"

PURE_INT_RE = re.compile(r"^[+-]?\d+$")
PURE_FLOAT_RE = re.compile(r"^[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?$")


# ---------------------------------------------------------------------------
# Missing-value matrix
# ---------------------------------------------------------------------------

def missing_value_matrix(df: pd.DataFrame) -> dict:
    """Per-column + per-row missingness summary."""
    n_rows, n_cols = df.shape
    is_null = df.isna()
    per_column = {}
    for col in df.columns:
        n_null = int(is_null[col].sum())
        per_column[col] = {
            "n_missing": n_null,
            "missing_pct": round(100.0 * n_null / n_rows, 2) if n_rows else 0.0,
        }
    per_row_counts = is_null.sum(axis=1)
    dist = {
        "rows_with_0_missing": int((per_row_counts == 0).sum()),
        "rows_with_1_missing": int((per_row_counts == 1).sum()),
        "rows_with_2_or_more_missing": int((per_row_counts >= 2).sum()),
        "max_missing_per_row": int(per_row_counts.max()) if n_rows else 0,
    }
    total_cells = n_rows * n_cols
    total_missing = int(is_null.sum().sum())
    return {
        "per_column": per_column,
        "per_row": dist,
        "total_cells": int(total_cells),
        "total_missing_cells": total_missing,
        "total_missing_pct": round(100.0 * total_missing / total_cells, 4) if total_cells else 0.0,
    }


# ---------------------------------------------------------------------------
# Data-type consistency check
# ---------------------------------------------------------------------------

def _value_matches_dtype(value, dtype: str) -> bool:
    """Check a single non-null value against an inferred logical dtype."""
    if dtype == "int":
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, np.integer)):
            return True
        if isinstance(value, (float, np.floating)):
            return bool(float(value).is_integer())
        return bool(PURE_INT_RE.match(str(value).strip()))
    if dtype == "float":
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, float, np.integer, np.floating)):
            return True
        return bool(PURE_FLOAT_RE.match(str(value).strip()))
    if dtype == "date":
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return True
        try:
            parsed = pd.to_datetime(value, errors="coerce", format="mixed")
            return not pd.isna(parsed)
        except Exception:
            return False
    if dtype == "boolean":
        return str(value).strip().lower() in {"true", "false", "0", "1", "0.0", "1.0"}
    # "string": everything renderable as text is consistent.
    return True


def dtype_consistency(df: pd.DataFrame, inferred: dict[str, str] | None = None) -> dict:
    """Check every non-null value against its column's inferred dtype.

    Returns per-column {inferred_dtype, n_checked, n_consistent,
    n_inconsistent, consistency_rate, inconsistent_examples}.
    """
    result = {}
    for col in df.columns:
        dtype = (inferred or {}).get(col) or infer_logical_dtype(df[col])
        series = df[col].dropna()
        n_checked = int(len(series))
        inconsistent_examples: list[str] = []
        n_inconsistent = 0
        for v in series.tolist():
            if not _value_matches_dtype(v, dtype):
                n_inconsistent += 1
                if len(inconsistent_examples) < 5 and str(v) not in inconsistent_examples:
                    inconsistent_examples.append(str(v))
        n_consistent = n_checked - n_inconsistent
        result[col] = {
            "inferred_dtype": dtype,
            "n_checked": n_checked,
            "n_consistent": int(n_consistent),
            "n_inconsistent": int(n_inconsistent),
            "consistency_rate": round(n_consistent / n_checked, 4) if n_checked else 1.0,
            "inconsistent_examples": inconsistent_examples,
        }
    return result


# ---------------------------------------------------------------------------
# Cardinality
# ---------------------------------------------------------------------------

def cardinality(df: pd.DataFrame, high_card_threshold: float = 0.5) -> dict:
    """Unique counts + cardinality ratio per column."""
    n_rows = len(df)
    result = {}
    for col in df.columns:
        n_unique = int(df[col].nunique(dropna=True))
        ratio = (n_unique / n_rows) if n_rows else 0.0
        if ratio >= high_card_threshold:
            level = "high (identifier-like)"
        elif ratio >= 0.05:
            level = "medium"
        else:
            level = "low (categorical-like)"
        top_values = {str(k): int(v) for k, v in df[col].value_counts(dropna=True).head(5).items()}
        result[col] = {
            "n_unique": n_unique,
            "cardinality_ratio": round(float(ratio), 6),
            "cardinality_level": level,
            "top_values": top_values,
        }
    return result


# ---------------------------------------------------------------------------
# Numeric correlation + distribution summary
# ---------------------------------------------------------------------------

NUMERIC_COLUMNS = ["Quantity", "UnitPrice"]


def numeric_summary(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Correlation matrix + distribution stats for numeric columns."""
    cols = [c for c in (columns or NUMERIC_COLUMNS) if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce") if cols else pd.DataFrame()
    distributions = {}
    for col in numeric.columns:
        s = numeric[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((s < lower) | (s > upper)).sum())
        distributions[col] = {
            "count": int(s.count()),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(float(iqr), 4),
            "iqr_lower": round(float(lower), 4),
            "iqr_upper": round(float(upper), 4),
            "n_iqr_outliers": n_outliers,
            "outlier_pct": round(100.0 * n_outliers / len(s), 2) if len(s) else 0.0,
            "n_negative": int((s < 0).sum()),
            "n_zero": int((s == 0).sum()),
            "skew": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurt()), 4),
        }
    corr = numeric.corr(method="pearson").round(4).fillna(0.0)
    return {
        "columns": list(numeric.columns),
        "pearson_correlation": corr.to_dict(),
        "distributions": distributions,
    }


# ---------------------------------------------------------------------------
# Visualizations (Agg backend -> PNG files)
# ---------------------------------------------------------------------------

def plot_missing_heatmap(df: pd.DataFrame, out_path: Path, sample_rows: int = 1000) -> Path:
    """Binary missingness map; CustomerID gaps show as a distinct band."""
    sample = df if len(df) <= sample_rows else df.sample(n=sample_rows, random_state=42).sort_index()
    mask = sample.isna().astype(int).T  # columns -> y axis, rows -> x axis
    fig_w = max(8, len(sample.columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, 6))
    im = ax.imshow(mask.values, aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_yticks(range(len(sample.columns)))
    ax.set_yticklabels(list(sample.columns))
    ax.set_xlabel(f"Sampled rows (n={len(sample)})")
    ax.set_title("Missing-value heatmap (bright = missing)\nCustomerID band shows ~30% missingness")
    # Annotate per-column missing % on the y labels.
    n = len(sample)
    labels = [
        f"{c} ({100.0 * sample[c].isna().sum() / n:.1f}% missing)" for c in sample.columns
    ]
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Missing (1) / Present (0)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _capped(series: pd.Series, cap_q: float = 0.995) -> pd.Series:
    cap = float(series.quantile(cap_q))
    return series.clip(upper=cap), cap


def plot_outlier_distribution(df: pd.DataFrame, out_path: Path) -> Path:
    """Histograms (99.5%-capped for readability) + boxplots for Quantity/UnitPrice."""
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        capped, cap = _capped(s)
        ax_h, ax_b = axes[i, 0], axes[i, 1]
        ax_h.hist(capped, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
        ax_h.axvline(float(s.median()), color="red", linestyle="--", label=f"median={s.median():.2f}")
        ax_h.set_title(f"{col} distribution (capped at 99.5% = {cap:.2f}; max={s.max():.2f})")
        ax_h.set_xlabel(col)
        ax_h.set_ylabel("Frequency")
        ax_h.legend(fontsize=8)
        ax_b.boxplot(s, patch_artist=True, orientation="horizontal",
                     boxprops=dict(facecolor="lightsteelblue"))
        ax_b.set_title(f"{col} boxplot (IQR outliers marked; neg/zero values kept)")
        ax_b.set_xlabel(col)
    # Hide unused row if only one numeric column present.
    if len(cols) < 2:
        for ax in axes[1]:
            ax.set_visible(False)
    fig.suptitle("Outlier distribution: Quantity / UnitPrice", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: Path) -> Path:
    """Pearson correlation heatmap over available numeric columns."""
    num_df = df.select_dtypes(include=[np.number]).copy()
    for col in NUMERIC_COLUMNS:  # ensure Quantity/UnitPrice present even if object-typed
        if col in df.columns and col not in num_df.columns:
            num_df[col] = pd.to_numeric(df[col], errors="coerce")
    num_df = num_df.dropna(axis=1, how="all")
    corr = num_df.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(max(5, corr.shape[1] * 1.4), max(4, corr.shape[0] * 1.1)))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)
    for r in range(corr.shape[0]):
        for c in range(corr.shape[1]):
            ax.text(c, r, f"{corr.values[r, c]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Pearson correlation heatmap (numeric columns)")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def profile_dataset(
    df: pd.DataFrame,
    visuals_dir: Path = DEFAULT_OUTPUT_DIR,
    heatmap_rows: int = 1000,
) -> dict:
    """Run all profiling steps + render the three PNGs. Returns report dict."""
    inferred = {col: infer_logical_dtype(df[col]) for col in df.columns}
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_value_matrix": missing_value_matrix(df),
        "dtype_consistency": dtype_consistency(df, inferred),
        "cardinality": cardinality(df),
        "numeric_profile": numeric_summary(df),
    }
    visuals_dir = Path(visuals_dir)
    visuals_dir.mkdir(parents=True, exist_ok=True)
    report["visuals"] = {
        "missing_heatmap": str(plot_missing_heatmap(df, visuals_dir / "missing_heatmap.png", sample_rows=heatmap_rows)),
        "outlier_distribution": str(plot_outlier_distribution(df, visuals_dir / "outlier_distribution.png")),
        "correlation_heatmap": str(plot_correlation_heatmap(df, visuals_dir / "correlation_heatmap.png")),
    }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sprint 2: profiling engine + visualizations.")
    p.add_argument("--input", default=str(_DEFAULT_INPUT), help="Path to online_retail_II.xlsx")
    p.add_argument(
        "--nrows-per-sheet", type=int, default=_DEFAULT_NROWS,
        help="Max rows read per sheet (0 = all rows; default 50000 for speed).",
    )
    p.add_argument("--visuals-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to save PNGs")
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="Where to save profiling_report.json")
    p.add_argument("--heatmap-rows", type=int, default=1000, help="Rows sampled in missing heatmap")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    df = load_dataset(args.input, nrows_per_sheet=args.nrows_per_sheet)
    # Canonical order first for stable heatmap/report layout.
    ordered = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df[ordered + [c for c in df.columns if c not in ordered]]
    report = profile_dataset(df, visuals_dir=Path(args.visuals_dir), heatmap_rows=args.heatmap_rows)
    report["source_file"] = str(args.input)
    report["nrows_per_sheet"] = args.nrows_per_sheet

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(
        {k: v for k, v in report.items() if k != "dtype_consistency"},
        indent=2, ensure_ascii=False,
    ))
    print(f"\nSaved profiling report -> {out}")
    for name, path in report["visuals"].items():
        print(f"  {name}: {path}")
    return report


if __name__ == "__main__":
    main()
