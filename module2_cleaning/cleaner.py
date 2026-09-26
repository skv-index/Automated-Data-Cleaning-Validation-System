"""Sprint 5 - Cleaning Pipeline (Module 2: Cleaning).

Actually fixes the data, enforcing the Sprint 4 contract
(``expected_schema.json``). Three stages:

  1. ``normalize_text`` - canonical formats: strip + collapse whitespace and
     upper-case StockCode/Description, title-case + alias-map Country
     (``EIRE`` -> Ireland, ``USA`` -> United States, ...), parse InvoiceDate
     to datetime64 (``%Y-%m-%d %H:%M:%S``, second resolution, tz-naive).
  2. ``impute_missing`` - two strategies:
       * ``statistical`` (default): Description from the StockCode lookup
         (mode per StockCode, global-mode fallback); median for numerics;
         CustomerID nulls are *preserved* - they are structural guest
         orders (schema rule rv-4), not dirt.
       * ``knn``: ``sklearn.impute.KNNImputer`` (k=5, distance-weighted)
         over the numeric measure columns. Chosen because it is
         non-parametric, preserves local structure and makes no
         distributional assumptions. This dataset has no numeric gaps, so
         on real data it is a verified no-op; ``verify_knn_imputer``
         proves the ML path by masked reconstruction on a copy.
  3. ``detect_exact_duplicates`` / ``detect_near_duplicates`` - exact
     full-row duplicates are removed (keep first); near-duplicates (same
     StockCode, ``difflib`` similarity >= 0.90 on the normalized
     Description, e.g. SPONGE/SPUNGE typos) are *flagged for review, never
     auto-merged* - similar names can be genuinely different products.

``build_sample_diff`` runs the pipeline on a stratified 10-20 row sample
and records before/after pairs into ``cleaning_sample_diff.json`` - the
visible proof the logic works before it touches the full dataset.

Usage:
    python cleaner.py
    python cleaner.py --input ..\\online_retail_II.xlsx --nrows-per-sheet 50000
    python cleaner.py --output cleaning_sample_diff.json --full-out cleaned.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


def _ensure_repo_root_on_path() -> None:
    """Put the repo root on ``sys.path`` so cross-module imports work no
    matter the working directory (repo root, ``module2_cleaning/``, ...)."""
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

try:
    from schema_inference import DEFAULT_OUTPUT as _DEFAULT_SCHEMA
except ImportError:  # allow `python module2_cleaning/cleaner.py` from repo root
    from module2_cleaning.schema_inference import (
        DEFAULT_OUTPUT as _DEFAULT_SCHEMA,
    )

from module1_profiling.metadata_extractor import (
    load_dataset as _load_raw,
)

MODULE_NAME = "module2_cleaning"
DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "online_retail_II.xlsx"
DEFAULT_DIFF = Path(__file__).resolve().parent / "cleaning_sample_diff.json"
DEFAULT_NROWS_PER_SHEET = 50000

#: ML imputer choice (noted per the sprint brief): KNN, k=5,
#: distance-weighted, over numeric measure columns only.
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_FEATURES = ["Quantity", "UnitPrice"]

#: Columns the KNN imputer must never touch: identifiers and the
#: structurally-missing guest-order marker.
KNN_EXCLUDED = ["CustomerID"]

#: Minimum SequenceMatcher ratio for a near-duplicate description pair
#: (same StockCode, normalized text, strictly below 1.0).
FUZZY_THRESHOLD = 0.90

#: Fraction of Quantity values masked in the KNN self-verification.
KNN_VERIFY_MASK_FRAC = 0.05
KNN_VERIFY_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_schema(path: str | Path = _DEFAULT_SCHEMA) -> dict:
    """Load Sprint 4's ``expected_schema.json`` contract."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Expected schema not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(path: str | Path = DEFAULT_INPUT,
                 nrows_per_sheet: int = DEFAULT_NROWS_PER_SHEET) -> pd.DataFrame:
    """Load the workbook with canonical headers (via Module 1's loader)."""
    return _load_raw(path, nrows_per_sheet=nrows_per_sheet)


# ---------------------------------------------------------------------------
# Stage 1: normalization (schema-driven)
# ---------------------------------------------------------------------------

def _country_alias_map(schema: dict) -> dict[str, str]:
    """Alias map keyed by title-cased spelling (matches normalized values)."""
    aliases = (schema.get("columns", {}).get("Country", {})
               .get("allowed_values", {}).get("aliases", {}))
    return {str(k).strip().title(): str(v) for k, v in aliases.items()}


def normalize_text(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the schema's canonical text/date formats.

    Returns the normalized frame plus a per-column change log. Nulls are
    never filled here - that is the imputation stage's job.
    """
    out = df.copy()
    log: dict = {"stage": "normalize_text", "columns": {}}

    for col in ("StockCode", "Description"):
        raw = out[col].astype("string")
        new = (raw.str.strip()
                  .str.replace(r"\s+", " ", regex=True)
                  .str.upper())
        changed = int(((raw.fillna("") != new.fillna(""))
                       & out[col].notna()).sum())
        out[col] = new
        log["columns"][col] = {
            "operation": "strip + collapse-internal-whitespace + upper",
            "n_changed": changed,
        }

    raw_country = out["Country"].astype("string")
    new_country = raw_country.str.strip().str.title()
    alias_map = _country_alias_map(schema)
    hits = new_country[new_country.isin(list(alias_map))]
    aliased = int(len(hits))
    applied = sorted({f"{k}->{alias_map[k]}" for k in set(hits)})
    new_country = new_country.replace(alias_map)
    changed = int(((raw_country.fillna("") != new_country.fillna(""))
                   & out["Country"].notna()).sum())
    out["Country"] = new_country
    log["columns"]["Country"] = {
        "operation": "strip + title-case + alias-map",
        "n_changed": changed,
        "n_aliased": aliased,
        "aliases_applied": applied,
    }

    raw_date = out["InvoiceDate"]
    parsed = pd.to_datetime(raw_date, errors="coerce", format="mixed")
    n_unparseable = int(parsed.isna().sum() - raw_date.isna().sum())
    out["InvoiceDate"] = parsed
    log["columns"]["InvoiceDate"] = {
        "operation": "to_datetime (canonical '%Y-%m-%d %H:%M:%S', "
                     "second resolution, tz-naive)",
        "n_unparseable": n_unparseable,
    }
    return out, log


# ---------------------------------------------------------------------------
# Stage 2: imputation (statistical + ML)
# ---------------------------------------------------------------------------

def build_description_lookup(df: pd.DataFrame) -> dict[str, str]:
    """Normalized StockCode -> mode Description (for Description gaps).

    ``df`` must already be normalized so keys match. Ties break to the
    first mode (deterministic: value_counts order).
    """
    frame = df[["StockCode", "Description"]].dropna()
    if frame.empty:
        return {}
    sc = frame["StockCode"].astype("string").str.strip().str.upper()
    de = frame["Description"].astype("string").str.strip().str.upper()
    return (de.groupby(sc).agg(lambda s: s.mode().iloc[0]).to_dict())


def _global_mode_description(df: pd.DataFrame) -> str | None:
    """Most frequent normalized Description (fallback imputation value)."""
    mode = (df["Description"].dropna().astype("string")
            .str.strip().str.upper().mode())
    return str(mode.iloc[0]) if len(mode) else None


def impute_missing(df: pd.DataFrame, schema: dict,
                   strategy: str = "statistical",
                   knn_neighbors: int = KNN_N_NEIGHBORS,
                   lookup: dict[str, str] | None = None,
                   global_mode: str | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    """Fill missing values with the chosen strategy.

    * ``statistical``: Description <- StockCode lookup (mode per StockCode,
      global-mode fallback); numerics <- median; CustomerID nulls preserved
      (structural guest orders, schema rv-4).
    * ``knn``: KNNImputer over numeric measure columns that actually have
      gaps; CustomerID is explicitly excluded (imputing it would fabricate
      customer identities).
    """
    if strategy not in ("statistical", "knn"):
        raise ValueError(f"unknown imputation strategy: {strategy!r}")
    out = df.copy()
    log: dict = {"stage": f"impute_missing[{strategy}]", "columns": {}}

    if strategy == "statistical":
        if lookup is None:
            lookup = build_description_lookup(out)
        # NOTE: callers imputing a *sample* must pass the full-frame
        # global_mode, otherwise the fallback reflects the sample only.
        if global_mode is None:
            global_mode = _global_mode_description(out)
        mask = out["Description"].isna()
        n_missing = int(mask.sum())
        keys = (out.loc[mask, "StockCode"].astype("string")
                .str.strip().str.upper())
        via_lookup = keys.map(lookup)
        n_lookup = int(via_lookup.notna().sum())
        filled = via_lookup.fillna(global_mode)
        out.loc[mask, "Description"] = filled.tolist()
        log["columns"]["Description"] = {
            "n_missing": n_missing,
            "n_imputed_via_stockcode_lookup": n_lookup,
            "n_imputed_via_global_mode": n_missing - n_lookup,
            "global_mode": global_mode,
        }
        for col in ("Quantity", "UnitPrice"):
            n = int(out[col].isna().sum())
            if n:
                out[col] = out[col].fillna(out[col].median())
                log["columns"][col] = {"n_missing": n,
                                       "action": "filled with median"}
            else:
                log["columns"][col] = {"n_missing": 0, "action": "none"}
        n_guest = int(out["CustomerID"].isna().sum())
        log["columns"]["CustomerID"] = {
            "n_missing": n_guest,
            "action": "preserved (structural guest orders, schema rv-4)",
        }
    else:  # knn
        num_cols = [c for c in KNN_FEATURES if c in out.columns]
        gapped = [c for c in num_cols if bool(out[c].isna().any())]
        log["excluded_columns"] = {
            c: "identifier / structural nulls - never imputed"
            for c in KNN_EXCLUDED if c in out.columns
        }
        if not gapped:
            log["columns"] = {
                c: {"n_missing": 0, "action": "none - no numeric gaps"}
                for c in num_cols
            }
            log["note"] = ("no-op on this dataset: Quantity/UnitPrice have "
                           "no nulls (verified); CustomerID excluded")
        else:
            imp = KNNImputer(n_neighbors=knn_neighbors, weights=KNN_WEIGHTS)
            out[num_cols] = imp.fit_transform(out[num_cols])
            if "Quantity" in gapped:
                out["Quantity"] = (pd.to_numeric(out["Quantity"])
                                   .round().astype("int64"))
            for col in gapped:
                log["columns"][col] = {
                    "n_missing": "see statistical log",
                    "action": f"KNNImputer(k={knn_neighbors}, "
                              f"weights={KNN_WEIGHTS})",
                }
    return out, log


def verify_knn_imputer(df: pd.DataFrame,
                       mask_frac: float = KNN_VERIFY_MASK_FRAC,
                       random_state: int = KNN_VERIFY_SEED,
                       n_neighbors: int = KNN_N_NEIGHBORS) -> dict:
    """Prove the ML imputation path on real distributions without touching
    real data: mask ``mask_frac`` of Quantity on a *copy*, reconstruct with
    KNNImputer, and report the masked reconstruction error."""
    work = df[KNN_FEATURES].apply(pd.to_numeric, errors="coerce").copy()
    rng = np.random.default_rng(random_state)
    mask = rng.random(len(work)) < mask_frac
    truth = work.loc[mask, "Quantity"].to_numpy(dtype=float)
    work.loc[mask, "Quantity"] = np.nan
    imp = KNNImputer(n_neighbors=n_neighbors, weights=KNN_WEIGHTS)
    filled = imp.fit_transform(work)
    pred = np.rint(filled[:, 0])[mask]
    return {
        "method": f"KNNImputer(k={n_neighbors}, weights={KNN_WEIGHTS})",
        "features": list(KNN_FEATURES),
        "n_rows_scored": int(len(work)),
        "n_masked": int(mask.sum()),
        "mask_frac": mask_frac,
        "random_state": random_state,
        "mae_masked_quantity": round(float(np.mean(np.abs(pred - truth))), 4),
        "exact_match_rate": round(float(np.mean(pred == truth)), 4),
        "note": ("self-test on a masked copy only; the real frame is "
                 "never modified by verification"),
    }


# ---------------------------------------------------------------------------
# Stage 3: duplicate detection (exact + fuzzy)
# ---------------------------------------------------------------------------

def detect_exact_duplicates(df: pd.DataFrame) -> tuple[pd.Series, dict]:
    """Full-row exact duplicates (NaN == NaN at the same position).

    Policy: remove, keep first occurrence - a byte-identical line item is a
    double-recorded row, not two genuinely identical purchases.
    """
    mask = df.duplicated(keep="first")
    n_groups = int(df[df.duplicated(keep=False)].fillna("").astype(str)
                   .agg("|".join, axis=1).nunique())
    return mask, {
        "n_duplicate_rows": int(mask.sum()),
        "n_duplicate_groups": n_groups,
        "policy": "removed (keep first)",
    }


def _union_find(pairs: list[tuple[str, str]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    groups: dict[str, set[str]] = {}
    for a in parent:
        groups.setdefault(find(a), set()).add(a)
    return [g for g in groups.values() if len(g) > 1]


def detect_near_duplicates(df: pd.DataFrame,
                           threshold: float = FUZZY_THRESHOLD) -> list[dict]:
    """Near-duplicate description variants within one StockCode.

    Compares *normalized* (stripped, upper-cased) Descriptions pairwise per
    StockCode; pairs with ``threshold <= similarity < 1.0`` (difflib ratio)
    form review groups. Policy: flag only - similar names (RETROSPOT vs
    SPOTTY) can be genuinely different products, and typos (SPONGE vs
    SPUNGE) need a human to pick the canonical spelling.
    """
    desc = df["Description"].astype("string").str.strip().str.upper()
    sc = df["StockCode"].astype("string").str.strip().str.upper()
    groups: list[dict] = []
    for code in sorted(set(sc.dropna())):
        rows = df.index[sc == code].tolist()
        values = sorted({str(desc.loc[i]) for i in rows
                         if pd.notna(desc.loc[i])})
        pairs: list[tuple[str, str]] = []
        for a_pos in range(len(values)):
            for b_pos in range(a_pos + 1, len(values)):
                ratio = SequenceMatcher(None, values[a_pos],
                                        values[b_pos]).ratio()
                if threshold <= ratio < 1.0:
                    pairs.append((values[a_pos], values[b_pos]))
        for component in _union_find(pairs):
            member_rows = sorted(
                i for i in rows
                if pd.notna(desc.loc[i]) and str(desc.loc[i]) in component)
            sims = [SequenceMatcher(None, a, b).ratio()
                    for a in component for b in component if a < b]
            groups.append({
                "group_id": f"ng-{len(groups) + 1}",
                "stockcode": code,
                "row_ids": member_rows,
                "descriptions": sorted(component),
                # One row per distinct spelling: the demo sample pairs the
                # typo against its canonical twin, not two typo rows.
                "representatives": {
                    d: next(i for i in member_rows
                            if str(desc.loc[i]) == d)
                    for d in sorted(component)
                },
                "max_similarity": round(max(sims), 4) if sims else 1.0,
                "recommendation": "review - never auto-merge",
            })
    return groups


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame, schema: dict,
                    knn_neighbors: int = KNN_N_NEIGHBORS,
                    fuzzy_threshold: float = FUZZY_THRESHOLD
                    ) -> tuple[pd.DataFrame, dict]:
    """Run normalize -> impute (statistical, then KNN) -> dedupe.

    Returns the cleaned frame (exact duplicates removed, keep first) and a
    stage-by-stage log. Near-duplicate groups are reported, not merged.
    """
    n_in = int(len(df))
    normalized, norm_log = normalize_text(df, schema)
    imputed, imp_log = impute_missing(normalized, schema,
                                      strategy="statistical")
    imputed, knn_log = impute_missing(imputed, schema, strategy="knn",
                                      knn_neighbors=knn_neighbors)
    dup_mask, dup_log = detect_exact_duplicates(imputed)
    cleaned = imputed.loc[~dup_mask].copy()
    near_groups = detect_near_duplicates(cleaned, threshold=fuzzy_threshold)
    log = {
        "n_rows_in": n_in,
        "n_rows_out": int(len(cleaned)),
        "n_exact_duplicates_removed": int(dup_mask.sum()),
        "n_near_duplicate_groups": len(near_groups),
        "stages": [norm_log, imp_log, knn_log, dup_log],
        "near_duplicate_groups": near_groups,
    }
    return cleaned, log


# ---------------------------------------------------------------------------
# Sample diff (visible proof before the full run)
# ---------------------------------------------------------------------------

def _jsonable(value):
    """Render one cell JSON-safe (Timestamp -> canonical string, NaN -> None)."""
    try:
        if value is None:
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(value, float) and (math.isnan(value) or pd.isna(value)):
            return None
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            v = float(value)
            return None if math.isnan(v) else v
        if pd.isna(value):
            return None
        return value
    except (TypeError, ValueError):
        return str(value)


def _json_row(row: pd.Series) -> dict:
    return {str(k): _jsonable(v) for k, v in row.items()}


def select_demo_sample(df: pd.DataFrame, schema: dict) -> list[int]:
    """Deterministically pick 10-20 row ids covering every defect class:
    an exact-duplicate pair, imputable + non-imputable Description gaps,
    Country aliases, padded/cased text, a fuzzy typo pair, a structural
    guest-order null, and clean control rows."""
    picked: list[int] = []

    def take(indices, n: int = 1) -> list:
        fresh = [i for i in indices if i not in picked][:n]
        picked.extend(fresh)
        return fresh

    # Exact-duplicate pair: rows of the first duplicated key in row order.
    key = df.fillna("").astype(str).agg("|".join, axis=1)
    dup_keys = key[key.duplicated(keep=False)]
    first_group = (dup_keys.index[dup_keys == dup_keys.iloc[0]].tolist()
                   if len(dup_keys) else [])
    exact_pair = take(sorted(first_group)[:2], 2)

    # Description gaps, split by lookup resolvability.
    norm, _ = normalize_text(df, schema)
    lookup = build_description_lookup(norm)
    gaps = df[df["Description"].isna()].index.tolist()
    gap_keys = (df.loc[gaps, "StockCode"].astype("string")
                .str.strip().str.upper().tolist())
    resolvable = [i for i, k in zip(gaps, gap_keys) if k in lookup]
    unresolvable = [i for i, k in zip(gaps, gap_keys) if k not in lookup]
    null_res = take(resolvable, 2)
    null_unres = take(unresolvable, 1)

    # Country aliases (raw spellings).
    alias_raw = {k for k in
                 _country_alias_map(schema) if k in
                 set(df["Country"].astype("string").str.strip().str.title())}
    alias_rows = [take(df[df["Country"].astype("string").str.strip()
                          .str.title() == a].index.tolist(), 1)
                  for a in sorted(alias_raw)]
    alias_rows = [i for sub in alias_rows for i in sub][:2]

    # Padded description + lowercase StockCode.
    desc = df["Description"].astype("string")
    padded = take(desc[desc.notna()
                       & (desc != desc.str.strip())].index.tolist(), 1)
    sc = df["StockCode"].astype("string")
    lower_sc = take(sc[sc.notna()
                       & (sc != sc.str.upper())].index.tolist(), 1)

    # Fuzzy typo pair: first near-duplicate group on the full frame, one
    # row per distinct spelling (typo vs canonical twin).
    near = detect_near_duplicates(norm)
    fuzzy_pair = (take(list(near[0]["representatives"].values())[:2], 2)
                  if near else [])

    # Guest-order null (CustomerID missing, not already picked).
    guest = take(df[df["CustomerID"].isna()].index.tolist(), 1)

    # Clean control row: no defect, not picked.
    defective = set(df[df.duplicated(keep=False)].index) | set(gaps)
    defective |= set(desc[desc.notna()
                          & (desc != desc.str.strip())].index)
    defective |= set(sc[sc.notna() & (sc != sc.str.upper())].index)
    defective |= {i for g in near for i in g["row_ids"]}
    defective |= set(df[df["CustomerID"].isna()].index)
    defective |= {i for i, v in zip(df.index, df["Country"].astype("string")
                                    .str.strip().str.title()) if v in alias_raw}
    clean = take([i for i in df.index if i not in defective], 1)

    ordered = sorted(set(exact_pair + null_res + null_unres + alias_rows
                         + padded + lower_sc + fuzzy_pair + guest + clean))
    roles = {
        "exact_pair": exact_pair, "null_resolvable": null_res,
        "null_unresolvable": null_unres, "country_alias": alias_rows,
        "padded": padded, "lower_stockcode": lower_sc,
        "fuzzy_pair": fuzzy_pair, "guest": guest, "clean": clean,
    }
    return ordered, roles


def build_sample_diff(df: pd.DataFrame, schema: dict,
                      knn_neighbors: int = KNN_N_NEIGHBORS,
                      fuzzy_threshold: float = FUZZY_THRESHOLD) -> dict:
    """Run the pipeline over the demo sample and record before/after pairs."""
    ordered, roles = select_demo_sample(df, schema)
    role_of = {i: r for r, ids in roles.items() for i in ids}

    norm_full, _ = normalize_text(df, schema)
    lookup = build_description_lookup(norm_full)
    full_global_mode = _global_mode_description(norm_full)

    raw_slice = df.loc[ordered].sort_index()
    norm_slice, norm_log = normalize_text(raw_slice, schema)
    imp_slice, imp_log = impute_missing(norm_slice, schema,
                                        strategy="statistical", lookup=lookup,
                                        global_mode=full_global_mode)
    imp_slice, knn_log = impute_missing(imp_slice, schema, strategy="knn",
                                        knn_neighbors=knn_neighbors)
    dup_mask, dup_log = detect_exact_duplicates(imp_slice)
    kept = imp_slice.loc[~dup_mask]
    near = detect_near_duplicates(kept, threshold=fuzzy_threshold)
    near_rows = {i: g["group_id"] for g in near for i in g["row_ids"]}
    knn_check = verify_knn_imputer(df, n_neighbors=knn_neighbors)

    # Which description rows were lookup- vs fallback-imputed.
    desc_gap = raw_slice["Description"].isna()
    gap_keys = (raw_slice.loc[desc_gap, "StockCode"].astype("string")
                .str.strip().str.upper())
    via_lookup = set(gap_keys[gap_keys.isin(lookup)].index)

    # Map each removed row to the kept twin it duplicates (sample scope).
    removed_ids = set(dup_mask.loc[dup_mask].index)
    twin_of: dict = {}
    for i in sorted(ordered):
        if bool(dup_mask.loc[i]):
            twin = [j for j in ordered
                    if j != i and j not in removed_ids
                    and _json_row(imp_slice.loc[j])
                    == _json_row(imp_slice.loc[i])]
            twin_of[i] = twin[0] if twin else None
    kept_note_for: dict = {}
    for removed, kept_id in twin_of.items():
        if kept_id is not None:
            kept_note_for.setdefault(kept_id, []).append(removed)

    records: list[dict] = []
    for i in sorted(ordered):
        before = _json_row(raw_slice.loc[i])
        operations: list[str] = []
        status = "unchanged"
        if bool(dup_mask.loc[i]):
            operations.append(
                "removed: exact duplicate"
                + (f" of row {twin_of[i]} (kept first)"
                   if twin_of[i] is not None else " (kept first occurrence)"))
            status = "removed-duplicate"
            after = None
        else:
            after = _json_row(kept.loc[i])
            # Text comparisons are string-coerced: int 85048 -> "85048" is
            # a storage cast, not a normalization fix.
            for col in ("StockCode", "Description"):
                if (before[col] is not None
                        and str(before[col]) != str(after[col])):
                    operations.append(
                        f"normalize:{col} strip+collapse+upper "
                        f"({before[col]!r} -> {after[col]!r})")
            if str(before["Country"]) != str(after["Country"]):
                operations.append(
                    f"normalize:Country title-case+alias "
                    f"({before['Country']!r} -> {after['Country']!r})")
            if before["Description"] is None and after["Description"] is not None:
                how = ("StockCode lookup" if i in via_lookup
                       else "global-mode fallback")
                operations.append(
                    f"impute:Description <- {how} "
                    f"({after['Description']!r})")
            if i in near_rows:
                operations.append(
                    f"flag:near-duplicate group {near_rows[i]} "
                    "(review - never auto-merged)")
            for r in kept_note_for.get(i, []):
                operations.append(
                    f"kept: first of exact-duplicate pair (row {r} removed)")
            if before["CustomerID"] is None:
                operations.append("preserve:null CustomerID "
                                  "(structural guest order, schema rv-4)")
            text_changed = any(str(before[k]) != str(after[k])
                               for k in before)
            if i in near_rows:
                status = "flagged-near-duplicate"
            elif before["Description"] is None and after["Description"] is not None:
                status = "imputed"
            elif text_changed:
                status = "normalized"
            elif before["CustomerID"] is None:
                status = "preserved"
        records.append({
            "row_id": int(i),
            "demo_role": role_of.get(i),
            "status": status,
            "before": before,
            "after": after,
            "operations": operations,
        })

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    _, full_dup_log = detect_exact_duplicates(
        impute_missing(norm_full, schema, strategy="statistical")[0])
    return {
        "module": MODULE_NAME,
        "artifact": "cleaning_sample_diff",
        "generated_at": datetime.now(timezone.utc).replace(
            microsecond=0).isoformat(),
        "source": {
            "input": str(DEFAULT_INPUT),
            "nrows_per_sheet": DEFAULT_NROWS_PER_SHEET,
            "n_rows": int(len(df)),
        },
        "imputation": {
            "statistical": ("Description <- StockCode lookup (mode per "
                            "StockCode, global-mode fallback); numerics <- "
                            "median; CustomerID nulls preserved (rv-4)"),
            "ml": {
                "method": "sklearn.impute.KNNImputer",
                "k": knn_neighbors,
                "weights": KNN_WEIGHTS,
                "features": list(KNN_FEATURES),
                "excluded": list(KNN_EXCLUDED),
                "choice_note": ("KNN chosen over regression/iterative "
                                "imputation: non-parametric, preserves "
                                "local structure, no distributional "
                                "assumptions, single-pass and fast. "
                                "CustomerID is excluded - imputing it "
                                "would fabricate customer identities."),
            },
            "ml_verification": knn_check,
        },
        "deduplication": {
            "exact_duplicates_removed_full_frame":
                int(full_dup_log["n_duplicate_rows"]),
            "policy_exact": "removed (keep first)",
            "policy_near": "flagged for review - never auto-merged",
        },
        "normalization": norm_log,
        "rows": records,
        "summary": {"n_rows": len(records), "by_status": by_status},
    }


# ---------------------------------------------------------------------------
# File-level API + CLI
# ---------------------------------------------------------------------------

def run_cleaning(
    input_path: str | Path = DEFAULT_INPUT,
    schema_path: str | Path = _DEFAULT_SCHEMA,
    output_path: str | Path = DEFAULT_DIFF,
    nrows_per_sheet: int = DEFAULT_NROWS_PER_SHEET,
    knn_neighbors: int = KNN_N_NEIGHBORS,
    fuzzy_threshold: float = FUZZY_THRESHOLD,
    full_out: str | Path | None = None,
) -> dict:
    """Load data + schema, build the sample diff (and optionally the fully
    cleaned dataset). Saves and returns the diff document."""
    df = load_dataset(input_path, nrows_per_sheet=nrows_per_sheet)
    schema = load_schema(schema_path)
    diff = build_sample_diff(df, schema, knn_neighbors=knn_neighbors,
                             fuzzy_threshold=fuzzy_threshold)
    diff["source"]["input"] = str(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diff, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    if full_out is not None:
        cleaned, log = clean_dataframe(df, schema,
                                       knn_neighbors=knn_neighbors,
                                       fuzzy_threshold=fuzzy_threshold)
        full_path = Path(full_out)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(full_path, index=False)
        log_path = full_path.with_suffix(".log.json")
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False,
                                       default=str), encoding="utf-8")
        diff["full_output"] = {"cleaned_csv": str(full_path),
                               "cleaning_log": str(log_path),
                               "n_rows_out": int(len(cleaned))}
        out.write_text(json.dumps(diff, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    return diff


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 5: Module 2 cleaning pipeline - normalize, "
                    "impute (statistical + KNN), dedupe (exact + fuzzy); "
                    "writes cleaning_sample_diff.json.")
    p.add_argument("--input", default=str(DEFAULT_INPUT),
                   help="Path to online_retail_II.xlsx")
    p.add_argument("--schema", default=str(_DEFAULT_SCHEMA),
                   help="Path to Sprint 4's expected_schema.json")
    p.add_argument("--output", default=str(DEFAULT_DIFF),
                   help="Where to save cleaning_sample_diff.json")
    p.add_argument("--nrows-per-sheet", type=int,
                   default=DEFAULT_NROWS_PER_SHEET,
                   help="Max rows read per sheet (0 = all rows).")
    p.add_argument("--knn-neighbors", type=int, default=KNN_N_NEIGHBORS,
                   help="KNNImputer neighbourhood size.")
    p.add_argument("--fuzzy-threshold", type=float, default=FUZZY_THRESHOLD,
                   help="Min difflib ratio for near-duplicate pairs.")
    p.add_argument("--full-out", default=None,
                   help="Optional: also write the fully cleaned CSV here.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    diff = run_cleaning(args.input, args.schema, args.output,
                        args.nrows_per_sheet, args.knn_neighbors,
                        args.fuzzy_threshold, args.full_out)
    summary = {
        "module": diff["module"],
        "n_rows": diff["summary"]["n_rows"],
        "by_status": diff["summary"]["by_status"],
        "ml_verification": diff["imputation"]["ml_verification"],
        "exact_duplicates_removed_full_frame": diff["deduplication"][
            "exact_duplicates_removed_full_frame"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nCleaning sample diff -> {args.output}")
    return diff


if __name__ == "__main__":
    main()

