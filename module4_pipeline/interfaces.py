"""Sprint 10 - Pipeline interfaces (Module 4: Pipeline).

Single source of truth for how data flows between modules: stage
contracts (which function runs, which config keys feed it, which
artifacts it must produce), config loading/validation, and preflight
checks. The orchestrator executes; this module defines the shape so a
misconfigured run fails fast with a clear message instead of dying
mid-pipeline.

Data flow (all handoffs are files, so each module output is inspectable)::

    online_retail_II.xlsx
        -> [profiling]   -> profiling_report.json (+ visuals/*.png)
        -> [cleaning]    -> cleaned_data.csv + cleaning_log.json
                             (reads expected_schema.json, the versioned
                              Module 2 contract - an input, not a stage output)
        -> [validation]  -> validation_report.json
                             (reads cleaned_data.csv)
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


#: Ordered pipeline stages. Outputs of one are inputs of the next.
STAGES = ("profiling", "cleaning", "validation")

#: Stage contracts: runnable entry point, config keys consumed, artifacts
#: the stage must leave behind (keys into the ``paths`` config section).
STAGE_CONTRACTS = {
    "profiling": {
        "runs": "module1_profiling.profiling_api:run_profiling",
        "params": ["input_xlsx", "profiling_report", "nrows_per_sheet",
                   "visuals_dir", "heatmap_rows"],
        "outputs": ["profiling_report"],
        "description": "Profile the raw workbook -> profiling_report.json.",
    },
    "cleaning": {
        "runs": "module2_cleaning.cleaning_api:run_cleaning_pipeline",
        "params": ["input_xlsx", "expected_schema", "cleaned_csv",
                   "cleaning_log", "nrows_per_sheet", "imputation_method",
                   "knn_neighbors", "fuzzy_threshold"],
        "outputs": ["cleaned_csv", "cleaning_log"],
        "description": "Clean + transform -> cleaned_data.csv + cleaning_log.json.",
    },
    "validation": {
        "runs": "module3_validation.validation_api:run_validation_report",
        "params": ["cleaned_csv", "validation_report", "contamination",
                   "classifier_sample_n"],
        "outputs": ["validation_report"],
        "description": "Validate cleaned data -> validation_report.json.",
    },
}

#: Canonical columns every stage must preserve (Module 1 contract).
EXPECTED_COLUMNS = [
    "InvoiceNo", "StockCode", "Description", "Quantity",
    "InvoiceDate", "UnitPrice", "CustomerID", "Country",
]

#: Imputation pipelines the cleaner implements (Sprint 5 design: the
#: statistical lookup always runs first; KNNImputer follows and is a
#: verified no-op when the numeric columns have no gaps, as here).
SUPPORTED_IMPUTATION_METHODS = ("statistical+knn",)

#: Log levels accepted in config ``logging.level``.
SUPPORTED_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")

DEFAULT_CONFIG_PATH = _repo_root() / "module4_pipeline" / "config.yaml"


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read the YAML config file (raw dict, unresolved paths)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping at top level.")
    return cfg


def _need(cfg: dict, section: str, key: str, kind: type, errors: list[str]):
    section_cfg = cfg.get(section)
    if not isinstance(section_cfg, dict) or key not in section_cfg:
        errors.append(f"config[{section}][{key}] is required")
        return None
    value = section_cfg[key]
    if not isinstance(value, kind):
        errors.append(
            f"config[{section}][{key}] must be {kind.__name__}, "
            f"got {type(value).__name__}")
        return None
    return value


def validate_config(cfg: dict) -> dict:
    """Validate settings and resolve ``paths.*`` against the repo root.

    Returns a normalized config with absolute path strings. Raises
    ``ValueError`` listing every problem (fail fast, all at once).
    """
    errors: list[str] = []
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")

    for section in ("paths", "profiling", "cleaning", "validation",
                    "logging", "pipeline"):
        if section not in cfg or not isinstance(cfg[section], dict):
            errors.append(f"config[{section}] section is required")

    out: dict = {"paths": {}, "profiling": {}, "cleaning": {},
                 "validation": {}, "logging": {}, "pipeline": {}}
    if errors:
        raise ValueError("invalid config:\n- " + "\n- ".join(errors))

    root = _repo_root()
    for key in ("input_xlsx", "expected_schema", "profiling_report",
                "visuals_dir", "cleaned_csv", "cleaning_log",
                "validation_report", "pipeline_log"):
        raw = _need(cfg, "paths", key, str, errors)
        if raw is not None:
            p = Path(raw)
            out["paths"][key] = str(p if p.is_absolute() else root / p)

    profiling = cfg["profiling"]
    nrows = profiling.get("nrows_per_sheet", 50000)
    heatmap = profiling.get("heatmap_rows", 1000)
    for label, value in (("nrows_per_sheet", nrows), ("heatmap_rows", heatmap)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"config[profiling][{label}] must be a non-negative int")
    out["profiling"] = {"nrows_per_sheet": nrows, "heatmap_rows": heatmap}

    cleaning = cfg["cleaning"]
    method = cleaning.get("imputation_method", "statistical+knn")
    if method not in SUPPORTED_IMPUTATION_METHODS:
        errors.append(
            f"config[cleaning][imputation_method] must be one of "
            f"{list(SUPPORTED_IMPUTATION_METHODS)} (got {method!r})")
    knn = cleaning.get("knn_neighbors", 5)
    fuzzy = cleaning.get("fuzzy_threshold", 0.90)
    if not isinstance(knn, int) or isinstance(knn, bool) or knn < 1:
        errors.append("config[cleaning][knn_neighbors] must be a positive int")
    if (not isinstance(fuzzy, (int, float)) or isinstance(fuzzy, bool)
            or not 0 < fuzzy <= 1.0):
        errors.append("config[cleaning][fuzzy_threshold] must be in (0, 1]")
    out["cleaning"] = {"imputation_method": method, "knn_neighbors": knn,
                       "fuzzy_threshold": float(fuzzy)}

    validation = cfg["validation"]
    contam = validation.get("contamination", 0.01)
    sample_n = validation.get("classifier_sample_n", 5000)
    if (not isinstance(contam, (int, float)) or isinstance(contam, bool)
            or not 0 < contam < 1):
        errors.append("config[validation][contamination] must be in (0, 1)")
    if (not isinstance(sample_n, int) or isinstance(sample_n, bool)
            or sample_n < 1):
        errors.append("config[validation][classifier_sample_n] must be a "
                      "positive int")
    out["validation"] = {"contamination": float(contam),
                         "classifier_sample_n": sample_n}

    logging_cfg = cfg["logging"]
    level = str(logging_cfg.get("level", "INFO")).upper()
    if level not in SUPPORTED_LOG_LEVELS:
        errors.append(f"config[logging][level] must be one of "
                      f"{list(SUPPORTED_LOG_LEVELS)} (got {level!r})")
    out["logging"] = {"level": level,
                      "console": bool(logging_cfg.get("console", True))}

    pipeline = cfg["pipeline"]
    out["pipeline"] = {"stop_on_stage_failure":
                       bool(pipeline.get("stop_on_stage_failure", True))}

    if errors:
        raise ValueError("invalid config:\n- " + "\n- ".join(errors))
    return out


# ---------------------------------------------------------------------------
# Preflight + artifact checks
# ---------------------------------------------------------------------------

def check_artifacts_exist(paths: dict, keys: list[str]) -> list[str]:
    """Return the subset of ``keys`` whose ``paths`` entry is missing on disk."""
    return [k for k in keys if not Path(paths[k]).exists()]


def preflight(cfg: dict) -> dict:
    """Fail fast when stage inputs are absent.

    Returns ``{"input_xlsx": ..., "expected_schema": ...}`` (resolved
    strings) when both exist; raises ``FileNotFoundError`` naming the
    missing file and the stage that needs it.
    """
    paths = cfg["paths"]
    for key, stage in (("input_xlsx", "profiling+cleaning"),
                       ("expected_schema", "cleaning")):
        if not Path(paths[key]).exists():
            raise FileNotFoundError(
                f"Stage input missing for {stage}: {key} -> {paths[key]}")
    return {"input_xlsx": paths["input_xlsx"],
            "expected_schema": paths["expected_schema"]}
