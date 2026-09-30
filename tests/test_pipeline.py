"""Sprint 10 - tests for the Module 4 pipeline (synthetic, self-contained).

The end-to-end test builds a small synthetic workbook, points a config at
a tmp dir, and runs the real orchestrator through all three module APIs.
"""

import numpy as np
import pandas as pd
import pytest
import yaml
import json

from module4_pipeline import interfaces, orchestrator
from module4_pipeline.logger import get_pipeline_logger


def _synthetic_workbook(path, n: int = 200, seed: int = 11) -> None:
    """Small realistic workbook: unique keys, both full years, valid values."""
    rng = np.random.default_rng(seed)
    half = n // 2
    df = pd.DataFrame({
        "InvoiceNo": [str(489434 + i) for i in range(n)],
        "StockCode": ["85048" if i % 5 else "79323P" for i in range(n)],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"
                        if i % 3 else "PINK CHERRY LIGHTS" for i in range(n)],
        "Quantity": rng.integers(1, 13, size=n),
        "InvoiceDate": ([pd.Timestamp("2009-12-01 07:45:00")] * half
                        + [pd.Timestamp("2010-06-03 12:00:00")] * (n - half)),
        "UnitPrice": np.round(rng.uniform(1.0, 10.0, size=n), 2),
        "CustomerID": [13085.0 if i % 3 else np.nan for i in range(n)],
        "Country": ["United Kingdom" if i % 4 else "France"
                    for i in range(n)],
    })
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Year 2009-2010")


def _test_config(tmp_path, xlsx) -> dict:
    return {
        "paths": {
            "input_xlsx": str(xlsx),
            "expected_schema": str(
                interfaces._repo_root()
                / "module2_cleaning" / "expected_schema.json"),
            "profiling_report": str(tmp_path / "profiling_report.json"),
            "visuals_dir": str(tmp_path / "visuals"),
            "cleaned_csv": str(tmp_path / "cleaned_data.csv"),
            "cleaning_log": str(tmp_path / "cleaning_log.json"),
            "validation_report": str(tmp_path / "validation_report.json"),
            "pipeline_log": str(tmp_path / "pipeline.log"),
        },
        "profiling": {"nrows_per_sheet": 0, "heatmap_rows": 50},
        "cleaning": {"imputation_method": "statistical+knn",
                     "knn_neighbors": 2, "fuzzy_threshold": 0.90},
        "validation": {"contamination": 0.05, "classifier_sample_n": 100},
        "logging": {"level": "INFO", "console": False},
        "pipeline": {"stop_on_stage_failure": True},
    }


def _write_config(tmp_path, xlsx) -> str:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(_test_config(tmp_path, xlsx)),
                        encoding="utf-8")
    return str(cfg_path)


# ---------------------------------------------------------------------------
# Config + interfaces
# ---------------------------------------------------------------------------

def test_validate_config_rejects_bad_imputation_method(tmp_path):
    cfg = _test_config(tmp_path, tmp_path / "missing.xlsx")
    cfg["cleaning"]["imputation_method"] = "knn"
    with pytest.raises(ValueError, match="imputation_method"):
        interfaces.validate_config(cfg)


def test_validate_config_rejects_missing_section(tmp_path):
    cfg = _test_config(tmp_path, tmp_path / "missing.xlsx")
    del cfg["validation"]
    with pytest.raises(ValueError, match="validation"):
        interfaces.validate_config(cfg)


def test_preflight_fails_fast_on_missing_input(tmp_path):
    cfg = interfaces.validate_config(
        _test_config(tmp_path, tmp_path / "no-such-file.xlsx"))
    with pytest.raises(FileNotFoundError, match="input_xlsx"):
        interfaces.preflight(cfg)


def test_logger_writes_readable_lines(tmp_path):
    log_file = tmp_path / "test.log"
    log = get_pipeline_logger("test-pipeline", log_file=log_file,
                              level="INFO", console=False)
    log.info("hello stage=%s", "profiling")
    log.warning("careful n=%s", 3)
    text = log_file.read_text(encoding="utf-8")
    assert "hello stage=profiling" in text
    assert "WARNING" in text and "INFO" in text


# ---------------------------------------------------------------------------
# End to end (synthetic workbook through all three module APIs)
# ---------------------------------------------------------------------------

def test_orchestrator_dry_run_runs_no_stages(tmp_path):
    xlsx = tmp_path / "shop.xlsx"
    _synthetic_workbook(xlsx)
    summary = orchestrator.run_pipeline(_write_config(tmp_path, xlsx),
                                        dry_run=True)
    assert summary["overall"]["status"] == "dry-run"
    assert summary["stages"] == {}
    assert not (tmp_path / "cleaned_data.csv").exists()


def test_orchestrator_end_to_end_produces_all_module_outputs(tmp_path):
    xlsx = tmp_path / "shop.xlsx"
    _synthetic_workbook(xlsx)
    summary = orchestrator.run_pipeline(_write_config(tmp_path, xlsx))

    assert summary["overall"]["status"] == "succeeded"
    assert {s: v["status"] for s, v in summary["stages"].items()} == {
        "profiling": "success", "cleaning": "success",
        "validation": "success"}

    # All three module outputs exist in one execution, per the contracts.
    expected = {key: interfaces.STAGE_CONTRACTS[stage]["outputs"]
                for stage in interfaces.STAGES
                for key in interfaces.STAGE_CONTRACTS[stage]["outputs"]}
    assert set(expected) == {"profiling_report", "cleaned_csv",
                             "cleaning_log", "validation_report"}
    assert (tmp_path / "profiling_report.json").exists()
    assert (tmp_path / "cleaned_data.csv").exists()
    assert (tmp_path / "cleaning_log.json").exists()
    assert (tmp_path / "validation_report.json").exists()

    cleaned = pd.read_csv(tmp_path / "cleaned_data.csv")
    assert len(cleaned) > 0
    assert set(interfaces.EXPECTED_COLUMNS) <= set(cleaned.columns)

    # Cross-stage consistency: the three reports describe ONE run together.
    prof = json.loads((tmp_path / "profiling_report.json").read_text())
    clog = json.loads((tmp_path / "cleaning_log.json").read_text())
    vrep = json.loads((tmp_path / "validation_report.json").read_text())
    # Same input rows seen by profiling and cleaning ...
    assert clog["source"]["n_rows_in"] == prof["n_rows"] == 200
    # ... and cleaning's output rows are exactly what validation scored.
    assert vrep["source"]["n_rows"] == clog["source"]["n_rows_out"]
    assert vrep["source"]["n_rows"] == len(cleaned)
    # Each report carries its module's real content, not a stub.
    assert vrep["rules"]["defined"] == 20
    assert vrep["scores"]["health"]["grade"] in {"Healthy", "Watch", "Review"}
    assert clog["quality_after"]["total"] >= clog["quality_before"]["total"]
    assert vrep["drift"]["verdict"] in {"none", "watch", "alert"}

    log_text = (tmp_path / "pipeline.log").read_text(encoding="utf-8")
    for marker in ("START stage=profiling", "START stage=cleaning",
                   "START stage=validation", "PIPELINE SUCCEEDED"):
        assert marker in log_text
