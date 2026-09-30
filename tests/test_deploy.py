"""Sprint 11 - tests for CLI, container files and versioning setup."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import pipeline
from module4_pipeline import interfaces

ROOT = Path(__file__).resolve().parent.parent


def _mini_csv(path: Path, n: int = 200, seed: int = 5) -> None:
    """Small realistic CSV input (canonical headers)."""
    rng = np.random.default_rng(seed)
    half = n // 2
    pd.DataFrame({
        "InvoiceNo": [str(489434 + i) for i in range(n)],
        "StockCode": ["85048" if i % 5 else "79323P" for i in range(n)],
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER"
                        if i % 3 else "PINK CHERRY LIGHTS" for i in range(n)],
        "Quantity": rng.integers(1, 13, size=n),
        "InvoiceDate": (["2009-12-01 07:45:00"] * half
                        + ["2010-06-03 12:00:00"] * (n - half)),
        "UnitPrice": np.round(rng.uniform(1.0, 10.0, size=n), 2),
        "CustomerID": [13085.0 if i % 3 else np.nan for i in range(n)],
        "Country": ["United Kingdom" if i % 4 else "France"
                    for i in range(n)],
    }).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# pipeline.py helpers
# ---------------------------------------------------------------------------

def test_ensure_workbook_stages_csv_without_data_loss(tmp_path):
    src = tmp_path / "in.csv"
    _mini_csv(src)
    staged = pipeline.ensure_workbook(src, tmp_path / "staging")
    assert staged.suffix == ".xlsx" and staged.exists()
    back = pd.read_excel(staged, sheet_name="data")
    assert len(back) == 200
    assert set(interfaces.EXPECTED_COLUMNS) <= set(back.columns)


def test_ensure_workbook_uses_xlsx_in_place_and_rejects(tmp_path):
    xlsx = tmp_path / "in.xlsx"
    _mini_csv(tmp_path / "in.csv")
    pd.read_csv(tmp_path / "in.csv").to_excel(xlsx, index=False)
    assert pipeline.ensure_workbook(xlsx, tmp_path / "staging") == xlsx
    bad = tmp_path / "in.txt"
    bad.write_text("nope", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input"):
        pipeline.ensure_workbook(bad, tmp_path / "staging")
    with pytest.raises(FileNotFoundError):
        pipeline.ensure_workbook(tmp_path / "ghost.csv",
                                 tmp_path / "staging")


def test_build_run_config_redirects_every_output(tmp_path):
    base = interfaces.load_config(ROOT / "module4_pipeline" / "config.yaml")
    cfg = interfaces.validate_config(
        pipeline.build_run_config(base, tmp_path / "in.xlsx",
                                  tmp_path / "results"))
    for key in pipeline.OUTPUT_FILENAMES:
        assert Path(cfg["paths"][key]).parent == tmp_path / "results"
    # The versioned schema contract still resolves to the repo file.
    assert cfg["paths"]["expected_schema"].endswith("expected_schema.json")


# ---------------------------------------------------------------------------
# Real CLI: `python pipeline.py --input data.csv --output results/`
# ---------------------------------------------------------------------------

def test_cli_end_to_end_csv_to_results_dir(tmp_path):
    src = tmp_path / "data.csv"
    out = tmp_path / "results"
    _mini_csv(src)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "pipeline.py"),
         "--input", str(src), "--output", str(out)],
        capture_output=True, text=True, cwd=str(ROOT), timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "PIPELINE SUCCEEDED" in proc.stdout
    for name in ("profiling_report.json", "cleaned_data.csv",
                 "cleaning_log.json", "validation_report.json",
                 "pipeline.log", "run_config.yaml"):
        assert (out / name).exists(), name
    assert (out / "visuals").is_dir()
    cleaned = pd.read_csv(out / "cleaned_data.csv")
    assert len(cleaned) > 0


# ---------------------------------------------------------------------------
# Container + versioning setup files
# ---------------------------------------------------------------------------

def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_requirements_pins_direct_dependencies():
    lines = {l.split("==")[0].strip().lower(): l for l in
             _read("requirements.txt").splitlines()
             if l.strip() and not l.startswith("#")}
    for dep in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
                "openpyxl", "pyyaml", "pytest"):
        assert dep in lines, dep
        assert "==" in lines[dep]  # pinned, not floating


def test_dockerfile_runs_pipeline_as_entrypoint():
    text = _read("Dockerfile")
    assert "pip install" in text and "requirements.txt" in text
    assert 'ENTRYPOINT ["python", "pipeline.py"]' in text
    assert "pipeline.py" in text and "module4_pipeline" in text
    # Data enters via --input mount: no COPY of datasets into the image.
    assert not any(line.strip().startswith("COPY")
                   and ("xlsx" in line or ".csv" in line)
                   for line in text.splitlines())


def test_git_lfs_tracks_dataset_and_outputs():
    attrs = _read(".gitattributes")
    assert "*.xlsx filter=lfs" in attrs
    assert "module2_cleaning/cleaned_data.csv filter=lfs" in attrs
    assert "*.png filter=lfs" in attrs
    gitignore = _read(".gitignore")
    # LFS-tracked files must be committable: git must NOT ignore them.
    for path in ("online_retail_II.xlsx",
                 "module2_cleaning/cleaned_data.csv"):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", path], cwd=str(ROOT))
        assert proc.returncode != 0, f"{path} is git-ignored"
