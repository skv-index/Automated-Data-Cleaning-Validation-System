"""Sprint 11 - production CLI for the data-cleaning pipeline.

Runnable exactly as ``python pipeline.py --input data.csv --output results/``:

  * ``--input`` accepts a workbook (``.xlsx``/``.xls``, used in place) or a
    CSV (converted to a staged workbook first, because Modules 1-2 read
    workbooks - see ``ensure_workbook``).
  * ``--output`` is a directory; EVERY file artifact (profiling report,
    cleaned CSV, both logs, validation report, visuals, pipeline log) is
    redirected under it, and the merged ``run_config.yaml`` that drove the
    run is saved alongside for provenance.

The run itself is delegated to ``module4_pipeline.orchestrator`` - this
file only maps CLI flags onto a config, so local and container runs share
one code path (the Dockerfile sets this file as ENTRYPOINT).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

from module4_pipeline.interfaces import load_config, validate_config
from module4_pipeline.orchestrator import run_pipeline

WORKBOOK_SUFFIXES = (".xlsx", ".xls")
CSV_SUFFIXES = (".csv",)
STAGING_DIRNAME = "_input_staging"
STAGED_WORKBOOK_NAME = "input.xlsx"
RUN_CONFIG_NAME = "run_config.yaml"

#: Output-path keys redirected under ``--output`` and their filenames.
OUTPUT_FILENAMES = {
    "profiling_report": "profiling_report.json",
    "visuals_dir": "visuals",
    "cleaned_csv": "cleaned_data.csv",
    "cleaning_log": "cleaning_log.json",
    "validation_report": "validation_report.json",
    "pipeline_log": "pipeline.log",
}


def ensure_workbook(input_path: str | Path, staging_dir: str | Path) -> Path:
    """Resolve ``--input`` to a workbook Modules 1-2 can read.

    Workbooks are used in place (no multi-GB copy); CSVs are converted to
    ``<staging_dir>/input.xlsx`` (single ``data`` sheet). Raises
    ``FileNotFoundError``/``ValueError`` with actionable messages.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    suffix = src.suffix.lower()
    if suffix in WORKBOOK_SUFFIXES:
        return src
    if suffix in CSV_SUFFIXES:
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        dest = staging / STAGED_WORKBOOK_NAME
        df = pd.read_csv(src)
        if df.empty:
            raise ValueError(f"Input CSV has no data rows: {src}")
        with pd.ExcelWriter(dest, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="data")
        print(f"Staged CSV -> workbook: {src} ({len(df)} rows) -> {dest}")
        return dest
    raise ValueError(
        f"Unsupported input type {src.suffix!r}: use one of "
        f"{list(WORKBOOK_SUFFIXES + CSV_SUFFIXES)}")


def build_run_config(base_cfg: dict, input_xlsx: str | Path,
                     output_dir: str | Path) -> dict:
    """Merge base config with CLI paths: input workbook + output redirection.

    Returns an UNVALIDATED dict (same shape as the YAML); callers pass it
    through ``interfaces.validate_config`` before running.
    """
    out = Path(output_dir)
    cfg = {section: dict(values)
           for section, values in base_cfg.items()}
    cfg["paths"] = dict(base_cfg.get("paths", {}))
    cfg["paths"]["input_xlsx"] = str(Path(input_xlsx))
    for key, filename in OUTPUT_FILENAMES.items():
        cfg["paths"][key] = str(out / filename)
    return cfg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run profiling -> cleaning -> validation end to end. "
                    "Example: python pipeline.py --input data.csv "
                    "--output results/")
    p.add_argument("--input", required=True,
                   help="Input dataset (.csv, .xlsx or .xls).")
    p.add_argument("--output", required=True,
                   help="Output directory for ALL artifacts.")
    p.add_argument("--config",
                   default=str(Path(__file__).resolve().parent
                               / "module4_pipeline" / "config.yaml"),
                   help="Base YAML config (CLI paths override it).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + preflight only; run no stages.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    out = Path(args.output)
    if out.exists() and not out.is_dir():
        raise ValueError(f"--output must be a directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    input_xlsx = ensure_workbook(args.input, out / STAGING_DIRNAME)
    run_cfg = build_run_config(load_config(args.config), input_xlsx, out)
    cfg = validate_config(run_cfg)

    run_config_path = out / RUN_CONFIG_NAME
    with run_config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({k: (v if k != "paths" else
                            {pk: str(pv) for pk, pv in v.items()})
                        for k, v in cfg.items()},
                       fh, sort_keys=False)

    summary = run_pipeline(str(run_config_path), dry_run=args.dry_run)
    print(f"\nResults -> {out.resolve()}")
    print(f"Run config -> {run_config_path.resolve()}")
    return summary


if __name__ == "__main__":
    main()
