"""Sprint 10 - Pipeline orchestrator (Module 4: Pipeline).

Runs the whole system in one execution, driven entirely by
``config.yaml``: Module 1 profiling -> Module 2 cleaning -> Module 3
validation, passing each stage's output files as the next stage's inputs
(per ``interfaces.STAGE_CONTRACTS``). Every step is logged to the config
log file via ``logger.get_pipeline_logger``.

Usage:
    from module4_pipeline.orchestrator import run_pipeline

    summary = run_pipeline()  # defaults to module4_pipeline/config.yaml

CLI:
    python module4_pipeline/orchestrator.py
    python module4_pipeline/orchestrator.py --config module4_pipeline/config.yaml
    python module4_pipeline/orchestrator.py --config ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)


_ensure_repo_root_on_path()

from module1_profiling.profiling_api import run_profiling
from module2_cleaning.cleaning_api import run_cleaning_pipeline
from module3_validation.validation_api import run_validation_report

from module4_pipeline.interfaces import (DEFAULT_CONFIG_PATH, STAGES,
                                         STAGE_CONTRACTS, check_artifacts_exist,
                                         load_config, preflight,
                                         validate_config)
from module4_pipeline.logger import get_pipeline_logger, stage_timer

MODULE_NAME = "module4_pipeline"


# ---------------------------------------------------------------------------
# Stages (one thin wrapper per module API - no logic lives here)
# ---------------------------------------------------------------------------

def _stage_profiling(cfg: dict, log) -> dict:
    paths, prof = cfg["paths"], cfg["profiling"]
    report = run_profiling(
        input_path=paths["input_xlsx"],
        output_path=paths["profiling_report"],
        nrows_per_sheet=prof["nrows_per_sheet"],
        visuals_dir=paths["visuals_dir"],
        heatmap_rows=prof["heatmap_rows"],
    )
    flags = report.get("flags", {}).get("summary", {})
    log.info("profiling rows=%s columns=%s flags=%s",
             report.get("n_rows"), report.get("n_columns"), flags)
    if flags.get("n_critical", 0):
        log.warning("profiling raised %s critical flag(s) - see %s",
                    flags["n_critical"], paths["profiling_report"])
    return {"n_rows": report.get("n_rows"),
            "outputs": [paths["profiling_report"]]}


def _stage_cleaning(cfg: dict, log) -> dict:
    paths, prof, clean = cfg["paths"], cfg["profiling"], cfg["cleaning"]
    log.info("cleaning imputation_method=%s knn_neighbors=%s fuzzy_threshold=%s",
             clean["imputation_method"], clean["knn_neighbors"],
             clean["fuzzy_threshold"])
    _cleaned, clog = run_cleaning_pipeline(
        input_path=paths["input_xlsx"],
        schema_path=paths["expected_schema"],
        cleaned_path=paths["cleaned_csv"],
        log_path=paths["cleaning_log"],
        nrows_per_sheet=prof["nrows_per_sheet"],
        knn_neighbors=clean["knn_neighbors"],
        fuzzy_threshold=clean["fuzzy_threshold"],
    )
    log.info("cleaning rows_in=%s rows_out=%s quality_before=%s "
             "quality_after=%s delta=%s",
             clog["source"]["n_rows_in"], clog["source"]["n_rows_out"],
             clog["quality_before"]["total"], clog["quality_after"]["total"],
             clog["quality_delta"])
    return {"n_rows_out": clog["source"]["n_rows_out"],
            "outputs": [paths["cleaned_csv"], paths["cleaning_log"]]}


def _stage_validation(cfg: dict, log) -> dict:
    paths, val = cfg["paths"], cfg["validation"]
    report = run_validation_report(
        input_path=paths["cleaned_csv"],
        report_path=paths["validation_report"],
        contamination=val["contamination"],
        sample_n=val["classifier_sample_n"],
    )
    health = report["scores"]["health"]
    log.info("validation health_score=%s grade=%s error_severity=%s "
             "anomaly_severity=%s drift=%s",
             health["health_score"], health["grade"],
             report["scores"]["error"]["error_severity"],
             report["scores"]["anomaly"]["anomaly_severity"],
             report["drift"]["verdict"])
    if health["grade"] != "Healthy":
        log.warning("validation grade=%s - see %s",
                    health["grade"], paths["validation_report"])
    return {"health_score": health["health_score"],
            "grade": health["grade"],
            "outputs": [paths["validation_report"]]}


_STAGE_RUNNERS = {
    "profiling": _stage_profiling,
    "cleaning": _stage_cleaning,
    "validation": _stage_validation,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(config_path: str | Path = DEFAULT_CONFIG_PATH,
                 dry_run: bool = False) -> dict:
    """Run all stages in contract order from a config file.

    Args:
        config_path: YAML config (see ``config.yaml``).
        dry_run: Validate config + preflight only; run no stages.

    Returns:
        Summary dict with per-stage status, timings and key metrics.
        Raises on misconfiguration, missing inputs, or (when configured)
        the first stage failure.
    """
    started_all = time.perf_counter()
    cfg = validate_config(load_config(config_path))
    log = get_pipeline_logger(
        "pipeline",
        log_file=cfg["paths"]["pipeline_log"],
        level=cfg["logging"]["level"],
        console=cfg["logging"]["console"],
    )
    log.info("pipeline config=%s dry_run=%s", config_path, dry_run)
    log.info("paths=%s", {k: v for k, v in cfg["paths"].items()})

    inputs = preflight(cfg)  # fail fast before any heavy work
    log.info("preflight ok input_xlsx=%s expected_schema=%s",
             inputs["input_xlsx"], inputs["expected_schema"])
    if dry_run:
        log.info("dry run - no stages executed")
        return {"config": str(config_path), "dry_run": True,
                "stages": {}, "overall": {"status": "dry-run"}}

    summary: dict = {"config": str(config_path), "dry_run": False,
                     "started_at": datetime.now(timezone.utc).replace(
                         microsecond=0).isoformat(), "stages": {}}
    failed: list[str] = []
    for stage in STAGES:
        stage_log = get_pipeline_logger(
            f"pipeline.{stage}",
            log_file=cfg["paths"]["pipeline_log"],
            level=cfg["logging"]["level"], console=False)
        stage_started = time.perf_counter()
        try:
            with stage_timer(stage_log, stage):
                detail = _STAGE_RUNNERS[stage](cfg, stage_log)
        except Exception:
            log.error("stage %s failed", stage)
            summary["stages"][stage] = {
                "status": "failed",
                "elapsed_s": round(time.perf_counter() - stage_started, 1)}
            failed.append(stage)
            if cfg["pipeline"]["stop_on_stage_failure"]:
                break
            continue
        missing = check_artifacts_exist(
            cfg["paths"], STAGE_CONTRACTS[stage]["outputs"])
        status = "success" if not missing else "success-with-missing-output"
        if missing:
            log.warning("stage %s missing expected output(s): %s",
                        stage, missing)
        summary["stages"][stage] = {
            "status": status,
            "elapsed_s": round(time.perf_counter() - stage_started, 1),
            **detail}
        log.info("stage %s status=%s", stage, status)

    overall = "succeeded" if not failed else "failed"
    summary["overall"] = {"status": overall,
                          "elapsed_s": round(time.perf_counter()
                                             - started_all, 1),
                          "failed_stages": failed}
    log.info("PIPELINE %s elapsed_s=%.1f",
             overall.upper(), summary["overall"]["elapsed_s"])
    if failed:
        raise RuntimeError(f"Pipeline failed at stage(s): {failed}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sprint 10: run profiling -> cleaning -> validation "
                    "from config.yaml.")
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                   help="Path to the pipeline YAML config.")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate config + preflight only; run no stages.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    summary = run_pipeline(args.config, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
