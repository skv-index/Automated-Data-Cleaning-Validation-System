"""Sprint 10 - Pipeline logging (Module 4: Pipeline).

One factory for the whole system: a stage-scoped logger writing to both
the config-chosen log file and (optionally) the console, plus a
``StageTimer`` context manager that brackets each stage with START/DONE
(or FAILED) lines including elapsed seconds. Repeat calls are idempotent -
handlers are rebuilt, never duplicated.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def get_pipeline_logger(name: str = "pipeline",
                        log_file: str | Path | None = None,
                        level: str = "INFO",
                        console: bool = True) -> logging.Logger:
    """Build (or rebuild) the pipeline logger.

    Args:
        name: Logger name; stage loggers use ``f"pipeline.{stage}"``.
        log_file: File the run is recorded to (parent dirs created).
            ``None`` means console only.
        level: One of DEBUG/INFO/WARNING/ERROR (validated upstream).
        console: Also emit to stdout.

    Returns:
        The configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


@contextmanager
def stage_timer(logger: logging.Logger, stage: str):
    """Log ``START <stage>`` on entry and ``DONE``/``FAILED`` + seconds."""
    logger.info("START stage=%s", stage)
    started = time.perf_counter()
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.error("FAILED stage=%s elapsed_s=%.1f error=%r",
                     stage, elapsed, exc)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("DONE stage=%s elapsed_s=%.1f", stage, elapsed)
