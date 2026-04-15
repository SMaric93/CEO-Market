"""Logging helpers used to make pipeline assumptions and stage timing explicit."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure package logging with a deterministic console format."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("whogetsconsidered")


@contextmanager
def log_stage(logger: logging.Logger, stage: str) -> Iterator[float]:
    """Log the start and end of a pipeline stage."""

    start = time.time()
    logger.info("starting stage=%s", stage)
    try:
        yield start
    finally:
        elapsed = time.time() - start
        logger.info("finished stage=%s elapsed_seconds=%.3f", stage, elapsed)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)
    return path
