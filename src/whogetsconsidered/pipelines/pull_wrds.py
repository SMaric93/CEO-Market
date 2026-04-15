"""Pipeline stage for optional WRDS data pulls."""

from __future__ import annotations

import logging

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.logging_utils import log_stage
from whogetsconsidered.wrds.puller import pull_wrds_bundle


def pull_wrds(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Pull project-relevant WRDS data and materialize canonical extracts where feasible."""

    with log_stage(logger, "pull-wrds"):
        pull_wrds_bundle(config, logger=logger)
