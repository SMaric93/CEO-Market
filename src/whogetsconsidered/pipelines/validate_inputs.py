"""Schema validation stage for canonical cleaned input data."""

from __future__ import annotations

import logging

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.constants import CORE_INPUT_TABLES, OPTIONAL_INPUT_TABLES
from whogetsconsidered.io.readers import read_input_table
from whogetsconsidered.logging_utils import log_stage


def validate_inputs_pipeline(
    config: WhoGetsConsideredConfig,
    *,
    logger: logging.Logger,
) -> None:
    """Validate all required inputs plus enabled optional inputs."""

    with log_stage(logger, "validate-inputs"):
        for name in CORE_INPUT_TABLES:
            read_input_table(name, getattr(config.inputs, name))
        for name in OPTIONAL_INPUT_TABLES:
            path = getattr(config.inputs, name)
            if path is not None:
                read_input_table(name, path)
