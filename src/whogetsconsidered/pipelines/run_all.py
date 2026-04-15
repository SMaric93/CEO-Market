"""Master pipeline driver that runs enabled stages in dependency order."""

from __future__ import annotations

import logging

from whogetsconsidered.config import WhoGetsConsideredConfig
from whogetsconsidered.outputs.figures import make_figures
from whogetsconsidered.outputs.tables import make_tables
from whogetsconsidered.pipelines.build_base_panel import build_base_panel
from whogetsconsidered.pipelines.build_candidate_sets import build_candidate_sets
from whogetsconsidered.pipelines.build_release_shocks import build_release_shocks
from whogetsconsidered.pipelines.build_succession_panel import build_succession_panel
from whogetsconsidered.pipelines.estimate_main_results import estimate_choice, estimate_iv, estimate_main
from whogetsconsidered.pipelines.score_fit import score_fit
from whogetsconsidered.pipelines.validate_inputs import validate_inputs_pipeline


def run_all(config: WhoGetsConsideredConfig, *, logger: logging.Logger) -> None:
    """Run the full enabled research pipeline."""

    validate_inputs_pipeline(config, logger=logger)
    build_base_panel(config, logger=logger)
    build_succession_panel(config, logger=logger)
    build_release_shocks(config, logger=logger)
    build_candidate_sets(config, logger=logger)
    score_fit(config, logger=logger)
    estimate_main(config, logger=logger)
    if config.features.iv_enabled:
        estimate_iv(config, logger=logger)
    if config.features.conditional_logit_enabled:
        estimate_choice(config, logger=logger)
    make_tables(config, logger=logger)
    make_figures(config, logger=logger)
