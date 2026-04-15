"""Run the full synthetic pipeline from Python."""

from whogetsconsidered.config import load_config
from whogetsconsidered.logging_utils import configure_logging
from whogetsconsidered.pipelines.run_all import run_all


def main() -> None:
    logger = configure_logging()
    config = load_config("examples/minimal_config.yml")
    run_all(config, logger=logger)


if __name__ == "__main__":
    main()
