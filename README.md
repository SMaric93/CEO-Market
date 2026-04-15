# whogetsconsidered

`whogetsconsidered` is an installable Python package for studying CEO succession under search frictions, candidate access constraints, and local executive-release shocks. The package is designed for empirical corporate-finance and strategy workflows that use cleaned CSV or Parquet extracts rather than direct proprietary APIs.

An optional WRDS bootstrap layer is also included for users who want to pull project-relevant
WRDS tables directly and materialize as many canonical inputs as WRDS can support.

The empirical backbone is intentionally high-feasibility:

- CEO identification from CRI-style executive panels
- succession-event construction from consecutive CEO changes
- local market access via MSA or distance-based geography
- released-candidate supply shocks from clean source-firm events
- internal bench measures from the focal firm's pre-event executive team
- transparent task-alignment fit scoring
- reduced-form validation, successor-origin, fit, and performance regressions

## What the package does

- Validates canonical input tables with explicit schema checks.
- Builds reproducible firm, executive, CEO, succession, release-shock, and candidate-set artifacts.
- Estimates reduced-form, validation, fit, IV, and choice-model specifications from typed config objects.
- Writes parquet intermediates plus CSV, JSON, LaTeX, and figure outputs under `artifacts/`.

## Installation

```bash
pip install -e .[dev]
```

## Quick start

```bash
wgc validate-inputs --config examples/minimal_config.yml
wgc run-all --config examples/minimal_config.yml
```

## Optional WRDS bootstrap

```bash
pip install -e .[dev,wrds]
wgc pull-wrds --config examples/wrds_bootstrap_config.yml
```

The WRDS stage writes staged source pulls under `artifacts/wrds/raw/`, canonicalized extracts
under `artifacts/wrds/canonical/`, and a manifest at `artifacts/wrds/manifest.json` showing
which package inputs were fully materialized versus which still require non-WRDS sources.

The synthetic configuration writes:

- parquet artifacts under `artifacts/`
- regression CSV/JSON bundles under `artifacts/outputs/results/`
- table CSV/JSON/LaTeX bundles under `artifacts/outputs/tables/`
- figure PNGs under `artifacts/outputs/figures/`

## Documentation

- [Package spec](docs/package_spec.md)
- [Data dictionary](docs/data_dictionary.md)
- [Estimation guide](docs/estimation_guide.md)
- [Identification guide](docs/identification_guide.md)
- [CLI examples](docs/cli_examples.md)
- [WRDS ingestion guide](docs/wrds_ingestion.md)
