# Package specification

## Design goal

The package is built around the idea that boards hire from a feasible set rather than a frictionless national market. The code therefore starts with data objects that are straightforward to build from cleaned extracts and only then layers on optional enrichments.

## Layout

- `src/whogetsconsidered/config.py`: typed YAML config
- `src/whogetsconsidered/schemas/`: explicit raw and canonical table schemas
- `src/whogetsconsidered/io/`: validated readers, artifact registry, and writers
- `src/whogetsconsidered/wrds/`: optional WRDS extraction, staging, and canonicalization
- `src/whogetsconsidered/executives/`: title parsing, CEO-ready logic, and person-year construction
- `src/whogetsconsidered/succession/`: CEO identification, succession detection, classification, and internal bench
- `src/whogetsconsidered/shocks/`: release shocks and noncompete/travel-time hooks
- `src/whogetsconsidered/candidates/`: candidate universe, accessible set, and pool metrics
- `src/whogetsconsidered/fit/`: task-alignment and optional predictive fit
- `src/whogetsconsidered/models/`: reduced-form, validation, IV, and choice estimators
- `src/whogetsconsidered/outputs/`: tables, figures, LaTeX, and JSON rendering
- `src/whogetsconsidered/pipelines/`: stage-oriented orchestration used by the CLI

## Artifact philosophy

Every major stage writes a parquet artifact plus a JSON metadata sidecar under `artifacts/`. The sidecar includes:

- row and column counts
- lineage notes for derived variables
- stage-level metadata used for reproducibility

## Pipeline stages

1. `validate-inputs`
   Checks required columns and data types before any transformations run.
2. `pull-wrds`
   Optional bootstrap stage that pulls project-relevant WRDS tables, stages them as parquet,
   and writes a manifest explaining which canonical inputs were successfully materialized.
3. `build-base-panel`
   Builds `firm_year_panel`, `executive_year_panel`, and `ceo_year_panel`.
4. `build-succession-panel`
   Builds `succession_events` and `internal_bench`.
5. `build-release-shocks`
   Builds `released_candidates` and `release_supply_metrics`.
6. `build-candidate-sets`
   Builds `candidate_universe` and `accessible_candidate_set`.
7. `score-fit`
   Scores transparent task alignment and optional predictive fit.
8. `estimate-main`
   Writes the event analysis panel and reduced-form results.
9. `estimate-iv`
   Runs 2SLS or a documented fallback if the synthetic sample is rank-deficient.
10. `estimate-choice`
   Runs candidate-choice estimation with conditional-logit-first logic.
11. `make-tables` and `make-figures`
   Render paper-facing outputs in multiple formats.

## Feasibility tiers

High-feasibility modules are on by default:

- CRI title parsing and CEO identification
- succession detection
- HQ market assignment
- internal bench
- released-candidate construction
- generic and relevant supply
- task-alignment fit
- reduced-form regressions

Medium-feasibility modules are implemented behind flags or separate commands:

- predictive cross-fitted fit
- BoardEx `known_to_board` enrichment
- IV estimation
- candidate choice estimation

Low-feasibility extensions remain optional entry points and do not block the default pipeline:

- travel-time shocks
- BLM bridge
- search-firm/private-firm enrichments
