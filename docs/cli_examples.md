# CLI examples

```bash
wgc pull-wrds --config examples/wrds_bootstrap_config.yml
wgc validate-inputs --config examples/minimal_config.yml
wgc build-base-panel --config examples/minimal_config.yml
wgc build-succession-panel --config examples/minimal_config.yml
wgc build-release-shocks --config examples/minimal_config.yml
wgc build-candidate-sets --config examples/minimal_config.yml
wgc score-fit --config examples/minimal_config.yml
wgc estimate-main --config examples/minimal_config.yml
wgc estimate-iv --config examples/minimal_config.yml
wgc estimate-choice --config examples/minimal_config.yml
wgc make-tables --config examples/minimal_config.yml
wgc make-figures --config examples/minimal_config.yml
wgc run-all --config examples/minimal_config.yml
```

## Typical workflow on user data

1. Map your cleaned extracts into the paths listed in `examples/minimal_config.yml`.
2. If you want to bootstrap available WRDS data first, run `wgc pull-wrds --config examples/wrds_bootstrap_config.yml` and inspect `artifacts/wrds/manifest.json`.
3. Run `wgc validate-inputs` once your canonical extracts are in place.
4. Run stages incrementally if you are debugging construction:
   - `build-base-panel`
   - `build-succession-panel`
   - `build-release-shocks`
   - `build-candidate-sets`
   - `score-fit`
5. Run `estimate-main` once the artifacts look correct.
6. Render paper-facing outputs with `make-tables` and `make-figures`.
7. Inspect the exact paper output bundle under:
   - `artifacts/panels/`
   - `artifacts/models/`
   - `artifacts/tables/`
   - `artifacts/figures/`
   - `artifacts/logs/`

## Optional modules

- Turn on `features.predictive_fit_enabled: true` to add cross-fitted predictive fit inside `score-fit`.
- Turn on `features.boardex_enabled: true` to populate `known_to_board_flag` in candidate sets.
- `estimate-iv` and `estimate-choice` can be run directly even when the default synthetic config keeps those flags off.
- Turn on `wrds.enabled: true` and fill the WRDS table specs if you want `pull-wrds` to stage WRDS data locally.
- Add `inputs.crsp_daily` and `inputs.ceo_announcement_dates` if you want announcement-window CARs populated instead of the placeholder Table 7 / figure notes.
