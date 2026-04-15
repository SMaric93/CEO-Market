# Data dictionary

## Required raw inputs

### `cri_exec_panel`

- `gvkey`: focal firm identifier
- `fyear`: fiscal year of the observation
- `exec_name_raw`: raw person name from the filing extract
- `title_raw`: raw executive title
- `exec_rank`: rank in the top team when available
- `filing_date`: filing date used for timing sanity checks

### `compustat_firm_year`

- `gvkey`, `fyear`, `sic`
- core balance-sheet and market variables used to build lagged controls
- `datadate`: accounting date anchor

### `hq_history`

- time-varying headquarters location with latitude, longitude, and optional MSA

### `release_events`

- clean source-firm events used to generate local supply shocks
- `clean_release_flag` is required; event tiers are parsed from `event_type`

### `noncompete_state_year`

- state-year noncompete score merged into lagged focal-firm controls

### `ff_industry_map`

- SIC to Fama-French mappings used for controls, residualization, and event-level fixed effects

## Core canonical artifacts

### `firm_year_panel`

One row per firm-year. Key derived variables:

- `tobin_q_raw = (prcc_f * csho + dltt + dlc) / at`
- `roa_raw = ebit / at`
- `tobin_q_resid`, `roa_resid`: residualized outcomes according to config
- `log_assets`, `rd_intensity`, `capital_intensity`, `leverage`
- `market_size_public_firms`, `same_industry_local_density`
- `noncompete_score`

### `executive_year_panel`

One row per person-year-firm. Key derived variables:

- `normalized_name`
- `person_id`
- title flags: `is_ceo`, `is_president`, `is_coo`, `is_cfo`, `is_interim`, `is_chair`
- `is_ceo_ready`, `is_ceo_ready_robust`
- `firm_tenure_years`

### `ceo_year_panel`

One row per firm-year with the selected CEO under the configured rule.

### `succession_events`

One row per CEO change event. Key fields:

- `outsider_flag`
- `local_external_flag`
- `interim_flag`
- `source_firm_gvkey`
- `source_hq_distance_km`
- `source_market_relation`

### `internal_bench`

Event-level t-1 successor-capacity measures:

- `has_president_tminus1`
- `has_coo_tminus1`
- `has_cfo_tminus1`
- `num_ceo_ready_insiders_tminus1`
- `avg_insider_tenure_tminus1`
- `heir_apparent_proxy_tminus1`

### `released_candidates`

Released CEO-ready candidates from clean source-firm events:

- `candidate_person_id`
- `source_gvkey`
- `release_year`
- `release_tier`
- `baseline_release_eligibility_flag`
- `main_sample_release_flag`

### `candidate_universe`

CEO-ready candidate-year observations with cumulative public-market features:

- prior CEO experience
- number of firms and industries
- mover indicator
- average prior firm quality and complexity
- `portable_quality_score`

### `accessible_candidate_set`

Candidate-event panel containing the feasible set:

- `internal_flag`
- `released_flag`
- `known_to_board_flag`
- `distance_km`
- `task_alignment_fit_score`
- `predictive_fit_score`
- `chosen_flag`

### `event_analysis_panel`

Event-level analysis panel used by the reduced forms:

- lagged controls at `t-1`
- `generic_supply`
- `RelevantSupply_e`
- realized fit and fit-gap metrics
- post-succession outcomes for horizons 1, 2, and 3
