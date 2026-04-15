# WRDS ingestion guide

`whogetsconsidered` can bootstrap a subset of the required research inputs directly from
WRDS. This layer is optional and is intentionally explicit about the boundary between:

- inputs that WRDS can supply directly,
- inputs that WRDS can proxy imperfectly,
- inputs that still need non-WRDS sources.

## Installation

```bash
pip install -e .[dev,wrds]
```

## CLI

```bash
wgc pull-wrds --config examples/wrds_bootstrap_config.yml
```

The stage writes:

- raw staged pulls to `artifacts/wrds/raw/`
- canonicalized extracts to `artifacts/wrds/canonical/`
- a manifest to `artifacts/wrds/manifest.json`

For non-interactive auth, set `wrds.username` and `wrds.pgpass_file` in your config.
The installed `wrds` client expects PostgreSQL-style credentials with:

- host: `wrds-pgdata.wharton.upenn.edu`
- port: `9737`
- database: `wrds`

So the credential line is:

```text
wrds-pgdata.wharton.upenn.edu:9737:wrds:YOUR_WRDS_USERNAME:YOUR_WRDS_PASSWORD
```

Create the file with owner-only permissions before running the pull:

```bash
mkdir -p .secrets
chmod 700 .secrets
printf 'wrds-pgdata.wharton.upenn.edu:9737:wrds:%s:%s\n' 'YOUR_WRDS_USERNAME' 'YOUR_WRDS_PASSWORD' > .secrets/wrds.pgpass
chmod 600 .secrets/wrds.pgpass
```

## What the WRDS stage can materialize

By default the stage is configured to try:

- `compustat_firm_year` from `comp.funda` plus `comp.company`
- `cri_exec_panel` as an explicit Execucomp proxy from `execcomp.anncomp`
- `release_events` from CRSP delistings plus CCM links
- optional BoardEx tables when licensed and enabled
- optional `ciq_pplintel.wrds_professional` people analytics
- optional merged `boardex_capiq_bridge` using `wrdsapps_plink_boardex_ciq.boardex_ciq`

The stage also writes a merged audit panel, `wrds_merged_company_year.parquet`, so you can
see how much executive coverage and release-event coverage WRDS is contributing by firm-year.

When the relevant subscriptions are enabled, the stage can also materialize:

- `capiq_people_analytics.parquet`
- `boardex_capiq_bridge.parquet`
- `boardex_people.parquet`
- `boardex_board_roles.parquet`
- `boardex_employment.parquet`

Because WRDS table layouts vary a bit across subscriptions and vintages, the package does not
assume every accounting or industry field lives on the same table. For example, `sic` is pulled
from `comp.company` when it is not present on `comp.funda`.

## What WRDS does not fully solve on its own

- `noncompete_state_year`
- `ff_industry_map`
- a fully geocoded historical HQ panel

WRDS company reference tables contain address metadata, but not a clean geocoded historical
HQ spell panel. The package therefore only builds `hq_history` if you provide a local geocode
crosswalk through `wrds.hq_geocode_crosswalk`.

## Conservative design choices

- WRDS pulls are staged exactly as parquet before any transformation.
- Canonical outputs are validated against the same explicit schemas used elsewhere in the package.
- The Execucomp-based `cri_exec_panel` is labeled as a proxy, not as a silent replacement.
- The CRSP-based `release_events` file uses a conservative default:
  only 2xx merger-style delisting codes are marked as `clean_release_flag = true`.

## Example config

See [examples/wrds_bootstrap_config.yml](/Users/stefanmaric/Papers/CEO%20Market/examples/wrds_bootstrap_config.yml).
