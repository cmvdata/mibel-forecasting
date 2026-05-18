# Data

This directory is intentionally empty in version control. Raw, processed,
and cached files are not tracked.

## Sources

### DAM (sub-objectives 3.1 and 3.2)

The DAM target and its sibling prices come from **ESIOS indicator 600**:

| Column | ESIOS geo_id | Description |
| --- | --- | --- |
| `price_es` | 3 | Spain (target) |
| `price_pt` | 1 | Portugal (exogenous) |
| `price_fr` | 2 | France (exogenous) |

The loader (`mibel_forecasting.data.esios.pull_indicator`) caches one
Parquet file per `(indicator_id, geo_id, year, month)` under
`data/cache/esios/`. Delete the directory to force a fresh re-pull;
otherwise subsequent calls read from disk and incur no API cost.

A token is required (`ESIOS_API_TOKEN` in `.env`).

### DAM exogenous (optional, v8 parquet)

The non-price exogenous features (demand / wind / solar forecasts, gas,
CO₂, NTC, French nuclear availability) are merged on request from the
consolidated v8 parquet of `mibel-congestion-monitor`. Set
`MIBEL_DAM_PARQUET` in `.env` to point at the file. The columns are
listed in `mibel_forecasting.data.loaders.DAM_V8_EXOGENOUS`.

**Do not consume `price_es` / `price_fr` from this parquet directly.** A
diagnostic in `reports/diagnostics/v8_vs_esios_2024_summary.md`
established that the labels in the v8 parquet are crossed at source:
`v8.price_es` matches ESIOS-PT at 99.66% (it is actually
Portugal / MIBEL-coupled), and `v8.price_fr` matches ESIOS-ES at 99.66%
(it is Spain — never France). The loader bypasses both columns and
exposes only the non-price exogenous fields.

### CID (sub-objective 3.3)

Reads `features_2022_2024.parquet` from
`mibel-congestion-monitor/neuro_detector/data/processed/`. Path is
configured via `MIBEL_CID_PARQUET` in `.env`. The CID target
(`mic_price`, ESIOS 1727) is unaffected by the v8 label bug; the
`price_es` / `price_fr` columns of this panel **do** inherit the bug
and are flagged in the loader docstring — sub-objective 3.3 will
revisit them when it is built.

## Tests

`tests/test_loaders.py` includes:
- Schema and tz checks on monkeypatched ESIOS pulls (no network).
- DST fall-back / spring-forward correctness on the CID parquet.
- An end-to-end `@pytest.mark.network` check that the loader's
  `price_es` matches a fresh ESIOS-ES pull for one week of 2024 to
  within `1e-6` €/MWh.
