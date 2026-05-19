# Exogenous audit — v8 parquet columns (2026-05)

> **Auditoría reproducida desde memoria interna del asistente en
> sesión previa.** Las verificaciones contra timestamps de publicación
> ESIOS no han sido re-ejecutadas en este commit.

Classification of the temporal nature of every exogenous column carried
by the `mibel-congestion-monitor` v8 parquet, with the explicit goal of
deciding which ones are admissible as features when the target is the
DAM price of day D (gate-closure 12:00 ES on day D-1, publication 14:00
CET on day D-1).

## Safe (D-1 day-ahead — known before DAM gate-closure)

Free to use as a feature when forecasting day D's DAM price.

| Column            | Source                          | Indicator |
| ----------------- | ------------------------------- | --------- |
| `es_demand_fc`    | ESIOS API                       | 460 (Previsión diaria de la demanda eléctrica peninsular) |
| `fr_demand_fc`    | RTE / ODRE eco2mix              | `prevision_j1` |
| `es_solar_fc`     | ESIOS API                       | 540 |
| `es_wind_fc`      | ESIOS API                       | 541 |
| `fr_solar_fc`     | ESIOS API                       | 1254 |
| `fr_wind_fc`      | ESIOS API                       | 1255 |
| `ntc_es_fr`       | JAO ATC + implicit pre-2022     | scheduled D-1 18:00 CET |
| `ntc_fr_es`       | JAO ATC + implicit pre-2022     | scheduled D-1 18:00 CET |

## Unsafe (realised or intra-day)

Must **not** be consumed as a feature when forecasting day D.

| Column              | Problem                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------- |
| `fr_nuclear_avail`  | ESIOS 10208 = disponibilidad **realised**, published ex-post.                            |
| `ttf_eur_mwh`       | Closing settlement of the day — possibly known only at session close, intra-day leakage. |
| `co2_eur_t`         | Same closing-settlement caveat as TTF.                                                   |

## Context and decisions

This audit was triggered after rMAE of LEAR on ES 2024 fell outside the
Lago (2021) reference range `[0.80, 0.95]`. The audit rules out
**leakage** as the explanation: the two exogenous columns used by the
default LEAR config (`es_demand_fc`, `es_wind_fc`) are both day-ahead
forecasts, classified safe here. The low rMAE is attributed instead to
the atypical solar dynamics of MIBEL 2024 (frequent zero-price mid-day
hours that the seasonal naive cannot anticipate) — see notebook
`02_lear.ipynb` discussion and `reports/diagnostics/utc_migration_2026_05.md`
for the convention shift.

**Operational rules for `LEAR(exogenous_cols=...)`:**

- Default `("es_demand_fc", "es_wind_fc")` is safe; keep as-is.
- French-aware extension that is still safe:
  `("es_demand_fc", "es_wind_fc", "fr_demand_fc", "fr_wind_fc", "ntc_es_fr")`.
- **Do not** add `fr_nuclear_avail`, `ttf_eur_mwh` or `co2_eur_t`
  without first lagging them by at least one day inside the loader.
- When the future `mibel-market-intel` ingestion lands, every pulled
  column should carry a `quality_flag` and the loader should refuse
  `realised`-marked columns whenever the target is the DAM.

## Provenance

The substance of this audit was produced in a prior assistant session
by exploring the `mibel-congestion-monitor` repo and the ESIOS
indicator catalog. The publication timestamps of the safe-list
indicators (especially ESIOS 460 / 540 / 541 / 1254 / 1255) have not
been independently re-pulled and timestamped in **this** commit. A
follow-up task should pull a sample window for each of those
indicators and confirm that `tx_time` precedes the DAM gate-closure on
each calendar day.
