# mibel-forecasting

Electricity price forecasting on the Iberian market (MIBEL). One of five modules
of the `mibel-portfolio` umbrella repository.

## Scope

Three sub-objectives, in priority order:

1. **DAM forecasting** — replicate Lago et al. (2021) on OMIE day-ahead Spain
   (ESIOS indicator 600). Models: seasonal naive, LEAR, DNN. Primary metric:
   rMAE versus naive; statistical comparison via Diebold–Mariano.
2. **Technical indicators** — replicate Demir et al. (2019): add EMA,
   Bollinger %B, MACD, MOM, ROC and Coppock as features to LEAR and DNN; report
   RMSE reduction versus the no-TI baseline.
3. **CID forecasting** — extend Demir et al. (2023) Chapter-8 feature
   engineering to the Iberian continuous intraday market (MIC), augmented with
   the microstructure features of Vilches (2026): `range_es`, `mic_volume_mwh`,
   `mic_price_missing`, `spread_da`, `umm_active_mw`.

## Bibliographic anchors

- Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). *Forecasting
  day-ahead electricity prices: A review of state-of-the-art algorithms, best
  practices and an open-access benchmark.* Applied Energy 293, 116983.
- Demir, S., Mincev, K., Kok, K., Paterakis, N. G. (2019). *Introducing
  technical indicators to electricity price forecasting.* Applied Sciences 10(1).
- Demir, S., Kok, K., Paterakis, N. G. (2023). *Statistical arbitrage trading
  across electricity markets using advantage actor-critic methods.* SEGAN 34.
- Monteiro, C. et al. (2015). *Explanatory Information Analysis for Day-Ahead
  Price Forecasting in the Iberian Electricity Market.* Energies 8(9).
- Aineto, D. et al. (2019). *On the Influence of Renewable Energy Sources in
  Electricity Price Forecasting in the Iberian Market.* Energies 12(11).
- Vilches, C. (2026). *Detecting Structural Manipulation in Electricity Markets.*

The LEAR and DNN implementations are ported from scratch following Lago et al.
(2021) Section 3.2. The `epftoolbox` library (AGPL-3.0) is consulted as the
reference implementation but not imported; this repository is MIT-licensed.

## Data

The DAM target (`price_es`) is pulled directly from **ESIOS indicator 600
geo=3** (`Precio mercado SPOT Diario`, España) — the canonical wholesale
Spanish day-ahead price. `price_pt` (geo=1) and `price_fr` (geo=2) are
pulled from the same indicator as exogenous features. Responses are cached
locally by month under `data/cache/esios/` so a re-run pays no API cost.

Exogenous non-price features (demand / wind / solar forecasts, NTC, TTF
gas, EU ETS CO₂) are merged on request from the v8 parquet produced by
the sibling repo `mibel-congestion-monitor`. The buggy `price_es` /
`price_fr` columns of that parquet are **never** consumed — see
`reports/diagnostics/v8_vs_esios_2024_summary.md` for the audit.

The CID panel (sub-objective 3.3) still reads `features_2022_2024.parquet`
from `mibel-congestion-monitor` for its microstructure features
(`range_es`, `mic_volume_mwh`, `umm_active_mw`, …). The CID target
(`mic_price`, ESIOS 1727) is unaffected by the v8 label bug.

| Panel | Source | Coverage |
| --- | --- | --- |
| DAM target + neighbour prices | ESIOS 600 geo=3/1/2 (cached locally) | unlimited |
| DAM exogenous (optional) | v8 parquet, non-price columns only | 2019-01-02 → 2024-12-31, hourly |
| CID | `features_2022_2024.parquet` | 2022 → 2024, hourly |

Paths and the ESIOS token are configured in `.env` (see `.env.example`).
Raw, processed and cached data files are not tracked in git.

## Limitations declared upfront

1. Public, aggregated data only — no sub-hourly limit order book.
2. The MIC regime changed on 2023-03-04 (time-of-arrival → price-priority).
   CID models limit their training window to 2022-2024 and treat the regime
   change explicitly.
3. The same-day DAM–CID spread is dominated by autocorrelation; a previous TFM
   saturated at AUC ≈ 0.88. CID features are audited for lookahead before
   training.
4. The 2022 energy crisis is atypical. Results are reported with and without
   that period where it affects conclusions.

## Quickstart

```bash
# 1. Install uv (one-time, user-level)
pip install --user uv

# 2. Create the environment and install
uv sync --extra dev --extra notebooks

# 3. Copy the env template and point it at your local parquets
cp .env.example .env  # then edit if needed

# 4. Run the test suite
uv run pytest

# 5. Run the first notebook (naive baseline on 2024)
uv run jupyter lab notebooks/01_dam_baselines.ipynb
```

## Repository layout

```
src/mibel_forecasting/
├── data/         loaders, calendar features, train/test splits
├── features/     technical indicators (Demir 2019), microstructure (Vilches 2026)
├── models/       naive, LEAR (ported), DNN (ported), optional GBM for CID
├── evaluation/   metrics, rolling recalibration loop, Diebold-Mariano
└── viz/          plots used in the reports
```

## Status

Early scaffolding. Currently implemented: project setup, CI, naive baseline on
DAM 2024. Next: LEAR port and Demir technical-indicator replication.
