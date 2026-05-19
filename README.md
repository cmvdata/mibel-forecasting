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

### Current status

DAM sub-objective (3.1) is partially shipped; sub-objectives 3.2 (TI) and
3.3 (CID) are not yet started.

Shipped:

- Seasonal-naive DAM baseline on Spain (2024 test window) — `notebooks/01_dam_baselines.ipynb`.
- LEAR (Lago 2021 §3.2) ported from scratch — `src/mibel_forecasting/models/lear.py`,
  walk-through in `notebooks/02_lear.ipynb`.
- Diebold-Mariano test with Newey-West (Andrews 1991 lag rule, floored at
  `h-1` for `h=24`) — `src/mibel_forecasting/evaluation/dm_test.py`.
- v8-vs-ESIOS label-bug audit (n = 8782 hourly observations on 2024,
  median |ESIOS-ES − v8.price_es| = 0 €/MWh) — see
  `reports/diagnostics/v8_vs_esios_2024_summary.md`. The buggy
  `price_es` / `price_fr` columns of v8 are blacklisted in the loader.
- Exogenous-column audit classifying every v8 non-price column as either
  day-ahead forecast (safe for LEAR) or realised/intra-day (leakage
  risk) — `reports/diagnostics/exogenous_audit_2026_05.md`.
- UTC as the canonical internal timezone of the loaders, with the
  before/after rMAE shift documented for traceability
  (Madrid → UTC: rMAE moved from 0.4283 to 0.3936 — same model, same
  features, different day boundary) — see
  `reports/diagnostics/utc_migration_2026_05.md`.
- Two-layer anti-leakage guard in the rolling-forecast evaluator:
  `LEAR.predict` reads price lags only from training history, and
  `rolling_forecast` masks the target column to NA before calling
  `predict`. Regression-tested with a multi-day-window poisoning test
  in `tests/test_lear.py`.

### Known limitations

1. **rMAE outside the Lago 2021 reference range.** Under the canonical
   UTC convention, LEAR DAM-ES 2024 lands at rMAE ≈ 0.39
   (`reports/diagnostics/utc_migration_2026_05.md`), well below the
   `[0.80, 0.95]` band that Lago et al. (2021) report on EPEX-BE/FR/DE,
   NordPool and PJM. The cause is structural: heavy solar penetration in
   MIBEL 2024 collapses mid-day prices to zero in a way the week-old
   seasonal naive cannot anticipate, inflating the denominator of the
   rMAE ratio. This is documented honestly in notebook 02 and is *not*
   a code or leakage issue.
2. **Notebook axis labels still assume Europe/Madrid** while the internal
   panel is now UTC. This is purely cosmetic for plots, but a reader of
   notebook 02 will see *"Hour of day (Europe/Madrid)"* on a UTC-indexed
   series. Slated for cleanup in P4.
3. **No formal robustness analysis yet.** A single test window
   (first two weeks of June 2024) is reported. Cross-window stability,
   bootstrap CIs over rMAE, and a regime-conditioned (high-solar vs
   low-solar day) breakdown are P6 work.

### How to reproduce

```bash
# 1. Sync the environment (uv must be installed; pip install --user uv)
uv sync --extra dev --extra notebooks

# 2. Run the unit/integration test suite (network tests are skipped
#    unless ESIOS_API_TOKEN is exported in .env)
uv run pytest

# 3. Execute the two shipped notebooks end-to-end
uv run jupyter nbconvert --to notebook --execute notebooks/01_dam_baselines.ipynb
uv run jupyter nbconvert --to notebook --execute notebooks/02_lear.ipynb
```

The ESIOS API token is read from `.env` (`ESIOS_API_TOKEN=...`).
Without it, the pure-network tests (`@pytest.mark.network`) are
deselected by the CI config and skipped locally.
