# UTC migration: rMAE shift on LEAR DAM-ES

**Date:** 2026-05-19
**Triggering commit:** `b38b456` — *refactor(data): use UTC as internal timezone for DAM/CID loaders*

## Headline

After switching the internal timezone of the loaders from `Europe/Madrid`
to UTC, rMAE on the LEAR day-ahead Spain benchmark moved:

| metric | before (Europe/Madrid) | after (UTC) |
|--------|------------------------:|-------------:|
| rMAE   | **0.4283**              | **0.3936**   |

**This is not an algorithmic improvement.** Same LEAR estimator, same
hyperparameters, same feature recipe, same evaluation horizon. The
number changed because the canonical boundary defining "a forecasting
day" changed.

## Attribution

Two changes landed close together in this branch:

1. **UTC migration** (`b38b456`) — DAM/CID loaders now expose UTC-indexed
   panels, and the rolling evaluator regroups predictions and targets
   by UTC calendar day instead of Madrid wall-clock calendar day.
2. **Target-leakage masking** in `LEAR.predict` (next commit on this
   branch) — guards the AR-lag block against reading realised targets
   that sit inside the test window.

Only (1) explains the rMAE delta. The masking in (2) is a **zero-op at
the standard 1-day-ahead recalibration horizon** used in this
benchmark: when the test window is a single day, the lag block reaches
back into the training window only, so masking changes no input value
and produces bit-identical predictions. The new regression test
`test_lear_multiday_window_no_leakage_from_within_test` only exposes
the underlying bug when the model is asked to predict ≥2 consecutive
days in one `predict` call.

## Why the boundary change moved the number

`Europe/Madrid` has DST. Under the old convention some calendar days
contained 23 or 25 hours; rMAE then averaged a non-uniform number of
hours per day, and the rolling split's day boundaries did not
coincide with the natural 24-hour publication block of ESIOS / OMIE.
Regrouping by UTC restores 24-hour panes everywhere and aligns day
boundaries with the publishers' own convention. The two rMAE numbers
are therefore computed under different denominators and are **not
directly comparable**.

## Framing for write-ups

> "We migrated the internal timezone of the data pipeline from
> Europe/Madrid to UTC (Lago 2021 §3.1 canonical convention). rMAE of
> LEAR on DAM-ES under the new convention is 0.3936; the previous 0.4283
> was computed under a non-canonical day boundary and is reported here
> only for traceability of the migration."

## Reproducibility

- Before: parent of `b38b456` on `main`.
- After: HEAD as of the commit chain that includes this report.
- LEAR benchmark notebook: `notebooks/02_lear.ipynb` (unchanged).
- Leakage-guard regression: `tests/test_lear.py`.
