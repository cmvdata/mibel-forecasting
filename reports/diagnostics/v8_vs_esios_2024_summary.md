# v8 vs ESIOS 600 - auditable diagnostic for 2024

**Scope** - all hours of 2024 where the v8 panel has a price (post-DST drop).
**N hours** - 8782.

## Summary statistics (abs differences, EUR/MWh)

|  | n | median | p75 | p90 | p95 | p99 | max | pct_eq_0 | pct_lt_0_01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| |ESIOS-ES - v8.price_es| | 8781.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 26.6100 | 96.7700 | 93.4176 | 93.4290 |
| |ESIOS-ES - v8.price_fr| | 8781.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.4300 | 99.6470 | 99.6584 |
| |ESIOS-FR - v8.price_es| | 8781.0000 | 13.2700 | 38.2000 | 63.3500 | 76.1100 | 98.3100 | 150.6000 | 29.8827 | 29.8941 |
| |ESIOS-FR - v8.price_fr| | 8781.0000 | 12.0900 | 37.4100 | 63.1500 | 75.9800 | 98.2880 | 150.6000 | 32.3312 | 32.3426 |
| |ESIOS-PT - v8.price_es| | 8781.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.4300 | 99.6470 | 99.6584 |
| |ESIOS-PT - v8.price_fr| | 8781.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 26.6100 | 96.7700 | 93.4176 | 93.4290 |

## Median |diff| matrix (EUR/MWh)

|  | vs ESIOS-ES | vs ESIOS-PT | vs ESIOS-FR |
| --- | --- | --- | --- |
| v8.price_es | 0.0000 | 0.0000 | 13.2700 |
| v8.price_fr | 0.0000 | 0.0000 | 12.0900 |

## Decision rule (registered before observation)

- **H1 (labels crossed)**: median(|ESIOS-ES - v8.price_fr|) ~ 0 AND median(|ESIOS-ES - v8.price_es|) > 0.
- **H2 (labels OK, decoupling drives noise)**: both v8.price_es and v8.price_fr have median ~ 0 vs ESIOS-ES.
- **H3**: ambiguous.

## Data
Full hourly panel at `reports/diagnostics/v8_vs_esios_2024.parquet` (8782 rows x 11 cols).