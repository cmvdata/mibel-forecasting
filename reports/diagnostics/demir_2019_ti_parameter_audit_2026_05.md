# Demir 2019 TI parameter audit (pre-implementation)

> Methodology audit produced **before** writing any technical-indicator
> code in this repo. The goal is to fix down which numerical
> parameters and formulas the implementation must follow, with verbatim
> citations to the paper, so the Phase 2 commits can be reviewed
> against an authoritative reference rather than against finance
> folklore.

## Source

Demir, S.; Mincev, K.; Kok, K.; Paterakis, N. G. (2020).
*Introducing Technical Indicators to Electricity Price Forecasting:
A Feature Engineering Study for Linear, Ensemble, and Deep Machine
Learning Models.* **Applied Sciences** 10(1), 255.
DOI: [10.3390/app10010255](https://doi.org/10.3390/app10010255).

Local copy at `reports/applsci-10-00255.pdf` (untracked by git).

## Paper-level decisions that constrain every indicator below

These are conventions the paper applies uniformly to all TIs, before
any single indicator's parameters become relevant. They must therefore
be replicated for the comparison with Demir's findings to be valid.

### 1. Grid-search per model — no universal canonical parameters

> "where n and s are hyperparameters tuned using grid-search during
> the modelling of DAM prices."
> *(Section 2.1, last line of the notations block, p. 3.)*

Every numerical hyperparameter (`n`, `s`, `s1`, `s2`, `n1`, `n2`) in
the formulas below is grid-searched **per model** in Demir 2019. The
paper does NOT prescribe a single canonical value of, e.g., `n` for
ROC or `s` for EMA — Table 1 reports the values that each model's
grid-search converged on.

Implication for us: our LEAR is a **linear model** (per-hour Lasso
following Lago 2021 §3.2). The Demir table column closest in spirit
is the linear-model column (LR / HR). When we propose a defensible
default below, we cite the linear-model row of Table 1 first; for
indicators that did not show up in the linear-model top-three the
paper offers no direct guidance and we either fall back to the
cross-model recurring optimum (with explicit citation) or list
grid-search on MIBEL as the principled alternative.

### 2. Hour-of-day grouping ("24 separate markets")

> "The DAM, however, is not sequential because it releases 24 prices
> simultaneously upon clearing. Consequently, we treat the dataset as
> an assortment of prices from 24 separate markets when calculating
> DAM TIs. Formally, we use h hour prices to calculate h hour TIs.
> For instance, the 12 h SMA (n = 3) at 9 November 2014 is calculated
> by taking the average of three 12 h prices: the 7, 8 and 9 of
> November."
> *(Section 2.3.3, p. 8.)*

Implication: every indicator is computed **24 times in parallel**,
once per hour-of-day, each on a daily-frequency series of that hour's
prices. The window parameter `n` (resp. `s`) is therefore in **days
of the same hour**, not in raw hours.

Implementation: `df['price_es'].groupby(df.index.hour).transform(...)`
inside each per-hour series.

### 3. Forecast convention and the resulting one-day shift

> "to forecast day d+1 prices, we use six prices from days d to d−5
> along with the average of d−6, d−13, ..., d−55 prices"
> *(Section 2.3.2, p. 8.)*

> "we train each model … once as a benchmark model accepting only
> lagged prices, and once as a TI model accepting both lagged prices
> and a lagged TI."
> *(Section 2.3.4, p. 8.)*

The TI fed into the model is a **lagged TI** — its value at the
forecast timestamp must depend only on prices known strictly before
the forecast timestamp.

Implementation rule we will follow uniformly: every TI is computed
with `shift(1)` on the per-hour-of-day daily series, so the value at
day D for hour h uses only prices `p_{d ≤ D-1, h}`. This is the
leakage-safe adaptation invoked by the user instruction "Use shift(1)
or closed-window operations exclusively". Lago's LEAR convention
(predict day D using D-1, D-2, D-3, D-7 lags) is automatically
compatible because the TI is now itself a `D-1`-safe quantity.

### 4. Pre-processing differs between Demir and us

> "using Min–Max scaling, both features and response variables are
> scaled."
> *(Section 2.3.2, p. 8.)*

Demir uses Min–Max. Our LEAR uses arcsinh-median (Lago 2021 §3.2).
The TI feature block enters our LEAR through the same arcsinh-median
scaler as the price lags. We do not need to replicate Demir's
Min–Max because the scaler is a model-internal step and is not
affected by the choice of feature.

### 5. Market, period and headline finding

- **Market**: Belgian DAM (Belpex) — *not* MIBEL.
- **Period**: 2014-01-01 → 2018-06-30 (4.5 years; train 3.5y / test 1y).
- **Headline result**: best TI reduced RMSE by **4.50 %** (HR), **4.49 %**
  (LR), **5.42 %** (AB), and 4.09 % (2NN). On average across all 10
  models tested, the best-per-model TI cut RMSE by 3.28 %.
  *(Abstract; Table 1; Section 3.)*

The 3-5 % magnitude is the figure we will compare our own LEAR
results against in Phase 2.

## Indicator-by-indicator audit

For each indicator we list (a) the formula and equation number from
the paper, (b) Demir's reported grid-search optimum(s) in Table 1,
focusing on linear models, (c) the pandas/numpy equivalent we will
use, and (d) any open decision.

### 1. Exponential Moving Average — EMA

**Formula (Section 2.1.2, Equation 2, p. 4):**
$$
\mathrm{EMA}(p_t, s) \;=\; \frac{p_t + \alpha\,p_{t-1} + \alpha^2 p_{t-2} + \cdots + \alpha^t p_0}{1 + \alpha + \alpha^2 + \cdots + \alpha^t},
\qquad
\alpha = \frac{s - 1}{s + 1}.
$$

The single hyperparameter is the **span** `s`.

**Pandas equivalent.** Demir's α equals pandas' decay factor
`(1 − α_pandas)` for `α_pandas = 2/(s+1)`, and the weighted-sum form
matches `adjust=True`. Therefore:

```python
df['price_es'].ewm(span=s, adjust=True).mean()
```

is numerically identical to Equation 2 above.

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | parameter |
|---|---|---|
| HR (linear) | **best** | `s = 2` |
| LR (linear) | second | `s = 2` |
| RF (ensemble) | best | `s = 6` |
| AB (ensemble) | third | `s = 6` |
| GB (ensemble) | third | `s = 6` |
| 2NN (deep, fully-connected) | best | `s = 22` |

**Proposed value for our LEAR (linear / Lasso):** `s = 2`. This
matches both linear models in Demir; it is also the value the paper
singles out for "short-term EMAs alongside lagged prices [used] to
identify short-term trend formations" (Section 3.3, p. 14).

### 2. Bollinger %B

**Formula (Section 2.1.5, Equation 9, p. 5):**
$$
\%\mathrm{B}(p_t, n) \;=\; \frac{p_t - \mathrm{BBAND}^{-}(p_t, n)}{\mathrm{BBAND}^{+}(p_t, n) - \mathrm{BBAND}^{-}(p_t, n)},
$$
where the Bollinger bands (Equations 7 and 8) and the moving standard
deviation (Equation 6) are built on the SMA of Equation 1:

$$
\mathrm{BBAND}^{\pm}(p_t, n) = \mathrm{SMA}(p_t, n) \pm 2 \cdot \mathrm{MSD}(p_t, n),
\quad
\mathrm{MSD}(p_t, n) = \sqrt{\tfrac{1}{n} \sum_{i=0}^{n-1} (p_{t-i} - \mathrm{SMA}(p_t, n))^2}.
$$

The single hyperparameter is the window `n`. The multiplier on MSD is
**hard-coded at 2** by Equations 7 and 8 — Demir does not tune it.

**Pandas equivalent.**

```python
sma = price_hour_series.rolling(n).mean()
msd = price_hour_series.rolling(n).std(ddof=0)   # population MSD per Eq. 6
upper, lower = sma + 2 * msd, sma - 2 * msd
pctB = (price_hour_series - lower) / (upper - lower)
```

Note `ddof=0` (population standard deviation) to match Equation 6
exactly; pandas defaults to `ddof=1`.

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | parameter |
|---|---|---|
| LR (linear) | **best** | `n = 58` |
| HR (linear) | second | `n = 58` |
| GB (ensemble) | second | `n = 54` |

**Proposed value for our LEAR:** `n = 58`. Consensus across both
linear models in Demir.

### 3. MACD (three components)

**Formulas (Section 2.1.3, Equations 3, 4, 5, p. 4):**

The paper splits MACD into the three series it is built from:

$$
\mathrm{Series}(p_t, s_1, s_2) = \mathrm{EMA}(p_t, s_1) - \mathrm{EMA}(p_t, s_2),
\quad s_2 > s_1
$$

$$
\mathrm{Signal}(p_t, s_1, s_2, s) = \mathrm{EMA}(\mathrm{Series}(p_t, s_1, s_2), s)
$$

$$
\mathrm{Histogram}(p_t, s_1, s_2, s) = \mathrm{Series}(p_t, s_1, s_2) - \mathrm{Signal}(p_t, s_1, s_2, s)
$$

Three hyperparameters: fast EMA span `s_1`, slow EMA span `s_2`,
signal EMA span `s`. The paper notes the "(e.g., s = 12)" and
"(e.g., s = 26)" wording in Section 2.1.3 (p. 4) as **examples
only** — the actual values are grid-searched, not fixed at the
finance-folklore 12/26/9 defaults.

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | component | parameters |
|---|---|---|---|
| LR (linear) | third | Histogram | `s_1 = 2, s_2 = 26, s = 9` |
| 2NN (deep) | third | Histogram | `s_1 = 58, s_2 = 116, s = 9` |

The Histogram never appears in any model's top-two, but it is the
**only** MACD component the linear-model grid-search retained at all.
The Series and Signal components do not show up in Demir's top-three
for any model.

**Proposed value for our LEAR:** `(s_1 = 2, s_2 = 26, s = 9)` for the
Histogram. **Confidence: high** — Demir's grid-search retained the
MACD Histogram with these exact parameters as the **third-best TI
for LR** (Linear Regression, Table 1, row "Third-Best TI", column
"LR", footnote `** = (s_1=2, s_2=26, s=9)`, p. 11), the same model
family our LEAR (per-hour Lasso) descends from. The 2NN model
independently retained the Histogram as its third-best TI at
different parameters (`s_1 = 58, s_2 = 116, s = 9`, footnote `***`),
reinforcing that the Histogram is the empirically preferred MACD
branch across model families.

We will compute all three components (Series, Signal, Histogram) at
the same parameters and let the Lasso decide whether to keep them;
expecting Series and Signal to be largely zeroed out, consistent
with Demir's empirical preference for the Histogram.

**Note on the "standard 12/26/9" finance default.** The paper
deliberately does *not* lock in those numbers — they appear only as a
parenthetical example in Section 2.1.3. We follow the grid-search
result, not the financial-charts default.

### 4. Momentum — MOM

**Formula (Section 2.1.6, Equation 11, p. 5):**
$$
\mathrm{MOM}(p_t, n) = p_t - p_{t-n}.
$$

A pure lagged difference. Single hyperparameter `n`.

**Pandas equivalent.** With the hour-of-day grouping (`n` in days of
the same hour):

```python
price_hour_series - price_hour_series.shift(n)
```

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | parameter |
|---|---|---|
| AB (ensemble) | **best** | `n = 58` |
| GB (ensemble) | **best** | `n = 58` |
| RF (ensemble) | second | `n = 58` |

**Open decision: not retained by any linear model in Demir's top-3.**
The paper does not provide a linear-model optimum for MOM. Two
defensible choices:

1. **Use Demir's cross-model recurring optimum `n = 58`**, with the
   caveat that this was selected for ensembles, not for linear
   models. Justification: 58 days is conspicuously similar to the
   `%B (n = 58)` linear-model optimum — both lie inside Demir's
   "behavioural-bias horizon" — and using the same `n` everywhere
   keeps the implementation auditable.
2. **Grid-search on MIBEL** over `n ∈ {1, 2, ..., 60}` per model,
   reporting the optimum alongside the Belgian comparison.

Phase-2 default: **`n = 58`** (option 1). A grid-search ablation is
listed as an open task for a later phase.

### 5. Rate of Change — ROC

**Formula (Section 2.1.7, Equation 12, p. 5):**
$$
\mathrm{ROC}(p_t, n) = \frac{p_t - p_{t-n}}{p_{t-n}}.
$$

Single hyperparameter `n`. Demir notes that ROC is "an oscillator,
comparable to the MOM indicator, that expresses change as a
percentage instead of an absolute value" (Section 2.1.7, p. 5).

**Pandas equivalent.**

```python
price_hour_series.pct_change(periods=n)
```

This guards against the `p_{t-n} = 0` edge case via pandas' standard
NaN propagation; in MIBEL DAM 2024 ~6 % of hours are exactly 0 €/MWh
so the division-by-zero risk is real. We will surface those NaNs and
let Lasso treat them as masked (or drop affected days through the
existing `_pivot_hourly` filter).

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | parameter |
|---|---|---|
| CNN | **best** | `n = 49` |
| 2CNN | **best** | `n = 49` |
| 2CNN_NN | best | `n = 9` |
| ResNet | best | `n = 27` |
| AB | second | `n = 57` |
| RF | third | `n = 57` |

**Open decision: not retained by any linear model in Demir's top-3.**
Two defensible choices, analogous to MOM:

1. **Use the cross-model favoured value `n = 49`** (the most common
   in Demir's deep-model selections), with the same caveat that no
   linear-model evidence supports it.
2. **Grid-search on MIBEL.**

Phase-2 default: **`n = 49`** (option 1). MOM and ROC both default
to "long-horizon" indicators (around 50 days of same-hour history),
which Demir's discussion in §3.1 explicitly ties to the
"behavioural-bias horizon" of DAM traders. Grid-search ablation
listed as future work.

### 6. Coppock Curve — COPP

**Formula (Section 2.1.8, Equation 13, p. 6):**
$$
\mathrm{COPP}(p_t, s, n_1, n_2) = \mathrm{EMA}\big( \mathrm{ROC}(p_t, n_1) + \mathrm{ROC}(p_t, n_2),\; s \big).
$$

Three hyperparameters: span `s` of the outer EMA and two ROC lags
`n_1`, `n_2`. The original Coppock (1962, equities, monthly data)
uses `(n_1 = 11, n_2 = 14, s = 10)`; Demir does **not** use those
values — they are tuned by grid-search on the DAM.

**Pandas equivalent.**

```python
roc1 = price_hour_series.pct_change(n1)
roc2 = price_hour_series.pct_change(n2)
copp = (roc1 + roc2).ewm(span=s, adjust=True).mean()
```

**Demir's grid-search optima (Table 1, p. 11):**

| model | rank | parameters |
|---|---|---|
| ResNet (deep) | second-best | `n_1 = 18, n_2 = 24, s = 18` (footnote `*`) |
| 2CNN_NN (deep) | third-best | `n_1 = 58, n_2 = 74, s = 54` (footnote `****`) |

**Open decision: Coppock is the only indicator on the six-indicator
list with no linear-model evidence at any parameter setting.** Demir
retained Coppock only by two deep models, and the two retained
parameter sets differ from each other — this is the weakest
indicator–parameter pairing in the audit. Two defensible choices:

1. **Adopt Demir's ResNet second-best parameters
   `(n_1 = 18, n_2 = 24, s = 18)`.** These are the highest-ranked
   Coppock appearance in Table 1 (second-best, above 2CNN_NN's
   third-best). The 2CNN_NN parameters `(n_1 = 58, n_2 = 74, s = 54)`
   are **not adopted**: they belong to a deeper CNN-based
   architecture whose capacity to absorb noisy long-horizon ROC
   components has no parallel in a linear Lasso, and adopting them
   would mix two unrelated grid-search optima.
2. **Grid-search on MIBEL.**

Phase-2 default: **`(n_1 = 18, n_2 = 24, s = 18)`** (option 1). We
explicitly expect Coppock to be the **most likely zero-coefficient
indicator** in the Lasso ablation — that itself is a finding worth
reporting, since Demir's Coppock evidence is already the marginal
case in the paper.

## Leakage-safe computation rule (uniform across all six indicators)

For each indicator, given a UTC-indexed hourly panel of `price_es`:

1. Split by hour-of-day: `df['price_es'].groupby(df.index.hour)`.
2. On each per-hour daily series, apply the pandas formula above.
3. **Apply `.shift(1)` on the daily series** (i.e. `shift(periods=1)`
   on the within-group series whose index is one observation per
   day at that hour).
4. Realign the 24 per-hour daily series back onto the hourly UTC
   index by simply taking each value at every timestamp of its hour.

Equivalent statement of the rule in plain English: the indicator
value attached to UTC timestamp `2024-06-10 14:00` is computed from
prices at hour 14 on days `2024-06-09, 2024-06-08, …`, i.e. with
strictly **no** information from `2024-06-10` itself.

This implementation rule will be verified in Phase 2 by the
no-leakage regression tests described in the user's brief: poison
`p_{T+1}` with 9999 and assert byte-identical indicator values at
`T`. With the rule above, this test passes by construction for all
six indicators.

## Summary table of Phase-2 parameters

| indicator | symbol(s) | value | Demir Table 1 source | confidence |
|---|---|---|---|---|
| EMA | `s` | **2** | LR/HR best | high (linear consensus) |
| Bollinger %B | `n` | **58** | LR best, HR second | high (linear consensus) |
| MACD Series, Signal, Histogram | `(s_1, s_2, s)` | **(2, 26, 9)** | LR third-best (Histogram, footnote `**` in Table 1) | high (linear-model evidence) |
| MOM | `n` | **58** | AB / GB best, RF second | medium (no linear evidence) |
| ROC | `n` | **49** | CNN / 2CNN best, AB second | medium (no linear evidence) |
| Coppock | `(n_1, n_2, s)` | **(18, 24, 18)** | ResNet second-best (footnote `*` in Table 1); 2CNN_NN third-best has different params `(58, 74, 54)` — not adopted | low (no linear evidence at any params) |

"Confidence" reflects how directly Demir's evidence supports the
parameter choice for a linear (Lasso) model on Belgian DAM. The
medium/low confidence items justify the open task of running a MIBEL
grid-search ablation in a later phase to confirm or revise.

## Open tasks (not addressed by this audit)

1. **Grid-search on MIBEL** over the medium- and low-confidence
   parameters (MOM `n`, ROC `n`, Coppock `(n_1, n_2, s)`). Phase-2
   uses Demir's defaults; a follow-up phase could retune.
2. **TI feature encoding into LEAR.** Demir uses "length-wise
   concatenation" (Section 2.3.4, p. 8) for linear models — TI values
   appended to the price-lag vector. We will follow this: extend
   LEAR's per-day feature row by 24 TI values per indicator (one
   per hour-of-day), gated behind an optional `include_ti`
   constructor flag so the existing 247-feature configuration stays
   bit-identical.
3. **Sanity check via DM test.** Phase 2 will run pairwise DM
   (TI vs no-TI) per regime, as the user's brief requires.

## Provenance

- Paper: Demir et al. 2019, DOI 10.3390/app10010255, local copy
  `reports/applsci-10-00255.pdf` (untracked).
- Audit author: assistant session, 2026-05-19, based on full
  paper text pasted by the user (cross-checked against the PDF's
  embedded metadata: title, authors, abstract).
- This audit was produced **before** any TI source code was written
  in this repository.
