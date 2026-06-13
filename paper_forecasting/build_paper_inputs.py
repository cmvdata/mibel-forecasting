"""
build_paper_inputs.py — tables + figure for the MIBEL LEAR forecasting paper.
Reuses the already-computed backtest results (reports/diagnostics/*.csv) and adds
the solar-saturation evidence (near-zero-price hour share by year) from this repo's
own ESIOS-600 price cache. No re-running of the heavy backtests.

Three grounded results:
  (1) LEAR beats seasonal naive across crisis/normal regimes (replication backbone).
  (2) Technical indicators (Demir 2019) systematically HURT (negative transfer).
  (3) Metric instability: as near-zero solar hours rise (0.3% -> 12.3%), sMAPE
      explodes (16% -> 51%) while MAE falls -- percentage metrics break.

Run: python paper_forecasting/build_paper_inputs.py
Outputs: paper_forecasting/
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DIAG = ROOT / "reports" / "diagnostics"
CACHE = ROOT / "data" / "cache" / "esios"
OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)


def _price_year(year):
    fs = sorted(glob.glob(str(CACHE / f"i600_geo3_{year}_*.parquet")))
    if not fs:
        return None
    s = pd.concat([pd.read_parquet(f) for f in fs])["value"]
    return s


def main():
    lines = []
    def p(*a):
        lines.append(" ".join(str(x) for x in a))

    lear = pd.read_csv(DIAG / "lear_robustness_2026_05.csv")
    ti = pd.read_csv(DIAG / "technical_indicators_2026_05.csv")

    p("=" * 76)
    p("FORECASTING PAPER INPUTS — LEAR on MIBEL (2022-2024)")
    p("=" * 76)

    # --- (3) solar-saturation: near-zero hour share by year ---
    p("\n[3] Solar saturation: near-zero-price hours by year (ESIOS 600, ES)")
    zrows = []
    for y in [2022, 2023, 2024]:
        s = _price_year(y)
        if s is None:
            p(f"   {y}: price cache missing")
            continue
        zrows.append(dict(year=y, hours=len(s), mean=round(s.mean(), 1),
                          share_le1=round(100 * (s <= 1).mean(), 1),
                          share_le0=round(100 * (s <= 0).mean(), 1)))
        p(f"   {y}: mean {s.mean():5.1f} EUR/MWh  share<=1EUR {100*(s<=1).mean():4.1f}%  "
          f"share<=0 {100*(s<=0).mean():4.1f}%  ({len(s)} h)")
    zdf = pd.DataFrame(zrows)
    zdf.to_csv(OUT / "zero_hour_share.csv", index=False)

    # --- (1) LEAR vs naive: best exogenous variant (demand+wind) by regime ---
    p("\n[1] LEAR (demand+wind) vs seasonal naive, by regime")
    best = lear[lear["model"] == "LEAR demand+wind"][
        ["regime", "MAE (EUR/MWh)", "sMAPE (%)", "rMAE vs naive", "DM p-value vs naive"]]
    p(best.to_string(index=False))

    # --- (2) TI degradation: rMAE with vs without TI ---
    p("\n[2] Technical-indicator transfer: rMAE without vs with TI (negative transfer)")
    base = "LEAR demand+wind"
    rows = []
    for reg in ["2022 (H2, crisis)", "2023 (full year)", "2024 (full year)", "2023-2024 pooled"]:
        b = ti[(ti.regime == reg) & (ti.model == base)]["rMAE vs naive"]
        t = ti[(ti.regime == reg) & (ti.model == base + " + TI")]["rMAE vs naive"]
        if len(b) and len(t):
            rows.append(dict(regime=reg, rMAE_base=round(b.iloc[0], 3),
                             rMAE_TI=round(t.iloc[0], 3),
                             delta_pts=round(100 * (t.iloc[0] - b.iloc[0]), 1)))
    tidf = pd.DataFrame(rows)
    p(tidf.to_string(index=False))
    tidf.to_csv(OUT / "ti_degradation.csv", index=False)

    (OUT / "paper_inputs.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # --- Figure: metric divergence under solar saturation ---
    yrs = [2022, 2023, 2024]
    mae = [float(lear[(lear.regime.str.startswith(str(y))) &
                      (lear.model == "LEAR demand+wind")]["MAE (EUR/MWh)"].iloc[0]) for y in yrs]
    smape = [float(lear[(lear.regime.str.startswith(str(y))) &
                        (lear.model == "LEAR demand+wind")]["sMAPE (%)"].iloc[0]) for y in yrs]
    zero = [zdf[zdf.year == y]["share_le1"].iloc[0] for y in yrs]

    fig, ax1 = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(len(yrs))
    ax1.bar(x - 0.2, mae, width=0.4, color="#1f4e79", label="MAE (EUR/MWh)")
    ax1.set_ylabel("MAE (EUR/MWh)", color="#1f4e79")
    ax1.set_ylim(0, 25)
    ax1.set_xticks(x)
    ax1.set_xticklabels(yrs)
    ax2 = ax1.twinx()
    ax2.plot(x, smape, "o-", color="#c0392b", lw=2, label="sMAPE (%)")
    for xi, (z, sm) in enumerate(zip(zero, smape)):
        ax2.annotate(f"{z:.1f}% hrs $\\leq$1\\euro" if False else f"{z:.1f}% hrs <=1EUR",
                     (xi, sm), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7, color="#7a1f12")
    ax2.set_ylabel("sMAPE (%)", color="#c0392b")
    ax2.set_ylim(0, 60)
    ax1.set_title("Same model, diverging metrics: MAE falls while sMAPE explodes\n"
                  "as near-zero solar hours rise (LEAR demand+wind)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_metric_divergence.pdf")
    print(f"\n[fig] {OUT/'fig_metric_divergence.pdf'}")


if __name__ == "__main__":
    main()
