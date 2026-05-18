"""One-off: regenerate the summary markdown from the saved parquet without re-pulling ESIOS."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "diagnostics"
parquet_path = OUT_DIR / "v8_vs_esios_2024.parquet"

merged = pd.read_parquet(parquet_path)

def stats(s: pd.Series) -> dict[str, float]:
    s = s.dropna()
    return {
        "n": len(s),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "pct_eq_0": float((s == 0).mean() * 100),
        "pct_lt_0_01": float((s < 0.01).mean() * 100),
    }

cols = [
    ("abs_es_v8es", "|ESIOS-ES - v8.price_es|"),
    ("abs_es_v8fr", "|ESIOS-ES - v8.price_fr|"),
    ("abs_fr_v8es", "|ESIOS-FR - v8.price_es|"),
    ("abs_fr_v8fr", "|ESIOS-FR - v8.price_fr|"),
    ("abs_pt_v8es", "|ESIOS-PT - v8.price_es|"),
    ("abs_pt_v8fr", "|ESIOS-PT - v8.price_fr|"),
]
summary = {label: stats(merged[col]) for col, label in cols}
df_summary = pd.DataFrame(summary).T

matrix = pd.DataFrame(
    {
        "vs ESIOS-ES": [merged["abs_es_v8es"].median(), merged["abs_es_v8fr"].median()],
        "vs ESIOS-PT": [merged["abs_pt_v8es"].median(), merged["abs_pt_v8fr"].median()],
        "vs ESIOS-FR": [merged["abs_fr_v8es"].median(), merged["abs_fr_v8fr"].median()],
    },
    index=["v8.price_es", "v8.price_fr"],
)

def df_to_md(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(["", *cols]) + " |"
    sep = "| " + " | ".join(["---"] * (len(cols) + 1)) + " |"
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)]
        for v in row:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])

lines = [
    "# v8 vs ESIOS 600 - auditable diagnostic for 2024",
    "",
    "**Scope** - all hours of 2024 where the v8 panel has a price (post-DST drop).",
    f"**N hours** - {len(merged)}.",
    "",
    "## Summary statistics (abs differences, EUR/MWh)",
    "",
    df_to_md(df_summary.round(4)),
    "",
    "## Median |diff| matrix (EUR/MWh)",
    "",
    df_to_md(matrix.round(4)),
    "",
    "## Decision rule (registered before observation)",
    "",
    "- **H1 (labels crossed)**: median(|ESIOS-ES - v8.price_fr|) ~ 0 AND median(|ESIOS-ES - v8.price_es|) > 0.",
    "- **H2 (labels OK, decoupling drives noise)**: both v8.price_es and v8.price_fr have median ~ 0 vs ESIOS-ES.",
    "- **H3**: ambiguous.",
    "",
    "## Data",
    f"Full hourly panel at `reports/diagnostics/v8_vs_esios_2024.parquet` ({merged.shape[0]} rows x {merged.shape[1]} cols).",
]
md_path = OUT_DIR / "v8_vs_esios_2024_summary.md"
md_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {md_path}")
print("\n=== Summary stats ===")
print(df_summary.round(4).to_string())
print("\n=== Median matrix ===")
print(matrix.round(4).to_string())
