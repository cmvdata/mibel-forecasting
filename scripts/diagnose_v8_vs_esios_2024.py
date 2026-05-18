"""Build an auditable v8 vs ESIOS 600 (ES/PT/FR) hourly panel for 2024.

Pulls ESIOS indicator 600 for three geos (España=3, Portugal=1, Francia=2),
merges with v8.price_es and v8.price_fr from the consolidated parquet, and
writes:
- reports/diagnostics/v8_vs_esios_2024.parquet  (full hourly table)
- reports/diagnostics/v8_vs_esios_2024_summary.md (summary stats Carlo asked for)

Intentionally limited to 2024 (post-Iberian-gas-mechanism, clean test of the
labels-crossed hypothesis).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from mibel_forecasting.data.loaders import load_dam_panel  # noqa: E402

TOKEN = os.environ["ESIOS_API_TOKEN"]
HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}
GEO_MAP = {3: "esios_es", 1: "esios_pt", 2: "esios_fr"}
OUT_DIR = ROOT / "reports" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_esios_chunk(start: str, end: str, geo_id: int) -> pd.Series:
    r = requests.get(
        "https://api.esios.ree.es/indicators/600",
        headers=HEADERS,
        params=[("start_date", start), ("end_date", end), ("geo_ids[]", str(geo_id))],
        timeout=120,
    )
    r.raise_for_status()
    rows = r.json()["indicator"]["values"]
    if not rows:
        return pd.Series(dtype=float)
    df = pd.DataFrame(rows)
    # ESIOS returns datetime_utc as ISO string in UTC
    df["dt_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    s = df.set_index("dt_utc")["value"].astype(float)
    # If granularity is sub-hourly, hourly-average; if already hourly, no-op.
    s = s.groupby(s.index.floor("h")).mean()
    return s


def fetch_esios_year(geo_id: int) -> pd.Series:
    chunks = []
    for month in range(1, 13):
        s = f"2024-{month:02d}-01T00:00:00Z"
        last = (pd.Timestamp(f"2024-{month:02d}-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")
        e = f"{last}T23:59:59Z"
        try:
            chunk = fetch_esios_chunk(s, e, geo_id)
            chunks.append(chunk)
            print(f"  geo={geo_id} 2024-{month:02d}: {len(chunk)} hours", flush=True)
        except requests.HTTPError as ex:
            print(f"  geo={geo_id} 2024-{month:02d} FAILED: {ex}", flush=True)
        time.sleep(0.4)
    out = pd.concat(chunks)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out.rename(GEO_MAP[geo_id])


def main() -> None:
    print("Loading v8 (Spain wall-clock → UTC) ...", flush=True)
    v8 = load_dam_panel(start="2024-01-01", end="2024-12-31", timezone="Europe/Madrid")
    v8 = v8[["price_es", "price_fr"]].rename(columns={"price_es": "v8_price_es", "price_fr": "v8_price_fr"})
    v8.index = v8.index.tz_convert("UTC")
    v8.index.name = "datetime_utc"
    print(f"  v8 hours: {len(v8)} (2024)", flush=True)

    panels = [v8]
    for geo_id in (3, 1, 2):
        print(f"Fetching ESIOS 600 geo={geo_id} ({GEO_MAP[geo_id]}) ...", flush=True)
        panels.append(fetch_esios_year(geo_id).to_frame())

    merged = pd.concat(panels, axis=1)
    merged.index.name = "datetime_utc"
    # Restrict to rows where v8 is present (the analysis target)
    merged = merged[merged["v8_price_es"].notna()]
    # Compute four absolute-difference columns
    merged["abs_es_v8es"] = (merged["esios_es"] - merged["v8_price_es"]).abs()
    merged["abs_es_v8fr"] = (merged["esios_es"] - merged["v8_price_fr"]).abs()
    merged["abs_fr_v8es"] = (merged["esios_fr"] - merged["v8_price_es"]).abs()
    merged["abs_fr_v8fr"] = (merged["esios_fr"] - merged["v8_price_fr"]).abs()
    merged["abs_pt_v8es"] = (merged["esios_pt"] - merged["v8_price_es"]).abs()
    merged["abs_pt_v8fr"] = (merged["esios_pt"] - merged["v8_price_fr"]).abs()

    parquet_path = OUT_DIR / "v8_vs_esios_2024.parquet"
    merged.to_parquet(parquet_path)
    print(f"\nWrote {parquet_path}  shape={merged.shape}", flush=True)

    # Summary stats
    def stats(s: pd.Series) -> dict[str, float]:
        s = s.dropna()
        n = len(s)
        return {
            "n": n,
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
        ("abs_es_v8es", "|ESIOS-ES − v8.price_es|"),
        ("abs_es_v8fr", "|ESIOS-ES − v8.price_fr|"),
        ("abs_fr_v8es", "|ESIOS-FR − v8.price_es|"),
        ("abs_fr_v8fr", "|ESIOS-FR − v8.price_fr|"),
        ("abs_pt_v8es", "|ESIOS-PT − v8.price_es|"),
        ("abs_pt_v8fr", "|ESIOS-PT − v8.price_fr|"),
    ]
    summary = {label: stats(merged[col]) for col, label in cols}
    df_summary = pd.DataFrame(summary).T

    matrix = pd.DataFrame(
        {
            "vs ESIOS-ES": [
                merged["abs_es_v8es"].median(),
                merged["abs_es_v8fr"].median(),
            ],
            "vs ESIOS-PT": [
                merged["abs_pt_v8es"].median(),
                merged["abs_pt_v8fr"].median(),
            ],
            "vs ESIOS-FR": [
                merged["abs_fr_v8es"].median(),
                merged["abs_fr_v8fr"].median(),
            ],
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

    # Write markdown summary
    lines = [
        "# v8 vs ESIOS 600 — auditable diagnostic for 2024",
        "",
        "**Scope** — all hours of 2024 where the v8 panel has a price (post-DST drop).",
        f"**N hours** — {len(merged)}.",
        "",
        "## Summary statistics (abs differences, €/MWh)",
        "",
        df_to_md(df_summary.round(4)),
        "",
        "## Median |diff| matrix (€/MWh)",
        "",
        df_to_md(matrix.round(4)),
        "",
        "## Decision rule (registered before observation)",
        "",
        "- **H1 confirmed (labels crossed)**: in 2024, median(|ESIOS-ES − v8.price_fr|) ≈ 0 *and* median(|ESIOS-ES − v8.price_es|) > 0.",
        "- **H2 (labels OK, decoupling drives noise)**: both v8.price_es and v8.price_fr have median ≈ 0 vs ESIOS-ES; outliers concentrate on specific hours.",
        "- **H3**: ambiguous → deeper investigation.",
        "",
        "## Data",
        "",
        f"Full hourly panel at `reports/diagnostics/v8_vs_esios_2024.parquet`",
        f"({merged.shape[0]} rows × {merged.shape[1]} cols).",
    ]
    md_path = OUT_DIR / "v8_vs_esios_2024_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")

    # Also print the summary to stdout
    print("\n=== Summary stats ===")
    print(df_summary.round(4).to_string())
    print("\n=== Median matrix ===")
    print(matrix.round(4).to_string())


if __name__ == "__main__":
    main()
