# Week 4 Review — Time Series & Cointegration (Days 22–27)

## Methodology
`plot_acf_pacf`/`ljung_box_test` (`src/data/stationarity.py`), `fit_arima` (`src/data/time_series.py`), `engle_granger_test`/`cointegration_spread`/`johansen_test`/`ou_half_life` (`src/signals/cointegration.py`).

## Findings
- No pair cointegrated at 5% (Engle-Granger or Johansen rank=0), 2011–2026 data. EUR/USD~GBP/USD borderline (p=0.0621) — resolves the Week 3 open question; Johansen confirms the Engle-Granger near-miss was a true null, not a power artifact.
- Rolling hedge ratio for EUR/USD~GBP/USD unstable: std=0.396, range −0.11 to +1.57 — answers Week 3's walk-forward stability question negatively.
- PCA structural note: post-2020 PC1 drift (~52%→~76%) confirmed as a real regime shift, not noise.
- Multi-Pair Stat Arb candidate weakened significantly by these results (later formally demoted Day 36).

## Interpretation
Johansen confirmed the ADF result from Week 3 wasn't a power artifact — rank=0 across all three pairs, and EUR/USD~GBP/USD's rolling hedge ratio is unstable enough (std=0.396) that it wouldn't have been tradeable even under a more favorable cointegration read. Combined with the PC1 structural break (52%→76% post-2020), the picture is consistent: these pairs share too much common variance for a mean-reversion spread trade to hold up.