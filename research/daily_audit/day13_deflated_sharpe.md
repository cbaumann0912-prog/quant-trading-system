# Day 13 Research Audit — Deflated Sharpe Ratio: Sandbox Strategy Per Pair

## Objective
Apply the Deflated Sharpe Ratio (Lopez de Prado & Bailey 2014) to per-pair daily returns rom the archived FVG_BoS_Reversal strategy. The DSR adjusts the observed Sharpe for selection bias across multiple trials and non-normality in the return distribution, returning a probability in [0, 1] that the observed Sharpe is a true positive.

## Methodology Notes
All four DSR inputs are derived from the same active-day return series to ensure consistency. Active days are defined as calendar days with non-zero daily_return_pct. Zero-return on-trading days describe when the strategy was flat, not the shape of its return distribution, and including them inflates kurtosis artificially.

The per-period SR conversion is handled by deflated_sharpe_ratio using self.ann_factor — the same annualization factor used by compute_sharpe — ensuring the two are always consistent regardless of data frequency. ann_factor is computed empirically as active observations divided by years in the sample rather than assuming 252. For this strategy trading 4–5 times per year, ann_factor = 252 would overstate the Sharpe

The Lopez de Prado formula is written in terms of raw kurtosis, so the V term uses (kurtosis + 2) rather than (kurtosis - 1) to convert correctly. For a normal distribution this produces (0 + 2) / 4 = 0.5, which matches the theoretical standard error of the Sharpe estimator under normality. Using excess kurtosis without this correction causes V to go negative for high SR inputs, breaking the formula.

Kurtosis was estimated as sample excess kurtosis on active returns after the Student-t theoretical formula returned undefined values due to df_fit <= 4 across all pairs. A V guard returns np.nan if V <= 0 to prevent downstream sqrt errors.

Caveat: zero-return days from breakeven exits are indistinguishable from non-trading days in this data source and are excluded. This is acceptable for this sandbox dataset where breakeven exits are rare. Phase 2 will use trade-level returns where every observation is active by definition.

## Hypothesis Setup

For each pair, DSR is computed at N = 2, 10, and 30 trials. Two threshold searches identify the trial count at which DSR drops below 0.95 and below 0.50 respectively.

## Results

| Pair | Sharpe | n_obs | Skewness | Kurtosis | DSR (N=2) | DSR (N=10) | DSR (N=30) | N where DSR < 0.95 | N where DSR < 0.5 |
|------|--------|-------|----------|----------|-----------|------------|------------|--------------------|-------------------|
| EUR/USD | 1.188 | 66 | -1.246 | 0.393 | 0.9982 | 0.9681 | 0.9122 | 16 | 1862 |
| GBP/USD | 1.006 | 75 | -1.039 | -0.118 | 0.9950 | 0.9358 | 0.8465 | 8 | 577 |
| USD/JPY | 1.011 | 79 | -1.105 | 0.003 | 0.9958 | 0.9431 | 0.8604 | 9 | 709 |

## Notes
- EUR/USD is the most robust of the three pairs, surviving 16 trials before DSR drops below 0.95. GBP/USD and USD/JPY fall below the institutional threshold at just 8 and 9 trials respectively. All three drop well below 0.5 before N=2000, meaning heavy parameter search would render any observed Sharpe statistically meaningless. Given that this is sandbox data from an unvalidated strategy, these thresholds carry no inferential weight — they confirm the methodology is functioning correctly.
- The per-period SR conversion is handled internally via self.ann_factor. An earlier iteration passed annualized Sharpe with per-period n_obs, saturating DSR near 1.0 regardless of trial count. Using ann_factor = 252 for a strategy trading 4–5 times per year overstated Sharpe by roughly an order of magnitude. The empirical ann_factor  eliminates this failure mode and makes the function frequency-agnostic — it works correctly for any strategy regardless of whether ann_factor is 252, 52, or 252 * 28.
- Kurtosis values are near zero or slightly negative across all pairs. This is expected given the strategy's dual take-profit exit structure. Trade outcomes cluster around three discrete points — stop loss, TP1, and TP2 — producing a trimodal return distribution. Probability mass spread across three modes rather than concentrated at the center produces platykurtic behavior by construction. This is structurally opposite to the fat-tailed raw market return distributions observed in the Day 4 analysis.
- The Student-t fit returned df_fit <= 4 for all pairs, making the theoretical kurtosis formula undefined. This is not a numerical error — it reflects the fact that fitting a unimodal symmetric distribution to a trimodal return series is a category mismatch. The t-distribution assumption embedded in the DSR non-normality correction is an approximation at best for dual-TP strategies and will remain a limitation in Phase 2 if the rebuilt strategy retains this exit structure.
- The contrast between Day 4 kurtosis and Day 13 kurtosis is methodologically meaningful. Day 4 measured the full distribution of raw market returns over 15 years — unbounded, unfiltered, capturing every extreme move. Day 13 measures a small filtered subset where entries meet specific structural criteria and exits are bounded by fixed TP and stop levels. Lower kurtosis is the expected consequence of bounded exits, not a data quality problem.
- This analysis will be rerun in Phase 2 on trade-level returns from properly built and  walk-forward validated strategies. At that stage the trimodal distribution structure will be documented explicitly and the DSR non-normality correction assessed for suitability given the exit structure of the rebuilt strategy.