# Research note — PC2 as a carry-factor candidate (7-pair USD-quoted universe)

## Question
Is PC2 in the 7-pair universe new, or the same PC2 already killed in Days 38/39/41? If new, does it hold up as a carry-trade mechanism?

## Method
7 USD-quoted pairs (EUR, GBP, JPY, CHF, AUD, CAD, NZD vs. USD). EUR-crosses dropped to avoid multicollinearity with pairs already in the set. Dev window 2011-2023, lockbox excluded by `DataLoader`'s date filter. PCA via the existing `pca()`, refit per year for stability. Rate-spread and crash-date checks via `pc2_carry_factor_analysis.py`.

## Findings

### Variance explained
| PC | Share | Cumulative |
|---|---|---|
| PC1 | 56.0% | 56.0% |
| PC2 | 16.0% | 72.0% |
| PC3 | 9.9% | 81.9% |
| PC4 | 7.1% | 89.0% |
| PC5 | 4.3% | 93.3% |
| PC6 | 3.7% | 97.0% |
| PC7 | 3.0% | 100% |

PC2 clears the noise floor; PC3-7 all sit under 10%.

### PC2 loadings
EUR -0.24, GBP -0.02, JPY +0.54, CHF +0.62, AUD +0.36, CAD -0.23, NZD +0.29.

Different from the old 3-pair PC2 (JPY-dominant, same sign across all three). Here, positive PC2 means JPY and CHF weaken while AUD, CAD, and NZD strengthen: funding currencies moving one way, target currencies the other.

### Year-by-year stability, 2011-2023
| Year | EUR | GBP | JPY | CHF | AUD | CAD | NZD | Var. |
|---|---|---|---|---|---|---|---|---|
| 2011 | 0.005 | 0.030 | 0.289 | 0.859 | 0.282 | -0.200 | 0.240 | 23.7% |
| 2012 | 0.469 | 0.157 | 0.531 | -0.442 | -0.322 | 0.085 | -0.409 | 14.8% |
| 2013 | 0.059 | -0.055 | 0.783 | 0.159 | 0.343 | -0.105 | 0.475 | 21.0% |
| 2014 | -0.451 | -0.242 | 0.292 | 0.535 | 0.468 | -0.244 | 0.297 | 16.6% |
| 2015 | 0.140 | 0.157 | 0.006 | 0.713 | 0.443 | -0.269 | 0.422 | 25.8% |
| 2016 | -0.215 | 0.329 | 0.840 | 0.279 | 0.143 | -0.177 | -0.099 | 24.4% |
| 2017 | -0.352 | -0.590 | 0.139 | 0.310 | 0.400 | -0.297 | 0.406 | 14.7% |
| 2018 | -0.213 | -0.379 | 0.458 | 0.428 | 0.435 | -0.344 | 0.332 | 13.3% |
| 2019 | -0.122 | 0.640 | 0.617 | 0.375 | -0.026 | 0.007 | -0.234 | 17.6% |
| 2020 | -0.432 | -0.268 | 0.573 | 0.520 | 0.251 | -0.150 | 0.239 | 14.2% |
| 2021 | -0.341 | 0.053 | 0.594 | 0.601 | 0.217 | -0.295 | 0.181 | 14.2% |
| 2022 | -0.076 | 0.043 | 0.842 | 0.272 | 0.307 | -0.269 | 0.205 | 10.1% |
| 2023 | -0.013 | 0.028 | 0.888 | 0.117 | 0.279 | -0.231 | 0.255 | 13.0% |

JPY, CHF, AUD, CAD, and NZD hold the same sign 10-13 of 13 years. EUR and GBP flip almost every year; treat them as noise.

## Hypothesis: funding-vs-target carry factor
JPY and CHF were the lowest-yielding G10 currencies for most of 2011-2023, the standard funding legs for a carry trade. AUD, CAD, and NZD carried higher, commodity-linked rates, the standard target legs (Menkhoff, Sarno, Schmeling & Schrimpf 2012; Brunnermeier, Nagel & Pedersen 2008). Positive PC2 would read as carry trades building, negative as unwind.

## Testing the hypothesis
Real JPY/CHF vs. AUD/CAD/NZD 3-month rate spread, 2-month publication lag.

| Test | Statistic | Result |
|---|---|---|
| Rate spread, level | r=0.019, p=0.24, n=4,029 | No relationship |
| Rate spread, MoM change | r=0.017, p=0.29, n=4,029 | No relationship |
| Lead-lag, naive, 21d | r=0.084, p<0.0001, n=4,008 | Looks significant |
| Lead-lag, naive, 63d | r=0.137, p<0.0001, n=3,966 | Looks significant |
| Lead-lag, naive, 126d | r=0.213, p<0.0001, n=3,903 | Looks significant |
| Lead-lag, non-overlapping subsample, 21d | r=0.074, p=0.31, n=191 | Null |
| Lead-lag, non-overlapping subsample, 63d | r=0.144, p=0.26, n=63 | Null |
| Lead-lag, non-overlapping subsample, 126d | r=0.271, p=0.14, n=31 | Null |
| Lead-lag, block-bootstrap 95% CI, 21d | [-0.03, 0.18] | Includes zero |
| Lead-lag, block-bootstrap 95% CI, 63d | [-0.03, 0.30] | Includes zero |
| Lead-lag, block-bootstrap 95% CI, 126d | [-0.06, 0.44] | Includes zero |
| SNB window (Jan 5-26, 2015) | single day -0.095 (~16σ), window z=-0.81 | One day, CHF-specific |
| COVID window (Feb 10-Apr 3, 2020) | z=-0.31, mixed sign | No pattern |

The naive lead-lag correlations looked clean at every horizon, but both series are heavily autocorrelated within their windows (the spread is a monthly step function forward-filled to daily, the target a rolling mean), so the naive p-values assume far more independent observations than exist — the same overlap problem Day 44 hit with 26-day forward returns. Correcting for it kills the effect entirely. SNB's move is real but explained by CHF's own outsized loading (0.62), not a broad carry mechanism.

## Verdict
Invalidated. The loadings look like a clean carry factor, but nothing holds up once tested: the rate spread is a flat null at every specification, and the one lead-lag result that looked real doesn't survive correcting for overlap. Closed as dead.

## Next steps
- More than two crash dates. One CHF shock and one ambiguous window isn't enough to judge a mechanism.
- Re-run the correlation excluding 2015-01-15, to check whether SNB alone is propping up the null.
- Trace PC4/PC6's extreme kurtosis to specific dates.
- If a real signal turns up on a future pass: trial #7, with its own written spec, before any implementation.
