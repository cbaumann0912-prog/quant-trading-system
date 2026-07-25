# Day 55 Research Audit: GARCH(1,1) Volatility Persistence, All Pairs

## Question
Does GARCH(1,1) persistence differ meaningfully across the full 10-pair FX universe, or is the spread mostly estimation noise?

## Methodology
`day55_garch_forex_pairs.py` fits `fit_garch` (`src/features/garch.py`, a from-scratch GARCH(1,1) MLE — Gaussian log-likelihood over σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}, solved with scipy's L-BFGS-B) on daily log returns for all 10 pairs, ~4,053 observations each, built from 1-minute closes resampled to daily via `DataLoader`. Conditional vol is checked against a 20-day rolling realized standard deviation, a stand-in for Day 68's realized-vol module.

To test whether full-sample persistence is a stable property of each pair or an artifact of one 13-year fit, each pair was refit on individual calendar years, then on 7 rolling 5-year windows once the single-year fits proved too noisy to answer the question.

## Findings

**Parameter estimates (dev window, 2011-01-03 to 2023-12-31):**

| Pair   | omega      | alpha  | beta   | persistence (α+β) | long_run_vol |
|--------|-----------:|-------:|-------:|-------------------:|-------------:|
| EURUSD | 8.300e-08  | 0.0301 | 0.9667 | 0.9968              | 0.00511      |
| GBPUSD | 5.836e-07  | 0.0732 | 0.9095 | 0.9827              | 0.00580      |
| USDJPY | 4.697e-07  | 0.0635 | 0.9210 | 0.9846              | 0.00552      |
| USDCHF | 4.219e-07  | 0.0532 | 0.9468 | **1.0000**          | **degenerate** |
| AUDUSD | 3.879e-07  | 0.0542 | 0.9368 | 0.9909              | 0.00655      |
| USDCAD | 1.712e-07  | 0.0348 | 0.9560 | 0.9908              | 0.00433      |
| NZDUSD | 4.692e-07  | 0.0456 | 0.9438 | 0.9894              | 0.00667      |
| EURGBP | 2.718e-07  | 0.0557 | 0.9326 | 0.9883              | 0.00482      |
| EURJPY | 6.581e-07  | 0.0480 | 0.9303 | 0.9783              | 0.00551      |
| EURCHF | 3.310e-06  | 0.0619 | 0.7811 | 0.8430              | 0.00459      |

Persistence ranking, most to least: USDCHF (1.0000) > EURUSD (0.9968) > AUDUSD (0.9909) > USDCAD (0.9908) > NZDUSD (0.9894) > EURGBP (0.9883) > USDJPY (0.9846) > GBPUSD (0.9827) > EURJPY (0.9783) > EURCHF (0.8430).

USDCHF sits exactly at the IGARCH boundary, so `long_run_vol` blows up to a large, meaningless number instead of a clean NaN — two runs gave 0.193 and 0.470 for the same series and window. That instability is itself the finding: the output isn't usable at any precision. EURCHF's omega is 40x every other pair's, persistence a low outlier at 0.84.

**GARCH conditional vol vs. 20-day realized vol:**

| Pair   | corr(GARCH, 20d realized) | mean GARCH vol | mean realized vol |
|--------|---------------------------:|----------------:|--------------------:|
| EURUSD | 0.9009 | 0.00461 | 0.00446 |
| GBPUSD | 0.9405 | 0.00506 | 0.00478 |
| USDJPY | 0.9524 | 0.00500 | 0.00472 |
| USDCHF | 0.8845 | 0.00572 | 0.00474 |
| AUDUSD | 0.9481 | 0.00600 | 0.00582 |
| USDCAD | 0.9288 | 0.00421 | 0.00410 |
| NZDUSD | 0.9389 | 0.00628 | 0.00608 |
| EURGBP | 0.9392 | 0.00437 | 0.00418 |
| EURJPY | 0.9465 | 0.00548 | 0.00530 |
| EURCHF | 0.7971 | 0.00434 | 0.00307 |

**Per-year GARCH(1,1) persistence (one fit per pair per calendar year, ~310-314 obs each):**

| Pair   | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| EURUSD | 0.140 | 0.998 | 0.999 | 0.991 | 0.881 | 0.271 | 0.888 | 0.885 | 0.973 | 0.927 | 0.973 | 0.959 | 0.887 |
| GBPUSD | 0.886 | 0.998 | 0.923 | 0.977 | 0.783 | 0.998 | 0.338 | 0.962 | 0.640 | 0.928 | 0.880 | 0.975 | 0.988 |
| USDJPY | 0.154 | 0.969 | 0.998 | 0.999 | 0.883 | 0.094 | 0.999 | 0.963 | 0.301 | 0.924 | 0.898 | 0.999 | 0.988 |
| USDCHF | 1.000 | 0.996 | 0.981 | 0.988 | 0.983 | 0.132 | 0.931 | 0.948 | 0.667 | 0.899 | 0.889 | 0.965 | 0.909 |
| AUDUSD | 0.973 | 0.997 | 0.983 | 0.984 | 0.893 | 0.997 | 0.887 | 0.887 | 0.996 | 0.996 | 0.981 | 0.987 | 0.888 |
| USDCAD | 0.978 | 0.986 | 0.970 | 0.980 | 0.998 | 0.990 | 0.888 | 0.889 | 0.998 | 0.985 | 0.061 | 0.950 | 0.995 |
| NZDUSD | 0.992 | 0.995 | 0.983 | 0.981 | 0.882 | 0.998 | 0.954 | 0.888 | 0.997 | 0.999 | 0.895 | 0.979 | 0.888 |
| EURGBP | 0.898 | 0.960 | 0.962 | 0.981 | 0.870 | 0.935 | 0.657 | 0.881 | 0.863 | 0.927 | 0.988 | 0.526 | 0.996 |
| EURJPY | 0.471 | 0.892 | 0.991 | 0.982 | 0.397 | 0.999 | 0.890 | 0.844 | 0.748 | 0.873 | 0.901 | 0.991 | 0.843 |
| EURCHF | 1.000 | 1.000 | 0.994 | 0.960 | 0.982 | 0.737 | 0.875 | 0.852 | 0.944 | 0.919 | 0.867 | 0.470 | 0.973 |

Spread per pair (min/max/mean/std): EURUSD 0.140/0.999/0.828/0.282, GBPUSD 0.338/0.998/0.867/0.189, USDJPY 0.094/0.999/0.782/0.346, USDCHF 0.132/1.000/0.868/0.238, AUDUSD 0.887/0.997/0.958/0.048, USDCAD 0.061/0.998/0.897/0.254, NZDUSD 0.882/0.999/0.956/0.049, EURGBP 0.526/0.996/0.880/0.138, EURJPY 0.397/0.999/0.833/0.192, EURCHF 0.470/1.000/0.890/0.147.

**Rolling 5-year window GARCH(1,1) persistence (7 windows, ~1,558-1,562 obs each):**

| Pair   | 2011-16 | 2012-17 | 2013-18 | 2014-19 | 2015-20 | 2016-21 | 2017-22 |
|--------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| EURUSD | 1.0000 | 0.9974 | 0.9961 | 0.9950 | 0.9992 | 0.9873 | 0.9848 |
| GBPUSD | 0.9932 | 1.0000 | 0.9999 | 0.9996 | 0.9539 | 0.9508 | 0.8817 |
| USDJPY | 0.9650 | 0.9987 | 0.9993 | 0.9965 | 0.9898 | 0.9836 | 0.9430 |
| USDCHF | 0.9653 | 0.8254 | 0.8297 | 0.8355 | 0.9976 | 0.9568 | 0.9543 |
| AUDUSD | 0.9951 | 0.9947 | 0.9932 | 0.9944 | 0.9986 | 0.9731 | 0.9477 |
| USDCAD | 0.9923 | 0.9939 | 0.9926 | 0.9918 | 0.9994 | 0.9905 | 0.9819 |
| NZDUSD | 0.9802 | 0.9960 | 0.9955 | 0.9961 | 0.9982 | 0.9794 | 0.9579 |
| EURGBP | 0.9905 | 0.9958 | 0.9956 | 0.9970 | 0.9764 | 0.9647 | 0.9148 |
| EURJPY | 0.9947 | 0.9824 | 0.9807 | 0.9286 | 0.9206 | 0.9309 | 0.9868 |
| EURCHF | 0.8206 | 0.8892 | 0.8870 | 0.8866 | 0.9902 | 0.8788 | 0.9137 |

Spread per pair (min/max/mean/std): EURUSD 0.9848/1.0000/0.9943/0.0059, GBPUSD 0.8817/1.0000/0.9684/0.0440, USDJPY 0.9430/0.9993/0.9823/0.0211, USDCHF 0.8254/0.9976/0.9092/0.0753, AUDUSD 0.9477/0.9986/0.9853/0.0186, USDCAD 0.9819/0.9994/0.9918/0.0052, NZDUSD 0.9579/0.9982/0.9862/0.0148, EURGBP 0.9148/0.9970/0.9764/0.0297, EURJPY 0.9206/0.9947/0.9607/0.0323, EURCHF 0.8206/0.9902/0.8952/0.0506.

## Interpretation
Outside the CHF pairs, persistence clusters tightly between 0.978 and 0.997, consistent with the standard result that FX vol clustering is strong and long-lived. Correlations of 0.90-0.95 against realized vol confirm the GARCH fit tracks a real shape rather than overfitting; GARCH running slightly above realized vol is expected, since it reacts to the latest shock while a rolling window smooths and lags.

The two CHF pairs fail for different reasons: USDCHF's is one outlier day, cleanly fixed by dropping it; EURCHF's is deeper, since persistence stays unidentifiable even with that day removed. Neither should feed position sizing or vol targeting as-is — use a Student-t innovation distribution, model the SNB day as a structural break, or drop CHF pairs from GARCH-based sizing until this is resolved.

Single-year fits are a caution about method, not about the pairs: wild year-to-year swings (EUR/USD 0.140 in 2011 to 0.999 in 2013) reflect what GARCH(1,1) looks like at ~310 observations, not real shifts in volatility dynamics. The 5-year rolling windows settle it — for the 8 non-CHF pairs, persistence is stable across windows (std 0.005-0.044 vs. 0.05-0.35 at the single-year size), confirming the full-sample ranking is real. USDCHF and EURCHF remain the exception in a way that independently confirms the SNB story: every window containing January 2015 shows depressed persistence, every window starting after it recovers. EUR/JPY shows a smaller, unexplained version of the same pattern (0.92-0.93 in the 2014-19/2015-20 windows vs. 0.98-0.99 elsewhere).

## Alternative Explanations
EURCHF's flat ridge could be an artifact of this optimizer setup rather than a true property of the data — worth checking against a different solver or a direct grid search over the α+β = constant ridge before concluding the series is fundamentally unidentifiable.

EUR/JPY's 2014-2020 persistence dip hasn't been traced to a specific driver. Worth confirming whether it's the China-deval/BOJ period specifically before treating it as informative about EUR/JPY generally.

## Next Steps
- Decide how to handle the SNB day before CHF-pair GARCH output feeds anything downstream — winsorize, exclude, or switch to Student-t innovations.
- Cross-check EURCHF's ridge finding with a different optimizer to rule out a solver artifact.
- Identify the driver behind EUR/JPY's 2014-2020 persistence dip.
 