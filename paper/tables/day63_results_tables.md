# Day 63 -- Results Tables (Intraday Overshoot, k=2.0, entry_delay=5min)

### Table 1 -- Full-sample metrics (2011-2023, unsplit, NOT out-of-sample)
| Pair | IC (full sample) | Sharpe (full sample, annualized) | n trades |
|---|---:|---:|---:|
| AUDUSD | 0.0419 | 0.3665 | 402 |
| EURCHF | 0.0103 | 0.0371 | 296 |
| EURGBP | 0.0318 | 0.4018 | 322 |
| EURJPY | -0.0424 | 0.5107 | 259 |
| EURUSD | -0.0921 | -0.0828 | 296 |
| GBPUSD | -0.0497 | 0.0269 | 372 |
| NZDUSD | -0.0211 | 0.3247 | 333 |
| USDCAD | -0.0293 | 0.1971 | 309 |
| USDCHF | 0.0570 | 0.0509 | 278 |
| USDJPY | -0.1021 | -0.3100 | 343 |

IC = Spearman(disp_k, trade_return) over every trade the pair produced, 2011-2023. Sharpe is the annualized Sharpe of the full, unsplit trade-return series -- raw, unsized, gross of costs. This table pools training and test periods and exists for comparability only; Tables 2 and 3 hold the out-of-sample numbers.

### Table 2 -- OOS walk-forward Sharpe distribution
| Pair | n windows | Mean | Std | Min | Max | % positive |
|---|---:|---:|---:|---:|---:|---:|
| AUDUSD | 30 | 0.5127 | 3.5893 | -6.8742 | 8.1429 | 47% |
| EURCHF | 21 | 2.2275 | 10.7605 | -22.8516 | 34.0937 | 62% |
| EURGBP | 30 | 1.0486 | 2.1120 | -2.5071 | 5.7512 | 67% |
| EURJPY | 28 | 1.8954 | 4.4844 | -10.3649 | 12.7658 | 71% |
| EURUSD | 30 | 0.3304 | 3.1446 | -4.0051 | 7.6058 | 53% |
| GBPUSD | 31 | 0.2373 | 2.6063 | -4.8526 | 5.1352 | 55% |
| NZDUSD | 29 | 4.6576 | 23.4769 | -5.4901 | 125.9207 | 66% |
| USDCAD | 30 | 0.0590 | 4.5131 | -6.1319 | 19.3362 | 50% |
| USDCHF | 27 | -2.7312 | 18.9680 | -91.9624 | 29.8196 | 59% |
| USDJPY | 31 | -4.1186 | 23.9652 | -132.3191 | 4.9378 | 55% |

Per-window annualized Sharpe of that window's triggered trades only. 5y train / 3mo test / 5-day embargo, k=2.0, entry_delay=5min.

### Table 3 -- Significance (pooled out-of-sample trades only)
n_trials = 10 (all 10 pairs actually run at k=2.0, delay=5min). BH alpha = 0.05.

| Pair | n obs (OOS) | Pooled OOS Sharpe | t-stat | p-value | BH-significant | DSR |
|---|---:|---:|---:|---:|:---:|---:|
| AUDUSD | 284 | 0.4976 | 0.5207 | 0.60296 | No | 0.1453 |
| EURCHF | 115 | 1.6607 | 1.1057 | 0.27119 | No | 0.3218 |
| EURGBP | 244 | 1.6820 | 1.6312 | 0.10415 | No | 0.5530 |
| EURJPY | 197 | 2.3077 | 2.0112 | 0.04567 | No | 0.6515 |
| EURUSD | 202 | 0.6323 | 0.5579 | 0.57750 | No | 0.1546 |
| GBPUSD | 296 | 0.5182 | 0.5535 | 0.58031 | No | 0.1540 |
| NZDUSD | 226 | 1.0426 | 0.9731 | 0.33157 | No | 0.2742 |
| USDCAD | 196 | -0.4638 | -0.4032 | 0.68728 | No | 0.0244 |
| USDCHF | 140 | -0.1974 | -0.1450 | 0.88488 | No | 0.0428 |
| USDJPY | 226 | -0.3681 | -0.3436 | 0.73149 | No | 0.0270 |

Pooled OOS Sharpe concatenates only trades inside a walk-forward test window, never a training window, across all windows for that pair. It is not the mean of Table 2's per-window Sharpes, and not Table 1's full-sample Sharpe.

### Table 4 -- Risk metrics (Day 65)
| Pair | VaR 95% | VaR 99% | CVaR 95% | Max drawdown | GARCH(1,1) persistence |
|---|---:|---:|---:|---:|---:|
| AUDUSD | 0.0055 | 0.0103 | 0.0085 | -5.05% | 0.9879 |
| EURCHF | 0.0027 | 0.0048 | 0.0039 | -0.97% | 0.9919 |
| EURGBP | 0.0037 | 0.0062 | 0.0052 | -2.09% | 0.9681 |
| EURJPY | 0.0038 | 0.0063 | 0.0066 | -4.46% | 0.9899 |
| EURUSD | 0.0050 | 0.0074 | 0.0070 | -2.40% | 0.9966 |
| GBPUSD | 0.0037 | 0.0079 | 0.0071 | -5.08% | 0.9591 |
| NZDUSD | 0.0058 | 0.0102 | 0.0086 | -4.44% | 0.9760 |
| USDCAD | 0.0042 | 0.0063 | 0.0055 | -4.41% | 0.9903 |
| USDCHF | 0.0043 | 0.0078 | 0.0066 | -3.59% | 0.9959 |
| USDJPY | 0.0047 | 0.0085 | 0.0082 | -5.73% | 0.9894 |

VaR/CVaR are historical, computed on the pooled-OOS trade-return series from Table 3 (per-trade log return, not daily). Max drawdown uses that same series' cumulative return path, test windows concatenated in chronological order. The path isn't continuous in calendar time since training and embargo periods are excluded, so read it as a trade-sequence drawdown, not a calendar-time one. GARCH(1,1) persistence (alpha+beta) is fit on the pair's own daily FX log returns, 2015-2022 -- a property of the price series, unrelated to the overshoot signal. Cross-reference against day62's GARCH figure.
