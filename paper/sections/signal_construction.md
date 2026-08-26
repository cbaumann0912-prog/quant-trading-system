# Signal Construction

*Rescoped Day 72 from the Day 59 draft. Covers Strategy 1 (PC2 Carry Regime) only; the remaining five strategies are treated in §4. Strategy 1 was closed null on Day 41. This section describes how the signal was built and why; the verdict and the evidence behind it belong to the results section. All figures are recomputed on the 2011-01-01 to 2023-12-31 development window.*

## Strategy 1 — PC2 Carry Regime

### Economic rationale

The three pairs this strategy was built on — EUR/USD, GBP/USD and USD/JPY — are not three independent assets. Each quotes the dollar, so a common dollar move drives all three simultaneously, and a principal component analysis of their daily log returns recovers that structure directly: the first component accounts for 58.4% of return variance over the development sample and stays above 65% in every year from 2020 onward.

The second component is the object of interest here. It explains 29.2% of variance and loads +0.865 on USD/JPY, +0.474 on GBP/USD and +0.167 on EUR/USD, an asymmetry that has a straightforward reading. The yen is the canonical funding currency of the carry trade: persistently low domestic rates make it the leg investors borrow in order to hold higher-yielding assets elsewhere. A factor dominated by USD/JPY, with the two European crosses contributing far less, is therefore a candidate proxy for the state of that funding trade rather than for the dollar itself.

The distributional evidence supports the reading. Over the development sample PC2 scores carry skewness of −1.04 and excess kurtosis of 28.56, and the most negative score, −0.0835, is 2.05 times the magnitude of the most positive, +0.0406. That asymmetry — long quiet accumulation punctuated by sharp, one-directional reversals — is the documented signature of carry unwinds, in which crowded positions are liquidated together and the funding currency appreciates abruptly. The 2016 Brexit referendum illustrates the mechanism from the other direction: UK-specific political risk weakened the shared dollar structure, PC2 absorbed the idiosyncratic sterling variance, and its variance share reached a sample maximum of 41.6% while PC1's fell to a sample minimum of 52.1%.

The hypothesis under test was that this factor, having identifiable economic content, would carry predictive information about its own subsequent returns.

### Factor construction

The factor is estimated by principal component analysis on the aligned daily log returns of the three pairs, using the implementation in `src/features/pca.py`. The procedure centres the return matrix, forms the covariance matrix, computes its eigendecomposition, and projects the centred returns onto the eigenvectors. All three components are retained.

Two properties were verified numerically rather than assumed. The correlation matrix of the resulting score series is the identity to within machine epsilon, with the largest off-diagonal entry at 1.4 × 10⁻¹⁶ against a float64 epsilon of 2.2 × 10⁻¹⁶. The variance of each score series matches its corresponding eigenvalue to six significant figures. Both checks confirm the hand-written decomposition behaves as the mathematics requires.

Eigenvectors are determined only up to sign, so a convention is required for the loadings to be comparable across estimation windows. Loadings are normalised so that the USD/JPY entry is positive. Without this, an arbitrary sign flip between two windows would invert the signal with no change in the underlying data.

Estimation is confined to the training period. Loadings are fitted on training returns alone, and test-period returns are centred using the **training-period mean** before projection. Centring with the test mean would leak test-period information into out-of-sample scores — a small leak in magnitude, but one that operates directly on the quantity being tested. On the 2011–2020 training split the resulting loadings are +0.1266 on EUR/USD, +0.4557 on GBP/USD and +0.8811 on USD/JPY.

The factor's own return series is constructed as a factor-mimicking portfolio: the training-period loadings applied as weights to contemporaneous pair returns,

    r_PC2,t = w_EUR · r_EUR,t + w_GBP · r_GBP,t + w_JPY · r_JPY,t

This is treated as a proxy for the return the factor represents, not as an investable, capital-allocated portfolio. No financing, sizing or capacity assumption attaches to it.

### The entry rule that was drafted, and the test that was actually run

A threshold entry rule was drafted: standardise the PC2 score, trigger on a crossing of ±z in either direction, and take the position in the direction of the crossing. Event frequency over the development sample:

| Threshold | Events | Per year | Positive | Negative |
|---|---|---|---|---|
| \|z\| > 1.5 | 403 | 31.0 | 194 | 209 |
| \|z\| > 2.0 | 187 | 14.4 | 94 | 93 |
| \|z\| > 2.5 | 95 | 7.3 | 47 | 48 |
| \|z\| > 3.0 | 50 | 3.9 | 27 | 23 |

The 2σ level was the working choice. It yields 14.4 events per year — enough to support inference over the sample — while remaining extreme enough that a crossing plausibly identifies a distinct factor state. At 3σ the rate falls to 3.9 per year, which over the evaluation window leaves too few events to distinguish an effect from noise regardless of what the returns look like.

**This rule was never tested.** Every result that produced Strategy 1's verdict — the Day 38 permutation test, the Day 39–40 IC/IR and breadth work, the Day 41 conditional-IC regression — is computed on the raw continuous PC2 score against one-day-forward factor returns, not on threshold crossings. The table above is a descriptive frequency count that motivated a rule the project then bypassed. It is reported here because the drafted rule is part of the honest record of what was considered, not because it carries evidence.

One constraint on execution follows from the data and is not a matter of preference. Lag-1 autocorrelation of PC2 scores is −0.0366, indicating no meaningful daily persistence in factor shocks. A rule that observes a crossing at the close and enters at the following open is therefore acting on information with no measured tendency to continue. Any tradable version of this signal would require execution at the point of crossing, intraday.

### Why the loadings must be re-estimated

Full-sample loadings are not usable for a signal evaluated on any sub-period. The eigenstructure drifts materially: PC1's variance share ranges from 52.1% in 2016 to 70.5% in 2023, and the full-sample figure of 58.4% understates the post-2020 level by 7 to 12 percentage points. Applying a single set of loadings estimated over all thirteen years to data from any one of them introduces error into both the signal and the risk estimate, and does so using information from the future.

The validation work reported in §4.1 uses a single train/test split at 2020-12-31 rather than rolling re-estimation. This removes the most severe form of lookahead but not all of it, and is treated as sufficient for the question of whether the raw factor carries any predictive content at all. It would not be sufficient for a deployment decision.

### Departure from the pre-registration process

Strategy 1 has no specification file. It emerged from the Day 18–19 factor work and was evaluated through daily audits rather than through the spec-first procedure applied to Strategies 2 through 6, under which a written hypothesis, economic rationale, entry and exit logic, sizing, risk controls and explicit death conditions are fixed before any test is run.

This is recorded as a departure rather than reconciled after the fact. Writing a specification for Strategy 1 now, with its results known, would produce a document indistinguishable in form from the genuinely pre-registered ones and would corrupt the evidential value of the entire roster. The absence is the honest artifact.

---

## Resolution of the four open items from the Day 59 draft

The Day 59 draft closed with four unresolved items. Each was traced to the code that produced the figures in question. All four are resolved; two of them changed what this section can claim.

**1. The standardisation window — resolved: there is no standardisation in the tested signal, and the σ in the threshold table is full-sample.**
`src/signals/pc2_carry.py` contains no division by any standard deviation. `pc2_scores` returns the raw projection of training-mean-centred test returns onto the PC2 loading vector, and that raw score is what every reported test consumes. The σ units in the threshold table come from a separate descriptive line in `research/applied_analysis/day19_pca_forex_pairs.py`, which computes `(pc2 − pc2.mean()) / pc2.std(ddof=1)` over the whole sample. That is a full-sample standardisation and therefore lookahead. It does not contaminate any reported result, because the tests that produced Strategy 1's verdict are rank-based — Spearman IC and sign-flip permutation — and invariant to any monotone rescaling of the score. The correct statement is narrow: the threshold table cannot be used to define a tradable entry rule without first re-deriving σ on a training window, and no such derivation exists in this project.

**2. The n = 1,638 versus n = 934 discrepancy — resolved: 934 is correct, and 1,638 was computed on lockbox data.**
The two figures are not two sample definitions. Commit `293a139` bounded 31 research scripts to `DEV_END = "2023-12-31"` and refreshed the results table in the Day 38 audit from 1,638 / 890 / 748 to 934 / 505 / 429, but left the prose "Sample size" paragraph in the same document carrying the old numbers. The 1,638 figure is a pre-bound count over a test window that ran from 2021-01-01 to the end of the raw files in 2026 — that is, through the reserved 2024–2026 lockbox. Re-running the current pipeline reproduces 934 pooled, 505 positive-signal and 429 negative-signal observations exactly, with training loadings 0.1266 / 0.4557 / 0.8811 matching the audit's own table. **Cite 934.** The stale paragraph is a documentation defect, not a second result, and it is flagged in §4.1.

**3. Exit logic — resolved: there is no exit rule, and the tested quantity is not a trade.**
Entry was drafted; holding period and exit condition never were. What the validation actually pairs is the score at *t* with the factor-mimicking portfolio return realised over *t+1*, via `align_signal_and_forward`. That is a forecast horizon inside a predictive test, not a holding period inside a strategy. The consequence for how the null is read is real and is stated in §4.1: Strategy 1's evidence bears on whether the raw factor score predicts the next day's factor return. It says nothing about whether a threshold-entry, defined-exit trading rule on the same factor would have worked, because no such rule was ever specified or run.

**4. Threshold selection as a trial — resolved: it is a trial, it was never counted, and counting it would not have moved any published p-value.**
The 2σ level was selected from four candidates. Under this project's own rule — trial count is cumulative and includes failed tests — that selection is a researcher degree of freedom and belongs in the count. It never entered it: the cumulative count used for the Benjamini-Hochberg correction in §5 is 6, one per strategy. The omission is recorded rather than retrofitted. It changes no published number, for the reason given under item 1: the threshold rule produced no reported test statistic, so there is no p-value for a larger trial count to correct. The item is kept because the *reason* it is harmless is an accident of which rule got tested, not evidence that the accounting was right.
