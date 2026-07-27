# Signal Construction

*First draft — Day 59, reclaimed time. Covers Strategy 1 (PC2 Carry Regime) only; the remaining five strategies follow in later drafts. Strategy 1 was closed null on Day 41. This section describes how the signal was built and why; the verdict and the evidence behind it belong to the results section.*

## Strategy 1 — PC2 Carry Regime

### Economic rationale

The three pairs in this study — EUR/USD, GBP/USD and USD/JPY — are not three independent assets. Each quotes the dollar, so a common dollar move drives all three simultaneously, and a principal component analysis of their daily log returns recovers that structure directly: the first component accounts for 58.4% of return variance over the full sample and above 65% in every year since 2020.

The second component is the object of interest here. It explains 29.2% of variance and loads +0.865 on USD/JPY, +0.474 on GBP/USD and +0.167 on EUR/USD, an asymmetry that has a straightforward reading. The yen is the canonical funding currency of the carry trade: persistently low domestic rates make it the leg investors borrow in order to hold higher-yielding assets elsewhere. A factor dominated by USD/JPY, with the two European crosses contributing far less, is therefore a candidate proxy for the state of that funding trade rather than for the dollar itself.

The distributional evidence supports the reading. Over the full sample PC2 scores carry skewness of −1.20 and excess kurtosis of 30.5, and the most negative score, −0.0874, is 2.2 times the magnitude of the most positive, +0.0399. That asymmetry — long quiet accumulation punctuated by sharp, one-directional reversals — is the documented signature of carry unwinds, in which crowded positions are liquidated together and the funding currency appreciates abruptly. The 2016 Brexit referendum illustrates the mechanism from the other direction: UK-specific political risk weakened the shared dollar structure, PC2 absorbed the idiosyncratic sterling variance, and its variance share reached a sample maximum of 41.6% while PC1's fell to a sample minimum.

The hypothesis under test was that this factor, having identifiable economic content, would carry predictive information about its own subsequent returns.

### Factor construction

The factor is estimated by principal component analysis on the aligned daily log returns of the three pairs, using the implementation in `src/features/pca.py`. The procedure centres the return matrix, forms the covariance matrix, computes its eigendecomposition, and projects the centred returns onto the eigenvectors. All three components are retained.

Two properties were verified numerically rather than assumed. The correlation matrix of the resulting score series is the identity to within machine epsilon, with the largest off-diagonal entry at 1.4 × 10⁻¹⁶ against a float64 epsilon of 2.2 × 10⁻¹⁶. The variance of each score series matches its corresponding eigenvalue to six significant figures. Both checks confirm the hand-written decomposition behaves as the mathematics requires.

Eigenvectors are determined only up to sign, so a convention is required for the loadings to be comparable across estimation windows. Loadings are normalised so that the USD/JPY entry is positive. Without this, an arbitrary sign flip between two windows would invert the signal with no change in the underlying data.

Estimation is confined to the training period. Loadings are fitted on training returns alone, and test-period returns are centred using the **training-period mean** before projection. Centring with the test mean would leak test-period information into out-of-sample scores — a small leak in magnitude, but one that operates directly on the quantity being tested.

The factor's own return series is constructed as a factor-mimicking portfolio: the training-period loadings applied as weights to contemporaneous pair returns,

    r_PC2,t = w_EUR · r_EUR,t + w_GBP · r_GBP,t + w_JPY · r_JPY,t

This is treated as a proxy for the return the factor represents, not as an investable, capital-allocated portfolio. No financing, sizing or capacity assumption attaches to it.

### Entry rule

The signal is the standardised PC2 score. An entry is triggered when the score crosses a threshold in either direction, with the position taken in the direction of the crossing. Event frequency over the full sample:

| Threshold | Events | Per year | Positive | Negative |
|---|---|---|---|---|
| \|z\| > 1.5 | 469 | 31.3 | 219 | 250 |
| \|z\| > 2.0 | 219 | 14.6 | 104 | 115 |
| \|z\| > 2.5 | 103 | 6.9 | 51 | 52 |
| \|z\| > 3.0 | 63 | 4.2 | 29 | 34 |

The 2σ threshold was the working choice. It yields 14.6 events per year — enough to support inference over the sample — while remaining extreme enough that a crossing plausibly identifies a distinct factor state. At 3σ the rate falls to 4.2 per year, which over the evaluation window leaves too few events to distinguish an effect from noise regardless of what the returns look like.

One constraint on execution follows from the data and is not a matter of preference. Lag-1 autocorrelation of PC2 scores is −0.037, indicating no meaningful daily persistence in factor shocks. A rule that observes a crossing at the close and enters at the following open is therefore acting on information with no measured tendency to continue. Any tradable version of this signal requires execution at the point of crossing, intraday, and the validation work below tests the signal's information content rather than a next-open implementation of it.

### Why the loadings must be re-estimated

Full-sample loadings are not usable for a signal evaluated on any sub-period. The eigenstructure drifts materially: PC1's variance share ranges from 52.1% in 2016 to 70.5% in 2022, and the full-sample figure of 58.4% understates the post-2020 level by roughly 10 to 15 percentage points. Applying a single set of loadings estimated over all thirteen years to data from any one of them introduces error into both the signal and the risk estimate, and does so using information from the future.

The validation work reported in the results section uses a single train/test split at 2020-12-31 rather than rolling re-estimation. This removes the most severe form of lookahead but not all of it, and is treated as sufficient for the question of whether the raw factor carries any predictive content at all. It would not be sufficient for a deployment decision.

### Departure from the pre-registration process

Strategy 1 has no specification file. It emerged from the Day 18–19 factor work and was evaluated through daily audits rather than through the spec-first procedure applied to Strategies 2 through 6, under which a written hypothesis, economic rationale, entry and exit logic, sizing, risk controls and explicit death conditions are fixed before any test is run.

This is recorded as a departure rather than reconciled after the fact. Writing a specification for Strategy 1 now, with its results known, would produce a document indistinguishable in form from the genuinely pre-registered ones and would corrupt the evidential value of the entire roster. The absence is the honest artifact.

---

## Open items for this draft

- **Standardisation window unspecified.** The threshold rule is stated in units of σ, but this draft does not fix whether σ is estimated full-sample, on the training period, or on a rolling window. A full-sample σ would be lookahead. This must be pinned down before the section is final.
- **Sample size discrepancy in the source audit.** `pc2_carry_regime_permutation_test.md` reports pooled n = 1,638 in its sample-size paragraph and pooled n = 934 in its results table. Each is internally consistent with its own subset counts (890 + 748, and 505 + 429), so these are two different sample definitions carrying the same label. Resolve before either number is cited.
- **Exit logic absent.** Entry is specified; holding period and exit condition are not. The validation tested one-day-forward returns, which implies a one-day hold, but that was a testing choice rather than a stated rule.
- **Threshold selection is itself a trial.** The 2σ level was chosen from a table of four candidates. That choice belongs in the cumulative trial count.
