# Underpowered by Design: Six Pre-Registered Null Results in Systematic FX Research

*Cornell Engineering, Summer 2026. Single author. All results computed on 1-minute FX data, 2 January 2011 to 29 December 2023.*

---

## Abstract

Six strategy hypotheses were pre-registered in writing and tested against thirteen years of 1-minute data on ten currency pairs. All six were closed as invalidated. The reserved 2024–2026 holdout was never used for evaluation.

The contribution is not the six nulls. It is four methodological results the testing produced, each anchored to the strategy that surfaced it and each reproducible from the scripts in the accompanying repository. First, a controlled leakage demonstration: moving a volatility-regime classifier's fit from full-sample to inside each walk-forward training window changes 44–79% of out-of-sample regime labels on three pairs and 39–68% across ten, on identical data and identical code. Second, portfolio aggregation manufacturing significance: ten signals at mean cross-correlation 0.3652 have an effective breadth of 2.33, and the pooled book posts a Sharpe of 1.0464 at bootstrap p < 0.00001 while no individual component clears p = 0.05 and the cross-pair mean information-ratio interval contains zero. Third, two pre-registered validation criteria that read as rigorous and could not discriminate — one tested whether a coefficient differed from zero but never tested its sign, and passed a strategy whose effect ran backwards; the other required an ordering among three cells whose confidence intervals all span zero, and has since returned FAIL, PASS and PASS on the same data under the same rule, its verdict turning on four trades out of 1,588. Fourth, the power ceiling as arithmetic rather than opinion: at thirteen years a true Sharpe near 0.55 is required to reach t = 2, intraday sampling does not relax it because the binding constraint is calendar span, and five of the six candidates were untestable at this sample size rather than merely wrong.

The last result implies a design error that is quantified rather than gestured at. The universe was chosen for breadth without checking what it cost in span. Dropping one pair would have bought 8.8 years and moved the required Sharpe from 0.555 to 0.428, which is the difference between a study that could resolve its own effect sizes and one that could not.

---

## 1. Introduction

This paper reports a systematic research program that produced no strategy.

Six hypotheses were written down before any of them was tested. Each specification fixed, in advance, the hypothesis, the economic rationale, the entry and exit logic, the sizing and risk controls, and the explicit conditions under which the strategy would be declared dead. Each was then tested. All six failed. A reserved slice of data — 1 January 2024 onward — was set aside for a single unbiased evaluation of anything that survived, and because nothing survived, it was never used to evaluate anything.

The question this paper asks is what that outcome is evidence about. Six failures could mean six bad hypotheses. They could equally mean one badly designed study, repeated six times. The answer matters, because the two readings imply opposite next moves: better hypotheses in the first case, a different study design in the second.

**The answer, stated up front, is mostly the second.** Five of the six candidates failed at effect sizes the sample could not have resolved even had they been real. That is not a rhetorical hedge. The t-statistic on an annualized Sharpe ratio grows roughly as the Sharpe times the square root of elapsed years, so thirteen years of data requires a true Sharpe near 0.55 before an estimate clears t = 2. Sampling the same thirteen years at one-minute rather than daily frequency does not help, because the binding quantity is calendar span, not observation count. The study was underpowered before it began, and it was underpowered because of a universe choice made for reasons that were never checked against their cost.

Four findings came out of the testing, each anchored to the strategy that produced it.

**A controlled leakage demonstration** (§5.1). The volatility-regime strategy's classifier was originally fit on the full sample and later refit inside each walk-forward training window. Nothing else changed — same data, same code, same fold geometry. Between 44% and 79% of out-of-sample regime labels changed, and all three pairs then tripped the strategy's own pre-registered turbulent-share failure condition. The leaked fit was not adding noise; it was systematically undercounting turbulence in one direction.

**Aggregation manufacturing significance** (§5.2). The intraday overshoot book pools ten pairs. Its Sharpe of 1.0464 carries a block-bootstrap p below 0.00001, and it contains no component that reaches significance on its own. At a mean cross-pair correlation of 0.3652 the effective breadth is 2.33 of 10, and had the trades been independent the book Sharpe would have been 0.4221. Nearly the whole book-level result is the correlation adjustment.

**Criteria that do not test their hypothesis** (§5.3). Two pre-registered rules failed in the same way: they looked rigorous and had no power to discriminate. One asked whether an interaction coefficient differed from zero and never asked its sign, passing a strategy whose momentum effect inside its own trading regime was −0.22% over 26 days. The other required a monotone ordering across three threshold cells whose confidence intervals all include zero; its deciding gap carries t = 0.05, and regenerating the trade list flips the verdict.

**The power ceiling is arithmetic** (§5.4). The constraint above is not a limitation to be noted and moved past. It is the study's central result, and it is quantified: the vendor carried every pair in the universe well before 2011, the binding constraint on a common start date was one pair, and dropping it would have bought 8.8 years.

### Positioning

Each of these failure modes is documented in the literature. This paper's claim is narrower and, I think, more useful: it reports a complete pre-registered program in which those failure modes were instrumented and measured as they occurred, including the ones that were caught only on re-examination after a verdict had already been issued.

Harvey, Liu and Zhu (2014) collect 316 factors proposed in the published literature and argue that, given the extent of the search behind them, a newly discovered factor needs a t-ratio above 3.0 rather than the conventional 2.0. Their sharpest observation for present purposes is about what the record cannot show: "we do not observe the factors that were tested but failed to pass the usual significance levels." This paper is a small attempt at the complementary record — six hypotheses, every test run on each, reported whether or not the result was flattering.

Bailey, Borwein, López de Prado and Zhu (2013) formalise how quickly a search over strategy configurations manufactures an in-sample result with zero expected out-of-sample Sharpe. Their Theorem 1 gives a minimum backtest length as a function of the number of independent configurations tried; the worked illustration is that with only five years of data, trying more than 45 independent configurations makes an in-sample annualized Sharpe of 1 with an expected out-of-sample Sharpe of zero close to guaranteed. §5.4 here is the same trade seen from the other side: not how many configurations a fixed sample supports, but how much sample a fixed effect size requires.

Bailey and López de Prado (2014) introduce the deflated Sharpe ratio, which corrects an observed Sharpe for selection bias under multiple testing and for non-normal returns, and state the principle this project adopted as a process rule: "Not controlling for the number of trials involved in a particular discovery leads to over-optimistic performance expectations." The trial count used here is cumulative across the roster and includes failed tests. §4 also records one trial that should have entered that count and did not.

Lo (2002) derives the distribution of the Sharpe ratio and shows that annualizing by the square root of the sampling frequency is valid only under restrictive conditions, with the error reaching 65% in his hedge-fund example once serial correlation is accounted for. This paper hits that result twice: once in the treatment of annualization factors, which are computed empirically from each series' own index rather than assumed (§2), and once in §6, where the surviving book's returns are shown to be serially dependent at a horizon that the block bootstrap's block length does not fully cover.

### What this study is

A single author, one retail data vendor, no replication, and no independent review of the code that produced any number reported here. That bounds the claims in specific ways. It does not bound them in others: the failure modes reported in §5 are properties of the design and of the arithmetic, not of the data vendor, and the leakage demonstration in particular is a before-and-after comparison on identical inputs where the only thing that changes is where the model is fit.

Every figure in this paper is computed by a script in the accompanying repository. Where a figure could not be verified, it is marked rather than estimated.

---

## 2. Data

### Source and coverage

Ten currency pairs of 1-minute OHLCV bars: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD, EUR/GBP, EUR/JPY and EUR/CHF. The development sample runs from 2 January 2011 to 29 December 2023 — 12.99 years — and contains 47,287,601 minute bars. Timestamps are local-time-naive, formatted `%Y%m%d %H%M%S`.

| Pair | Minute bars | Daily closes | Flat bars | Duplicate stamps |
|---|---:|---:|---:|---:|
| EUR/USD | 4,766,619 | 4,056 | 1.49% | 300 |
| GBP/USD | 4,759,785 | 4,055 | 1.51% | 300 |
| USD/JPY | 4,700,657 | 4,056 | 2.12% | 300 |
| USD/CHF | 4,722,674 | 4,055 | 3.29% | 299 |
| AUD/USD | 4,749,069 | 4,055 | 1.76% | 300 |
| USD/CAD | 4,691,825 | 4,055 | 2.49% | 299 |
| NZD/USD | 4,715,513 | 4,055 | 2.49% | 300 |
| EUR/GBP | 4,725,060 | 4,055 | 1.96% | 300 |
| EUR/JPY | 4,782,753 | 4,056 | 0.55% | 300 |
| EUR/CHF | 4,673,646 | 4,055 | 3.65% | 300 |

Coverage is continuous. The largest gap between consecutive daily observations is 3 days in every pair — ordinary weekend closure — and no pair has a gap exceeding 4 days anywhere. Across all 47.3 million bars there are no missing closes, no non-positive prices, and no bars violating the OHLC ordering constraints.

Eight three-month interbank rate series accompany the prices, covering the United States, euro area, United Kingdom, Japan, Switzerland, Canada, Australia and New Zealand. These are **monthly**, not daily: 156 observations each over 2011-01 to 2023-12. Any carry or rate-differential factor built from them inherits that frequency, and a monthly rate aligned onto a daily price panel is a step function rather than a daily observation.

### Three hard limits

**Volume is identically zero** in all 47,287,601 rows across all ten files. No volume-based feature is constructible, and none is used anywhere in this study.

**No bid or ask is recorded.** Each bar carries a single price series. Every transaction-cost figure in this paper therefore rests on assumed ECN spreads rather than observed ones. No spread reported here has been measured.

**Timestamps duplicate across daylight-saving transitions.** Each pair contains 299 or 300 duplicated minute stamps, occurring exclusively in October of 2019 through 2023. The stamps carry no timezone, so the repeated autumn hour appears twice. Daily resampling collapses these silently; they remain a live concern for the intraday work in §4.6, which reads minute bars directly.

Between 0.55% and 3.65% of minute bars are flat, high equal to low. These concentrate in thin sessions and, for EUR/CHF, in the Swiss National Bank floor period: that pair is the highest in the universe at 3.65% overall, with flat shares of 7.00% in 2012 and 7.55% in 2014 against 1.75% to 3.62% in every year from 2015 on. Flat bars are retained rather than filtered.

### Resampling and the evaluation window

Statistical work outside §4.6 is conducted at daily frequency. Minute bars are resampled by last observation within each calendar day, dropping empty days, giving 4,055 or 4,056 daily closes per pair. Aligned across all ten, the panel is 4,056 rows from 2011-01-02 to 2023-12-29 with 7 missing cells; complete-case daily log returns number 4,053.

Returns are log differences of consecutive daily closes. Log returns are additive across time, which is what allows aggregation over arbitrary lookback and holding windows by summation, and are closer to normal for small daily FX moves. Where returns are aggregated across pairs at a single date rather than across time, simple returns are used, since log returns are not additive across a portfolio.

The evaluation window is **2011-01-01 to 2023-12-31**. An earlier draft of this section named 2015–2022; that window describes no result in this study and has been dropped. Twenty-six research scripts carry `DEV_END = "2023-12-31"` as a module constant with `START = "2011-01-01"` as the corresponding convention, and every verdict in §4 was computed on that range. Two 2015 dates survive in the repository and neither defines an evaluation window: a GARCH warm-up boundary in the figure generators, and a command-line default for `--train-start`.

### The Sunday session, and two annualization factors

The FX week opens at 17:00 ET on Sunday. Median minute-bar counts per calendar day are 1,435 on Monday, 1,437 Tuesday through Thursday, 1,020 on Friday and 416 on Sunday, with no Saturday bars. Daily resampling therefore emits a Sunday observation built from roughly a seven-hour session — 671 to 672 such bars per pair, 16.5% to 16.6% of the daily sample. They behave as their length implies: on EUR/USD the mean absolute daily log return is 0.00159 on Sunday bars against 0.00319 to 0.00428 on weekdays.

The daily index consequently yields an empirical annualization factor of 312.19 or 312.27 for every pair, against the 260.57 a weekday-only index would give. This study annualizes using the factor computed from each series' own index rather than any fixed convention.

The larger factor is not a bias, and the arithmetic is worth stating because it is counterintuitive. Excluding Sunday bars does not remove the Sunday price move; it merges that move into the Friday-to-Monday return. Both partitions describe the same path — total EUR/USD log return over the sample is −0.1861 including Sunday bars and −0.1904 excluding them. Annual variance is the sum of per-period variances and that sum is invariant to how the year is partitioned, so σ√K estimates the same annual volatility either way: the coarser partition raises the per-period standard deviation from 0.004718 to 0.005116 by almost exactly the factor by which it lowers √K. Annualized Sharpe on EUR/USD is −0.1719 including Sunday bars and −0.1776 excluding them, a ratio of 0.97.

The residual three percent is aggregation and serial dependence rather than session structure, and is an instance of Lo's (2002) general result that √K annualization is sensitive to autocorrelation in the underlying returns. That sensitivity applies at any frequency and is treated in §6.

What the empirical factor protects against is a fixed constant. Annualizing the same series with a hardcoded 252 gives −0.1544 against the measured −0.1719 — a discrepancy with nothing behind it.

Two different factors appear in this paper and the distinction matters. The daily panel measures 312.19–312.27 observations per year because the vendor buckets the Sunday open as its own date. The intraday session book of §4.6 measures 259.44 sessions per year over the same 12.99 years, because that strategy trades one 09:00–13:00 session per weekday and never trades the Sunday stub. Neither is a correction of the other; they describe two observation streams drawn from the same bars, each computed from its own index. Applying either to the other's returns would misstate the annualized Sharpe by roughly 10% — the kind of error that survives review because both constants look plausible.

### Embargo, purging, and overlapping outcomes

Walk-forward evaluation enforces a directional gap between the end of each training window and the start of the corresponding test window. The embargo is 5 rows of the resampled daily index throughout — weekends are already dropped, so an embargo "day" is a row, not a calendar day — sized to exceed the longest feature lookback so no test-set feature reads training data. The gap is directional: trailing training rows adjacent to the boundary are not purged, because features are causal and the leakage risk runs only forward.

A second leakage channel arises from the target rather than the features. Forward returns over h bars are forward-looking by construction, so for the final h bars of any test window the target price lies beyond the window boundary. Those observations are masked rather than scored, and every scored bar resolves its full holding horizon inside its own test window.

Overlapping forward returns remain a design limitation. Consecutive h-bar forward returns share h−1 bars, so the effective number of independent observations in a window of n scored bars is roughly n/h rather than n. §5.2 and §4.2 both turn on this quantity.

### Reserved holdout

The interval from 1 January 2024 onward is reserved for a single unbiased evaluation of a strategy that has survived every development test. No research script reads it, no strategy was evaluated or selected on it, no parameter was fit to it, and no result in this paper is computed on it.

One qualification belongs on the record even though it has since been closed. Until late in the project four real-data tests parsed the raw files with no end date, so the test suite computed on the reserved rows on every run. All four are numerical-equivalence checks — a Johansen trace test against the hand-derived eigenvalue problem, and three comparisons of the `scipy` Markowitz solution against the closed-form KKT solution — so nothing was ever selected, fitted or evaluated on that data, but the claim as originally written was stronger than the code supported. Those tests are now bounded to the development window like every research script. What remains true of the whole project is that reading a file loads all of its rows before a date filter applies; no computation anywhere uses a row dated after 29 December 2023.

### The universe was chosen without checking its cost

One design decision conditions everything that follows, and belongs here rather than only in the limitations. Ten pairs at thirteen years was chosen for breadth. The breadth it delivered, measured on the one book that pooled all ten, was 2.33 effective independent signals out of 10 (§5.2).

The span that breadth cost is quantified in §5.4. The 2011 start is a download choice rather than a vendor limit, the binding constraint on a common start across all ten is NZD/USD at 2005-08, and dropping that single pair moves the common start to 2002-03 — 8.8 additional years, taking the Sharpe required for t = 2 from 0.555 to 0.428.

---

## 3. Research Design

This section describes the protocol as a protocol, before any result. The nulls in §4 are only interpretable against it.

### Pre-registration

Each strategy has a specification written before any test was run, in twelve fixed sections: hypothesis, economic rationale, prior evidence, signal definition, entry, exit, sizing, risk controls, cost assumptions, the falsification battery, the failure conditions, and the verdict rule. The specifications are unedited from when they were written; only the status line is amended on closure. They are reproduced verbatim in Appendix A.

The section that does the work is the one naming the conditions under which the strategy is declared dead. Fixing those in advance is what converts a disappointing result into a verdict rather than a reason to keep looking. §5.3 reports two cases where that machinery ran correctly and still produced the wrong answer, because the conditions themselves could not discriminate.

### The seven-stage pipeline

1. **Pre-registration**, as above.
2. **Power gate.** `compute_required_sample_size` and `compute_achieved_power` are run before data is touched. If the sample cannot resolve the effect size the hypothesis implies, the hypothesis is not tested.
3. **Signal construction.** `SignalBuilder` takes any `signal_fn(data, lookback) -> pd.Series` under a causal-by-convention contract, with `validate_no_lookahead` as a mechanical leakage check. That check is a falsification test: passing it does not certify the absence of leakage.
4. **Walk-forward validation.** Expanding or rolling windows, with every fit — including regime classifiers and volatility models — refit inside the training window. Purged cross-validation with overlap purging and embargo is the research tool; walk-forward is the deployment-realistic backtest. They are not substitutes.
5. **Statistical validation.** Permutation tests, block-bootstrap intervals that respect serial dependence, deflated Sharpe, and Benjamini-Hochberg correction applied across the whole roster rather than within a strategy.
6. **Cost gate.** Round-trip spread, rollover and implied trade count against a breakeven Sharpe.
7. **Verdict**, documented either way, with the failure mode named.

### Trial counting

The trial count is cumulative across the roster and includes failed tests, following the principle Bailey and López de Prado (2014) make explicit: not controlling for the number of trials leads to over-optimistic performance expectations. The count used for the Benjamini-Hochberg correction in §5 is 6, one per strategy.

That count is not complete, and the gap is recorded rather than repaired. Strategy 1's entry threshold was selected from four candidate levels (§4.1). Under this project's own rule that is a researcher degree of freedom and belongs in the count. It never entered it. The omission moves no published p-value, for a reason that is an accident rather than a defence: the threshold rule was drafted and then bypassed, so it produced no test statistic for a larger trial count to correct. Strategy 4b's audit separately records that roughly 26 window, lookback, regime and universe configurations were explored in one session-variant pass, best of them t = +0.89. That exploration is logged in the repository but is likewise not in the roster count.

### The lockbox

The interval from 2024-01-01 is opened only on a PASS. Six candidates failed without it being opened, which means one clean out-of-sample test remains unspent.

The reasoning for keeping it sealed through six failures is worth stating, because the opposite instinct is strong. Opening the holdout to confirm a failure spends the only unseen data in the project and buys nothing, since the strategy is already dead by its own pre-registered criteria. Worse, it creates a live temptation: if the confirmation came back positive by chance, there would be a dead strategy with a flattering out-of-sample number attached to it and no clean way to discount that number afterwards.

### Process failures, stated here rather than buried

Four departures from the intended discipline are part of the record.

**The specification deadline slipped by eleven days.** The plan required a written one-page specification per candidate by day 30. No specification work is dated to day 30 anywhere in the repository. The specifications for two of the three original candidates were written during a backlog sprint on day 41, and both strategies were tested and discarded the following day.

**Strategy 1 never got a specification at all.** It originated in principal-component work on days 18–19 and was evaluated through daily audits rather than the spec-first process. Writing one now, with the results known, would produce a document formally indistinguishable from the genuinely pre-registered ones and would corrupt the evidential value of the whole roster. The absence is the honest artifact.

**Strategy 4b's specification is still marked draft on sizing and risk.** It was a redeployment of an existing leg rather than a new hypothesis, and the specification was never completed to the standard of the other five.

**The power gate became binding only at day 57** — after five of the six candidates had already been tested. It is stage 2 of the pipeline above because that is where it belongs, not because it was there from the start. It is the single largest process change the six nulls produced, and it arrived late enough to be useless for the program it came out of. §5.4 is the argument for why it should have been first.

### What is computed from scratch, and what is not

The statistical machinery is derived and implemented rather than imported: ordinary and regularized regression, eigendecomposition and principal components, GARCH by maximum likelihood, purged cross-validation, block bootstrap, permutation testing, multiple-testing correction, deflated Sharpe, information coefficient and ratio, Markowitz and risk parity, Kelly, value-at-risk and conditional value-at-risk, and geometric-Brownian and Ornstein-Uhlenbeck simulation.

A short list of standard test statistics is called through established libraries rather than rebuilt, and is named explicitly in Appendix C: the augmented Dickey-Fuller and KPSS tests, the Johansen trace test, ARIMA state-space fitting, Ljung-Box and Jarque-Bera, distribution CDFs, and the numerical optimizers themselves. In each case the critical-value tables or the solver are the imported part and the objective functions are not. Two components were originally scoped for first-principles implementation and ended up on that list; they are listed rather than quietly reclassified.

---

## 4. Strategy Roster and Verdicts

Six hypotheses, one subsection each, in the same five-part shape: hypothesis, economic rationale, pre-registered criterion, outcome, and what killed it. The subsections are deliberately of similar length. No strategy gets extra space for having a prettier number.

There is no common performance table, and the omission is deliberate. The six criteria are heterogeneous by design — information-coefficient tests, three different operationalizations of a nonlinearity claim, two-leg and three-way interaction regressions, and a Sharpe and permutation battery. Forcing a shared metric would mean re-running each strategy on a basis it was not pre-registered under, which is the precise move the pre-registration exists to prevent. The comparison below reports criterion and outcome, not a common Sharpe.

| # | Strategy | Closed | Criterion that failed |
|---|---|---|---|
| 1 | PC2 Carry Regime | Day 41 | Unconditional and conditional predictability both null across three tests |
| 2 | Momentum with ML Regime Filter | Day 42 | Base momentum IC null under two independent methods |
| 3 | OU Half-Life Mean Reversion | Day 42 | Nonlinearity null in all three operationalizations |
| 4 | Vol Regime Breakout / Mean-Reversion | Day 49 | Reversion leg null at p = 0.563; both legs required |
| 4b | Momentum-Only Pooled Book | Day 57 | Robustness-1 null at ten pairs; original pass ran backwards |
| 5 | Month-End FX Rebalancing Flow | Day 57 | Significant with the sign inverted |
| 6 | Intraday Overshoot Reversal | Day 57 | R1 monotonicity, R3 per-trade permutation, and H2 |

### 4.1 PC2 Carry Regime

**Hypothesis.** The second principal component of daily returns on EUR/USD, GBP/USD and USD/JPY carries predictive information about its own subsequent returns.

**Rationale.** The three pairs all quote the dollar, so a common dollar move drives all three and the first component recovers it: 58.42% of return variance over the development sample. The second component explains 29.22% and loads +0.865 on USD/JPY, +0.474 on GBP/USD and +0.167 on EUR/USD. The yen is the canonical funding currency of the carry trade, so a factor dominated by USD/JPY with the European crosses contributing far less is a candidate proxy for the state of that funding trade rather than for the dollar. The distributional evidence supports the reading: PC2 scores carry skewness of −1.04 and excess kurtosis of 28.56, and the most negative score, −0.0835, is 2.05 times the magnitude of the most positive, +0.0406 — the long-quiet-then-sharp-reversal signature of carry unwinds.

**Criterion.** Significance of the information coefficient between the PC2 score and one-day-forward factor-mimicking returns, under a permutation test, surviving multiple-testing correction across the tests run that day. Then, conditionally, significance of the interaction term in a regression of forward returns on the score, a volatility regime variable, and their product.

**Outcome.** Three independent tests, all null. Loadings are fit on training returns to 2020-12-31 and test returns are centred on the training mean; the resulting test sample is 934 aligned observations. Pooled IC is 0.0378 at permutation p = 0.2557. Split by signal sign, the positive subset gives IC 0.0405 at p = 0.1928 and the negative subset IC 0.0910 at p = 0.0400 — the only sub-threshold value among three exploratory splits, and neither Bonferroni nor Benjamini-Hochberg rejects it. A purged cross-validation comparison found no leakage-inflation artifact. The conditional test, on 909 observations, returns an interaction coefficient of −18.868 at p = 0.365 with R² = 0.0031, and four robustness cells across two regime definitions — 0.0693 at p = 0.153, 0.0004 at p = 0.986, 0.0866 at p = 0.070, −0.0477 at p = 0.335 — that are unanimously indistinguishable from zero.

**What killed it.** Not the p-values, which are merely unremarkable, but the reliability of the design that produced them: the interaction regression's condition number is 2.10 × 10¹⁰, above the threshold at which the design matrix is treated as near-singular, so the standard errors and the p-values derived from them are numerically unreliable in either direction. A model explaining a third of one percent of forward-return variance, from a near-singular fit, with four unanimous robustness nulls behind it, has nothing in it.

**A trial not counted, and a threshold never tested.** An entry rule was drafted on threshold crossings of the standardised score, with event counts of 403, 187, 95 and 50 at 1.5σ, 2.0σ, 2.5σ and 3.0σ. The 2σ level was chosen from those four. That selection is a trial and belongs in the count. It never entered it, and — separately — the rule itself was never tested: every result above uses the raw continuous score. The σ in that table is also a full-sample standard deviation, which is lookahead; it does not contaminate the reported results only because they are rank-based and invariant to monotone rescaling.

### 4.2 Momentum with ML Regime Filter

**Hypothesis.** Test A: the Spearman information coefficient between a trailing 26-day return and a forward 5-day return is significantly positive on EUR/USD, GBP/USD and USD/JPY. Test B, conditional on A: a machine-learning regime filter improves it.

**Rationale.** Time-series momentum is among the most widely documented effects in currencies. If it is present at this horizon in these pairs, a regime classifier conditioning on volatility and rate differentials should sharpen it.

**Criterion.** Test A was pre-registered as fatal: Test B would not be run unless a base effect existed unconditionally. A pass required p < 0.05 **and** IC > 0.

**Outcome.** Six tests, zero significant, five of six point estimates carrying the wrong sign. Two independent methods were run. Method 1 subsamples every fifth row so no two observations share a forward window, then permutes the forward returns 10,000 times: EUR/USD IC −0.0393 at p = 0.2617, GBP/USD −0.0471 at p = 0.1868, USD/JPY −0.0167 at p = 0.6361, all on n = 805. Method 2 keeps all overlapping daily rows and permutes in contiguous blocks of 31, preserving the dependence the overlap induces: EUR/USD −0.0292 at p = 0.3501, GBP/USD −0.0453 at p = 0.1490, USD/JPY +0.0017 at p = 0.9599, on n ≈ 4,024.

**What killed it.** No base effect, so Test B was moot — a regime filter cannot condition an effect that is absent unconditionally, and searching for one anyway is the standard route to manufacturing a finding from noise.

**The overlap correction earns its place here**, and this subsection is where the paper's methodological argument first bites. Comparing each permutation p against the naive Spearman p implied by SE(ρ) = 1/√(n−1): on the non-overlapping subsample the two agree to three decimals on all three pairs, ratios of 1.0. On the overlapping sample GBP/USD gives a naive t of −2.87 and a naive p of 0.0041 — conventionally significant — against a block-permutation p of 0.1490, an inflation factor of 36.4. Treating 4,023 overlapping observations as independent counts roughly 130 independent windows about 31 times each. The non-overlapping arm functions as a control: where dependence is removed by construction the two p-values coincide, which rules out the permutation machinery itself as the source of the gap.

**Window alignment was the original result.** An earlier construction overlapped the trailing and forward windows and returned an IC near +0.30, roughly six times anything observed after the fix. A result that large in daily FX should be treated as evidence of a coding error before it is treated as evidence of an anomaly.

### 4.3 OU Half-Life Mean Reversion

**Hypothesis.** Test 1: the z-scored deviation of price from its rolling moving average mean-reverts. Test 2: large-magnitude deviations are *more reliable* entry signals than small ones.

**Rationale.** If a price series reverts to a trailing mean with an estimable half-life, the magnitude of a deviation should carry information about the speed or reliability of the reversion, and a threshold entry can exploit it.

**Criterion.** The nonlinearity claim was pre-registered in three independent operationalizations, with any single failure fatal: reversion *time* by pool, forward *return* by pool, and the information coefficient between deviation magnitude and subsequent return.

**Outcome.** Test 1 passed and Tests 2, 2b and 2c all failed. On the z-series built from a 100-day moving average and rolling standard deviation, the augmented Dickey-Fuller statistic is −6.1475, −6.1831 and −6.8496 across the three pairs at p ≈ 0, and KPSS fails to reject stationarity at 0.1510, 0.1028 and 0.4076 — the two tests agree, which is the stronger form of the conclusion since their null hypotheses are opposite. Fitted OU θ is 0.01953, 0.01966 and 0.02187, giving half-lives of 35.5, 35.3 and 31.7 days.

Test 2, on reversion time, gives differences of −2.390 days at p = 0.6199, +5.023 at p = 0.2679 and −0.006 at p = 1.0000 — signs disagreeing across pairs, with USD/JPY's two pools differing by six thousandths of a day. Test 2b, on 5-day forward return, gives −0.6493% at p = 0.0262, +0.2179% at p = 0.7073 and −0.2461% at p = 0.3874. Test 2c, on the information coefficient, gives −0.2365 at p = 0.0643, −0.0674 at p = 0.6270 and −0.0295 at p = 0.8280.

**What killed it.** EUR/USD's Test 2b result is *significant in the wrong direction*: the hypothesis predicts large deviations earn more, and they earn 0.65 percentage points less. Across nine tests the estimates disagree in sign and, where consistent, lean negative.

**Two defects are recorded rather than repaired.** The block-bootstrap interval on θ is unusable and is reported as such: [0.0604, 0.0781] for EUR/USD against a point estimate of 0.0195, an interval excluding its own point estimate by a factor of three. Resampling 20-day blocks injects artificial discontinuities at every join and an OU fit reads those as fast reversion. Separately, the moving-average window search that was supposed to select the parameter swept its grid against minute-indexed prices with no resampling, so a grid intended as 78 to 624 trading days actually swept 78 to 624 minutes. Repeated at daily frequency it still finds no plateau — the half-life to window ratio stays inside 0.29 to 0.56 across an eightfold range — so the defect did not change the conclusion drawn from it, which is the lucky case rather than a defence.

### 4.4 Volatility Regime Breakout / Mean-Reversion

**Hypothesis.** Momentum works better in turbulent volatility regimes and mean-reversion works better in calm ones, so a regime-gated two-leg strategy beats either leg unconditionally.

**Rationale.** Regime dependence in currency return dynamics is documented, and a classifier that identifies the state should let each leg trade only where it works.

**Criterion.** Two interaction regressions, pooled across three pairs and all out-of-sample walk-forward folds, with the regime composite refit inside every training fold. A leg passes only if a reliability gate on condition number and variance-inflation factors passes for both the primary and alternate-window fits, the primary interaction p is below 0.05, the alternate-window fit is also below 0.05 with the same sign, and a 1,000-permutation dummy shuffle is below 0.05. Any single null kills the leg, and a strategy-level pass requires **both** legs.

**Outcome.** The momentum leg passed on every criterion — primary b3 = −0.00356 at p < 0.0001, alternate window −0.00170 at p = 0.0043 with matching sign, permutation p = 0.0010, condition numbers 4.06 and 4.17 with all variance-inflation factors near 1.00. The reversion leg failed: primary b3 = +0.00028 at p = 0.56308, and the permutation test agrees at p = 0.5485. The alternate-window fit alone reaches significance at p = 0.0006, but the specification requires primary significance regardless of what the alternate window shows. Both fits are on 6,563 pooled out-of-sample observations.

**What killed it.** Section 10 requires both legs. One leg passing is a different, smaller strategy than the one hypothesized, and adopting it would be selecting on the result.

**The leg that passed is the subject of §5.1 and §5.3.** Its pass was obtained under a classifier fit on the full sample; refitting inside each training window changed 44% to 79% of out-of-sample regime labels. And the criterion that passed it tested only whether b3 differed from zero, never its sign.

### 4.5 Momentum-Only Pooled Book

**Hypothesis.** The momentum leg from strategy 4, redeployed as a standalone pooled book across all ten pairs, retains its edge.

**Rationale.** The leg had passed its own two-leg battery. Extending it from three pairs to ten should raise breadth without changing the mechanism. This is a redeployment of an existing leg, not a new trial.

**Criterion.** The same Section 10 battery as strategy 4, applied at ten pairs: reliability gate, primary interaction p, alternate-window robustness with matching sign, and permutation.

**Outcome.** FAIL on both counts — the extension and the re-examination. Across ten pairs on 21,873 pooled out-of-sample observations the primary fit gives b3 = −0.00137 at p = 0.000032 and the permutation agrees at p = 0.0010, but the alternate 156-day regime window gives b3 = −0.00030 at p = 0.3992. Any single null kills the leg, and moving the regime window from 78 to 156 days shrinks the coefficient roughly sixfold. At three pairs the same swap shrank it about twofold and held significance; more pairs made the result fragile to one design choice rather than uniformly noisier.

**What killed it, and it is the more important half.** The sign problem sits in the *original three-pair pass*, not in the extension. The hypothesis is that momentum works better in turbulent regimes. For the validated three, b1 = +0.0014 and b3 = −0.00356, so momentum's total effect inside the turbulent regime is b1 + b3 = −0.0022. The signal is ±1, so a +1 momentum reading predicts about −0.22% over the following 26 days in exactly the regime the strategy trades. The ten-pair fit agrees at −0.0021, and an independently computed information coefficient agrees again at −0.104 and −0.094. The strategy passed its criterion while running backwards. §5.3 develops this.

**The specification is still marked draft on sizing and risk**, which is recorded here rather than repaired.

### 4.6 Intraday Overshoot Reversal

**Hypothesis.** Intraday price displacement beyond a volatility-scaled threshold during the 09:00–12:00 window overshoots and partially reverts by 13:00, and the effect is stronger when the displacement happens faster (H2).

**Rationale.** Fast displacement is more likely to reflect liquidity demand than information, and liquidity-driven moves revert.

**Criterion.** Nine checks including a block-bootstrap p on the book Sharpe, the predicted sign, threshold monotonicity across k ∈ {1.5, 2.0, 2.5} (R1), sign stability and post-break significance across a structural break (R2), a per-trade permutation of the fade direction (R3), a reliability gate, a cost gate at 2.0 pips, Benjamini-Hochberg rank against the roster, and the H2 speed interaction. Section 10 fails on any single failed criterion.

**Outcome.** FAIL on two, on figures regenerated in the project's declared environment. The primary result looks strong — book Sharpe +1.0464 with block-bootstrap p < 0.00001, ten of thirteen years positive, both structural-break halves significant at +1.1980 and +0.9555, and the cost gate cleared at 4.54× with a maximum viable round-trip spread of 9.078 pips against a 2.0-pip gate. Two checks fail. R3's per-trade permutation, which is the test the specification actually named, gives p = 0.1349; the plain t-test agrees at t = +1.52, p = 0.1281. H2 gives b3 = −0.0465 at p = 0.679 with Cohen's d = −0.2686 — the prediction was b3 > 0 and significant, and the point estimate runs backwards.

A third check, R1's threshold monotonicity, was recorded as a failure when this strategy was closed and passes on regeneration: mean returns of +0.135, +0.832 and +0.874 basis points across k = 1.5, 2.0 and 2.5, on 6,706, 3,211 and 1,588 trades. §5.3 treats that reversal as its own finding rather than as a correction.

**What killed it.** Either of the two would have. The mechanism failure is the most consequential: the specification pre-committed to treating a null H2 as fatal to the mechanism story even if H1 passed, and a second mechanism — an NYSE-open alternative — failed as well, leaving no working explanation for the effect.

**This strategy is not the near-miss, and §5.2 explains why.** Its Sharpe is the artifact. At a cross-pair correlation of 0.3652 the effective breadth is 2.33 of 10; no individual pair reaches significance, the largest |t| among the ten is 1.69, the bootstrap interval on cross-pair mean information ratio contains zero, and had trades been independent within a day the book Sharpe would have been +0.4221 rather than +1.0464. Achieved power is 0.6419 against 1,279 active days where 1,860 are needed for 80%.

---

## 5. Four Findings

This section is why the paper exists. Each result is cross-cutting rather than specific to the strategy that produced it, and each is anchored to that strategy so the reader can check it.

### 5.1 A controlled leakage demonstration

Leakage is usually argued about rather than measured, because the leaked and unleaked versions of a study are rarely both available. Here they are, and the comparison is unusually clean: identical data, identical code, identical fold geometry. The only thing that changes is where the volatility-regime classifier is fit.

Strategy 4's classifier was originally fit once on the full sample and applied across all folds. Refitting it inside each walk-forward training window — the correct construction, and the one the specification had always called for — changes the out-of-sample regime label on a large fraction of days. On the three pairs where it was first measured, walk-forward and full-sample labels agree on 49.41%, 55.74% and 20.52% of out-of-sample days, so **44% to 79% of labels flip**. In absolute terms that is 1,107 of 2,188 bars on EUR/USD, 968 of 2,187 on GBP/USD and 1,739 of 2,188 on USD/JPY. Repeating the same check across all ten pairs gives agreement between 32.2% and 60.8%, so **39% to 68% flip**. Both ranges are legitimate; they are the three-pair and ten-pair runs of the same procedure.

USD/JPY is the worst case and worth stating plainly: 20.52% agreement means the two classifiers disagree roughly four days in five, which is worse than a coin flip among three labels would produce.

Two features make this more than a large number. First, the disagreement is directional, not noise: the leaked fit was systematically undercounting turbulence, and once the classifier is refit correctly all three pairs trip the strategy's own pre-registered 40% turbulent-share failure condition — a condition the leaked version passed. A leak that scattered labels randomly would not move a share statistic in one direction. Second, the classifier's underlying correlation input is itself unstable across folds: every pair's rho flips sign at least once across the window transitions, with per-window values ranging from −0.197 to +0.824 on EUR/USD alone. The regime signal was never stable enough to justify a single full-sample estimate, and the full-sample fit concealed exactly that.

The general point is about what a full-sample fit is doing. It is not merely using a little future information at the margin. Where the quantity being fit is itself regime-dependent, the full-sample estimate is an average over states that the strategy is supposed to be distinguishing between, and the resulting classification can be worse than uninformative — it can be confidently wrong in a direction that flatters the hypothesis.

### 5.2 Aggregation manufacturing significance

The intraday overshoot book is the cleanest instance of this in the project, and it is worth being explicit that it is presented as the cleanest *instance of an artifact*, not as the strategy that almost worked.

The book pools ten pairs, equally weighted, one trade per pair per triggering session. Its headline numbers are strong: Sharpe +1.0464, block-bootstrap p < 0.00001 on a 21-day block, bootstrap interval [+0.5019, +1.5785], ten of thirteen years positive, and both halves of a structural-break split significant at +1.1980 and +0.9555.

Underneath, nothing is significant. No individual pair reaches p = 0.05. The largest |t| among the ten is 1.69, on EUR/JPY. The bootstrap interval on the cross-pair mean information ratio is [−0.0011, +0.0551] and includes zero. Two of the ten pairs have negative mean trade returns.

The reconciliation is correlation. Mean cross-pair correlation of the daily book components is +0.3652, so the effective breadth is

    BR_eff = N / (1 + (N − 1) ρ) = 10 / (1 + 9 × 0.3652) = 2.33

rather than 10. Had the trades been independent within a day, the same per-trade statistics would imply a book Sharpe of +0.4221. The observed +1.0464 is therefore mostly the difference between treating ten correlated series as ten independent draws and treating them as 2.33.

Diversification is a real source of risk-adjusted return and this is not an argument that it is illusory. The argument is narrower: the *p-value* is the artifact. A book-level significance test that ignores cross-sectional dependence is asking whether the aggregate beat zero given a sample size it does not have, and the strategy is left resting on a directional edge that nothing in the analysis separates from zero.

The permutation split says the same thing twice, and the specification chose the harder of the two. Permuting the fade direction per trade gives p = 0.1349. Permuting per day — which preserves which pairs traded together — gives p = 0.0010. The per-day test asks whether the book beats a coin flip; the per-trade test asks whether the *direction* does. Preserve the cross-pair structure and the book looks decisive; strip it and the direction does not clear. The specification pre-committed to the per-trade version, which is the one that fails.

The same mechanism appears one level down in §4.2, where treating 4,023 overlapping daily observations as independent inflates a p-value by a factor of 36. Dependence treated as independence is a single error that shows up at the observation level, the trade level and the book level, and it is the most common way a null result acquires a small p-value.

### 5.3 Criteria that do not test their hypothesis

Both instances below were pre-registered, both read as rigorous, and both issued a PASS on evidence that could not support one. They failed in different ways.

**Sign untested.** Strategy 4's day-48 rule asked whether the interaction coefficient b3 differed from zero and never asked its sign. The momentum leg passed it: primary b3 = −0.00356 at p < 0.0001, alternate window −0.00170 at p = 0.0043, permutation p = 0.0010, reliability gate clean. The hypothesis was that momentum works *better* in turbulent regimes. Momentum's total effect inside the turbulent regime is b1 + b3 = +0.0014 − 0.00356 = −0.0022 — a +1 momentum reading predicting about −0.22% over 26 days in exactly the regime the strategy trades.

The audit that issued the pass recorded "sign is consistent (negative) across the primary and alternate-window fits" as *support* for it. The observation was correct and its significance was inverted: the consistency was real and the consistent sign was backwards. The honest description of that pass is that the interaction is reliably non-zero and reliably points the wrong way.

**No power to discriminate.** Strategy 6's R1 required a monotone ordering of mean returns across three entry thresholds. When the strategy was closed the realised ladder was +0.149, +0.852 and +0.795 basis points at k = 1.5, 2.0 and 2.5, on 6,708, 3,206 and 1,584 trades. It was not monotone, and R1 was recorded as one of three failures.

The check could not have been informative either way. All three confidence intervals span zero. The deciding gap, between k = 2.0 and k = 2.5, was +0.050 basis points against a standard error of 1.014 — a t-statistic of 0.05. Four trades out of 1,588 flip the ordering, and at an excess kurtosis of +22.5 a handful of extremes decides it. Year by year the ladder is monotone in 2 of 11 tradeable years.

**The demonstration is that it then flipped, twice.** Regenerating the trade list from the same raw bars with the same committed code — once on Linux under Python 3.10, once on the project's declared Windows and Python 3.12 environment — produces ladders of +0.134 / +0.825 / +0.874 and +0.135 / +0.832 / +0.874. Both are monotone. **R1 passes in both.** The strategy verdict is unchanged in both, because Section 10 fails on any single criterion and R3 per-trade and H2 fail throughout.

The size of the move is four to nine trades in roughly 3,200. Its origin is only partly the platform. The two regenerations agree with each other on eight of ten pairs and differ from the closing figures on the same four pairs in the same direction, which rules out the linear-algebra backend for those four; only two pairs genuinely differ between the two builds, and there the GARCH maximum-likelihood optimum is the explanation. For the other four no committed change accounts for the difference at all — the validation script, the signal module, the volatility module, the pinned dependencies and the raw files are all unchanged since the strategy was closed, and the pipeline consumes no randomness anywhere between the minute bars and the trade list. The closing figures appear to have come from a state that is not in the version history.

That is a second and worse problem, and §6 treats it as a limitation. For R1 the consequence is simply this: a pre-registered criterion has now returned FAIL, PASS and PASS on the same hypothesis, the same data and the same rule. It is uninformative in both directions, and the fact that it currently passes is a cleaner demonstration of that than its original failure was. A criterion that can be flipped by regenerating the inputs it was computed from is not measuring the market.

The general claim is that a criterion with no power to discriminate is worse than having no criterion at all, because it emits a PASS that reads as evidence. Both failures were caught only on re-examination, after the criteria had already been used to issue verdicts — which is the part that should worry a reader, since the re-examination was discretionary.

**One consequence was adopted.** A direction requirement was added to the verdict rule, and strategy 5 became its first use: its three-way interaction coefficient is b4 = −1.200 × 10⁻⁴ at p = 0.00093, with a block-bootstrap interval of [−2.52 × 10⁻⁴, −2.03 × 10⁻⁵] at p = 0.024 and a narrow-window robustness check at p = 0.024, on 67,057 pooled observations with a condition number of 48.2 and maximum variance-inflation factor of 1.000. Every criterion passed except the sign, and the prediction was b4 > 0. Under the old rule it would have been a pass.

### 5.4 The power ceiling is arithmetic

The t-statistic on an annualized Sharpe ratio is approximately

    t ≈ SR × √(years)

so thirteen years requires a true Sharpe near 0.555 to reach t = 2. This is not a modelling choice. It follows from the estimator's standard error and it does not care what the hypothesis is.

The consequence that matters is that **intraday sampling does not relax it.** Sampling the same calendar span at one-minute rather than daily frequency multiplies the observation count without multiplying the elapsed years, and the elapsed years are what the expression contains. Strategy 6 is the demonstration: it trades an intraday session, has 3,211 trades, and still has achieved power of 0.6419 against a requirement of 1,860 active days where it has 1,279.

This changes what the six nulls mean. Five of the six candidates were **untestable at this sample size rather than wrong**, and that distinction changes what to do next. A wrong hypothesis should be abandoned. An untestable one should be either abandoned or given a sample that can resolve it, and choosing between those requires knowing which situation you are in.

**The fix was available the whole time.** The 2011 start is a download choice, not a vendor limit; the vendor carries every pair in the universe well before it. The binding constraint on a common start date across all ten pairs is NZD/USD, available from 2005-08.

| Universe | Pairs | Common start | Dev years | Sharpe for t = 2 | Active days | Adequately powered |
|---|---:|---|---:|---:|---:|---|
| As downloaded | 10 | 2011.00 | 13.0 | 0.555 | 1,278 | no |
| Vendor maximum | 10 | 2005.58 | 18.4 | 0.466 | 1,811 | no |
| Drop NZD/USD | 9 | 2002.17 | 21.8 | 0.428 | 2,147 | **yes** |
| USD majors only | 6 | 2000.42 | 23.6 | 0.412 | 2,319 | **yes** |

Active-day counts assume the overshoot book's realised trigger rate and 259.44 sessions per year, and are reproduced from the project's own span-constraint note, which was computed against the strategy's closing figures. The regenerated run of §4.6 gives 1,279 active days against an 80% threshold of 1,860, rather than 1,278 against 1,871 — differences of one day and eleven days, which move no row's verdict and are recorded for consistency rather than because they matter.

Dropping one pair buys 8.8 years and takes the hurdle from 0.555 to 0.428, a 23% reduction. It moves the overshoot book from roughly 1,278 active days against a requirement near 1,870 to 2,147 against the same requirement. **The study would have been adequately powered.**

The trade is breadth against span, and §5.2 has already priced the breadth: ten pairs delivered an effective breadth of 2.33. The marginal pair was worth far less than the marginal year, and the universe was chosen without anyone computing either quantity. That is the design error this paper is named for.

Two caveats keep this from being a free lunch, and both are real. Pre-2005 currency markets differ in structure — electronic execution was less dominant, spreads were wider, the participant mix differed — so a longer sample buys power at the cost of assuming the effect is stable across a market-structure change. And retail vendor history that old is lower quality and would need validating for gaps and stale quotes before it could be trusted. Neither caveat rescues the original choice, because neither was the reason for it.

---

## 6. Limitations

Ordered by how much each threatens the conclusions above.

**1. The universe design error.** This is the central limitation and §5.4 quantifies it: ten pairs at thirteen years delivered an effective breadth of 2.33 while costing 8.8 years of span, and a nine-pair universe from 2002 would have been adequately powered for the effect sizes tested. Stated as a specific error rather than a general note about sample size, because it is one. Every conclusion in §4 about a hypothesis being null should be read as conditional on a study that could not have resolved most of them.

**2. No observed spreads.** Every cost figure rests on assumed ECN quotes. Strategy 6 cleared its gate at 4.54× with a maximum viable round-trip spread of 9.078 pips, which is a wide margin — but it is a margin against an assumption. Entry slippage is unmeasured throughout, and 1-minute closes are not tradeable prices. A cost gate computed against assumed spreads can only tell you that costs are not obviously binding; it cannot tell you they are not binding.

**3. Serial dependence, and a block length that does not fully cover it.** The overshoot book's daily returns are autocorrelated: Ljung-Box on the 3,369-day series gives p = 0.00053 at 10 lags, p = 0.00075 at 21 lags and p = 0.00091 at 40 lags. This invalidates the naive Sharpe t-statistic outright, which is why every significance claim on that book is quoted from the block bootstrap rather than the t-test.

The block length itself needs stating and justifying, and the honest answer is partial. The block used is 21 days. The n^(1/3) heuristic for a block bootstrap of the mean gives 15 at this sample size, so 21 is conservative relative to that rule. But the measured autocorrelation function exceeds its ±0.0338 significance band at lags 2, 6, 19, 30, 33 and 34 — six exceedances in 40 lags where roughly two would be expected by chance, and three of them beyond the block length. That lag-by-lag measurement was taken on a rebuild of the book whose trade list differs from the published one by nine trades in 3,211, or 0.3%, which is far too small to move an autocorrelation function but is stated because it was not recomputed on the published set. Dependence at lags 30–34 is not preserved by 21-day blocks. The direction of the resulting error is toward significance: a block shorter than the dependence horizon manufactures independence the data does not have, which narrows the interval and shrinks the p-value. The p < 0.00001 on the book Sharpe should therefore be read as an upper bound on the evidence, not a measurement of it. Estimating the block length from the series' own autocorrelation structure rather than fixing it by hand is the standard remedy, and is not implemented here.

**4. Reproducibility, and one result that failed it.** No number in this paper has been independently reproduced, and no line of the code that produced them has been reviewed by anyone else. Three specific consequences are visible in the record.

The suite's coverage was never measured, so "tests pass" is weaker evidence than it appears. The numerical results are platform-sensitive in places: two of the ten pairs in strategy 6's book produce different trade counts under a different linear-algebra backend, because the volatility model is fit by maximum likelihood.

The third is worse than either. Strategy 6's closing figures **cannot be regenerated from any committed state of the repository**, on the author's own machine or elsewhere. Four of the ten pairs move identically under two independent rebuilds, and no change to the validation script, the signal module, the volatility module, the pinned dependencies or the raw files accounts for it; the pipeline consumes no randomness at all. The figures published in §4.6 and §5.2 are the regenerated ones and the closing figures are recorded as superseded. This is the same defect an earlier audit found in the bootstrap confidence interval — a published result that could not be reproduced from the code alleged to produce it — one level further up the stack, and it is the strongest argument in this paper for treating reproducibility as something to be tested rather than assumed.

**5. Two strategies fall short of the pre-registration standard.** Strategy 1 has no specification at all and was evaluated through daily audits; strategy 4b's specification is still marked draft on sizing and risk. Both are recorded as departures in §3 rather than repaired, because a specification written after the results are known is formally indistinguishable from a genuine one and would devalue the four that are genuine.

**6. Overlapping forward returns.** Consecutive h-bar forward returns share h−1 bars, so the effective observation count in a test window is roughly n/h rather than n. Every per-window estimate in §4 carries that discount, and §4.2 shows what ignoring it costs: a factor of 36 on a single p-value.

**7. Retail vendor data quality, and the structure break any extension must assume across.** The three hard limits in §2 — zero volume, no bid or ask, duplicated daylight-saving timestamps — bound what can be asked of this data at all. And the fix proposed in §5.4 comes with its own assumption: pre-2005 currency markets differ enough in structure that extending the sample backward trades one problem for another. A structural-break test across the 2005 boundary should be a precondition of that extension rather than an afterthought.

---

## 7. Conclusion

Six hypotheses were pre-registered and tested. All six were closed as invalidated. The reserved holdout was never used for evaluation and remains available for a single unbiased test.

The transferable claim is not that these six strategies do not work. On the arithmetic in §5.4, five of them were never resolvable on the sample chosen to test them, so their nulls are weak evidence about the strategies and strong evidence about the study. What transfers is narrower and, I think, more useful:

- **A validation protocol can pass a strategy that runs backwards.** Strategy 4's momentum leg cleared a four-part battery — reliability gate, primary significance, alternate-window robustness with matching sign, and a permutation test — while predicting −0.22% over 26 days in the regime it traded. Every check fired correctly. None of them checked the direction.
- **A book-level p-value can be manufactured by correlation.** Ten marginal signals at ρ = 0.3652 produced a Sharpe of 1.0464 at p < 0.00001 while no component cleared p = 0.05 and the cross-pair information-ratio interval contained zero. The aggregate was not evidence about the strategy; it was evidence about the covariance matrix.
- **A criterion can be unfalsifiable in practice while looking rigorous on paper.** Strategy 6's threshold-monotonicity check turns on four trades in 1,588, and has returned FAIL, PASS and PASS on the same data under the same rule.
- **Power is a design parameter, not a diagnostic.** It was treated as something to compute after a disappointing result for the first five candidates, and became a binding pre-condition only at day 57. Reversing that order is the single change most likely to have altered this program's outcome.

**What changes for a seventh attempt.** The power calculation comes before the hypothesis, and it is applied to the data window as well as to the effect size: fix the required Sharpe first, then choose the universe and span that deliver it. Nine pairs from 2002 is the starting design, with a structural-break test across 2005 as a precondition. Every coefficient-based criterion carries a direction requirement. Every criterion is checked for discriminating power before it is adopted — the specific test being whether the criterion's own confidence intervals could distinguish the outcomes it is supposed to order.

**What the sealed holdout is worth.** One clean out-of-sample test, unspent after six failures. That is the whole of it, and it is worth more than six confirmations of results that are already dead by their own criteria. The temptation to spend it on a failure — to check, just once, whether the null was real — is exactly the temptation the protocol exists to remove, and the fact that it was never spent is the one process claim in this paper that the record fully supports.

---

# Appendices

## Appendix A — The pre-registrations

Six specifications for seven entries in the roster. Strategy 1 has none, for the reason given in §3: it originated in the day 18–19 factor work and was evaluated through daily audits rather than the spec-first process, and writing one now with the results known would produce a document formally indistinguishable from the genuine ones.

The text below is reproduced mechanically from the repository rather than retyped. The specifications are unedited from when they were written; only the status line is amended on closure. The authority for that claim is the git history, not this copy — `git log --follow` on each file shows the content commit and the status amendment separately.


### `research/strategies/s02_momentum_ml_regime/spec.md`

#### Strategy Specification — FX Momentum with ML Regime Filter

**Strategy name:** FX Momentum with ML Regime Filter
**Date drafted:** Day 42 (2026-07-07)
**Status:** **DISCARDED** — Test A (base momentum effect) failed after correcting a window-alignment bug; Claim B (regime-dependence) never reached, moot

---

##### 1. Hypothesis

**Hypothesis:** Currencies exhibiting recent positive (negative) returns tend to continue rising (falling) over a subsequent period — a persistent anomaly (Menkhoff et al.: up to 10% p.a. spread between winners and losers) not explained by standard risk factors. Momentum's predictive power is regime-dependent: it works better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. An ML classifier gates exposure based on predicted regime state (trending vs. choppy).

**What would falsify this hypothesis?**

| Test | Question | Result |
|------|----------|--------|
| Test A — Base momentum effect | Is Spearman IC between trailing 26-day return and forward 5-day return significantly positive? (permutation test, tested via two independent methodologies) | **FAILED** — all 3 pairs, both methods |
| Test B — Regime-dependence | Does an ML regime filter improve momentum's predictive power in trending vs. choppy states? | **NOT RUN — moot**, no base effect to condition |

Test A failing was pre-registered as fatal to the base momentum claim, which in turn makes Claim B untestable in any meaningful sense — a regime filter cannot rescue a signal that does not exist unconditionally, and searching for regime-conditional effects in the absence of an unconditional one is a known path to manufacturing spurious findings from noise.

---

##### 2. Economic Rationale

**Why should this edge exist?**
Framed by Menkhoff et al. as "limits to arbitrage" — exploiting the momentum anomaly exposes traders to risks (crash risk, funding risk, unpredictable reversals) not captured by simple covariance-based risk measures, which is why professional capital hasn't arbitraged it away. **This project's implementation did not find evidence of the base effect in the tested construction (Section 1); see Section 10 for the full record, including a construction bug that initially produced a spurious positive result.**

**Who are the natural counterparties?** Not explicitly stated in the source write-up.

**Why hasn't this been arbitraged away already?**
Limits to arbitrage — crash risk, funding risk, and unpredictable reversal risk are not captured by simple covariance-based risk measures.

**What known macro/structural regime does this depend on?**
Momentum tends to work better in trending macro environments and tends to fail or reverse violently around regime transitions or crowded-trade unwinds. This was the explicit motivation for the ML regime gate — the classifier was meant to capture exactly the condition under which momentum's underlying mechanism (slow information diffusion, underreaction) is actually operative versus not. **Moot given Test A's failure — there is no base mechanism for a regime filter to condition on.**

---

##### 3. Data Required

**Instruments:** EUR/USD, GBP/USD, USD/JPY (all three).

**Data frequency:** Daily close, resampled from 1-minute OHLCV via `.resample('D').last()`.

**Lookback window required:** 26 trading days (1 month, per project's 26-days/month convention, standardized with OU Half-Life strategy work).

**Any external/exogenous data needed?** None used in Test A. Candidate ML classifier features for a regime filter (Claim B) were never specified or sourced, since that stage was never reached.

---

##### 4. Signal Logic

**What was computed, step by step (Test A only; live signal never designed):**

1. Compute daily log returns: `log_returns_t = log(price_t / price_{t-1})`
2. Compute cumulative sum of log returns for index-based windowing: `cumsum_t = sum(log_returns_{1..t})`
3. Trailing signal: `trailing_return_t = cumsum_t - cumsum_{t-26}` (26-day cumulative log return)
4. Forward outcome: `forward_return_t = cumsum_{t+5} - cumsum_t` (5-day forward cumulative log return, zero overlap with trailing window)
5. Test A computed Spearman IC between trailing_return and forward_return across all valid rows

**Signal frequency:** Would recompute daily in a live signal; Test A itself was evaluated both as a non-overlapping weekly subsample (Method 1) and as full daily overlapping observations with block-based permutation (Method 2).

**Parameters and values used:**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| Lookback | 26 trading days | 1 month, per project's 26-days/month convention |
| Holding period | 5 trading days | 1 week |
| Significance threshold | α = 0.05, permutation-based | Consistent with framework convention given confirmed fat tails/volatility clustering |

---

##### 5. Entry Rule

**Never designed — moot given discard.** No live entry rule was built; Test A operated only on the trailing/forward return relationship, not on an executable signal.

---

##### 6. Exit Rule

**Never designed — moot given discard.**

---

##### 7. Position Sizing Rule

**Never designed — moot given discard.**

---

##### 8. Risk Controls

**Never designed — moot given discard.**

---

##### 9. Failure Conditions

**Realized.** Per the pre-registered falsification criterion (Section 1), Test A found no significant positive IC between trailing and forward returns after correcting a construction bug — condition met, strategy discarded.

---

##### 10. Statistical Validation Plan — Executed Record

**In-sample testing — results:**

- **Test A** (base momentum effect): Spearman IC between 26-day trailing return and 5-day forward return, tested via two independent methodologies to check robustness to overlapping-window handling.
  - **Initial construction contained a window-alignment bug**: a `.shift(HOLDING-1)` operation was misapplied, causing the "forward return" window to overlap almost entirely with the trailing window (differing by only ~1 day) rather than being genuinely forward-looking. This produced a spuriously large IC (~0.29–0.33, p≈0.00000, all pairs, both methods) that was flagged as implausibly high relative to published FX momentum literature (typically much weaker effects) and investigated before being trusted.
  - **After correction** (explicit cumulative-sum index construction, verified by manual inspection of window date ranges to confirm zero overlap between trailing and forward windows):
    - Method 1 (non-overlapping weekly subsample, n=805/pair): EUR/USD IC=-0.039 (p=0.262), GBP/USD IC=-0.047 (p=0.187), USD/JPY IC=-0.017 (p=0.636)
    - Method 2 (overlapping daily data, block permutation with block_size=31, n≈4,024/pair): EUR/USD IC=-0.029 (p=0.350), GBP/USD IC=-0.045 (p=0.149), USD/JPY IC=+0.002 (p=0.960)
  - **Both methods agree**: no significant positive IC in any pair. **Test A: FAILED.**

**Multiple-testing correction:** N/A — strategy discarded before reaching cross-candidate correction stage. This candidate's null result should still be counted when computing multiple-testing correction across the full three-candidate shortlist, per project rules.

**Deflated Sharpe Ratio:** Not computed — no return-generating signal was built.

**Confidence intervals:** Not computed for this strategy — Test A used permutation-based significance testing directly, no bootstrap CI was needed at this stage.

**Out-of-sample / walk-forward validation:** Not reached.

**Leakage check:** The *corrected* trailing/forward return construction has zero window overlap, confirmed via manual date-range inspection (Section 4). The *initial, buggy* construction had a de facto leakage-like flaw — the "forward" window improperly reused most of the trailing window's own days, which is functionally similar to a lookahead/overlap leakage bug even though it arose from an indexing error rather than a genuine future-information leak. Worth flagging as a general lesson: an implausibly strong initial result (IC ~0.30, well above literature norms) was the signal that prompted the check that caught this.

**Transaction costs / breakeven:** Not reached.

---

##### 11. Open Questions / Known Gaps

- Strategy is discarded; Sections 5–9 were never completed, consistent with not investing further design effort in a strategy whose base premise failed.
- Claim B (ML regime-dependence) was never tested — this was a deliberate methodological choice (testing a regime filter on a non-existent base effect risks manufacturing a spurious regime-conditional finding from noise), not an oversight.
- The initial construction bug (window-alignment error producing spurious IC ~0.30) is a useful cautionary case for future strategy work: an unusually strong result relative to published literature benchmarks should trigger verification before being trusted, not be accepted as a good outcome.
- If momentum is revisited in a future 90-day cycle or as a new candidate, the corrected cumsum-based windowing construction (Section 4) is reusable, validated infrastructure — independent of this specific strategy's failed result.

---

##### 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None applied. No statistically-derived input was ever adjusted based on forward-looking judgment, since no return-generating signal was built before the strategy was discarded.

**Reasoning:** N/A.
---

### `research/strategies/s03_ou_halflife_mean_reversion/spec.md`

#### Strategy Specification — OU Half-Life Mean-Reversion (Z-Score Threshold)

**Strategy name:** OU Half-Life Mean-Reversion (Z-Score Threshold, PPP-Motivated)
**Date drafted:** Day 42 (2026-07-07)
**Status:** **DISCARDED** — Test 1 passed; Tests 2, 2b, 2c (nonlinearity claim, three independent operationalizations) all failed

---

##### 1. Hypothesis

**Hypothesis:** The deviation of EUR/USD, GBP/USD, and USD/JPY's price from its rolling moving average, normalized by rolling volatility (z-score), mean-reverts over time. Large-magnitude z-score deviations are more reliable entry signals than small deviations, because small deviations are statistically indistinguishable from noise around a slowly-drifting equilibrium, while large deviations more plausibly reflect a temporary dislocation likely to correct. MA window, volatility window, and z-threshold are determined via purged-CV grid search on in-sample data (Section 10); working parameters used for hypothesis testing (100-day MA, 100-day vol window, |z| ≥ 1.5 split) were never carried forward to live-signal optimization, since the underlying hypothesis was falsified before that stage.

**What would falsify this hypothesis?**

| Test | Question | Result |
|------|----------|--------|
| Test 1 — Base existence | Does the z-score deviation series show statistically significant mean-reversion? (ADF rejects unit root, KPSS fails to reject stationarity) | **PASSED** — all 3 pairs |
| Test 2 — Nonlinearity (reversion speed) | Is mean reversion time for large-\|z\| excursions (peak ≥ 1.5) statistically distinguishable from small-\|z\| excursions? (permutation test on the difference in mean reversion time) | **FAILED** — all 3 pairs |
| Test 2b — Nonlinearity (forward return magnitude) | Is mean 5-day signed forward return from peak significantly larger for large-\|z\| excursions than small-\|z\| excursions? (permutation test on difference in means) | **FAILED** — all 3 pairs (2 of 3 wrong-signed) |
| Test 2c — Nonlinearity (forward return rank) | Is there a significant positive Spearman IC between peak \|z\| magnitude and signed 5-day forward return? (permutation test on IC) | **FAILED** — all 3 pairs, all wrong-signed |

Any one of these tests failing was pre-registered as fatal to the strategy as specified. All three nonlinearity operationalizations failed. Per standing research-integrity rules, this is treated as a closed, discarded result — not retuned or retried further.

---

##### 2. Economic Rationale

**Why should this edge exist?**
Deviations from equilibrium exchange rate levels are constrained by transaction costs and capital requirements on international arbitrage — small deviations aren't worth correcting given those costs, but larger dislocations eventually attract capital that pulls price back. This implementation targeted a related but distinct claim: even without modeling true PPP fair value, the *degree* of statistical dislocation from a smoothed price anchor should predict reversion behavior, because large deviations more plausibly reflect genuine order-flow imbalance (forced hedging, temporary liquidity shortfalls) than small deviations, which are more likely routine noise around a slow-moving equilibrium. **This claim was tested three ways and did not hold in any of them (Section 1).**

**Who are the natural counterparties?**
Hedgers and corporates transacting at prevailing rates regardless of short-term deviation (inelastic flow), other systematic mean-reversion strategies competing for the same edge, and momentum-driven flow that pushes price further from equilibrium in the short run (the source of the dislocations this strategy waits out).

**Why hasn't this been arbitraged away already?**
Capacity constraints and the difficulty of confidently distinguishing "large deviation about to revert" from "start of a genuine trend/regime shift" in real time — this was precisely the discrimination problem Tests 2/2b/2c were designed to validate. Transaction costs on frequent small-deviation trading make pure noise-trading unprofitable, which was part of why this strategy explicitly avoided acting on small deviations. **Given the test results, an equally plausible reading is that this discrimination problem is not actually solvable at this magnitude scale using this z-score construction — i.e., there may be no exploitable difference between "large dislocation" and "ordinary noise" for these three pairs at this timescale.**

**What known macro/structural regime does this depend on?**
Requires a regime where price genuinely reverts to a moving-average-defined equilibrium rather than trending persistently (e.g., sustained rate-differential-driven currency trends, as in strong dollar cycles, would violate this). No explicit regime filter was ever built into this strategy — moot now given the strategy's discard, but worth noting as a possible confound: without a regime filter, the excursion sample (Section 4) pools observations across whatever mix of trending and mean-reverting regimes occurred in this data, which could partially explain why the magnitude-conditioning signal didn't emerge cleanly.

---

##### 3. Data Required

**Instruments:** EUR/USD, GBP/USD, USD/JPY (all three; framework does not support USD/CHF).

**Data frequency:** Daily close. Resampled from 1-minute OHLCV via `.resample('D').last()`. Justification: MA/vol-window construction and OU half-life estimates (~32–35 days) operate on a multi-week timescale, making daily resolution appropriate; intraday noise would not meaningfully improve signal quality at this holding-period scale.

**Lookback window required:** Minimum ~100 trading days before a valid MA/vol-window reading is available (MA_WINDOW = VOL_WINDOW = 100, working values used for hypothesis testing; never advanced to optimization given the discard).

**Any external/exogenous data needed?** None used. True PPP-based fair value (CPI differentials) was considered and explicitly rejected in favor of a moving-average anchor due to frequency mismatch (monthly CPI vs. daily FX) and added data-acquisition burden not justified for this implementation.

---

##### 4. Signal Logic

**What was computed, step by step (for hypothesis-testing purposes; never advanced to a live signal):**

1. Compute rolling moving average of daily close price: `MA_t = mean(price, window=100)`
2. Compute deviation: `deviation_t = price_t - MA_t`
3. Compute rolling volatility of the deviation series: `vol_t = std(deviation, window=100)`
4. Compute z-score: `z_t = deviation_t / vol_t`
5. Track excursions: an excursion begins when `|z_t| ≥ 1.0`, continues while same-sign, tracking a running same-sign peak; a peak-segment ends when magnitude reverts by 1.0 z-unit from the current peak (fixed-absolute reversion rule), or a new peak is set
6. Tests 2/2b/2c each computed a different outcome variable (reversion time, 5-day forward return, IC) conditional on this excursion structure

**Signal frequency:** Recomputed daily, using only trailing data (rolling windows are one-sided/backward-looking — confirmed no lookahead in MA/vol construction).

**Parameters and working values used (hypothesis-testing only, never optimized further):**

| Parameter | Value | Justification |
|-----------|-------|----------------|
| MA window | 100 days | Working value; purged-CV grid search never run given discard |
| Vol window | 100 days | Matched to MA window after diagnosing that a shorter window (originally 20) mechanically undersized the denominator relative to the slow-moving deviation series, inflating z-scores (76–80% of observations above \|z\|=1.0, vs. expected ~32%). Resulting z-score std (~1.7) is elevated above the naive target of 1.0, accepted as an expected property of z-scoring an autocorrelated series (34-day half-life), not treated as a residual bug. |
| Entry threshold | \|z\| ≥ 1.0 | — |
| Large/small pool split | peak \|z\| ≥ 1.5 | Confirmed via threshold-separation script to produce a balanced split (36–38% of observations above, consistent across all three pairs) |
| Reversion-confirmation distance (X) | 1.0 z-unit | Chosen by convention (matches entry threshold scale) after an empirical diagnostic (varying X from 0.3–1.0) showed no plateau, only a monotonic decline attributable to a distance-to-target mechanical confound, not genuine signal |
| Censoring cap | 3× pair-specific OU half-life (~95–106 days) | Computed directly on the z-score series, replacing an earlier, materially longer (~1 year) half-life computed on a different, unrelated price-level deviation object from Day 26 |
| Forward horizon (Test 2b/2c) | 5 trading days | Chosen because it approximated Test 2's observed mean reversion time; **flagged as informed by already-observed data, not independently pre-registered** — logged honestly as a minor deviation from strict pre-registration discipline |

---

##### 5. Entry Rule

**Not finalized — moot given discard.** The hypothesis-testing entry threshold (|z| ≥ 1.0) was used only for excursion detection in Tests 1–2c. No live trading entry rule was ever designed, since the strategy was discarded before reaching that stage.

---

##### 6. Exit Rule

**Not finalized — moot given discard.** The excursion-end construction (Section 4) was retrospective/analytical only, never translated into a real-time exit rule.

---

##### 7. Position Sizing Rule

**Never designed — moot given discard.**

---

##### 8. Risk Controls

**Never designed — moot given discard.**

---

##### 9. Failure Conditions

**Realized.** Per the pre-registered falsification criteria (Section 1), Tests 2, 2b, and 2c all failed to support the nonlinearity claim central to this strategy's differentiation from vanilla mean-reversion. Per standing rule, a failed pre-registered test is fatal and not subject to retuning — this condition was met and the strategy is discarded accordingly.

---

##### 10. Statistical Validation Plan — Executed Record

**In-sample testing — results:**

- **Test 1** (base mean-reversion existence): ADF + KPSS on z-score deviation series. **PASSED** — both tests agree across all three pairs (ADF p ≈ 0.00000, stat -6.76 to -7.51; KPSS p ≈ 0.09–0.10).
- **Test 2** (nonlinearity — reversion time): permutation test, mean reversion time, large (peak ≥1.5) vs. small (peak <1.5) pool. **FAILED.**
  - EUR/USD: n=63, small mean=5.29d, large mean=7.68d, diff=-2.39d (wrong-signed), p=0.620
  - GBP/USD: n=54, small mean=12.05d, large mean=7.03d, diff=+5.02d, p=0.268
  - USD/JPY: n=58, small mean=7.05d, large mean=7.05d, diff=-0.01d, p=1.000
- **Test 2b** (nonlinearity — forward return magnitude): permutation test, mean 5-day signed forward return from peak, large vs. small pool. **FAILED.**
  - EUR/USD: diff=-0.65% (wrong-signed), p=0.026 — significant at 5% in the direction opposite the hypothesis
  - GBP/USD: diff=+0.22%, p=0.707
  - USD/JPY: diff=-0.25% (wrong-signed), p=0.387
- **Test 2c** (nonlinearity — forward return rank): Spearman IC, peak magnitude vs. signed 5-day forward return, permutation test. **FAILED.** All three ICs are negative, i.e. larger excursions rank *lower* on subsequent signed return — the opposite of the hypothesis.
  - EUR/USD: IC=-0.237 (wrong-signed), p=0.064
  - GBP/USD: IC=-0.067 (wrong-signed), p=0.627
  - USD/JPY: IC=-0.030 (wrong-signed), p=0.828
- Significance threshold used throughout: α = 0.05, permutation-based (not parametric), given confirmed fat tails/volatility clustering in this data (Day 4, Day 37 findings).

**n_trials for this hypothesis family: 3** (Tests 2, 2b, 2c) — to be carried forward if any future strategy references this line of research or reuses this z-score construction.

**Multiple-testing correction:** N/A — strategy discarded before reaching cross-candidate correction stage. This candidate's null result should still be counted when computing multiple-testing correction across the full three-candidate shortlist, per project rules (correction applies to the full set tested, not just survivors).

**Deflated Sharpe Ratio:** Not computed — no return-generating signal was ever built; Tests 1–2c operated on the z-score/excursion structure directly, not on realized strategy returns.

**Confidence intervals:** Block bootstrap CI attempted on OU θ (block_size=20) — **result unreliable**, diagnosed as an artifact of block-bootstrap resampling introducing artificial discontinuities at block boundaries, inflating apparent θ in every replicate (bootstrap CI [0.061, 0.078] did not bracket point estimate 0.020). Not used as the basis for any conclusion. Left as an open technical note for any future series with similar autocorrelation structure — appropriate block length likely needs to be ≥ the series' own half-life (~35 days here), consistent with the Politis & White (2004) block-size-selection question already deferred to buffer days (Day 37 audit).

**Out-of-sample / walk-forward validation:** Not reached — strategy discarded at in-sample hypothesis-testing stage, before `WalkForwardValidator`, put off for a later date, would have been relevant.

**Leakage check:** The z-score construction itself (rolling MA, rolling vol) used only trailing data — no leakage in that component. The excursion-based tests (2/2b/2c) were inherently retrospective (require observing the future path to identify a peak and measure outcomes) — valid for hypothesis testing as executed, but this construction could never have been lifted directly into real-time signal logic without modification, which is now moot.

**Transaction costs / breakeven:** Not reached.

---

##### 11. Open Questions / Known Gaps

- Strategy is discarded; Sections 5–8 were never completed and are not being completed retroactively, consistent with not investing further design effort in a falsified strategy.
- Three independent, pre-registered tests of the nonlinearity claim (reversion time, forward-return magnitude, forward-return rank) all failed — this is treated as a well-evidenced negative result, not an inconclusive one.
- Forward horizon (5 days, Tests 2b/2c) was chosen informed by Test 2's observed reversion times — a minor deviation from strict pre-registration, logged honestly rather than treated as a fully blind choice.
- No regime filter or ML component was ever incorporated (Tony's standing requirement) — moot given the discard, but would have been a gap had the strategy survived.
- Reusable groundwork from this candidate, in case referenced by future strategy work: the corrected z-score construction methodology (MA/vol window sizing diagnosis), the OU half-life-on-z-score-series approach to deriving a censoring cap, and the excursion-detection algorithm itself (Section 4) — all validated as sound infrastructure, independent of this specific strategy's failed hypothesis.
- Two earlier diagnostic dead-ends (MA-window-via-half-life-plateau search; X-confirmation-buffer search) were also negative results, documented in the Rolling Decision Log, and are relevant if similar window/threshold-selection procedures are attempted for other candidates — both failed for a related mechanical-confound reason (distance-to-target scaling dominating the signal being measured).

---

##### 12. Judgmental Adjustments to Statistical Inputs

**Adjustment:** None applied. No statistically-derived input was ever adjusted based on forward-looking judgment, since no return-generating signal was built before the strategy was discarded.

**Reasoning:** N/A.
---

### `research/strategies/s04_vol_regime_breakout/spec.md`

#### Strategy Specification: Volatility Regime Breakout/Mean-Reversion

**Date drafted:** Day 42 (2026-07-07), finalized Day 43
**Status:** Complete

##### 1. Hypothesis
FX volatility and correlation structure go through genuine regime shifts (Ang & Bekaert, 2002), not continuous scaling of one process. The best trading rule differs by regime: mean-reversion when calm, momentum/breakout when turbulent. A 2-feature classifier (rolling realized volatility + interest rate differential) identifies which regime is active. The vol-index-proxy leg from the original 3-feature design was dropped: no free, historically complete, FX-specific vol index exists.

**Falsification criteria** (pre-registered Day 43):

1. Test statistic: conditional IC of the trading signal within the classifier-identified regime (must be conditional, not unconditional, since the hypothesis is regime-dependence itself).
2. Primary threshold: p < 0.05, uncorrected.
3. Reliability gate (from the Day 41 PC2 failure): interaction regression condition number < ~1e10 (VIF < 10), main effects mean-centered before the interaction term. This strategy needs two such regressions, one per leg (`signal x regime_indicator`), since it has two signals rather than PC2's one.
4. Robustness: primary test + both robustness checks (Section 10) must unanimously agree. Any single null kills the hypothesis.
5. Multiple-testing: final PASS/FAIL must survive Benjamini-Hochberg correction across all 4 strategies tested project-wide. Since the other 3 are already null, a real finding here needs p < 0.0125 (BH critical value, rank 1 of 4) to survive.

##### 2. Economic Rationale
If the edge exists, it comes from identifying a regime switch before the market fully adjusts. Natural counterparties and why this hasn't been arbitraged away are not addressed in the source write-up. Depends on there being two genuinely distinct regimes, per Ang & Bekaert's state-change claim.

##### 3. Data Required
**Instruments:** EUR/USD, GBP/USD, USD/JPY, daily, via existing DataLoader.

**Rate differentials:** FRED OECD 3-month interbank rates (US: IR3TIB01USM156N, Euro Area: IR3TIB01EZM156N, UK: IR3TIB01GBM156N, Japan: IR3TIB01JPM156N), monthly, in `data/{region}_3m_interbank.csv`. Refresh via `fetch_rate_differentials()` in `src/framework/data_loader.py`.

`eurusd_rate_diff = r_EA - r_US`, `gbpusd_rate_diff = r_UK - r_US`, `usdjpy_rate_diff = r_US - r_JP`. Shifted forward 2 months before forward-filling to daily (OECD publication lag, ~6 weeks) to avoid look-ahead. Creates a 59-day warm-up gap; first valid value 2011-03-01.

Full-sample means are not centered near zero or symmetric across pairs (`eurusd_rate_diff` -1.00, `gbpusd_rate_diff` -0.16, `usdjpy_rate_diff` +1.49), which is why the regime threshold below is z-scored rather than an absolute cutoff.

##### 4. Signal Logic
A simple rolling-threshold classifier, not a Hamilton-filter Markov-switching model (that's put off for a later date per the roadmap). This tests a simplified proxy for Ang & Bekaert's regime concept, not their actual model.

1. **Regime features:** 78-day (1 quarter, `trading_days_per_year // 12 * 3`) rolling realized volatility of log returns, plus the rate differential (2-month-lagged, forward-filled).
2. **Regime combination:** both z-scored and combined via PCA (1st component), sign-normalized so volatility loading is positive. Must be refit inside each walk-forward window in production; the Day 43 script fits full-sample for threshold selection only. Full findings and numbers: `research/strategies/validation_falsification/vol_regime_composite_threshold_selection.md`.
3. **Regime classification (hard switch):** `|composite z| > 1.5` = turbulent, `< 1.0` = calm, `1.0-1.5` = deadzone (no trade). Threshold of 1.5 preserves test power (turbulent is 9.2-15.9% of observations vs. 3.4-5.0% at 2.0).
4. **Turbulent rule, time-series momentum** (Moskowitz, Ooi & Pedersen 2012): `signal_t = sign(P_t/P_{t-78} - 1)`.
5. **Calm rule, price z-score mean-reversion:** `z_price_t = (P_t - mean_26(P)) / std_26(P)`, entry `|z_price| > 2.0`. Threshold chosen from conditional-forward-return evidence (audit doc addendum), not rarity alone. GBP/USD and USD/JPY show a real, increasing reversion effect; EUR/USD shows none, a real finding, not something to explain away.

**Caveat:** thresholds above (1.0/1.5/2.0) were chosen from full-sample descriptive analysis, a mild form of look-ahead in the design process. The actual test of whether the strategy works is Section 10's out-of-sample validation.

**Parameters:**

| Parameter | Value |
|---|---|
| `trading_days_per_year` | 312 |
| Regime window / momentum lookback | 78 trading days |
| Regime thresholds | turbulent > 1.5, calm < 1.0, deadzone 1.0-1.5 |
| Mean-reversion window | 26 trading days |
| Mean-reversion entry threshold | \|price z\| > 2.0 |

##### 5. Entry Rule
Regime classifier (Section 4) gates which rule is active; no trade in the deadzone.

**Momentum (turbulent):** `sign(P_t/P_{t-78} - 1)`, re-evaluated daily, no confirmation lag. Literal MOP convention.

**Mean-reversion (calm), 3-rung ladder:** each rung independent, all conditioned on `|composite z| < 1.0`.

| Rung | Trigger |
|---|---|
| 1 (initial) | `\|price z\| > 2.0` |
| 2 (add) | `\|price z\| > 2.5` |
| 3 (add, cap) | `\|price z\| > 3.0` |

Rung 1 is exceedance-triggered (from flat); rungs 2-3 are crossing-triggered (fire once, on first cross above that level, not every day price stays there). The ladder has no direct academic citation, Gatev, Goetzmann & Rouwenhorst (2006) use single-shot entry. This is a practitioner convention that adds 3 free parameters (rung count, spacing, cap) versus the single-threshold version. Sizing per rung is in Section 7.

##### 6. Exit Rule
**Deadzone is not a forced-exit zone.** An open position rides through the deadzone; it only force-closes on a flip to the *opposite* regime (turbulent to calm, or vice versa).

**Momentum exit:** sign flip only (mirrors entry, no separate stop-loss at this layer).

**Mean-reversion exit, whichever hits first:**
1. Reversion to target band: `|price z| < 0.5`, closes all rungs together (not staged).
2. Time-stop: 26 trading days from the initial rung-1 entry, force-close regardless of price.

If a position rides through the deadzone and the regime never flips before the data ends, mark-to-market and close at the data boundary (bookkeeping, not a trading decision).

##### 7. Position Sizing Rule
**Base method (both legs), ex-ante vol targeting:** `size_t = (target_vol / realized_vol_t) x base_capital`, target vol 40% annualized (MOP's own value, a normalization convention, not a capital recommendation). Uses the same 78-day rolling vol already computed for the classifier rather than MOP's literal EWMA estimate, for parameter parsimony.

**Ladder sizing:** equal weight per rung, `size_t / 3` each, so a fully-built ladder sums to the same `size_t` a momentum position would get at the same vol.

**Hard cap:** required in addition to vol-targeting (which can otherwise blow up as realized vol -> 0). Value set in Section 8.

**Cross-pair:** shared capital pool. If `n` pairs have live positions simultaneously, each position's `size_t` is additionally scaled by `1/n`, rescaled dynamically as positions open/close across pairs.

##### 8. Risk Controls
- **Position-size cap:** 2x the vol-targeted size at this strategy's own historical median realized vol (per pair).
- **Max drawdown:** 25% from peak (strategy-level) triggers an automatic halt for manual review. Looser than a typical 15% single-sleeve default, to accommodate this strategy's regime-timing risk.
- **Concentration:** the Section 7 `1/n` scaling is treated as sufficient; no separate net-USD-exposure cap. Known blind spot (doesn't distinguish 3 independent bets from 3 correlated USD bets), accepted for now.
- **Capital allocation:** not set here, deferred to portfolio-construction time. No capital gets committed before Section 10 clears.

##### 9. Failure Conditions
**Strategy decay:** realized live Sharpe falling materially below the backtest's deflated-Sharpe confidence band (Day 13 standard). Chosen over a pure IC re-test since it also catches execution/cost decay, not just statistical decay.

**Classifier decay:** live regime proportions drifting from the Day 43 baselines (turbulent ~9.2-15.9%, deadzone ~13.6-24.6%). A classifier producing, say, 40%+ turbulent days means the thresholds no longer match current conditions, a distinct failure from the trading rules themselves breaking.

##### 10. Statistical Validation Plan
No separate full-sample screening stage. The primary test runs directly inside each `WalkForwardValidator` test fold, out-of-sample from the start.

**Primary test**, two interaction regressions, pooled across walk-forward out-of-sample folds:
- Momentum: `R_{t+26} = b0 + b1*momentum_signal_t + b2*turbulent_dummy_t + b3*(momentum_signal_t x turbulent_dummy_t) + eps_t`
- Reversion: `R_{t+26} = b0 + b1*price_z_t + b2*calm_dummy_t + b3*(price_z_t x calm_dummy_t) + eps_t`

Both mean-centered before the interaction term. `b3` is the term of interest. Forward horizon: 26 trading days, shared across both legs. Reliability gate: condition number < ~1e10 per leg.

**Robustness checks** (both must agree; either null kills that leg):
1. Alternate regime window: re-run with 156 trading days instead of 78.
2. Permutation test: shuffle regime-dummy labels (preserving base rates), rebuild the null for `b3`, 1000 permutations.

**Verdict:** each leg evaluated independently; strategy-level PASS requires both legs to pass. A momentum-only pass is a different strategy, not this one. Subject to project-wide BH correction (Section 1, item 5) before appearing as a PASS anywhere.

**Lockbox holdout:** a recent slice (e.g. 2024-2026) is reserved and never enters the walk-forward folds at all, development or robustness checks. It is opened once, only if this strategy passes everything above and becomes a genuine deployment candidate. This is a single-use test per hypothesis, not another round of tuning; if the lockbox result disagrees with the walk-forward verdict, that disagreement is reported as-is, not explained away.

##### 11. Open Questions / Known Gaps
- PCA adds no real value over equal-weighting (kept for citability/consistency with PC2, not because it's doing work).
- The ladder (Section 5) has no academic citation; adds 3 free parameters versus single-threshold entry.
- Sizing's vol estimator deviates from MOP's literal EWMA convention (parsimony trade-off).
- Cross-pair concentration doesn't distinguish independent bets from correlated USD bets (accepted gap, Section 8).
- EUR/USD shows no reversion signal; Section 10's per-leg (not per-pair) verdict could mask a pair-level failure if EUR/USD alone fails.
- Threshold selection (1.0/1.5/2.0) has a look-ahead caveat: chosen from the same descriptive data the strategy will later be tested against.
- Dynamic `1/n` rescaling and per-window classifier refitting are not yet implemented; both are real work for SignalBuilder, put off for a later date.

##### 12. Judgmental Adjustments to Statistical Inputs
None of these come from a hypothesis test; each is a documented judgment call.

| Adjustment | Value | Basis |
|---|---|---|
| Position-size hard cap | 2x vol-targeted size at median vol | Risk convention, multiple not tested |
| Max drawdown halt | 25% from peak | Judgmental, accommodates regime-timing risk |
| Ladder rung count/spacing | 3 rungs, z=2.0/2.5/3.0, equal-weighted | Simplicity judgment |
| Cross-pair concentration limit | 1/n scaling only | Accepted blind spot |
| Mean-reversion exit target | \|z\| < 0.5 | Judgmental, trades citability for practicality |
| Fixed time-stop | 26 trading days | Reuses tested horizon, not a derived half-life |
| Shared forward-return horizon (Section 10) | 26 trading days, both legs | Comparability over leg-specific precision |
---

### `research/strategies/s04b_momentum_only_book/spec.md`

#### Strategy Specification: Momentum-Only, Pooled 3-Pair Book

**Date drafted:** 2026-07-19, post-Day-49 decision (not backdated — see Provenance below)
**Status:** Draft, sizing/risk sections need explicit review before this can clear to deployment-candidate status

##### Provenance — why this isn't a new trial

This is not a new hypothesis. It is a narrower redeployment of one leg of strategy #4 (`volatility_regime_breakout_mean_revert.md`), which failed at the strategy level (Section 10 requires both legs; reversion's primary p=0.563 once the lockbox leak was excluded — see `research/strategies/validation_falsification/vol_regime_two_leg_section10_validation.md`). The momentum leg's test — primary interaction regression, both robustness checks, reliability gate — was pre-registered under that spec and passed blind, before either leg's result was known. Nothing here is evidence generated after seeing what worked.

Consequence: this spec does not reset the project-wide multiple-testing count. `n_trials` stays at 4. It also does not include GBP/USD-only reversion as a companion leg — that idea was generated by looking at per-pair results after the pooled reversion test already failed, and there is no unseen dev data left to test it against (the full pre-2024 range was already used in the Day 43 exploratory pass). It's logged as an open, untested idea in `research/strategies/validation_falsification/vol_regime_signal_report_audit.md`, not deployed here.

##### 1. Hypothesis
Time-series momentum (Moskowitz, Ooi & Pedersen 2012) has a real, regime-conditional edge in FX: `sign(P_t/P_{t-78}-1)` predicts 26-day forward returns specifically when the 2-feature volatility/rate-differential composite classifies the pair as turbulent (`|composite z| > 1.5`). Already tested and passed at the leg level (Day 48).

##### 2. Economic Rationale
Unchanged from strategy #4: trend persistence following initial under-reaction to news, plausibly concentrated in turbulent periods when information flow and repricing are both faster. Same unaddressed gap as before — natural counterparties and why this hasn't been arbitraged away are not established, only assumed from the regime-dependence literature.

##### 3. Data Required
Unchanged from strategy #4 Section 3: EUR/USD, GBP/USD, USD/JPY daily via `DataLoader`; rate differentials per the same construction (2-month publication lag, forward-filled).

##### 4. Signal Logic
Unchanged from strategy #4 Sections 4.1–4.3 and 4.4: same 2-feature composite regime classifier (refit per walk-forward window per Day 47), same `|z|>1.5` turbulent threshold, same momentum signal. No reversion leg, no calm-regime logic, no deadzone handling (deadzone and calm are both simply flat for this strategy).

##### 5. Entry Rule
`sign(P_t/P_{t-78}-1)` when `|composite z| > 1.5`, re-evaluated daily, no confirmation lag. Unchanged from strategy #4 Section 5's momentum branch.

##### 6. Exit Rule
Sign flip, or regime flip out of turbulent (to calm or deadzone) forces a close. Unchanged from strategy #4 Section 6's momentum branch — no separate stop-loss at this layer.

##### 7. Position Sizing Rule — OPEN, needs explicit decision
Strategy #4's vol-targeting formula (`target_vol / realized_vol_t x base_capital`, 40% annualized target) and cross-pair `1/n` shared-capital scaling both generalize cleanly to a single-leg book and can likely carry over unchanged. Not yet decided:
- A single pair is turbulent 9.2–15.9% of the time. Pooled across 3 pairs, days with *at least one* pair active are more frequent, but turbulent episodes are not independent across pairs (they share global vol shocks) — the realized pooled activity rate hasn't been measured, only assumed to be higher than any single pair's number. Needs an empirical check before sizing assumptions are trusted.
- Whether idle capital during non-turbulent stretches is acceptable as-is, or needs a secondary use (e.g., a genuinely separate, independently-tested sleeve) — out of scope for this spec, flagged for later.

##### 8. Risk Controls
Carried over unchanged from strategy #4 Section 8 (position-size cap 2x vol-targeted size at median realized vol, 25% max drawdown halt, `1/n` as the only concentration control). These were judgmental, not derived, in the original spec, and momentum-only doesn't change that — they haven't been re-examined for a single-leg book specifically. The correlated-turbulence gap noted in Section 7 above also weakens the `1/n` concentration control here: if turbulent regimes cluster across pairs, `1/n` understates true concentration risk more than it did for the original 2-leg design.

##### 9. Failure Conditions
Same as strategy #4 Section 9 (live Sharpe decay vs. DSR band, classifier regime-proportion drift). Worth flagging plainly: momentum's DSR was 0.1646 (n_trials=4) against an input observed Sharpe of only +0.0289 — the p-value passed cleanly, but the raw signal-level Sharpe this is built on is weak. A passed significance test is not the same claim as "this is a strong edge." Sizing decisions in Section 7 should account for that, not just for the p-value.

##### 10. Statistical Validation Plan
No new test is run. Momentum's primary regression, both robustness checks, and the reliability gate already passed under strategy #4's Section 10 (`research/strategies/validation_falsification/vol_regime_two_leg_section10_validation.md`) — reused here, not repeated.

**Open decision, not resolved by this spec:** the lockbox holdout (2024-2026) was reserved for "this strategy," meaning the original 2-leg design. Momentum-only is a redefinition — narrower scope, different sizing/risk assumptions once Section 7 is settled. Whether momentum's already-passed per-leg test is sufficient to treat this as cleared for the lockbox, or whether the redefinition itself needs a full review pass first, is a judgment call to make explicitly before opening it — not something to default into either direction.

##### 11. Open Questions / Known Gaps
- Pooled-book turbulent-activity rate (Section 7) is asserted, not measured.
- Cross-pair concentration during correlated turbulence episodes is unaddressed (same blind spot as strategy #4, arguably worse here since there's no reversion leg to be active when momentum isn't).
- DSR is weak (0.1646) despite a clean p-value — sizing shouldn't be set as if this were a high-conviction edge.
- Lockbox-eligibility decision (Section 10) not yet made.

##### 12. Judgmental Adjustments to Statistical Inputs
| Adjustment | Value | Basis |
|---|---|---|
| Position-size hard cap | 2x vol-targeted size at median vol | Carried over from strategy #4, not re-derived for single-leg book |
| Max drawdown halt | 25% from peak | Carried over, not re-derived |
| Cross-pair concentration limit | 1/n scaling only | Carried over; weaker justification here per Section 8 |
| Turbulent-regime threshold | \|composite z\| > 1.5 | Reused from Day 43, unchanged |

---

### `research/strategies/s05_month_end_fx_flow/spec.md`

#### Strategy Specification: Month-End FX Rebalancing Flow

**Date drafted:** Day 57 (2026-07-25)
**Status:** Pre-registered. Written before any test was run.

##### Provenance
Nothing in `research/` has examined month-end effects, fix windows, or calendar-timed flows. Every threshold and window in Section 4 is fixed here, before the first test.

This is strategy #5. Four were tested previously (PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion, Volatility Regime Breakout/Mean-Reversion, the last surviving as `momentum_only_pooled_book.md` until `day57_momentum_book_invalidation.md` closed it). Day 56-57 session work explored roughly 26 configurations of that dead hypothesis, all null and all logged there. Honest `n_trials` for the deflated Sharpe here is 5.

**What is borrowed and what is not.** The directional month-end hypothesis (H1) is a replication of Melvin & Prins. 

Three things are not in that literature as far as this repo's reading goes, and they are the contribution:

1. The post-fix reversal test (H2) as a *falsification* of the mechanism rather than a confirmation of the effect. The published work establishes the flow exists; H2 asks whether the price impact is temporary, which is what "non-informational" actually commits you to.
2. Volatility-conditioned flow magnitude (H3), derived from the mechanism rather than fitted, using the GARCH regime classifier built on Days 55-56.
3. A cross-sectional dollar-neutral construction across 10 pairs, where the published treatment is largely USD-centric and per-pair.

If H1 replicates and H2 and H3 are null, the honest result is "known effect reproduced, proposed mechanism refinement unsupported." That is still a result.

##### 1. Hypothesis
FX spot drifts predictably in the final trading days of each calendar month, concentrated around the 16:00 London fix, driven by mechanical portfolio-hedge rebalancing rather than information.

Direction is conditioned on the preceding month's hedging need, proxied by the month's cumulative FX return (Section 4). The effect must be stronger at month end than on other days, and stronger inside the fix window than outside it. Either alone is consistent with something other than the stated mechanism.

Falsification criteria, binding:

1. Test statistic: the month-end × fix-window interaction coefficient from the Section 10 primary regression.
2. Direction requirement: the coefficient must be significant and carry the predicted sign. A significant coefficient of the wrong sign is a FAIL. Added in response to Day 48/57, where a significance-only rule passed a leg whose effect ran backwards.
3. Primary threshold: p < 0.05, uncorrected.
4. Reliability gate: condition number < 1e10, all VIF < 10, main effects mean-centered before the interaction. Precedent: Day 41 PC2 discarded at condition number 2.27e10.
5. Robustness: primary plus both robustness checks must agree unanimously, same sign. Any single null kills it.
6. Multiple testing: must survive Benjamini-Hochberg across 5 strategies. The other 4 are null, so this needs p < 0.01 at rank 1 of 5.
7. Cost gate: net-of-cost annualized return must stay positive at 1.0 pip round trip. Added because the Day 57 session variant produced a positive gross Sharpe entirely consumed by turnover.

##### 2. Economic rationale
The mechanism is a structural, non-informational flow. Institutions holding foreign equity and bond portfolios hedge the currency exposure with forwards. When foreign markets rise over a month, the hedge is now too small relative to the asset position, so the hedger sells foreign currency to restore the ratio. These adjustments cluster at month end and execute disproportionately at the 16:00 London fix, because the fix is the benchmark most hedging mandates are judged against.

Documented in Melvin & Prins (Journal of Empirical Finance, 2015) and related BIS work on fix-window liquidity. Unlike the momentum hypothesis this project spent Days 42-57 on, there is a named counterparty and a reason the flow persists: the hedgers are price-insensitive and mandate-bound, executing at a benchmark regardless of level.

Why it may still fail. Post-2013 manipulation scandals changed fix execution, the window widened from 1 minute to 5 in 2015, and banks changed how they internalize flow. Any effect may have decayed or moved. The sample spans that break, which is why the structural-break check in Section 10 is required rather than optional.

What would make it uninteresting even if significant: an effect smaller than the round-trip spread is a fact about microstructure, not a strategy. Criterion 7 binds.

##### 3. Data
All 10 pairs from the outset, so universe selection cannot become a post-hoc degree of freedom the way it did in `momentum_book_regime_conditional_robustness.md`.

1-minute bars, `data/{pair}.csv`, 2011-01-01 to 2023-12-31 development. Lockbox 2024-01-01 to 2026-05-01 sealed.

Timestamps: files are fixed-offset UTC-5, so convert per `src/features/sessions.py` then DST-aware convert to Europe/London. The fix is a London-local 16:00 event and moves in UTC across the year; a naive UTC cut would smear it.

No rate or equity data needed. The hedging-need proxy is built from FX returns alone, which avoids the publication-lag machinery that complicated the previous strategy.

##### 4. Signal logic
All parameters fixed as of this document. None may be tuned after seeing results.

1. Fix window: 15:30-16:15 Europe/London. Wider than the official 5-minute window because pre-positioning and unwind fall outside it, and 5 minutes of 1-minute bars is too thin to estimate on.
2. Control window: 10:00-10:45 Europe/London. Same duration, liquid, no scheduled benchmark event.
3. Month end: the last 2 trading days of the calendar month. Binary indicator.
4. Hedging-need proxy: `sign(cumulative log return from month start through t-1)`. Uses only information available before the window opens. A month where the base currency appreciated implies hedgers must sell it, so the predicted direction is opposite the month's accumulated move.
5. Signal: `-hedge_need_t` on month-end days, 0 otherwise.
6. Return measured: log return of the fix window, and separately the control window, per pair per day.

| Parameter | Value |
|---|---|
| Fix window | 15:30-16:15 Europe/London |
| Control window | 10:00-10:45 Europe/London |
| Month end | last 2 trading days |
| Hedge-need lookback | month-to-date through t-1 |
| Universe | all 10 pairs |
| Development sample | 2011-01-01 to 2023-12-31 |

##### 5. Entry rule
On each of the last 2 trading days of a month, enter at the open of the 15:30 London bar in the direction `-sign(month-to-date return through t-1)`. No entry otherwise. The control window is measured, never traded.

##### 6. Exit rule
Flat at the close of the 16:15 London bar. No overnight holding, no stop, no discretionary exit. Holding period is 45 minutes by construction.

##### 7. Position sizing
Not decided here, deliberately. Sizing only matters if Section 10 passes, and pre-committing a sizing scheme to an unvalidated signal was a documented weakness of the previous book, whose Section 7 stayed open right through invalidation. If this validates, sizing gets specified in an amendment before any sized backtest runs.

For validation only, the test uses unit exposure per pair, equal-weighted. That is a measurement convention, not a sizing proposal.

##### 8. Risk controls
Deferred with Section 7. The structural risk worth naming now: the strategy is concentrated in about 24 trading days per year with a 45-minute holding period, so sample size accumulates slowly. Expect roughly 312 pair-days per pair in development, 3,120 pooled.

##### 9. Failure conditions
- Net-of-cost return non-positive at 1.0 pip round trip.
- Effect present in the control window at similar magnitude, which would mean general month-end drift rather than fix-specific flow.
- Effect present on non-month-end days at similar magnitude, same reasoning.
- Sign opposite to prediction, regardless of significance.
- Post-2015 subsample null while pre-2015 carries the result, dating the effect to the old fix regime and making it untradeable now.

##### 10. Statistical validation plan

Structured as a primary endpoint plus a fixed sequence of secondary tests, borrowing the gatekeeping design used in clinical trials. Only H1 decides PASS/FAIL. H2 and H3 are tested only if the test before them passed, so family-wise error stays controlled without raising H1's bar. Testing them out of sequence, or reporting one as a standalone finding after H1 fails, voids the correction.

**H1, primary.** Pooled OLS across all 10 pairs and all development days:

`window_return = b0 + b1*signal + b2*month_end + b3*fix_window + b4*(signal × month_end × fix_window) + eps`

via `interaction_regression_centered` extended to a 3-way interaction, reliability gate per criterion 4. Predicts b4 > 0. Significance and sign both required.

Three robustness checks on H1, all of which must agree:

- *Window definition.* Narrow the fix window to 15:45-16:05. A real fix-driven flow should survive tightening around the event.
- *Structural break.* Split at 2015-02-15, fit both subsamples. Same sign in both, and the post-reform subsample significant on its own. This decides whether the effect is still live.
- *Permutation.* 1,000-permutation shuffle of the month-end label, two-sided empirical p on b4, via `permutation_test_interaction_coefficient`.

**H2, post-fix reversal.** Tested only if H1 passes. Regress the return of the 16:15-17:00 London window on the realized fix-window return, restricted to month-end days. Predicts a negative coefficient: mechanical pressure from a price-insensitive counterparty moves price temporarily and must partly unwind.

Estimate the reversion speed with `ou_half_life` on the month-end fix-to-post-fix series. A half-life materially longer than one session is inconsistent with a 45-minute liquidity effect and would point at something informational instead.

This is the sharpest test in the spec because it discriminates rather than confirms. A generic month-end drift, an unmodelled risk premium, or a calendar artifact all predict H1 while predicting no reversal. Only the flow mechanism predicts both. **A null or positive H2 does not fail the strategy, but it does falsify the stated mechanism**, and the write-up must then say the effect exists for reasons the spec does not explain.

**H3, volatility conditioning.** Tested only if H2 passes. Fit `fit_garch` per pair on daily log returns, label each month by `classify_vol_regime` applied to the month's mean conditional volatility, then re-fit H1 separately within high-vol and low-vol months.

Predicts a larger b4 in high-vol months. Hedge-ratio drift away from target is proportional to realized volatility over the month, so a high-vol month leaves a bigger gap to close and demands a bigger rebalancing trade. Direct implication of the mechanism, not a free parameter.

GARCH must be fit walk-forward, not full-sample, per the Day 47 finding that full-sample fitting changed regime labels by 39-68%. Both the 4-way interaction and the split-sample version will be reported; the 4-way carries real collinearity risk and the reliability gate may reject it, in which case the split-sample version stands.

**Portfolio construction variant.** The signal above is traded per pair, time-series. A cross-sectional version will also be reported: rank all 10 pairs by hedging need each month-end day, long the top 3, short the bottom 3, dollar-neutral. This strips common USD moves and raises breadth per the Fundamental Law (`information_ratio`, Day 40).

This is not a fifth hypothesis. It is a second construction of the same signal, so it does not enter the gatekeeping sequence and does not increment `n_trials`. The time-series version is designated primary in advance; if the cross-sectional version looks better, that is an observation for a future spec, not a result to promote after the fact.

**Standard errors.** All pooled p-values use `block_bootstrap` with date-blocks rather than OLS analytic errors. Month-end flows are common across pairs on the same date, so ten pairs on one date are nowhere near ten independent observations and analytic errors would be optimistic. Block length selected by the same rule as `research/notes/bootstap_block_length_selection.md`, fixed before testing.

**Verdict rule.** PASS requires: reliability gate on H1 and its window-definition check; H1 p(b4) < 0.05 with b4 > 0 on block-bootstrap errors; all three robustness checks agreeing in sign, with the window check p < 0.05 and the post-2015 subsample p < 0.05; permutation p < 0.05; and the criterion 7 cost gate. Any single failure kills the strategy. H2 and H3 do not affect PASS/FAIL; they determine whether the mechanism claim survives alongside it.

**Lockbox.** 2024-2026 stays sealed until the development verdict is written down. Opened once, for a PASS, and the result stands either way.

##### 11. Open questions and known gaps
The hedge-need proxy is a crude stand-in for actual equity-portfolio hedging demand, which would properly need foreign equity index returns and holdings data. A pure-FX proxy may be too noisy to detect the flow even if it is real. Most likely route to a false negative.

45-minute holding on 1-minute bars raises execution-realism questions, particularly slippage inside a high-volume window, that unit-exposure backtesting does not capture. H2's fade trade is worse on this count, since it trades immediately after the most crowded window of the day.

The 2-day month-end definition comes from the literature rather than being derived here. Fixed, not tuned, but also not optimized, and the true flow window may differ.

H3 splits an already-thin sample. Roughly 3,120 pooled pair-days become two halves of about 1,560, and the power calculation from `day57_momentum_book_invalidation.md` applies with full force. A null H3 is close to uninformative and should not be read as evidence against the volatility mechanism.

The cross-sectional variant needs at least 6 pairs with valid signals on a given date to form 3-and-3 legs. Days failing that are dropped, which is a mild selection effect on which dates enter that book.

---

### `research/strategies/s06_intraday_overshoot/spec.md`

#### Strategy Specification: Intraday Overshoot Reversal, London-NY Overlap

**Date drafted:** Day 57 (2026-07-25)
**Status:** CLOSED, Day 57. Section 10 verdict FAIL. Figures regenerated Day 72: the strategy fails on robustness 3 per-trade permutation (p = 0.1349) and H2 (b3 = -0.0465, p = 0.679). Robustness 1 threshold monotonicity, recorded as a third failure on the Day 57 run, passes on regeneration and is uninformative in either direction. Result in `research/strategies/s06_intraday_overshoot/intraday_overshoot_section10_validation.md`. Lockbox never opened.

Written and pre-registered before any test was run; the sections below are unedited from that pre-registration.

##### Provenance
Strategy #6. Five tested previously, all null: PC2 Carry Regime, Momentum w/ ML Regime, OU Half-Life Mean Reversion, Volatility Regime Breakout/Mean-Reversion (momentum-only successor closed in `momentum_book_invalidation.md`), Month-End FX Flow (closed in `month_end_fx_flow_h1_result.md`). Honest `n_trials` is 6.

##### 0. Feasibility, computed before the hypothesis
**Cost hurdle.** EUR/USD 09:00-12:00 ET session return std is 27.6bp, annualizing to 4.39% vol. Against a 0.9bp round trip:

| Trades/yr | Cost %/yr | Break-even Sharpe |
|---|---|---|
| 252 (unconditional) | 2.27 | 0.517 |
| 60 | 0.54 | 0.123 |
| 30 | 0.27 | 0.062 |
| 15 | 0.14 | 0.031 |

Selectivity is the entire cost argument. Trading the window unconditionally requires Sharpe 0.52 before earning anything, which is why the Day 57 session variants failed. A 2-sigma entry threshold fires on 9.6% of days, about 24 trades/year/pair, dropping the hurdle to 0.062.

Because this strategy fades a fast move, entry slippage will exceed the quoted spread. The cost gate is therefore evaluated at **1.0, 2.0 and 3.0 pips**, and the strategy must clear at 2.0 pips to pass. Clearing only at 1.0 pip is recorded as a fail.

**Power hurdle.** t on an annualized Sharpe is SR x sqrt(years), so 12.9 years still requires **SR >= 0.55** for t = 2. Sampling intraday does not change this: the constraint is calendar span, not observation count, as verified by simulation during the carry work.

The lever available here that carry lacked is breadth. Ten pairs with imperfectly correlated intraday shocks give an effective breadth well above 1, so a per-pair Sharpe near 0.30 can plausibly produce a book Sharpe above 0.55 (Fundamental Law, `information_ratio`, Day 40). Compute realized effective breadth from the cross-pair correlation of trade returns and report it. If effective breadth turns out near 1, the power argument collapses and the result should be read as inconclusive.

Achieved power is reported with every result via `compute_achieved_power`.

##### 1. Hypothesis
**H1.** Within the 09:00-12:00 ET window, price displacement away from the session open partially reverses before the window closes. Fading a large displacement earns a positive net-of-cost return.

**H2, the contribution.** The reversal is stronger following *fast* displacement than slow displacement of the same magnitude. Fast moves are impatient liquidity demand and overshoot; slow moves of equal size are more likely informed and should not revert.

Falsification criteria, binding:

1. Direction requirement: significant **and** predicted sign. A significant coefficient of the wrong sign is a FAIL. Two prior strategies produced exactly that.
2. Primary threshold: p < 0.05 on H1, block-bootstrap standard errors clustered by date.
3. Reliability gate on any regression: condition number < 1e10, all VIF < 10, main effects mean-centered before interaction.
4. Multiple testing: Benjamini-Hochberg across 6 strategies, so p < 0.0083 at rank 1 of 6.
5. Cost gate: net-of-cost Sharpe positive at **2.0 pips** round trip on realized trade count.
6. Power disclosure: achieved power and realized effective breadth reported alongside every result.

##### 2. Economic rationale
The counterparty is a market maker with unwanted inventory. When a large directional order arrives in a three-hour window, dealers absorb it, price moves further than the information warrants, and dealers unwind into the reversion. The trader on the other side is buying immediacy and is price-insensitive about it.

This mechanism does not get arbitraged away because the liquidity demander is not trying to predict anything, and the compensation is payment for bearing inventory risk. That is a structurally different situation from the momentum hypothesis this project spent Days 42-57 on, which never identified a counterparty at all.

Choosing 09:00-12:00 ET is not arbitrary within that story: it is the London-New York overlap, the highest-flow window of the FX day, so it should carry both the most liquidity demand and the most dealer inventory turnover.

**What would falsify the mechanism rather than the effect.** If reversal is equally strong for slow and fast displacement (H2 null), the story about impatient flow is wrong even if H1 passes, and the write-up must say the effect exists for reasons this spec does not explain.

##### 3. Data
All 10 pairs from the outset, so universe selection cannot become a post-hoc degree of freedom as it did in `momentum_book_regime_conditional_robustness.md`.

1-minute bars, `data/{pair}.csv`. Timestamps carry a fixed UTC-5 file offset, converted per `src/features/sessions.py` then DST-aware converted to America/New_York, since 09:00 ET is a local-clock event.

Development 2011-01-01 to 2023-12-31. Lockbox 2024-01-01 to 2026-05-01, sealed.

##### 4. Signal logic
All parameters fixed as of this document. None may be tuned after seeing results.

1. Entry scan runs 09:00 to 12:00 ET, exit at 13:00 ET. Reference price is the 09:00 bar open.
2. Fit `fit_garch` on daily log returns per pair, walk-forward refit per the Day 47 finding that full-sample fitting shifts regime labels by 39-68%. Scale to a session-equivalent sigma.
3. Entry threshold is `k x sigma_t` where sigma_t is the GARCH conditional vol as of t-1, with **k = 2.0**.
4. Entry fires the first time within the window that `|log(P_t / P_open)| > k x sigma_t`. Direction is the fade: short if price is above the open, long if below.
5. One entry per pair per day, maximum. No re-entry after exit.

GARCH is used to **scale the threshold, not to gate trading**. A fixed threshold fires constantly in high vol and never in low vol, so trade count and risk swing by regime. Scaling holds the trigger rate roughly stable across regimes, which is what a liquidity-provision book requires.

| Parameter | Value |
|---|---|
| Entry scan window | 09:00-12:00 America/New_York |
| Exit | 13:00 America/New_York |
| Reference | 09:00 bar open |
| Threshold k | 2.0 conditional sigma |
| Max entries | 1 per pair per day |
| Universe | all 10 pairs |
| Development sample | 2011-01-01 to 2023-12-31 |

##### 5. Entry rule
Enter at the close of the first 1-minute bar whose displacement from the 09:00 open exceeds `2.0 x sigma_t`, in the direction opposite the displacement. Unit exposure per pair.

##### 6. Exit rule
Flat at the 13:00 ET bar close. No target, no stop, no discretion. Holding period is whatever remains after entry.

This is deliberately parameter-free. A reversion target or stop would improve the return distribution but each adds a tunable degree of freedom to a pre-registered test, and the project's history is that free parameters are where results go to die.

##### 7. Position sizing
Deferred until H1 passes, per the pattern that left the momentum book's Section 7 open through invalidation. Validation uses unit exposure per pair, equal weighted, as a measurement convention.

##### 8. Risk controls
Deferred with Section 7. Named now, because this strategy's risk profile is its central weakness: liquidity provision is structurally negative skew. Many small wins, occasional large losses when the move keeps running, and a hard 13:00 exit with no stop guarantees the left tail is uncapped within the holding period. Report realized skew, excess kurtosis, and CVaR (`src/analysis/portfolio.py`) alongside Sharpe, not after.

##### 9. Failure conditions
- Net-of-cost Sharpe non-positive at 2.0 pips on realized trade count.
- Effect present only in the smallest displacement bucket, which would suggest bid-ask bounce rather than genuine overshoot.
- Realized effective breadth near 1, collapsing the power argument.
- H2 null, falsifying the impatient-flow mechanism even if H1 survives.
- Trade count materially below 15/year/pair, which would leave too few observations to say anything.

##### 10. Statistical validation plan
Gatekeeping. H2 runs only if H1 passes.

**H1 primary.** Pooled mean trade return across all pairs and trades, tested against zero. Block-bootstrap standard errors clustered by **date**, since displacement events cluster across pairs on common shocks and 10 pairs on one day are nowhere near 10 independent observations. Report pooled book Sharpe, achieved power, effective breadth, deflated Sharpe at `n_trials=6`, skew, excess kurtosis, and max drawdown.

**H1 robustness 1, threshold.** Repeat at k = 1.5 and k = 2.5. The effect should not depend on the specific threshold, and the monotonicity matters: if reversal is stronger at larger k, that supports overshoot; if stronger at smaller k, it suggests bid-ask bounce and fails Section 9.

**H1 robustness 2, structural break.** Split at 2017-06-30 and fit both halves. Same sign required in both, with the later half significant on its own, since an effect that died mid-sample is untradeable now.

**H1 robustness 3, permutation.** 1,000-permutation shuffle of the fade direction, holding entry timing fixed, two-sided empirical p.

**H2.** Split trades by time-to-threshold (fast = threshold crossed within 30 minutes of the open, slow = after). Interaction regression of trade return on displacement size, speed dummy, and their interaction, via `interaction_regression_centered`. Prediction: fast trades revert more.

**Verdict.** PASS requires reliability gate on H1 and robustness 1; H1 p < 0.05 on block-bootstrap errors with the predicted sign; all three robustness checks agreeing in sign with robustness 1 and the post-break half both p < 0.05; permutation p < 0.05; and the 2.0-pip cost gate. Any single failure kills it.

**Lockbox.** Opened once, only on a PASS, only after the development verdict is written down.

##### 11. Open questions and known gaps
Entry slippage is the biggest threat. Fading a fast move means crossing the spread into adverse flow, and realized slippage on a 2-sigma displacement bar will exceed the quoted spread by an unknown amount. The 2.0-pip gate is a guess at this, not a measurement, and only tick data would settle it.

1-minute OHLCV closes are not tradeable prices. At a 2-sigma threshold the displacement is around 55bp against a 0.9bp spread, so bid-ask bounce should not dominate, but robustness check 1's monotonicity is the actual test of that and it should be read carefully.

GARCH conditional vol is fit on daily returns and rescaled to a session horizon, which assumes the daily-to-intraday vol ratio is stable. It is not, since intraday vol has strong time-of-day structure. A session-native vol estimate would be better and is not built here.

The hard 13:00 exit is honest but almost certainly suboptimal. If H1 passes, the first extension worth specifying is an exit study, as its own hypothesis with its own trial count.
---

## Appendix B — Framework inventory and test coverage

Counts as of the commit that introduced this paper. The outline that specified this appendix estimated 30 modules and 372 tests; both were stale and the measured figures are below.

| | |
|---|---|
| Framework | 8,475 lines across 36 modules in `src/` |
| Tests | 488 tests across 34 files, 5,864 lines |
| Research record | 42 daily audits, 38 reproducible analysis scripts, 32 validation and falsification files |
| Universe | 10 FX pairs of 1-minute OHLCV, plus 8 three-month interbank rate series |
| Strategies pre-registered | 6 |
| Strategies surviving validation | 0 |

Module inventory by package:

- `src/stats/` — distributions and tail-mass comparison; Wald and t-tests, effect size, required sample size and achieved power; OLS by normal equations with R², adjusted R² and residual diagnostics, ridge, lasso by numerical minimization, interaction regression with variance-inflation factors and centering; rolling correlation and a bootstrap-calibrated CUSUM regime-shift detector; gradient descent and an SLSQP constrained optimizer validated against the closed-form KKT solution; geometric Brownian motion and Ornstein-Uhlenbeck simulation with parameter fitting.
- `src/evaluation/` — bootstrap confidence intervals and block bootstrap; purged k-fold cross-validation with three-condition overlap purging and embargo; permutation tests including paired-sign and interaction-coefficient variants; Bonferroni and Benjamini-Hochberg correction.
- `src/signals/` — `SignalBuilder` with information coefficient, rolling IC, forward returns and lookahead validation; momentum, price z-score and ladder mean-reversion, regime-gated composites, per-window regime refit, vol-scaled triple-barrier labels, and the intraday overshoot session builder; Engle-Granger, Johansen, cointegration spreads and OU half-life.
- `src/features/` — eigendecomposition and PCA from first principles with SVD-based inversion; GARCH(1,1) by maximum likelihood with a 1-D k-means volatility regime classifier; session construction with local-open-to-UTC handling.
- `src/analysis/` — `PerformanceAnalyzer` (empirical annualization, Sharpe, deflated Sharpe, Sortino, max drawdown, Calmar, hit rate, profit factor, t-statistic, Jarque-Bera, Ljung-Box, tracking error, regime-conditional attribution); the transaction-cost model; Markowitz closed-form and numerical, efficient frontier, minimum-variance and equal-risk-contribution risk parity, Kelly, and historical, parametric and Monte Carlo VaR and CVaR; CAPM decomposition and PCA factor extraction.
- `src/framework/` — `DataLoader` with composition-root embargo and leakage-safe splitting; `WalkForwardValidator` window generation and slicing.
- `src/utils/` — centralized logging configuration and a single project-level random seed with a `get_rng` factory.

Two coverage caveats belong here rather than in a footnote. Test coverage has never been measured, so the suite passing is evidence that nothing obviously broke and not evidence that the branches added are correct. And four of these tests are real-data integration tests living in the unit suite: they parse roughly 900 MB of minute bars through a shared session-scoped fixture, are bounded to the development window, and sit behind a `slow` marker so the rest of the suite runs without them.

## Appendix C — What is not implemented from scratch

Reproduced from the repository README, because a blanket from-scratch claim would not survive thirty seconds of checking and should not be made.

| Component | Source | Why |
|---|---|---|
| ADF, KPSS | `statsmodels.tsa.stattools` | Critical-value tables are interpolated from published simulations; reimplementing reproduces a lookup, not the theory |
| Johansen trace test | `statsmodels.tsa.vector_ar.vecm` | Same reason — the eigenvalue problem is hand-derived in `pca.py`, the critical values are not |
| ARIMA fitting | `statsmodels.tsa.arima` | Order selection logic and AIC comparison are mine; the state-space MLE is not |
| Ljung-Box, Jarque-Bera | `statsmodels`, `scipy.stats` | Standard test statistics |
| Distribution CDFs, `spearmanr` | `scipy.stats` | Numerical primitives |
| SLSQP, MLE optimizer | `scipy.optimize` | The objective functions — GARCH log-likelihood, lasso penalty, risk-parity ERC — are hand-written; the solver is not |

Two of these were originally scoped for first-principles implementation and ended up here instead. They are listed rather than quietly reclassified.

## Appendix D — Reproduction

Python 3.12. Dependencies pinned in `requirements.txt`: `numpy` 1.26.4, `pandas` 2.2.2, `scipy` 1.13.0, `statsmodels` 0.14.2, `scikit-learn` 1.4.2, `matplotlib` 3.8.4, `click` 8.4.1, `pytest` 8.2.0.

```bash
git clone https://github.com/cbaumann0912-prog/summer2026.git
cd summer2026
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Minute-bar CSVs are roughly 300 MB per pair, live outside the repository, and are mounted at a configurable path. The Docker image pins the full environment including the interpreter:

```bash
docker build -t quant-research:v1.0.0 .
docker run --rm -v /absolute/path/to/data:/data:ro -v "$(pwd)/results:/out" \
  quant-research:v1.0.0 momentum --pairs EURUSD --data-dir /data --output /out
```

Every analysis script rebuilds from raw 1-minute bars on each run; there is no session caching anywhere in the project. `research/run_research.py` refuses any end date reaching the reserved 2024-01-01 boundary unless `--allow-lockbox` is passed explicitly.

Two reproduction caveats, both from §5.3. Results depending on the GARCH maximum-likelihood fit are sensitive to the platform's linear-algebra backend at the level of a few trades in a few thousand: a Linux build with the same pinned library versions agrees with the published Windows figures on eight of ten pairs and differs on two. And strategy 6's original closing figures are not reproducible from any committed state at all; §4.6 and §5.2 publish regenerated figures, with the superseded set recorded in the strategy's own validation document.

---

## References

Every work below was read for this paper. Where the copy consulted is a preprint or working
paper rather than the published version, that is stated, because the quotations in the text
come from the copy consulted. Sources are in `research/resources/`.

Bailey, D. H., J. M. Borwein, M. López de Prado and Q. J. Zhu (2014). "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." *Notices of the American Mathematical Society* 61(5), 458–471. Theorem 1 gives the minimum backtest length needed to avoid selecting a strategy whose in-sample Sharpe ratio has an expected out-of-sample value of zero. *Copy consulted: the preprint of 7 October 2013.*

Bailey, D. H. and M. López de Prado (2014). "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality." *The Journal of Portfolio Management* 40(5), 94–107. Source of the trial-counting principle this project adopted as a process rule. *Copy consulted: the preprint of 31 July 2014.*

Harvey, C. R., Y. Liu and H. Zhu (2016). "…and the Cross-Section of Expected Returns." *The Review of Financial Studies* 29(1), 5–68. Collects 316 published factors and argues for a t-ratio hurdle above 3.0. *Copy consulted: NBER Working Paper No. 20592, October 2014.*

Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4), 36–52. Derives the distribution of the Sharpe ratio and shows that √K annualization holds only under restrictive conditions, with the error reaching 65% in a hedge-fund example once serial correlation is accounted for.

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Source of the purged k-fold cross-validation with embargo implemented in `src/evaluation/cross_validation.py` (§7.4, pp. 105–108, with the embargo at p. 107) and of the triple-barrier labelling in `src/signals/triple_barrier.py` (§3.4, p. 45).
