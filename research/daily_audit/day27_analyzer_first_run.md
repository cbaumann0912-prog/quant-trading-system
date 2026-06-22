# Day 27 Audit — PerformanceAnalyzer First Run

## 1. Question Investigated
- What is actually being tested here: the analyzer's mechanics, or the pairs' performance? (Given no trade log exists for raw pair returns, this should be a mechanics question, not a performance verdict — state that explicitly.)

## 2. Why It Matters
- Why does verifying the analyzer on real data (vs. synthetic data, which is what you tested with me) matter before it gets used on actual strategy candidates at Day 49?
- What could go wrong with real data that synthetic data wouldn't expose? (gaps, holidays, NaNs, non-uniform spacing)

## 3. Methodology
- State the data source, date range, resampling convention, and log-return calculation used.
- State explicitly: trades=None for all three pairs, and why.

## 4. Assumptions
- What does compute_ann_factor assume about self.returns that's relevant here? (Choice A — daily mark-to-market series; for raw pair returns this holds trivially since every day has a return.)
- Any assumptions about data continuity (weekends, holidays already dropped by the resample step)?

## 5. Findings
- Report the actual numbers for each pair: sharpe, sortino, max_drawdown, calmar, t_stat, annualized_return, annualized_vol.
- Confirm which fields returned NaN (win_rate, profit_factor) and that this matches expectation given trades=None.
- Note anything unexpected: crashes, inf values, suspiciously large/small numbers.

## 6. Alternative Explanations
- If any metric looks surprising (e.g., a pair's Sharpe being unexpectedly high/low for an untraded buy-and-hold series), what else could explain it besides a bug? (e.g., known carry trade drift in USD/JPY, EUR weakness over the sample period)

## 7. Next Steps
- List the 5 most important gaps to close tomorrow, per the calendar event's instruction.
- Does this run reveal anything that should change scope before Day 28's full analyzer run on all strategies?