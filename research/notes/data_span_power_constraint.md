# Data Span as the Binding Power Constraint

## Question
The Day 57 validation found the study underpowered for its own effect size. How much of that is fixable by downloading more history, and is it worth doing inside this 90-day window?

## Finding
**The 2011 start is a download choice, not a data limit.** The vendor carries every pair in the universe well before that. The binding constraint on a common start date across all ten is NZD/USD at 2005-08. Dropping that one pair moves the common start back to 2002-03.

| Universe | pairs | common start | dev years | Sharpe for t=2 | active days | powered |
|---|---|---|---|---|---|---|
| current download | 10 | 2011.00 | 13.0 | 0.555 | 1,278 | no |
| vendor maximum | 10 | 2005.58 | 18.4 | 0.466 | 1,811 | no |
| drop NZD/USD | 9 | 2002.17 | 21.8 | 0.428 | 2,147 | **yes** |
| USD majors only | 6 | 2000.42 | 23.6 | 0.412 | 2,319 | **yes** |

Active days assume the intraday overshoot book's realized 37.9% trigger rate and 259.44 sessions/year. The 80% power threshold for that book's effect size is 1,871 active days.

Vendor earliest availability for the current universe: EUR/USD, GBP/USD, USD/JPY, USD/CHF at 2000-05; AUD/USD, USD/CAD at 2000-06; EUR/GBP, EUR/JPY, EUR/CHF at 2002-03; NZD/USD at 2005-08.

## Interpretation
Sacrificing one pair buys 8.8 years and takes the required Sharpe from 0.555 to 0.428, a 23% reduction in the hurdle. That single swap moves the overshoot book from 1,278 active days against a 1,871 requirement to 2,147 against the same requirement. The study would have been adequately powered.

That is the honest size of the constraint. Five of six candidates in this project failed at effect sizes the sample could not resolve, and the fix was available the whole time at the cost of one pair from the universe.

The trade is breadth against span. Ten pairs at 13 years gave realized effective breadth of 2.34, so the marginal pair was contributing far less than the marginal year would have. Nine pairs at 21.8 years is the better design and was not considered when the universe was set.

## Why this is not being fixed now
Expanding is not an additive operation. Every audit from Day 4 onward is computed on 2011-2023, so new history invalidates the existing result set rather than extending it. The work is: re-download roughly 5 GB of 1-minute bars, validate quality on pre-2005 history from a retail vendor, complete the Parquet migration deferred to the buffer block since the current loader reads whole files into memory, then re-run every audit. That is a week or more of the remaining project days and it would consume the paper block.

It also cannot change any verdict already reached. All six candidates are closed. Re-running a closed hypothesis on a larger sample to see whether it passes the second time is the post-hoc move the pre-registration discipline exists to prevent, and the overshoot book failed on sign, mechanism, and permutation grounds that added power does not address.

## Consequences
- **Paper.** This belongs in the limitations section as a quantified design error, not a vague note about sample size. The claim is specific: the universe was chosen for breadth without checking what it cost in span, and the resulting study could not resolve the effects it was built to test.
- **Strategy #7.** The power-first rule adopted on Day 57 should be applied to the data window as well as to the hypothesis. Fix the required Sharpe first, then choose the universe and span that deliver it.
- **Next 90 days.** Nine pairs from 2002 is the starting design. The pre-2005 regime differs in market structure, spreads and participant mix, so a structural break test across the 2005 boundary should be a precondition rather than an afterthought.

## Caveats
Pre-2005 FX is not the same market. Electronic execution was less dominant, spreads were wider, and the participant mix differed. A longer sample buys power at the cost of assuming the effect is stable across a market-structure change, which is a stronger assumption than it looks.

Retail vendor history that old is also lower quality than recent data. Anything before 2005 should be validated for gaps and stale quotes before it is trusted, which is part of why the fix is not free.
