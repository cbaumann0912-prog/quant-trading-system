# Day 05 — Rolling Correlation Regimes

## Goal
The goal for today was to implement a code which flags historical regime shifts between the forex pairs EURUSD, USDJPY, and GBPUSD. 

## Why Correlation Regimes Matter for Risk Management
Running three forex pairs at the same time doesn't mean there are three independent bets. EURUSD and GBPUSD have a historical correlation of 0.60. The problem gets worse when the relationship changes. When EURUSD and GBPUSD decouple below 0.4, the macro assumption a strategy may be built on no longer holds. The pairs stop moving together and signals that made sense under normal conditions start conflicting. The negative pairs flip the risk. When those relationships drift toward zero, whatever natural hedge existed between the positions disappears. I'm now long two things that used to partially offset each other and no longer do. Both are reasons to stop and check whether my position sizes still reflect what I actually intend to risk.

## Methodology
I loaded 13 years of 1-minute OHLCV data for each pair, 2011-01-01 through 2023-12-31, resampled to daily closes, and computed log returns. Those three return series were aligned by combining them into a single DataFrame and dropping any dates missing from any pair. A 30-day rolling Pearson correlation was then computed between each pair combination using the rolling_correlation function built earlier today. Regime breaks were detected by flagging periods where correlation crossed a pair-specific threshold — below 0.4 for EURUSD/GBPUSD, above -0.1 for GBPUSD/USDJPY, and above -0.2 for EURUSD/USDJPY — and stayed there for at least 14 consecutive days. Each break was recorded as a start and end date.

## Results

**EURUSD/GBPUSD** — 10 regime breaks detected with an average duration of 27 days. 
The longest break ran from December 2013 to March 2014 at 96 days.

**GBPUSD/USDJPY** — 25 regime breaks with an average duration of 31 days. 
The longest ran from May 2016 to August 2016 at 102 days, coinciding with the Brexit referendum.

**EURUSD/USDJPY** — 21 regime breaks with an average duration of 38 days. 
The longest ran from October 2012 to May 2013 at 202 days during the European sovereign debt crisis.

## Average Regime Duration

| Pair | Avg Duration |
|------|-------------|
| EURUSD/GBPUSD | 27 days |
| GBPUSD/USDJPY | 31 days |
| EURUSD/USDJPY | 38 days |

## Verdict

### Event 1 — Brexit (June–August 2016)
GBPUSD/USDJPY broke down around the June 2016 Brexit referendum. GBP sold off on the UK-specific shock while JPY rallied as a safe haven, decoupling two pairs that normally move inversely.

### Event 2 — European Sovereign Debt Crisis (2012–2013)
The longest detected break ran nearly seven months in EURUSD/USDJPY. EUR was under sustained pressure from the eurozone debt crisis while JPY moved independently on domestic policy.

### Event 3 — COVID Recovery Divergence (June–July 2020)
EURUSD/USDJPY decoupled for six weeks as EUR rallied on Europe's fiscal response while JPY weakened as safe haven demand faded.

## Implications for the Trading System
For positive pairs like EURUSD/GBPUSD, a correlation drop below 0.4 means the pairs are decoupling — they are behaving more independently than usual. This is an opportunity to increase position sizes on both since the assumed concentration risk no longer applies. Conversely, a spike toward 1.0 on the same pair means two positions are effectively 
one trade — cut size on one. For the negative pairs, a move toward zero means the natural hedge between positions is breaking down — reduce exposure on both until the relationship stabilizes.

On a $100K book, ignoring these signals means running position sizes calibrated to a market structure that no longer exists. That's not a edge case — the data shows an average of 19 regime breaks per pair over 13 years, roughly one every eight months.

Current limitation: the system only flags correlation breakdown. Detection of correlation spikes on positive pairs is not yet implemented and will be added as a separate module.