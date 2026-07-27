# Day 57 Research — Transaction Cost Breakeven

## Question
At what round-trip spread does the strategy's edge disappear entirely? That number is the maximum viable spread; if a broker quotes higher, the strategy is not deployable.

## Verdict
**Cost gate: PASS.** Intraday Overshoot Reversal clears its pre-registered 2.0-pip gate with a **4.54x margin** — maximum viable round-trip spread is **9.078 pips** against assumed spreads of 0.9–1.8. Costs are not the binding constraint on this strategy.

**Only 3 of 10 pairs clear their own assumed spread standalone.** EUR/USD and USD/JPY have negative gross returns before any cost is charged. The margin is a portfolio property, not a signal property.

Scope: the live candidate only. The five closed strategies get no breakeven number — computing one for a hypothesis with no established edge is arithmetic on noise.

## Methodology
Module: `round_trip_cost_bps`, `breakeven_annual_return`, `breakeven_sharpe`, `cost_report`, `max_viable_spread_pips`, `rollover_bps_per_day` in `src/analysis/performance_analyzer.py`. Signal construction: `src/signals/intraday_overshoot.py`. Script: `research/applied_analysis/day57_transaction_cost_breakeven.py`.

Sessions are rebuilt from the raw 1-minute bars on every run. Book economics: 3,206 trades, 24.69 round trips/yr/pair, gross annual return 2.2542%, gross annual vol 2.1605%, gross Sharpe +1.0434, trade-weighted 1.0058 bp/pip.

Universe is all 10 pairs. The calendar event's "3 pairs only, USD/CHF is NOT supported" is stale — `DataLoader.SUPPORTED_PAIRS` has held 10 since the expansion.

**Spreads are assumed, not measured.** The 1-minute OHLCV files carry no bid or ask, so no spread in this project has ever been observed. EUR/USD at 0.9 is carried from Section 0 of the spec; the rest are typical ECN quotes for the London–NY overlap. The headline does not rest on them — maximum viable spread is inverted out of realized gross return and realized trade count with no spread input at all.

Rollover is zero in practice: the book is flat by 13:00 ET and never rolls. The column below shows what a full day held would cost, from 3-month interbank differentials, and is included because `cost_report` returns it.

## Cost report — 10 pairs
Hold 0.167 days, 24.69 round trips/yr/pair.

| Pair | spread (pips) | spread_bps | rollover_bps | total_bps | breakeven_return |
|---|---|---|---|---|---|
| EUR/USD | 0.9 | 0.7784 | 0.0410 | 0.8193 | 0.2023% |
| GBP/USD | 1.3 | 0.9512 | 0.0084 | 0.9596 | 0.2369% |
| USD/JPY | 0.9 | 0.8234 | 0.0499 | 0.8733 | 0.2156% |
| USD/CHF | 1.5 | 1.5850 | 0.0693 | 1.6543 | 0.4084% |
| AUD/USD | 1.2 | 1.5885 | 0.0492 | 1.6376 | 0.4043% |
| USD/CAD | 1.5 | 1.1736 | 0.0019 | 1.1756 | 0.2902% |
| NZD/USD | 1.8 | 2.5581 | 0.0582 | 2.6163 | 0.6459% |
| EUR/GBP | 1.2 | 1.4007 | 0.0326 | 1.4333 | 0.3539% |
| EUR/JPY | 1.5 | 1.1651 | 0.0090 | 1.1740 | 0.2899% |
| EUR/CHF | 1.6 | 1.4552 | 0.0284 | 1.4836 | 0.3663% |

## Maximum viable spread — book level
| Round-trip spread | cost %/yr | net ann return | net Sharpe | breakeven SR |
|---|---|---|---|---|
| 1.00 pip | 0.2483% | 2.0059% | +0.9284 | 0.1149 |
| 2.00 pips (gate) | 0.4966% | 1.7576% | +0.8135 | 0.2299 |
| 3.00 pips | 0.7449% | 1.5093% | +0.6986 | 0.3448 |
| **9.078 pips** | 2.2547% | −0.0004% | −0.0002 | 1.0436 |

**Maximum viable round-trip spread: 9.078 pips. Pre-registered gate: 2.0 pips. Margin: 4.54x. Cost gate PASS.**

Against a gross Sharpe of +1.0434, the hurdle at the gate is 0.2299 — cleared by more than 4x on a risk-adjusted basis too.

## Maximum viable spread — per pair
| Pair | trades | rt/yr | gross %/yr | max pips | assumed | margin | verdict |
|---|---|---|---|---|---|---|---|
| EUR/USD | 296 | 22.8 | −0.1109% | 0.000 | 0.9 | 0.00 | not viable |
| GBP/USD | 372 | 28.6 | +0.0409% | 0.195 | 1.3 | 0.15 | not viable |
| USD/JPY | 343 | 26.4 | −0.4617% | 0.000 | 0.9 | 0.00 | not viable |
| USD/CHF | 278 | 21.4 | +0.0048% | 0.021 | 1.5 | 0.01 | not viable |
| AUD/USD | 402 | 31.0 | +0.6282% | 1.533 | 1.2 | 1.28 | viable |
| USD/CAD | 309 | 23.8 | +0.2536% | 1.362 | 1.5 | 0.91 | not viable |
| NZD/USD | 333 | 25.6 | +0.5868% | 1.610 | 1.8 | 0.89 | not viable |
| EUR/GBP | 318 | 24.5 | +0.4257% | 1.489 | 1.2 | 1.24 | viable |
| EUR/JPY | 259 | 19.9 | +0.6977% | 4.504 | 1.5 | 3.00 | viable |
| EUR/CHF | 296 | 22.8 | +0.0389% | 0.188 | 1.6 | 0.12 | not viable |

Three of ten clear standalone. Two have negative gross returns. Four more clear a spread below 0.2 pips, which is no margin at all.

The book's max viable spread is 9.078 pips against a per-pair mean of 1.090. That gap is the result: equal weighting across 10 pairs cuts volatility faster than it cuts return, so the book's cost tolerance is several times any individual pair's. This is the cost-side view of the diversification dependence the H1 note found via effective breadth (2.22 of 10).

Two consequences. The margin is not robust to universe reduction — dropping to the majors leaves EUR/USD, GBP/USD and USD/JPY with a combined gross return near zero. And the pairs carrying the result (AUD/USD, EUR/GBP, EUR/JPY, NZD/USD) are the ones with the widest real spreads and thinnest liquidity during the London–NY overlap, since their local sessions are closed. The assumed spreads for exactly those pairs are the least reliable numbers here.

## Section 10 status
The cost gate is criterion 5 of the spec's falsification list. Where the full verdict stands:

| Requirement | Status |
|---|---|
| H1 primary — p < 0.05 block-bootstrap, predicted sign | PASS, pending rerun |
| H1 robustness 1 — threshold monotonicity k = 1.5/2.0/2.5 | PASS, pending rerun |
| H1 robustness 2 — structural break, later half significant | PASS, pending rerun |
| H1 robustness 3 — 1,000-permutation shuffle | PASS, pending rerun |
| Reliability gate — condition number, VIF | PASS |
| Cost gate — net Sharpe positive at 2.0 pips | **PASS (this audit)** |
| Multiple testing — BH across 6 strategies, p < 0.0083 | Not yet run |
| H2 — speed interaction | FAIL, mechanism unsupported |

Every H1 row reads "pending rerun" because `intraday_overshoot_reversal_h1.md` does not reproduce from a fresh rebuild: 3,616 published trades against 3,206 rebuilt, −11.3%. Commit a84039f flags the note as "pending walk-forward rerun," so the note is the stale side. The gap is not uniform across pairs, which is a changed GARCH sigma path rather than a truncation, and it is far too large to be numerical noise — a 0.03% sigma perturbation moves roughly 1 trigger day in 311.

**The verdict remains incomplete.** A passing cost gate does not advance the strategy toward deployment on its own, and the lockbox stays sealed.

## Alternative explanations
The 9.078-pip margin assumes the realized trade count is stable out of sample. A regime where 2-sigma displacements fire more often would raise turnover and cut the margin proportionally. Trigger-rate stability has not been tested.

Slippage is not spread. Fading a fast move means crossing into adverse flow, and realized fills on a 2-sigma displacement bar will be worse than the quoted spread by an unknown amount. A 4.5x margin against *quoted* spread is not a 4.5x margin against *realized* cost. The spec already names this as its largest unquantified risk; this audit does not reduce it, only shows there is room to absorb a fairly large amount of it.

The per-pair ranking could be a power artifact. No individual pair reached significance in the H1 note (largest |t| = 1.40), so "EUR/USD has a negative edge" and "EUR/USD has a small positive edge measured with noise" are not distinguishable here. The diversification conclusion holds either way; the specific ordering should not be read as reliable.

## Next steps
- Rerun the H1 pipeline and update `intraday_overshoot_reversal_h1.md`. Until then the project's only live result is unverifiable, which blocks the paper's results section more than anything else on the list.
- Re-examine `intraday_overshoot_h1.py`'s flat `PIP_BP = 0.9`. The trade-weighted figure is 1.0058, so published net Sharpes are about 12% optimistic on the cost term. Small, but it is a known bias with a known sign.
- Test cost margin under universe reduction. If the strategy is only viable at 10 pairs, that is a deployment constraint worth stating explicitly in the spec's Section 7 before sizing work begins.
- Run the 6-way Benjamini-Hochberg correction. It is the one Section 10 criterion never attempted, and at rank 1 of 6 it needs p < 0.0083.
- Nothing here justifies opening the lockbox. The cost gate is one of the spec's Section 10 requirements, not all of them.
