import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.signals.intraday_overshoot import build_overshoot_sessions, overshoot_trades

from src.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    bps_per_pip,
    breakeven_sharpe,
    cost_report,
    max_viable_spread_pips,
    rollover_bps_per_day,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]

ASSUMED_SPREAD_PIPS = {
    "EURUSD": 0.9, "USDJPY": 0.9, "GBPUSD": 1.3, "AUDUSD": 1.2,
    "USDCHF": 1.5, "USDCAD": 1.5, "NZDUSD": 1.8, "EURGBP": 1.2,
    "EURJPY": 1.5, "EURCHF": 1.6,
}

RATE_FILES = {
    "USD": "us", "EUR": "ea", "JPY": "jp", "GBP": "uk",
    "CHF": "ch", "AUD": "au", "CAD": "ca", "NZD": "nz",
}

START, END = "2011-01-01", "2023-12-31"
NOTIONAL = 100_000.0
K_PRIMARY = 2.0
PRIMARY_DELAY = 5
KS = [K_PRIMARY]
ENTRY_DELAYS = [PRIMARY_DELAY]
SCAN_OPEN, SCAN_CLOSE, EXIT_MIN = 9 * 60, 12 * 60, 13 * 60
VOL_RATIO_MIN_OBS = 250
GARCH_MIN_TRAIN = 500
HOLD_DAYS = 4 / 24
COST_PIPS = [1, 2, 3]
GATE_PIPS = 2.0

RULE = "=" * 78

sessions = {}
for _pair in PAIRS:
    print(f"staging {_pair} ...", flush=True)
    sessions[_pair] = build_overshoot_sessions(
        pair=_pair, data_dir=DATA_DIR, start=START, end=END,
        ks=KS, entry_delays=ENTRY_DELAYS,
        scan_open=SCAN_OPEN, scan_close=SCAN_CLOSE, exit_min=EXIT_MIN,
        vol_ratio_min_obs=VOL_RATIO_MIN_OBS, garch_min_train=GARCH_MIN_TRAIN,
    )

all_days = pd.Index(
    sorted(set().union(*[set(c.index) for c in sessions.values()])), name="date"
)
SPAN_YEARS = (all_days.max() - all_days.min()).days / 365.25
ANN_FACTOR = len(all_days) / SPAN_YEARS

REF_QUOTE = {p: float(sessions[p]["open"].median()) for p in PAIRS}


def load_rate(stem):
    df = pd.read_csv(DATA_DIR / f"{stem}_3m_interbank.csv", parse_dates=["date"])
    window = df[(df["date"] >= START) & (df["date"] <= END)]
    return float(window["value"].mean()) / 100.0


RATES = {ccy: load_rate(stem) for ccy, stem in RATE_FILES.items()}
ROLLOVER = {
    p: rollover_bps_per_day(RATES[p[:3]], RATES[p[3:]], direction=1) for p in PAIRS
}

trades = overshoot_trades(sessions, K_PRIMARY, PRIMARY_DELAY)
wide = trades.pivot_table(index="date", columns="pair", values="ret")
book = wide.mean(axis=1).reindex(all_days).fillna(0.0)

gross_sharpe = PerformanceAnalyzer(returns=book).compute_sharpe()
gross_ann_ret = book.mean() * ANN_FACTOR
gross_vol = book.std(ddof=1) * np.sqrt(ANN_FACTOR)

tpy_per_pair = len(trades) / SPAN_YEARS / len(PAIRS)

trade_share = trades["pair"].value_counts(normalize=True)
book_bps_per_pip = sum(trade_share[p] * bps_per_pip(p, REF_QUOTE[p]) for p in PAIRS)
pip_ref = 0.0001 / (book_bps_per_pip / 1e4)

print(f"\n{RULE}")
print("DAY 57 -- TRANSACTION COST BREAKEVEN")
print(f"{len(PAIRS)} pairs | {START} to {END} | notional ${NOTIONAL:,.0f}")
print(f"Intraday Overshoot Reversal, +{PRIMARY_DELAY}min entry, k={K_PRIMARY}, "
      f"{tpy_per_pair:.2f} round trips/yr/pair")
print(RULE)


print("\n[1] COST REPORT -- all 10 pairs at assumed spreads")
print(f"{'pair':>8}{'spread':>8}{'spread_bps':>12}{'rollover_bps':>14}"
      f"{'total_bps':>11}{'breakeven_return':>18}")

for p in PAIRS:
    r = cost_report(
        p, notional=NOTIONAL, holding_period_days=HOLD_DAYS,
        trades_per_year=tpy_per_pair, spread_pips=ASSUMED_SPREAD_PIPS[p],
        quote_price=REF_QUOTE[p], rollover_bps_per_day_=abs(ROLLOVER[p]),
    )
    print(f"{p:>8}{ASSUMED_SPREAD_PIPS[p]:>8.1f}{r['spread_bps']:>12.4f}"
          f"{r['rollover_bps']:>14.4f}{r['total_bps']:>11.4f}"
          f"{r['breakeven_return']:>18.4%}")


max_spread = max_viable_spread_pips(
    gross_annual_return=gross_ann_ret, pair="EURUSD",
    quote_price=pip_ref, trades_per_year=tpy_per_pair,
)

print(f"\n[2] MAXIMUM VIABLE SPREAD -- book level, no spread assumption")
print(f"  gross annual return       {gross_ann_ret:>12.4%}")
print(f"  gross annual vol          {gross_vol:>12.4%}")
print(f"  gross Sharpe              {gross_sharpe:>+12.4f}")
print(f"  max viable round trip     {max_spread:>12.3f}  pips")
print(f"  pre-registered gate       {GATE_PIPS:>12.3f}  pips")
print(f"  margin over gate          {max_spread / GATE_PIPS:>12.2f}x")

print(f"\n{'pips':>9}{'cost %/yr':>12}{'net ann ret':>13}{'net Sharpe':>12}"
      f"{'breakeven SR':>14}")
for pips in COST_PIPS + [round(max_spread, 2)]:
    cost_yr = tpy_per_pair * pips * book_bps_per_pip / 1e4
    hurdle = breakeven_sharpe(
        cost_bps=pips * book_bps_per_pip, holding_period_days=HOLD_DAYS,
        annualized_vol=gross_vol, trades_per_year=tpy_per_pair,
    )
    print(f"{pips:>9.2f}{cost_yr:>12.4%}{gross_ann_ret - cost_yr:>13.4%}"
          f"{(gross_ann_ret - cost_yr) / gross_vol:>+12.4f}{hurdle:>14.4f}")

print(f"\n{'cost gate @' + str(GATE_PIPS) + ' pips':>32}"
      f"{'PASS' if max_spread > GATE_PIPS else 'FAIL':>10}")


print(f"\n[3] MAXIMUM VIABLE SPREAD -- per pair")
print(f"{'pair':>8}{'trades':>8}{'rt/yr':>8}{'gross %/yr':>12}"
      f"{'max pips':>10}{'assumed':>9}{'margin':>8}  verdict")

n_viable = 0
per_pair_max = []
for p in PAIRS:
    tp = trades[trades["pair"] == p]
    series = tp.set_index("date")["ret"].reindex(all_days).fillna(0.0)
    p_ann_ret = series.mean() * ANN_FACTOR
    p_rt_yr = len(tp) / SPAN_YEARS
    p_max = max_viable_spread_pips(p_ann_ret, p, REF_QUOTE[p], trades_per_year=p_rt_yr)
    assumed = ASSUMED_SPREAD_PIPS[p]
    ok = p_max > assumed
    n_viable += ok
    per_pair_max.append(p_max)
    print(f"{p:>8}{len(tp):>8}{p_rt_yr:>8.1f}{p_ann_ret:>+12.4%}"
          f"{p_max:>10.3f}{assumed:>9.1f}{p_max / assumed:>8.2f}"
          f"  {'viable' if ok else 'NOT viable'}")

print(f"\n{'pairs viable standalone':>32}{n_viable:>6} / {len(PAIRS)}")
print(f"{'per-pair mean max pips':>32}{np.mean(per_pair_max):>10.3f}")
print(f"{'book max pips':>32}{max_spread:>10.3f}")

print(f"\n{RULE}\n")
