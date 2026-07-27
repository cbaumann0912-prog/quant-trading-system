import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.performance_analyzer import (
    PerformanceAnalyzer,
    bps_per_pip,
    information_ratio,
)
from src.analysis.portfolio import cvar, var_historical, var_parametric
from src.analysis.signal_report import build_signal_report
from src.evaluation.bootstrap import block_bootstrap, bootstrap_confidence_interval
from src.evaluation.significance import (
    benjamini_hochberg_correction,
    paired_sign_permutation_test,
)
from src.signals.intraday_overshoot import build_overshoot_sessions
from src.stats.hypothesis_tests import (
    compute_achieved_power,
    compute_effect_size_cohens_d,
    compute_required_sample_size,
    t_test_mean,
)
from src.stats.regression import interaction_regression_centered

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]
START = "2011-01-01"
END = "2023-12-31"

SCAN_OPEN = 9 * 60
SCAN_CLOSE = 12 * 60
EXIT_MIN = 13 * 60

K_PRIMARY = 2.0
KS = [1.5, 2.0, 2.5]
ENTRY_DELAYS = [0, 1, 5, 15]
PRIMARY_DELAY = 5
VOL_RATIO_MIN_OBS = 250
GARCH_MIN_TRAIN = 500

COST_PIPS = [1, 2, 3]
GATE_PIPS = 2
BREAK_DATE = pd.Timestamp("2017-06-30")
N_BOOTSTRAP = 3000
BLOCK_DAYS = 21
N_PERMUTATIONS = 1000
H2_FAST_MINUTES = 30
SEED = 42
ALPHA = 0.05
N_TRIALS = 6
MIN_TRADES_PER_PAIR_YEAR = 15
TARGET_POWER = 0.80

PRIOR_STRATEGY_P = {
    "PC2 Carry Regime": 1.0,
    "Momentum w/ ML Regime": 1.0,
    "OU Half-Life Mean Reversion": 1.0,
    "Vol Regime Breakout/Mean-Rev": 0.563,
    "Month-End FX Flow": 1.0,
}

sessions = {}
for pair in PAIRS:
    print(f"staging {pair} ...", flush=True)
    sessions[pair] = build_overshoot_sessions(
        pair=pair, data_dir=DATA_DIR, start=START, end=END,
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


def trades(k, delay):
    frames = []
    for p in PAIRS:
        px = f"px_{k}_d{delay}"
        x = sessions[p].dropna(subset=[px, "exit_px", f"disp_{k}"])
        frames.append(pd.DataFrame({
            "pair": p, "date": x.index, "tmin": x[f"t_{k}"].values,
            "ret": (-np.sign(x[f"disp_{k}"]) * np.log(x["exit_px"] / x[px])).values,
        }))
    return pd.concat(frames, ignore_index=True)


def book(t):
    wide = t.pivot_table(index="date", columns="pair", values="ret")
    return wide.mean(axis=1).reindex(all_days).fillna(0.0), wide


def analyzer(b):
    return PerformanceAnalyzer(returns=b)


def bps_per_pip_weighted(t):
    share = t["pair"].value_counts(normalize=True)
    return sum(share.get(p, 0.0) * bps_per_pip(p, REF_QUOTE[p]) for p in PAIRS)


def cost_per_year(t, pips):
    return (len(t) / SPAN_YEARS) * pips * bps_per_pip_weighted(t) / 1e4 / len(PAIRS)


def net_sharpe(t, b, pips):
    return ((b.mean() * ANN_FACTOR - cost_per_year(t, pips))
            / (b.std(ddof=1) * np.sqrt(ANN_FACTOR)))


def sharpe_of(x):
    s = pd.Series(x, index=all_days[:len(x)])
    return PerformanceAnalyzer(returns=s).compute_sharpe()


def boot_p_two_sided(samples):
    return float(2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))


RULE = "=" * 78
print(f"\n{RULE}")
print("INTRADAY OVERSHOOT REVERSAL -- SECTION 10 VALIDATION")
print(f"{len(PAIRS)} pairs | {START} to {END} | k={K_PRIMARY} | entry +{PRIMARY_DELAY}min")
print(f"span {SPAN_YEARS:.2f}y | ann factor {ANN_FACTOR:.2f} | n_trials {N_TRIALS}")
print(RULE)


print("\n[1] ENTRY DELAY")
print(f"{'entry':>14}{'trades':>8}{'mean bp':>10}{'t':>8}{'p':>9}"
      f"{'hit rate':>10}{'gross SR':>10}{'net SR':>9}")
for delay in ENTRY_DELAYS:
    t_d = trades(K_PRIMARY, delay)
    b_d, _ = book(t_d)
    tt = t_test_mean(t_d["ret"], null_mean=0.0, confidence=1 - ALPHA)
    label = "crossing bar" if delay == 0 else f"+{delay} min"
    print(f"{label:>14}{len(t_d):>8}{t_d['ret'].mean() * 1e4:>+10.3f}"
          f"{tt['t_stat']:>+8.2f}{tt['p_value']:>9.4f}{(t_d['ret'] > 0).mean():>10.1%}"
          f"{analyzer(b_d).compute_sharpe():>+10.3f}{net_sharpe(t_d, b_d, GATE_PIPS):>+9.3f}")


print("\n[2] ROBUSTNESS 1 -- ENTRY THRESHOLD")
print(f"{'k':>6}{'trades':>8}{'mean bp':>10}{'hit rate':>10}{'gross SR':>10}{'net SR':>9}")
mono = []
for k in KS:
    t_k = trades(k, PRIMARY_DELAY)
    b_k, _ = book(t_k)
    mono.append(t_k["ret"].mean())
    print(f"{k:>6}{len(t_k):>8}{t_k['ret'].mean() * 1e4:>+10.3f}"
          f"{(t_k['ret'] > 0).mean():>10.1%}{analyzer(b_k).compute_sharpe():>+10.3f}"
          f"{net_sharpe(t_k, b_k, GATE_PIPS):>+9.3f}")
mono_ok = mono[0] < mono[1] < mono[2]
sign_ok = all(np.sign(m) == np.sign(mono[1]) for m in mono)
print(f"{'monotone increasing':>34}{str(mono_ok):>10}")
print(f"{'same sign across k':>34}{str(sign_ok):>10}")


print("\n[3] TRIGGER TIMING")
tt_all = pd.concat([sessions[p][f"t_{K_PRIMARY}"].dropna() for p in PAIRS])
print(f"{'window':<18}{'triggers':>10}{'share':>9}")
for lo, hi in [(0, 30), (30, 31), (31, 45), (45, 60),
               (60, 90), (90, 120), (120, 150), (150, 181)]:
    n = int(((tt_all >= lo) & (tt_all < hi)).sum())
    lab = ("09:30 exactly" if (lo, hi) == (30, 31)
           else f"{9 + lo // 60:02d}:{lo % 60:02d}-{9 + hi // 60:02d}:{hi % 60:02d}")
    print(f"{lab:<18}{n:>10}{n / len(tt_all):>9.1%}")
print(f"{'median min':<18}{tt_all.median():>10.0f}")
print(f"{'q25 min':<18}{tt_all.quantile(.25):>10.0f}")
print(f"{'q75 min':<18}{tt_all.quantile(.75):>10.0f}")


for delay, tag in [(0, "[4] CROSSING BAR ENTRY"),
                   (PRIMARY_DELAY, "[5] PRIMARY -- +5 MIN ENTRY")]:
    t_d = trades(K_PRIMARY, delay)
    b_d, wide_d = book(t_d)
    report = analyzer(b_d).run_report(n_trials=N_TRIALS)
    obs = report.sharpe_ratio
    wins, losses = t_d.loc[t_d["ret"] > 0, "ret"], t_d.loc[t_d["ret"] < 0, "ret"]
    dd = analyzer(b_d).compute_max_drawdown()

    print(f"\n{RULE}")
    print(tag)
    print(RULE)
    print(f"{'ann return':<24}{report.annualized_return * 100:>+10.3f}%")
    print(f"{'ann volatility':<24}{report.annualized_vol * 100:>10.3f}%")
    print(f"{'Sharpe':<24}{obs:>+10.3f}")
    print(f"{'Sharpe t':<24}{report.t_stat:>+10.2f}")
    print(f"{'Sortino':<24}{report.sortino_ratio:>+10.3f}")
    print(f"{'Calmar':<24}{report.calmar_ratio:>+10.3f}")
    print(f"{'max drawdown':<24}{report.max_drawdown * 100:>+10.2f}%")
    print(f"{'max DD days':<24}{dd['duration_days']:>10}")
    print(f"{'max DD start':<24}{str(dd['start_date'].date()):>10}")
    print(f"{'max DD end':<24}{str(dd['end_date'].date()):>10}")
    print(f"{'VaR 95% daily':<24}{var_historical(b_d, 0.95) * 100:>10.3f}%")
    print(f"{'VaR 95% parametric':<24}{var_parametric(b_d, 0.95) * 100:>10.3f}%")
    print(f"{'CVaR 95% daily':<24}{cvar(b_d, 0.95) * 100:>10.3f}%")
    print(f"{'trades':<24}{len(t_d):>10}")
    print(f"{'trades/yr':<24}{len(t_d) / SPAN_YEARS:>10.1f}")
    print(f"{'trades/yr/pair':<24}{len(t_d) / SPAN_YEARS / len(PAIRS):>10.2f}")
    print(f"{'active days':<24}{(b_d != 0).sum():>10}")
    print(f"{'active day share':<24}{(b_d != 0).mean():>10.1%}")
    print(f"{'hit rate':<24}{(t_d['ret'] > 0).mean():>10.1%}")
    print(f"{'mean win bp':<24}{wins.mean() * 1e4:>+10.2f}")
    print(f"{'mean loss bp':<24}{losses.mean() * 1e4:>+10.2f}")
    print(f"{'win/loss ratio':<24}{abs(wins.mean() / losses.mean()):>10.3f}")
    print(f"{'profit factor':<24}{wins.sum() / abs(losses.sum()):>10.3f}")
    print(f"{'median entry min':<24}{t_d['tmin'].median():>10.0f}")
    print(f"{'skew':<24}{report.skewness:>+10.3f}")
    print(f"{'excess kurtosis':<24}{report.excess_kurtosis:>+10.3f}")
    print(f"{'Jarque-Bera p':<24}{report.jb_p_value:>10.1e}")
    print(f"{'Ljung-Box p':<24}{report.lb_p_value:>10.4f}")
    print(f"{'deflated Sharpe':<24}{report.deflated_sharpe:>10.4f}")

    print(f"\n{'pips':>8}{'cost %/yr':>12}{'net SR':>10}")
    for pips in COST_PIPS:
        print(f"{pips:>8}{cost_per_year(t_d, pips) * 100:>12.3f}"
              f"{net_sharpe(t_d, b_d, pips):>+10.3f}")


t = trades(K_PRIMARY, PRIMARY_DELAY)
b, wide = book(t)
a = analyzer(b)
obs = a.compute_sharpe()
bpp = bps_per_pip_weighted(t)

boot_sr = block_bootstrap(series=b.values, block_size=BLOCK_DAYS,
                          n_samples=N_BOOTSTRAP, statistic_fn=sharpe_of, seed=SEED)
sr_lo, sr_hi = np.percentile(boot_sr, [2.5, 97.5])
sr_p = boot_p_two_sided(boot_sr)

boot_mean = block_bootstrap(series=b.values, block_size=BLOCK_DAYS,
                            n_samples=N_BOOTSTRAP, statistic_fn=np.mean, seed=SEED)
mean_lo, mean_hi = np.percentile(boot_mean, [2.5, 97.5])
mean_p = boot_p_two_sided(boot_mean)

print(f"\n{RULE}")
print("[6] H1 PRIMARY -- date-clustered block bootstrap")
print(RULE)
print(f"{'pooled mean trade bp':<28}{t['ret'].mean() * 1e4:>+12.4f}")
print(f"{'mean daily book bp':<28}{b.mean() * 1e4:>+12.4f}")
print(f"{'mean CI low bp':<28}{mean_lo * 1e4:>+12.4f}")
print(f"{'mean CI high bp':<28}{mean_hi * 1e4:>+12.4f}")
print(f"{'mean bootstrap p':<28}{mean_p:>12.5f}")
print(f"{'book Sharpe':<28}{obs:>+12.4f}")
print(f"{'Sharpe CI low':<28}{sr_lo:>+12.4f}")
print(f"{'Sharpe CI high':<28}{sr_hi:>+12.4f}")
print(f"{'Sharpe bootstrap p':<28}{sr_p:>12.5f}")
print(f"{'block size days':<28}{BLOCK_DAYS:>12}")
print(f"{'n bootstrap':<28}{N_BOOTSTRAP:>12}")

rho = np.nanmean(wide.corr().values[np.triu_indices(len(PAIRS), 1)])
br_eff = len(PAIRS) / (1 + (len(PAIRS) - 1) * rho)
power = compute_achieved_power(n=int((b != 0).sum()),
                               effect_size=abs(obs) / np.sqrt(ANN_FACTOR), alpha=ALPHA)
print(f"{'cross-pair corr':<28}{rho:>+12.4f}")
print(f"{'effective breadth':<28}{br_eff:>12.2f}")
print(f"{'achieved power':<28}{power:>12.4f}")

pt = t["ret"].mean() / t["ret"].std(ddof=1)
iid = pt * np.sqrt(t.groupby("date").size().mean()) * np.sqrt((b != 0).sum() / SPAN_YEARS)
print(f"{'book SR if iid':<28}{iid:>+12.4f}")
ir_pairs = [t.loc[t["pair"] == p, "ret"].mean() / t.loc[t["pair"] == p, "ret"].std(ddof=1)
            for p in PAIRS]
ir_lo, ir_hi = bootstrap_confidence_interval(
    np.array(ir_pairs), np.mean, n_bootstrap=N_BOOTSTRAP, confidence=1 - ALPHA
)
print(f"{'per-pair IR':<28}{information_ratio(ir_pairs, method='empirical'):>+12.4f}")
print(f"{'per-pair mean IR CI low':<28}{ir_lo:>+12.4f}")
print(f"{'per-pair mean IR CI high':<28}{ir_hi:>+12.4f}")
print(f"{'pairs positive':<28}{sum(1 for v in ir_pairs if v > 0):>12} / {len(PAIRS)}")
req_n = compute_required_sample_size(
    effect_size=abs(obs) / np.sqrt(ANN_FACTOR), alpha=ALPHA, power=TARGET_POWER
)
print(f"{'n for 80% power':<28}{req_n:>12}")
print(f"{'n active days':<28}{int((b != 0).sum()):>12}")


print(f"\n{RULE}")
print("[7] ROBUSTNESS 2 -- STRUCTURAL BREAK")
print(RULE)
pre, post = b[b.index < BREAK_DATE], b[b.index >= BREAK_DATE]
pre_sr = PerformanceAnalyzer(returns=pre).compute_sharpe()
post_sr = PerformanceAnalyzer(returns=post).compute_sharpe()


def half_bootstrap_p(half):
    idx = half.index

    def sr_half(x):
        return PerformanceAnalyzer(returns=pd.Series(x, index=idx[:len(x)])).compute_sharpe()

    samples = block_bootstrap(series=half.values, block_size=BLOCK_DAYS,
                              n_samples=N_BOOTSTRAP, statistic_fn=sr_half, seed=SEED)
    return boot_p_two_sided(samples)


pre_p = half_bootstrap_p(pre)
post_p = half_bootstrap_p(post)
print(f"{'break date':<28}{str(BREAK_DATE.date()):>12}")
print(f"{'pre Sharpe':<28}{pre_sr:>+12.4f}")
print(f"{'pre bootstrap p':<28}{pre_p:>12.5f}")
print(f"{'pre days':<28}{len(pre):>12}")
print(f"{'post Sharpe':<28}{post_sr:>+12.4f}")
print(f"{'post bootstrap p':<28}{post_p:>12.5f}")
print(f"{'post days':<28}{len(post):>12}")
print(f"{'same sign':<28}{str(np.sign(pre_sr) == np.sign(post_sr)):>12}")
print(f"{'post significant':<28}{str(post_p < ALPHA):>12}")


print(f"\n{RULE}")
print("[8] ROBUSTNESS 3 -- PERMUTATION")
print(RULE)
perm_trade = paired_sign_permutation_test(
    t["ret"].values, n_permutations=N_PERMUTATIONS, seed=SEED, alternative="two-sided"
)
perm_day = paired_sign_permutation_test(
    b[b != 0].values, n_permutations=N_PERMUTATIONS, seed=SEED, alternative="two-sided"
)
p_trade = perm_trade["p_value"]
p_day = perm_day["p_value"]
print(f"{'n permutations':<28}{N_PERMUTATIONS:>12}")
print(f"{'per-trade observed mean bp':<28}{perm_trade['observed_mean_diff'] * 1e4:>+12.4f}")
print(f"{'per-trade sign flip p':<28}{p_trade:>12.4f}")
print(f"{'per-day observed mean bp':<28}{perm_day['observed_mean_diff'] * 1e4:>+12.4f}")
print(f"{'per-day sign flip p':<28}{p_day:>12.4f}")


print(f"\n{RULE}")
print("[9] PER-PAIR")
print(RULE)
print(f"{'pair':<9}{'trades':>8}{'tr/yr':>8}{'mean bp':>10}{'t':>8}"
      f"{'hit rate':>10}{'Sharpe':>9}")
for p in PAIRS:
    s = t.loc[t["pair"] == p]
    tt = t_test_mean(s["ret"], null_mean=0.0, confidence=1 - ALPHA)
    solo = s.set_index("date")["ret"].reindex(all_days).fillna(0.0)
    print(f"{p:<9}{len(s):>8}{len(s) / SPAN_YEARS:>8.1f}{s['ret'].mean() * 1e4:>+10.3f}"
          f"{tt['t_stat']:>+8.2f}{(s['ret'] > 0).mean():>10.1%}"
          f"{PerformanceAnalyzer(returns=solo).compute_sharpe():>+9.3f}")
min_tr_yr = min(len(t.loc[t["pair"] == p]) / SPAN_YEARS for p in PAIRS)
print(f"{'min trades/yr/pair':<35}{min_tr_yr:>9.1f}")


print(f"\n{RULE}")
print("[10] ANNUAL")
print(RULE)
print(f"{'year':<8}{'days':>7}{'trades':>8}{'return':>10}{'Sharpe':>9}{'maxDD':>9}")
for yr, grp in b.groupby(b.index.year):
    n_tr = int((t["date"].dt.year == yr).sum())
    ddy = PerformanceAnalyzer(returns=grp).compute_max_drawdown()["value"]
    print(f"{yr:<8}{len(grp):>7}{n_tr:>8}{grp.sum() * 100:>+9.2f}%"
          f"{PerformanceAnalyzer(returns=grp).compute_sharpe():>+9.3f}{ddy * 100:>+8.2f}%")
pos = sum(1 for _, g in b.groupby(b.index.year) if g.sum() > 0)
print(f"{'years positive':<23}{pos} / {b.index.year.nunique()}")


print(f"\n{RULE}")
print("[11] H2 -- DISPLACEMENT SPEED")
print(RULE)
h2 = t.copy()
h2["fast"] = (h2["tmin"] <= H2_FAST_MINUTES).astype(float)
h2 = h2.merge(
    pd.concat([sessions[p][[f"disp_{K_PRIMARY}"]].assign(pair=p).reset_index()
               for p in PAIRS]),
    on=["pair", "date"], how="left",
)
h2["size"] = h2[f"disp_{K_PRIMARY}"].abs()
h2 = h2.dropna(subset=["size"])

print(f"{'bucket':<10}{'trades':>8}{'mean bp':>10}{'t':>8}{'hit rate':>10}")
for lab, mask in [("fast", h2["fast"] == 1.0), ("slow", h2["fast"] == 0.0)]:
    s = h2.loc[mask, "ret"]
    tt = t_test_mean(s, null_mean=0.0, confidence=1 - ALPHA)
    print(f"{lab:<10}{len(s):>8}{s.mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}"
          f"{(s > 0).mean():>10.1%}")

h2_fit = interaction_regression_centered(
    h2["ret"].reset_index(drop=True), h2["size"].reset_index(drop=True),
    h2["fast"].reset_index(drop=True), x1_label="displacement", x2_label="fast",
)
h2_pass = bool(h2_fit["p_values"]["interaction"] < ALPHA
               and h2_fit["coefficients"]["interaction"] > 0
               and h2_fit["reliability_gate_passed"])
cohens_d = compute_effect_size_cohens_d(
    h2.loc[h2["fast"] == 1.0, "ret"], h2.loc[h2["fast"] == 0.0, "ret"]
)
print(f"\n{'fast vs slow Cohens d':<28}{cohens_d:>+12.4f}")
print(f"{'b3 interaction':<28}{h2_fit['coefficients']['interaction']:>+12.4f}")
print(f"{'b3 p':<28}{h2_fit['p_values']['interaction']:>12.4f}")
print(f"{'n_obs':<28}{h2_fit['n_obs']:>12}")
print(f"{'condition number':<28}{h2_fit['condition_number']:>12.2e}")
print(f"{'max VIF':<28}{max(h2_fit['vif'].values()):>12.3f}")
print(f"{'reliability gate':<28}{str(h2_fit['reliability_gate_passed']):>12}")
print(f"{'H2':<28}{'PASS' if h2_pass else 'FAIL':>12}")

print(f"\n{'bucket':<32}{'trades':>8}{'mean bp':>10}{'t':>8}{'hit rate':>10}")
for lab, mask in [("t < 30", t["tmin"] < 30),
                  ("t == 30", t["tmin"] == 30),
                  ("30 < t <= 45", (t["tmin"] > 30) & (t["tmin"] <= 45)),
                  ("t > 45", t["tmin"] > 45)]:
    s = t.loc[mask, "ret"]
    tt = t_test_mean(s, null_mean=0.0, confidence=1 - ALPHA)
    print(f"{lab:<32}{len(s):>8}{s.mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}"
          f"{(s > 0).mean():>10.1%}")


print(f"\n{RULE}")
print("[12] MULTIPLE TESTING")
print(RULE)
bh_names = list(PRIOR_STRATEGY_P) + ["Intraday Overshoot Reversal"]
bh_ps = list(PRIOR_STRATEGY_P.values()) + [sr_p]
bh_flags = benjamini_hochberg_correction(bh_ps, alpha=ALPHA)
bh_pass = bool(bh_flags[-1])
print(f"{'strategy':<32}{'p':>10}{'BH reject':>12}")
for name, pv, flag in zip(bh_names, bh_ps, bh_flags):
    print(f"{name:<32}{pv:>10.5f}{str(flag):>12}")
print(f"{'strategies tested':<32}{len(bh_ps):>10}")
print(f"{'rank-1 bar':<32}{ALPHA / len(bh_ps):>10.5f}")

sr_rep = build_signal_report(
    strategy_name="Intraday Overshoot Reversal",
    leg_ic_by_window={"overshoot": ir_pairs},
    leg_sharpe_by_window={"overshoot": [
        PerformanceAnalyzer(returns=t.loc[t["pair"] == p].set_index("date")["ret"]
                            ).compute_sharpe() for p in PAIRS]},
    leg_primary_p_value={"overshoot": sr_p},
    leg_regime_gated_returns={"overshoot": b},
    n_trials=N_TRIALS,
)
leg = sr_rep.legs["overshoot"]
print(f"{'cross-pair IC mean':<28}{leg.ic_mean:>+12.4f}")
print(f"{'cross-pair IC std':<28}{leg.ic_std:>12.4f}")
print(f"{'IC-derived IR':<28}{leg.ic_ir:>+12.4f}")
print(f"{'pairs positive IC':<28}{leg.ic_frac_positive:>12.1%}")
print(f"{'per-pair Sharpe mean':<28}{leg.sharpe_mean:>+12.4f}")
print(f"{'per-pair Sharpe std':<28}{leg.sharpe_std:>12.4f}")
print(f"{'deflated Sharpe':<28}{leg.dsr:>12.4f}")
print(f"{'DSR n_obs':<28}{leg.dsr_n_obs:>12}")


print(f"\n{RULE}")
print("[13] SECTION 9 FAILURE CONDITIONS")
print(RULE)
smallest_k_strongest = mono[0] > mono[1]
fails = {
    "net Sharpe non-positive at gate": net_sharpe(t, b, GATE_PIPS) <= 0,
    "effect only in smallest displacement": smallest_k_strongest,
    "effective breadth near 1": br_eff < 1.5,
    "H2 null": not h2_pass,
    "trades/yr/pair below 15": min_tr_yr < MIN_TRADES_PER_PAIR_YEAR,
}
for label, triggered in fails.items():
    print(f"{'TRIGGERED' if triggered else 'clear':>10}  {label}")


print(f"\n{RULE}")
print("[14] SECTION 10 VERDICT")
print(RULE)
checks = {
    "H1 primary p < 0.05": sr_p < ALPHA,
    "H1 predicted sign": obs > 0,
    "R1 threshold monotone": mono_ok,
    "R1 same sign across k": sign_ok,
    "R2 same sign both halves": bool(np.sign(pre_sr) == np.sign(post_sr)),
    "R2 post-break p < 0.05": post_p < ALPHA,
    "R3 permutation per-trade p < 0.05": p_trade < ALPHA,
    "R3 permutation per-day p < 0.05": p_day < ALPHA,
    "reliability gate": bool(h2_fit["reliability_gate_passed"]),
    f"cost gate at {GATE_PIPS} pips": net_sharpe(t, b, GATE_PIPS) > 0,
    "BH rank 1 of 6": bh_pass,
}
for label, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL':>6}  {label}")

h1_strict = all(checks.values())
h1_perday = all(v for k, v in checks.items() if k != "R3 permutation per-trade p < 0.05")
print(f"\n{'H1 strict all checks':<34}{'PASS' if h1_strict else 'FAIL':>10}")
print(f"{'H1 excl per-trade permutation':<34}{'PASS' if h1_perday else 'FAIL':>10}")
print(f"{'H2':<34}{'PASS' if h2_pass else 'FAIL':>10}")
print(f"{'Section 10 strategy verdict':<34}"
      f"{'PASS' if (h1_strict and h2_pass) else 'FAIL':>10}")
print(f"{'lockbox':<34}{'SEALED':>10}")
print(f"\n{RULE}\n")
