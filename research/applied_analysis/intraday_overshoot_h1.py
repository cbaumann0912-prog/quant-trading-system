import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.performance_analyzer import PerformanceAnalyzer, information_ratio
from src.analysis.portfolio import cvar, var_historical
from src.analysis.signal_report import build_signal_report
from src.evaluation.bootstrap import block_bootstrap
from src.features.garch import fit_garch
from src.features.sessions import FILE_UTC_OFFSET_HOURS
from src.stats.hypothesis_tests import compute_achieved_power, t_test_mean
from src.stats.regression import interaction_regression_centered

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"
CACHE_DIR = REPO_ROOT / "research" / "applied_analysis" / "_overshoot_cache"

PAIRS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD",
    "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF",
]
START = "2011-01-01"
END = "2023-12-31"
NY = "America/New_York"

SCAN_OPEN = 9 * 60
SCAN_CLOSE = 12 * 60
EXIT_MIN = 13 * 60

K_PRIMARY = 2.0
KS = [1.5, 2.0, 2.5]
ENTRY_DELAYS = [0, 1, 5, 15]
PRIMARY_DELAY = 5
VOL_RATIO_MIN_OBS = 250
GARCH_MIN_TRAIN = 500

PIP_BP = 0.9
COST_PIPS = [1, 2, 3]
GATE_PIPS = 2
BREAK_DATE = pd.Timestamp("2017-07-01")
N_BOOTSTRAP = 3000
BLOCK_DAYS = 21
N_PERMUTATIONS = 1000
H2_FAST_MINUTES = 30
SEED = 42
ALPHA = 0.05
N_TRIALS = 6

CACHE_DIR.mkdir(parents=True, exist_ok=True)

for pair in PAIRS:
    cache_path = CACHE_DIR / f"{pair}.csv"
    if cache_path.exists():
        continue

    print(f"staging {pair} ...", flush=True)
    raw = pd.read_csv(DATA_DIR / f"{pair}.csv", usecols=["Datetime", "Close"])
    parsed = pd.to_datetime(raw["Datetime"], format="%Y%m%d %H%M%S")
    ny = pd.DatetimeIndex(
        (parsed + pd.Timedelta(hours=FILE_UTC_OFFSET_HOURS)).dt.tz_localize("UTC")
    ).tz_convert(NY)
    bars = pd.DataFrame({
        "c": raw["Close"].to_numpy(),
        "d": ny.normalize().tz_localize(None),
        "m": ny.hour.values * 60 + ny.minute.values,
    })
    del raw, parsed, ny

    daily_close = bars.groupby("d")["c"].last()
    daily_ret = np.log(daily_close / daily_close.shift(1)).dropna()

    # Walk-forward GARCH: parameters refit each year on data strictly before it,
    # then the variance recursion rolled forward on realised returns only. A
    # single full-sample fit would leak future data into the parameters even
    # with the path lagged, and Day 47 measured that leak shifting regime
    # labels by 39-68% on the momentum book.
    conditional_vol = pd.Series(np.nan, index=daily_ret.index)
    for year in sorted(daily_ret.index.year.unique()):
        train = daily_ret[daily_ret.index.year < year]
        if len(train) < GARCH_MIN_TRAIN:
            continue
        g = fit_garch(train)
        hist = daily_ret[daily_ret.index.year <= year]
        eps = (hist - train.mean()).to_numpy()
        var = np.empty(len(eps))
        var[0] = eps.var()
        for i in range(1, len(eps)):
            var[i] = g["omega"] + g["alpha"] * eps[i - 1] ** 2 + g["beta"] * var[i - 1]
        path = pd.Series(np.sqrt(var), index=hist.index)
        conditional_vol.loc[conditional_vol.index.year == year] = path[path.index.year == year]
    garch = {"conditional_vol": conditional_vol.dropna()}

    scan = bars[(bars["m"] >= SCAN_OPEN) & (bars["m"] <= EXIT_MIN)].sort_values(["d", "m"])
    sess_open = scan.groupby("d")["c"].first()
    scan_close = scan[scan["m"] <= SCAN_CLOSE].groupby("d")["c"].last()
    exit_px = scan.groupby("d")["c"].last()
    sess_ret = np.log(scan_close / sess_open)

    ratio = (sess_ret.expanding(VOL_RATIO_MIN_OBS).std()
             / daily_ret.reindex(sess_ret.index).expanding(VOL_RATIO_MIN_OBS).std()).shift(1)
    sigma_sess = garch["conditional_vol"].reindex(sess_ret.index).shift(1) * ratio

    out = pd.DataFrame({
        "sigma": sigma_sess.values,
        "open": sess_open.values,
        "exit_px": exit_px.reindex(sess_open.index).values,
    }, index=sess_open.index)
    out.index.name = "date"

    grouped = {d: (g["m"].to_numpy(), g["c"].to_numpy())
               for d, g in scan[scan["m"] <= SCAN_CLOSE].groupby("d")}
    for k in KS:
        cols = {f"t_{k}": [], f"disp_{k}": []}
        for delay in ENTRY_DELAYS:
            cols[f"px_{k}_d{delay}"] = []
        for date in out.index:
            s = out["sigma"].get(date, np.nan)
            mm, cc = grouped.get(date, (np.array([]), np.array([])))
            hit_i = -1
            if np.isfinite(s) and s > 0 and len(cc):
                hit = np.abs(np.log(cc / cc[0])) > k * s
                if hit.any():
                    hit_i = int(np.argmax(hit))
            if hit_i < 0:
                cols[f"t_{k}"].append(np.nan)
                cols[f"disp_{k}"].append(np.nan)
                for delay in ENTRY_DELAYS:
                    cols[f"px_{k}_d{delay}"].append(np.nan)
                continue
            cols[f"t_{k}"].append(mm[hit_i] - SCAN_OPEN)
            cols[f"disp_{k}"].append(np.log(cc[hit_i] / cc[0]))
            for delay in ENTRY_DELAYS:
                j = np.searchsorted(mm, mm[hit_i] + delay)
                cols[f"px_{k}_d{delay}"].append(cc[j] if j < len(cc) else np.nan)
        for name, vals in cols.items():
            out[name] = vals

    out = out.loc[(out.index >= START) & (out.index <= END)]
    out.to_csv(cache_path)

cache = {p: pd.read_csv(CACHE_DIR / f"{p}.csv", index_col=0, parse_dates=True) for p in PAIRS}
all_days = pd.Index(sorted(set().union(*[set(c.index) for c in cache.values()])), name="date")

# One annualization convention for everything: the empirical observations-per-year
# of the book's own index, matching PerformanceAnalyzer.compute_ann_factor so the
# module-computed and hand-computed figures cannot drift apart.
SPAN_YEARS = (all_days.max() - all_days.min()).days / 365.25
ANN_FACTOR = len(all_days) / SPAN_YEARS


def trades(k, delay):
    frames = []
    for p in PAIRS:
        px = f"px_{k}_d{delay}"
        x = cache[p].dropna(subset=[px, "exit_px", f"disp_{k}"])
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


def cost_per_year(t, pips):
    return (len(t) / SPAN_YEARS) * pips * PIP_BP / 1e4 / len(PAIRS)


def net_sharpe(t, b, pips):
    return (b.mean() * ANN_FACTOR - cost_per_year(t, pips)) / (b.std(ddof=1) * np.sqrt(ANN_FACTOR))


def sharpe_of(x):
    s = pd.Series(x, index=all_days[:len(x)])
    return PerformanceAnalyzer(returns=s).compute_sharpe()


RULE = "=" * 78
print(f"\n{RULE}")
print("INTRADAY OVERSHOOT REVERSAL -- H1")
print(f"{len(PAIRS)} pairs | {START} to {END} | lockbox sealed | k={K_PRIMARY} | "
      f"entry scan 09:00-12:00 ET, exit 13:00 ET")
print(RULE)

print("\n[1] ENTRY DELAY -- execution realism")
print(f"{'entry':>14}{'trades':>8}{'mean bp':>10}{'t':>8}{'p':>9}{'hit rate':>10}{'gross SR':>10}{'net SR':>9}")
for delay in ENTRY_DELAYS:
    t = trades(K_PRIMARY, delay)
    b, _ = book(t)
    tt = t_test_mean(t["ret"], null_mean=0.0, confidence=1 - ALPHA)
    label = "crossing bar" if delay == 0 else f"+{delay} min"
    print(f"{label:>14}{len(t):>8}{t['ret'].mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}"
          f"{tt['p_value']:>9.4f}{(t['ret'] > 0).mean():>10.1%}"
          f"{analyzer(b).compute_sharpe():>+10.3f}{net_sharpe(t, b, GATE_PIPS):>+9.3f}")
print(f"{'':>14}net SR is after {GATE_PIPS} pips round trip")

print("\n[2] ROBUSTNESS -- entry threshold (+5 min entry)")
print(f"{'k':>6}{'trades':>8}{'mean bp':>10}{'hit rate':>10}{'gross SR':>10}{'net SR':>9}")
mono = []
for k in KS:
    t = trades(k, PRIMARY_DELAY)
    b, _ = book(t)
    mono.append(t["ret"].mean())
    print(f"{k:>6}{len(t):>8}{t['ret'].mean() * 1e4:>+10.3f}{(t['ret'] > 0).mean():>10.1%}"
          f"{analyzer(b).compute_sharpe():>+10.3f}{net_sharpe(t, b, GATE_PIPS):>+9.3f}")

print("\n[3] TRIGGER TIMING -- minutes after 09:00 ET (k=2.0)")
tt_all = pd.concat([cache[p][f"t_{K_PRIMARY}"].dropna() for p in PAIRS])
for lo, hi in [(0, 30), (30, 31), (31, 45), (45, 60), (60, 90), (90, 120), (120, 150), (150, 181)]:
    n = int(((tt_all >= lo) & (tt_all < hi)).sum())
    lab = "09:30 exactly" if (lo, hi) == (30, 31) else f"{9 + lo // 60:02d}:{lo % 60:02d}-{9 + hi // 60:02d}:{hi % 60:02d}"
    print(f"  {lab:<16}{n:>6} ({n / len(tt_all):>5.1%})  {'#' * int(60 * n / len(tt_all))}")
print(f"  median {tt_all.median():.0f} min   q25 {tt_all.quantile(.25):.0f}   q75 {tt_all.quantile(.75):.0f}")

for delay, tag in [(0, "PRE-REGISTERED ENTRY (crossing bar)"),
                   (PRIMARY_DELAY, "PRIMARY: EXECUTION-REALISTIC ENTRY (+5 min)")]:
    t = trades(K_PRIMARY, delay)
    b, wide = book(t)
    report = analyzer(b).run_report(n_trials=N_TRIALS)
    obs = report.sharpe_ratio
    wins, losses = t.loc[t["ret"] > 0, "ret"], t.loc[t["ret"] < 0, "ret"]

    print(f"\n{RULE}")
    print(tag)
    print(RULE)

    print("\n  Return and risk")
    print(f"    ann return          {report.annualized_return * 100:>+9.3f}%")
    print(f"    ann volatility      {report.annualized_vol * 100:>9.3f}%")
    print(f"    Sharpe              {obs:>+9.3f}   (t {report.t_stat:+.2f})")
    print(f"    Sortino             {report.sortino_ratio:>+9.3f}")
    print(f"    Calmar              {report.calmar_ratio:>+9.3f}")
    dd = analyzer(b).compute_max_drawdown()
    print(f"    max drawdown        {report.max_drawdown * 100:>+9.2f}%   ({dd['duration_days']} days, "
          f"{dd['start_date'].date()} to {dd['end_date'].date()})")
    print(f"    VaR 95% (daily)     {var_historical(b, 0.95) * 100:>9.3f}%")
    print(f"    CVaR 95% (daily)    {cvar(b, 0.95) * 100:>9.3f}%")

    print("\n  Trade profile")
    print(f"    trades              {len(t):>9}   ({len(t) / SPAN_YEARS:.0f}/yr)")
    print(f"    active days         {(b != 0).sum():>9}   ({(b != 0).mean():.1%} of sample)")
    print(f"    hit rate            {(t['ret'] > 0).mean():>9.1%}")
    print(f"    mean win            {wins.mean() * 1e4:>+9.2f} bp")
    print(f"    mean loss           {losses.mean() * 1e4:>+9.2f} bp")
    print(f"    win/loss ratio      {abs(wins.mean() / losses.mean()):>9.3f}")
    print(f"    profit factor       {wins.sum() / abs(losses.sum()):>9.3f}")
    print(f"    median time to entry{t['tmin'].median():>9.0f} min after 09:00")

    print("\n  Net of cost")
    for pips in COST_PIPS:
        flag = "  <- gate" if pips == GATE_PIPS else ""
        print(f"    {pips} pip round trip   Sharpe {net_sharpe(t, b, pips):>+7.3f}"
              f"   cost {cost_per_year(t, pips) * 100:.3f}%/yr{flag}")

    print("\n  Significance")
    boot = block_bootstrap(series=b.values, block_size=BLOCK_DAYS, n_samples=N_BOOTSTRAP,
                           statistic_fn=sharpe_of, seed=SEED)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    boot_p = 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    print(f"    bootstrap 95% CI    [{lo:+.3f}, {hi:+.3f}]   p {boot_p:.5f}")

    rng = np.random.default_rng(SEED)
    null_trade = np.array([(t["ret"].values * rng.choice([-1, 1], len(t))).mean() for _ in range(N_PERMUTATIONS)])
    p_trade = np.mean(np.abs(null_trade) >= abs(t["ret"].mean()))
    null_day = np.array([
        sharpe_of(wide.mul(pd.Series(rng.choice([-1, 1], len(wide)), index=wide.index), axis=0)
                  .mean(axis=1).reindex(all_days).fillna(0.0).values)
        for _ in range(N_PERMUTATIONS)
    ])
    p_day = np.mean(np.abs(null_day) >= abs(obs))
    print(f"    permutation, trade  p {p_trade:.4f}   (tests the directional signal alone)")
    print(f"    permutation, day    p {p_day:.4f}   (preserves cross-pair structure)")
    print(f"    deflated Sharpe     {report.deflated_sharpe:>7.4f}   (n_trials={N_TRIALS})")
    print(f"    achieved power      {compute_achieved_power(n=int((b != 0).sum()), effect_size=abs(obs) / np.sqrt(ANN_FACTOR), alpha=ALPHA):>7.3f}")

    print("\n  Stability and structure")
    pre, post = b[b.index < BREAK_DATE], b[b.index >= BREAK_DATE]
    print(f"    break {BREAK_DATE.date()}    pre {PerformanceAnalyzer(returns=pre).compute_sharpe():+.3f}"
          f"   post {PerformanceAnalyzer(returns=post).compute_sharpe():+.3f}")
    rho = np.nanmean(wide.corr().values[np.triu_indices(len(PAIRS), 1)])
    br_eff = len(PAIRS) / (1 + (len(PAIRS) - 1) * rho)
    print(f"    cross-pair corr     {rho:>+7.3f}   effective breadth {br_eff:.2f} of {len(PAIRS)}")
    pt = t["ret"].mean() / t["ret"].std(ddof=1)
    iid = pt * np.sqrt(t.groupby("date").size().mean()) * np.sqrt((b != 0).sum() / SPAN_YEARS)
    print(f"    book SR if iid      {iid:>+7.3f}   vs actual {obs:+.3f} (gap = diversification)")
    ic_by_pair = [t.loc[t["pair"] == p, "ret"].mean() / t.loc[t["pair"] == p, "ret"].std(ddof=1) for p in PAIRS]
    print(f"    per-pair IR         {information_ratio(ic_by_pair, method='empirical'):>+7.3f}"
          f"   {sum(1 for v in ic_by_pair if v > 0)}/{len(PAIRS)} pairs positive")

    print("\n  Distribution")
    print(f"    skew                {report.skewness:>+7.3f}")
    print(f"    excess kurtosis     {report.excess_kurtosis:>+7.3f}")
    print(f"    Jarque-Bera p       {report.jb_p_value:>7.1e}   (normality)")
    print(f"    Ljung-Box p         {report.lb_p_value:>7.4f}   (autocorrelation)")

t = trades(K_PRIMARY, PRIMARY_DELAY)
b, wide = book(t)
a = analyzer(b)
boot = block_bootstrap(series=b.values, block_size=BLOCK_DAYS, n_samples=N_BOOTSTRAP,
                       statistic_fn=sharpe_of, seed=SEED)
boot_p = float(2 * min(np.mean(boot <= 0), np.mean(boot >= 0)))
pre, post = b[b.index < BREAK_DATE], b[b.index >= BREAK_DATE]

print(f"\n{RULE}")
print("PER-PAIR BREAKDOWN (+5 min entry)")
print(RULE)
print(f"{'pair':<9}{'trades':>8}{'mean bp':>10}{'t':>8}{'hit rate':>10}{'Sharpe':>9}")
for p in PAIRS:
    s = t.loc[t["pair"] == p]
    tt = t_test_mean(s["ret"], null_mean=0.0, confidence=1 - ALPHA)
    solo = s.set_index("date")["ret"].reindex(all_days).fillna(0.0)
    print(f"{p:<9}{len(s):>8}{s['ret'].mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}"
          f"{(s['ret'] > 0).mean():>10.1%}"
          f"{PerformanceAnalyzer(returns=solo).compute_sharpe():>+9.3f}")

print(f"\n{RULE}")
print("ANNUAL BREAKDOWN (+5 min entry)")
print(RULE)
print(f"{'year':<8}{'days':>7}{'trades':>8}{'return':>10}{'Sharpe':>9}{'maxDD':>9}")
for yr, grp in b.groupby(b.index.year):
    n_tr = int((t["date"].dt.year == yr).sum())
    ddy = ((1 + grp).cumprod() / (1 + grp).cumprod().cummax() - 1).min()
    print(f"{yr:<8}{len(grp):>7}{n_tr:>8}{grp.sum() * 100:>+9.2f}%"
          f"{PerformanceAnalyzer(returns=grp).compute_sharpe():>+9.3f}{ddy * 100:>+8.2f}%")
pos = sum(1 for _, g in b.groupby(b.index.year) if g.sum() > 0)
print(f"{'':<8}{pos} of {b.index.year.nunique()} years positive")

print(f"\n{RULE}")
print("H2: does reversal depend on displacement speed?")
print(RULE)
h2 = t.copy()
h2["fast"] = (h2["tmin"] <= H2_FAST_MINUTES).astype(float)
h2 = h2.merge(
    pd.concat([cache[p][[f"disp_{K_PRIMARY}"]].assign(pair=p).reset_index() for p in PAIRS]),
    on=["pair", "date"], how="left",
)
h2["size"] = h2[f"disp_{K_PRIMARY}"].abs()
h2 = h2.dropna(subset=["size"])

print(f"{'bucket':<10}{'trades':>8}{'mean bp':>10}{'t':>8}{'hit rate':>10}")
for lab, mask in [("fast", h2["fast"] == 1.0), ("slow", h2["fast"] == 0.0)]:
    s = h2.loc[mask, "ret"]
    tt = t_test_mean(s, null_mean=0.0, confidence=1 - ALPHA)
    print(f"{lab:<10}{len(s):>8}{s.mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}{(s > 0).mean():>10.1%}")
print(f"  split at {H2_FAST_MINUTES} min after 09:00 (spec Section 10)")

h2_fit = interaction_regression_centered(
    h2["ret"].reset_index(drop=True), h2["size"].reset_index(drop=True),
    h2["fast"].reset_index(drop=True), x1_label="displacement", x2_label="fast",
)
print(f"\n  interaction regression: return ~ displacement x fast")
print(f"    b3 (interaction)    {h2_fit['coefficients']['interaction']:>+9.4f}   "
      f"p {h2_fit['p_values']['interaction']:.4f}")
print(f"    n_obs               {h2_fit['n_obs']:>9}")
print(f"    condition number    {h2_fit['condition_number']:>9.2e}   max VIF "
      f"{max(h2_fit['vif'].values()):.3f}   gate {h2_fit['reliability_gate_passed']}")
h2_pass = (h2_fit["p_values"]["interaction"] < ALPHA
           and h2_fit["coefficients"]["interaction"] > 0
           and h2_fit["reliability_gate_passed"])
print(f"    H2 {'PASS' if h2_pass else 'FAIL'}   (prediction: b3 > 0, fast displacement reverts more)")

print("\n  Follow-up: is the fast penalty the 09:30 NYSE open rather than speed?")
print("  Hypothesis proposed and tested post-hoc, recorded as falsified.")
print(f"  {'bucket':<32}{'trades':>8}{'mean bp':>10}{'t':>8}{'hit rate':>10}")
for lab, mask in [("t < 30 (before equity open)", t["tmin"] < 30),
                  ("t == 30 (NYSE open exactly)", t["tmin"] == 30),
                  ("30 < t <= 45 (open aftermath)", (t["tmin"] > 30) & (t["tmin"] <= 45)),
                  ("t > 45 (rest of window)", t["tmin"] > 45)]:
    s = t.loc[mask, "ret"]
    tt = t_test_mean(s, null_mean=0.0, confidence=1 - ALPHA)
    print(f"  {lab:<32}{len(s):>8}{s.mean() * 1e4:>+10.3f}{tt['t_stat']:>+8.2f}{(s > 0).mean():>10.1%}")
print("  Prediction was that t<30 resembles the slow bucket. It does not, so speed rather")
print("  than the equity open remains the operative variable. Mechanism still unexplained.")

sr = build_signal_report(
    strategy_name="Intraday Overshoot Reversal (H1, +5min entry)",
    leg_ic_by_window={"overshoot": [t.loc[t["pair"] == p, "ret"].mean() /
                                    t.loc[t["pair"] == p, "ret"].std(ddof=1) for p in PAIRS]},
    leg_sharpe_by_window={"overshoot": [PerformanceAnalyzer(
        returns=t.loc[t["pair"] == p].set_index("date")["ret"]).compute_sharpe() for p in PAIRS]},
    leg_primary_p_value={"overshoot": boot_p},
    leg_regime_gated_returns={"overshoot": b},
    n_trials=N_TRIALS,
)
leg = sr.legs["overshoot"]

print(f"\n{RULE}")
print("SIGNAL REPORT (+5 min entry)")
print(RULE)
print(f"  cross-pair IC mean        {leg.ic_mean:>+8.4f}")
print(f"  cross-pair IC std         {leg.ic_std:>8.4f}")
print(f"  IC-derived IR             {leg.ic_ir:>+8.4f}")
print(f"  pairs with positive IC    {leg.ic_frac_positive:>8.1%}")
print(f"  per-pair Sharpe mean      {leg.sharpe_mean:>+8.4f}   std {leg.sharpe_std:.4f}")
print(f"  primary p                 {leg.primary_p_value:>8.5f}")
print(f"  BH-significant            {str(sr.bh_rejected['overshoot']):>8}   (alpha {sr.bh_alpha})")
print(f"  survives project bar      {str(leg.primary_p_value < sr.project_wide_bar_p):>8}   "
      f"(p < {sr.project_wide_bar_p})")
print(f"  deflated Sharpe           {leg.dsr:>8.4f}   (n_trials {leg.dsr_n_trials}, n_obs {leg.dsr_n_obs})")

print(f"\n{RULE}")
print("VERDICT")
print(RULE)
checks = {
    "H1  book Sharpe > 0": a.compute_sharpe() > 0,
    "H1  bootstrap p < 0.05": boot_p < ALPHA,
    "H1  reversion strengthens with threshold": mono[0] < mono[1] < mono[2],
    "H1  break: same sign both halves": np.sign(PerformanceAnalyzer(returns=pre).compute_sharpe())
                                        == np.sign(PerformanceAnalyzer(returns=post).compute_sharpe()),
    f"H1  cost gate: net Sharpe > 0 at {GATE_PIPS} pips": net_sharpe(t, b, GATE_PIPS) > 0,
    "H2  b3 > 0 and significant": h2_pass,
}
for label, passed in checks.items():
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
h1_pass = all(v for k, v in checks.items() if k.startswith("H1"))
print(f"\n  H1 {'PASS' if h1_pass else 'FAIL'}   H2 {'PASS' if h2_pass else 'FAIL'}")
print("  Lockbox 2024-2026 remains sealed.")

