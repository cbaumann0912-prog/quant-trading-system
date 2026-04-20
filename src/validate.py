# =============================================================================
# validate.py — Phase 1 Portfolio Validation (modular CLI + interactive menu)
# =============================================================================
# Runs validation on the COMBINED portfolio (EURUSD + GBPUSD + USDJPY under
# the portfolio open-risk cap) — the configuration that ships to deployment.
#
# TRUE MODULAR DISPATCH
# ---------------------
# Each --section runs ONLY that section and exits.  There is no fall-through
# into later stages.  Baseline is executed on demand when a section requires
# a trade log; nothing else is implied.
#
# USAGE
# -----
# CLI (explicit, non-interactive):
#   python validate.py --section full
#   python validate.py --section baseline
#   python validate.py --section walkforward
#   python validate.py --section robustness
#   python validate.py --section montecarlo
#   python validate.py --section montecarlo --n-sim 2000 --seed 7
#   python validate.py --section full --no-plots --quiet
#   python validate.py --section full --no-export
#
# Interactive (no CLI args -> menu):
#   python validate.py
#     1) Full validation
#     2) Baseline only
#     3) Walk-forward only
#     4) Robustness only
#     5) Monte Carlo only (bootstrap)
#
# PORTFOLIO BACKTEST CALLS PER SECTION
# ------------------------------------
#   baseline     : 1        (3 per-pair engine calls)
#   walkforward  : 1        (baseline reused)
#   robustness   : 9        (baseline reused + 8 TP variants = 27 engine calls)
#   montecarlo   : 1        (baseline reused)
#   full         : 9        (same as robustness; all other sections reuse)
# =============================================================================

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import io
import os
import sys

# Force UTF-8 so Windows cp1252 doesn't choke on box-drawing chars printed
# by the robustness module.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

import numpy as np
import pandas as pd

from trailing     import STARTING_EQUITY, MAX_RISK_PCT
from robustness   import (
    run_phase1_walk_forward,
    print_phase1_decision,
    _metrics,
)
from monte_carlo  import run_monte_carlo, summary_to_frame
from portfolio    import run_portfolio_backtest
from sim_costs    import reset_rng
from run_portfolio import PAIRS_CONFIG, RISK_PCT, PORTFOLIO_CAP
from validation_plots import (
    plot_baseline,
    plot_walkforward,
    plot_robustness,
    plot_monte_carlo,
    plot_validation_summary,
)


# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_START = "2011-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2026-12-31"

TP1_VARIANTS = [0.70, 0.75, 0.80]
TP2_VARIANTS = [1.40, 1.50, 1.60]
TP1_BASE     = 0.75
TP2_BASE     = 1.50

DD_LIMIT = 10.0   # max acceptable drawdown % for final decision + MC threshold

PORTFOLIO_LABEL    = "PORTFOLIO (EURUSD + GBPUSD + USDJPY)"
PORTFOLIO_RNG_SEED = 42
MC_SEED_DEFAULT    = 42
MC_N_SIM_DEFAULT   = 1000

_BACKTEST_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT      = os.path.dirname(_BACKTEST_DIR)
# Each validation section writes its CSVs + paper-ready PNGs into its own
# subfolder under this root.
DEFAULT_EXPORT_DIR = os.path.join(_PROJECT_ROOT, "results", "validation")

# Sub-folder names per section.  `summary` holds the cross-section dashboard
# produced only by `run_full_validation`.
SECTION_SUBDIRS = {
    "baseline":    "baseline",
    "walkforward": "walkforward",
    "robustness":  "robustness",
    "montecarlo":  "montecarlo",
    "summary":     "summary",
}


def _section_dir(base: str | None, section: str) -> str | None:
    """Return the per-section subfolder under `base`, creating it on demand."""
    if not base:
        return None
    sub = SECTION_SUBDIRS.get(section, section)
    path = os.path.join(base, sub)
    os.makedirs(path, exist_ok=True)
    return path


# =============================================================================
# Printing helpers
# =============================================================================

def _banner(title: str, width: int = 72) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title: str, width: int = 72) -> None:
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print("-" * width)


# =============================================================================
# Portfolio runners (shared by multiple sections)
# =============================================================================

def _run_portfolio_once(
    tp1_r: float | None = None,
    tp2_r: float | None = None,
    verbose: bool = False,
    out_dir: str | None = None,
) -> dict:
    """
    Run one full portfolio backtest (3 per-pair engine calls + event replay).
    TP overrides are forwarded via engine_kwargs.  When None, engine defaults
    (trained values) are used -- the baseline cell.
    """
    ek: dict = {}
    if tp1_r is not None: ek["tp1_r"] = tp1_r
    if tp2_r is not None: ek["tp2_r"] = tp2_r

    reset_rng(PORTFOLIO_RNG_SEED)
    return run_portfolio_backtest(
        pairs_config      = PAIRS_CONFIG,
        risk_pct_by_pair  = RISK_PCT,
        portfolio_cap_pct = PORTFOLIO_CAP,
        starting_equity   = STARTING_EQUITY,
        spread_mult       = 1.0,
        slip_mult         = 1.0,
        out_dir           = out_dir,
        verbose           = verbose,
        engine_kwargs     = ek or None,
    )


def _portfolio_trade_df(portfolio_results: dict) -> pd.DataFrame:
    """
    Flatten the portfolio trade log, sorted by exit_time.  Robustness helpers
    want `total_pnl`; alias it from `total_portfolio_pnl` when absent.
    """
    log = portfolio_results.get("portfolio_trade_log") or []
    if not log:
        return pd.DataFrame()
    df = (
        pd.DataFrame(log)
          .sort_values("exit_time")
          .reset_index(drop=True)
    )
    if "total_pnl" not in df.columns and "total_portfolio_pnl" in df.columns:
        df["total_pnl"] = df["total_portfolio_pnl"]
    return df


# =============================================================================
# Section: baseline
# =============================================================================

def run_baseline_portfolio(
    verbose: bool = True,
    export_dir: str | None = None,
) -> tuple[pd.DataFrame, float, dict]:
    """
    Run the baseline portfolio backtest.  Returns (trade_df, final_equity, raw).
    When `export_dir` is given, the portfolio module writes its CSV bundle
    (trades, equity, yearly, pairs, summary, overlap) there.
    """
    _banner(f"BASELINE PORTFOLIO BACKTEST  |  {PORTFOLIO_LABEL}")
    if verbose:
        print(f"  DD limit: <= {DD_LIMIT:.0f}%   Cap: {PORTFOLIO_CAP:.1%}")
        print(f"  Per-pair risk : "
              + "  ".join(f"{p} {r:.2%}" for p, r in RISK_PCT.items()))

    results  = _run_portfolio_once(verbose=verbose, out_dir=export_dir)
    trade_df = _portfolio_trade_df(results)
    final_eq = float(
        trade_df["equity_after"].iloc[-1]
        if not trade_df.empty else STARTING_EQUITY
    )

    if verbose:
        print(f"\n  Baseline trades      : {len(trade_df)}")
        print(f"  Baseline final equity: ${final_eq:,.2f}")
        if export_dir:
            print(f"  Exported baseline CSVs -> {export_dir}")

    if export_dir:
        paths = plot_baseline(
            trade_df     = trade_df,
            equity_curve = results.get("equity_curve", []),
            yearly_df    = results.get("yearly_df"),
            pair_df      = results.get("pair_df"),
            out_dir      = export_dir,
        )
        if verbose:
            for p in paths:
                print(f"  Chart   : {p}")

    return trade_df, final_eq, results


# =============================================================================
# Section: walkforward
# =============================================================================

def run_walkforward(
    base_trade_df: pd.DataFrame,
    starting_equity: float = STARTING_EQUITY,
    train_start: str = TRAIN_START, train_end: str = TRAIN_END,
    test_start:  str = TEST_START,  test_end:  str = TEST_END,
    export_dir: str | None = None,
) -> dict:
    """
    Walk-forward validation on the portfolio trade log.  Slices by entry_time
    and replays pnl_pct; no extra backtests needed.
    """
    _subheader("WALK-FORWARD VALIDATION  (portfolio)")
    print("  (sliced from baseline — no extra portfolio runs)")

    result = run_phase1_walk_forward(
        frames               = None,
        pair                 = PORTFOLIO_LABEL,
        run_backtest_fn      = None,
        train_start          = train_start, train_end = train_end,
        test_start           = test_start,  test_end  = test_end,
        starting_equity      = starting_equity,
        precomputed_trade_df = base_trade_df,
    )

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        rows = []
        for phase in ("TRAIN", "TEST"):
            m = result.get(phase)
            if isinstance(m, dict) and m:
                rows.append({"phase": phase, **m})
        if rows:
            path = os.path.join(export_dir, "walkforward_summary.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  Exported : {path}")

        paths = plot_walkforward(
            walk_result     = result,
            trade_df        = base_trade_df,
            train_start     = train_start, train_end = train_end,
            test_start      = test_start,  test_end  = test_end,
            starting_equity = starting_equity,
            out_dir         = export_dir,
        )
        for p in paths:
            print(f"  Chart   : {p}")

    return result


# =============================================================================
# Section: robustness
# =============================================================================

def run_robustness(
    base_trade_df: pd.DataFrame,
    base_final_eq: float,
    starting_equity: float = STARTING_EQUITY,
    tp1_variants: list | None = None,
    tp2_variants: list | None = None,
    export_dir: str | None = None,
) -> pd.DataFrame:
    """
    Test the TP1/TP2 grid on the FULL PORTFOLIO.  Baseline cell reused; each
    other cell is one full portfolio backtest.
    """
    tp1_variants = tp1_variants or TP1_VARIANTS
    tp2_variants = tp2_variants or TP2_VARIANTS
    n_cells      = len(tp1_variants) * len(tp2_variants)

    W   = 80
    SEP = "=" * W
    DIV = "  " + "-" * (W - 2)

    print(f"\n{SEP}")
    print(f"  PARAMETER ROBUSTNESS CHECK  —  {PORTFOLIO_LABEL}")
    print(f"  TP1 variants : {tp1_variants}   TP2 variants : {tp2_variants}")
    print(f"  Baseline ({TP1_BASE:.2f} / {TP2_BASE:.2f}) reused — "
          f"{n_cells - 1} extra portfolio runs")
    print(SEP)
    print(f"  {'TP1':>5} {'TP2':>5}  {'Trades':>7} {'Return%':>9} "
          f"{'MaxDD%':>8} {'Sharpe':>8} {'PF':>7}")
    print(DIV)

    rows = []
    for tp1 in tp1_variants:
        for tp2 in tp2_variants:
            is_base = abs(tp1 - TP1_BASE) < 0.001 and abs(tp2 - TP2_BASE) < 0.001

            if is_base:
                trade_df, final_eq = base_trade_df, base_final_eq
            else:
                res      = _run_portfolio_once(tp1_r=tp1, tp2_r=tp2, verbose=False)
                trade_df = _portfolio_trade_df(res)
                final_eq = float(
                    trade_df["equity_after"].iloc[-1]
                    if not trade_df.empty else starting_equity
                )

            m   = _metrics(trade_df, final_eq, starting_equity)
            tag = "  <- base" if is_base else ""
            print(f"  {tp1:>5.2f} {tp2:>5.2f}  "
                  f"{m['trades']:>7}  "
                  f"{m['total_return']:>+8.2f}%  "
                  f"{m['max_drawdown']:>+7.2f}%  "
                  f"{m['sharpe']:>8.3f}  "
                  f"{m['profit_factor']:>6.3f}"
                  f"{tag}")
            rows.append({"tp1_r": tp1, "tp2_r": tp2, "is_baseline": is_base, **m})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ret_mean  = float(df["total_return"].mean())
    ret_std   = float(df["total_return"].std())
    shr_std   = float(df["sharpe"].std())
    cv        = ret_std / abs(ret_mean) if ret_mean != 0 else float("inf")
    prof_frac = float((df["total_return"] > 0).mean())
    n_prof    = int(prof_frac * len(df))

    print(DIV)
    print(f"\n  Return range       : [{df['total_return'].min():+.2f}%, "
          f"{df['total_return'].max():+.2f}%]")
    print(f"  Return sigma / CV  : {ret_std:.2f}%  /  {cv:.2f}")
    print(f"  Sharpe sigma       : {shr_std:.3f}")
    print(f"  Profitable configs : {n_prof} / {len(df)}")

    if cv < 0.20 and shr_std < 0.30 and prof_frac >= 0.80:
        verdict, detail = "STABLE", "performance consistent across all TP variants"
    elif cv < 0.40 and prof_frac >= 0.60:
        verdict, detail = "STABLE", "acceptable sensitivity to TP placement"
    else:
        verdict, detail = "SENSITIVE", "results depend heavily on TP levels — possible overfit"

    print(f"\n  ROBUSTNESS RESULT: {verdict}  ({detail})")
    print(f"{SEP}\n")

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "parameter_robustness_grid.csv")
        df.to_csv(path, index=False)
        print(f"  Exported : {path}")

        paths = plot_robustness(
            param_df = df,
            out_dir  = export_dir,
            base_tp1 = TP1_BASE, base_tp2 = TP2_BASE,
        )
        for p in paths:
            print(f"  Chart   : {p}")

    return df


# =============================================================================
# Section: montecarlo (bootstrap only)
# =============================================================================

def run_mc(
    base_trade_df: pd.DataFrame,
    n_simulations: int = MC_N_SIM_DEFAULT,
    seed: int = MC_SEED_DEFAULT,
    drawdown_threshold: float = DD_LIMIT,
    starting_equity: float = STARTING_EQUITY,
    plot: bool = True,
    verbose: bool = True,
    export_dir: str | None = None,
) -> dict:
    """
    Bootstrap Monte Carlo on the portfolio trade log.  Writes
    `monte_carlo_bootstrap_summary.csv` to `export_dir` when given.
    """
    _subheader("MONTE CARLO SIMULATION  (bootstrap, portfolio)")
    print(f"  Trades in input : {len(base_trade_df)}")
    print(f"  Simulations     : {n_simulations:,}")
    print(f"  DD threshold    : {drawdown_threshold:.1f}%")
    print(f"  Seed            : {seed}")

    # Disable the monte_carlo module's interactive plot when we are going to
    # save paper-ready charts ourselves (avoids duplicate/popup figures).
    mc_plot = plot and (export_dir is None)

    result = run_monte_carlo(
        trade_df           = base_trade_df,
        n_simulations      = n_simulations,
        starting_equity    = starting_equity,
        drawdown_threshold = drawdown_threshold,
        seed               = seed,
        plot               = mc_plot,
        verbose            = verbose,
        pnl_scale          = "percent",
    )

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        path = os.path.join(export_dir, "monte_carlo_bootstrap_summary.csv")
        summary_to_frame(result).to_csv(path, index=False)
        print(f"  Exported : {path}")

        if plot:
            paths = plot_monte_carlo(
                mc_result          = result,
                out_dir            = export_dir,
                drawdown_threshold = drawdown_threshold,
            )
            for p in paths:
                print(f"  Chart   : {p}")

    return result


# =============================================================================
# Final decision (used by full)
# =============================================================================

def print_final_decision(
    walk_result: dict,
    param_df: pd.DataFrame,
    mc_result: dict | None,
) -> None:
    risk_result = {
        "rows_df":        pd.DataFrame(),
        "best_risk":      MAX_RISK_PCT,
        "keep_current":   True,
        "recommendation": "KEEP current risk mix — not re-tested in this run",
    }
    print_phase1_decision(
        walk_result      = walk_result,
        param_result_df  = param_df,
        risk_result      = risk_result,
        pair             = PORTFOLIO_LABEL,
        current_risk_pct = MAX_RISK_PCT,
    )

    if not mc_result:
        return

    s   = mc_result["summary"]
    thr = s["drawdown_threshold_pct"]
    print("  Monte Carlo (bootstrap):")
    print(f"    median final equity : ${s['median_final_equity']:,.0f}"
          f"   (5th pct ${s['p5_final_equity']:,.0f}"
          f"   95th ${s['p95_final_equity']:,.0f})")
    print(f"    median return       : {s['median_total_return']*100:+.2f}%"
          f"   (5th pct {s['p5_total_return']*100:+.2f}%"
          f"   95th {s['p95_total_return']*100:+.2f}%)")
    print(f"    P(loss) / P(dd>{thr:.0f}%) : "
          f"{s['prob_loss']*100:.2f}%  /  "
          f"{s['prob_dd_exceeds_threshold']*100:.2f}%")
    print()


# =============================================================================
# Full pipeline
# =============================================================================

def run_full_validation(
    n_sim: int = MC_N_SIM_DEFAULT,
    seed: int = MC_SEED_DEFAULT,
    plot: bool = True,
    verbose: bool = True,
    export_dir: str | None = None,
) -> dict:
    _banner(f"PHASE 1 — FULL VALIDATION  |  {PORTFOLIO_LABEL}")
    print(f"  Portfolio runs : 1 baseline + {len(TP1_VARIANTS)*len(TP2_VARIANTS)-1} TP variants = "
          f"{len(TP1_VARIANTS)*len(TP2_VARIANTS)}  (MC reuses baseline)")

    base_df, base_eq, base_raw = run_baseline_portfolio(
        verbose=verbose, export_dir=_section_dir(export_dir, "baseline"),
    )
    walk = run_walkforward(
        base_df, starting_equity=STARTING_EQUITY,
        export_dir=_section_dir(export_dir, "walkforward"),
    )
    params = run_robustness(
        base_df, base_eq, starting_equity=STARTING_EQUITY,
        export_dir=_section_dir(export_dir, "robustness"),
    )
    mc = run_mc(
        base_df, n_simulations=n_sim, seed=seed,
        drawdown_threshold=DD_LIMIT, starting_equity=STARTING_EQUITY,
        plot=plot, verbose=verbose,
        export_dir=_section_dir(export_dir, "montecarlo"),
    )

    _subheader("FINAL VALIDATION DECISION")
    print_final_decision(walk, params, mc)

    if export_dir:
        summary_dir = _section_dir(export_dir, "summary")
        paths = plot_validation_summary(
            baseline_metrics = (base_raw or {}).get("metrics", {}),
            walk_result      = walk,
            param_df         = params,
            mc_result        = mc,
            dd_limit         = DD_LIMIT,
            out_dir          = summary_dir,
        )
        for p in paths:
            print(f"  Chart   : {p}")

    return {
        "trade_df":     base_df,
        "final_equity": base_eq,
        "walk_result":  walk,
        "param_df":     params,
        "mc_result":    mc,
    }


# =============================================================================
# Dispatch — ONE place that maps section -> function, no fall-through
# =============================================================================

SECTIONS = ("full", "baseline", "walkforward", "robustness", "montecarlo")


def dispatch(
    section: str,
    n_sim: int = MC_N_SIM_DEFAULT,
    seed: int = MC_SEED_DEFAULT,
    plot: bool = True,
    verbose: bool = True,
    export_dir: str | None = None,
) -> None:
    """
    Run exactly ONE section and return.  Sections that need a trade log run
    the baseline on demand and then stop.  There is no fall-through.
    """
    if section == "full":
        run_full_validation(
            n_sim=n_sim, seed=seed, plot=plot,
            verbose=verbose, export_dir=export_dir,
        )
        return

    if section == "baseline":
        run_baseline_portfolio(
            verbose=verbose,
            export_dir=_section_dir(export_dir, "baseline"),
        )
        return

    # walkforward / robustness / montecarlo all need the baseline trade log.
    if section in {"walkforward", "robustness", "montecarlo"}:
        base_df, base_eq, _ = run_baseline_portfolio(
            verbose=verbose, export_dir=None,  # don't export baseline CSVs for sub-runs
        )

        if section == "walkforward":
            run_walkforward(
                base_df, starting_equity=STARTING_EQUITY,
                export_dir=_section_dir(export_dir, "walkforward"),
            )
            return

        if section == "robustness":
            run_robustness(
                base_df, base_eq, starting_equity=STARTING_EQUITY,
                export_dir=_section_dir(export_dir, "robustness"),
            )
            return

        if section == "montecarlo":
            run_mc(
                base_df, n_simulations=n_sim, seed=seed,
                drawdown_threshold=DD_LIMIT, starting_equity=STARTING_EQUITY,
                plot=plot, verbose=verbose,
                export_dir=_section_dir(export_dir, "montecarlo"),
            )
            return

    raise ValueError(f"Unknown section: {section!r} (choose from {SECTIONS})")


# =============================================================================
# Interactive menu (only shown when no CLI args were provided)
# =============================================================================

_MENU = [
    ("1", "full",        "full"),
    ("2", "baseline",    "baseline"),
    ("3", "walkforward", "walkforward"),
    ("4", "robustness",  "robustness"),
    ("5", "montecarlo",  "monte carlo"),
]


def _interactive_pick() -> str:
    print("\n" + "=" * 72)
    print("  VALIDATE.PY  —  What would you like to run?")
    print("=" * 72)
    for key, _, desc in _MENU:
        print(f"   {key}. {desc}")
    print("   q. Quit")
    print("-" * 72)

    valid = {k: s for k, s, _ in _MENU}
    valid_by_name = {s: s for _, s, _ in _MENU}
    while True:
        try:
            choice = input("  Choice [1-5, or q]: ").strip().lower()
        except EOFError:
            # Non-interactive stdin (piped / CI) -- default to full.
            print("\n  (no stdin -> defaulting to full validation)")
            return "full"

        if choice in {"q", "quit", "exit"}:
            sys.exit(0)
        if choice in valid:
            return valid[choice]
        if choice in valid_by_name:
            return valid_by_name[choice]
        print(f"  invalid choice {choice!r} — enter 1-5 or q")


# =============================================================================
# CLI
# =============================================================================

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Portfolio validation — modular runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--section",
        choices=list(SECTIONS),
        default=None,
        help="Which validation section to run.  If omitted, an interactive "
             "menu is shown.",
    )
    p.add_argument(
        "--n-sim", type=int, default=MC_N_SIM_DEFAULT,
        help="Number of Monte Carlo simulations (bootstrap).",
    )
    p.add_argument(
        "--seed", type=int, default=MC_SEED_DEFAULT,
        help="Monte Carlo RNG seed (portfolio seed is fixed separately).",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose progress logs (summary blocks still print).",
    )
    p.add_argument(
        "--no-plots", action="store_true",
        help="Disable Monte Carlo diagnostic plots.",
    )
    p.add_argument(
        "--export-dir", default=DEFAULT_EXPORT_DIR,
        help="Directory for CSV exports (baseline, walk-forward, robustness, MC).",
    )
    p.add_argument(
        "--no-export", action="store_true",
        help="Disable all CSV exports (overrides --export-dir).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args = _build_argparser().parse_args(argv)

    # If the user provided *no* CLI args at all, drop into the menu.
    # (--section is the one way to be fully non-interactive.)
    section = args.section if args.section else _interactive_pick()

    verbose    = not args.quiet
    plot       = not args.no_plots
    export_dir = None if args.no_export else args.export_dir

    dispatch(
        section    = section,
        n_sim      = args.n_sim,
        seed       = args.seed,
        plot       = plot,
        verbose    = verbose,
        export_dir = export_dir,
    )


if __name__ == "__main__":
    main()
