# =============================================================================
# sim_costs.py — Execution cost model: spread + slippage simulation
# =============================================================================
# Provides realistic fill prices for all trade events.
#
# PUBLIC API
# ----------
# get_default_costs(pair)                              -> (spread, slippage_std)
# apply_entry_fill(price, direction, spread, slip_std) -> filled price
# apply_exit_fill(price, direction, spread, slip_std)  -> filled price
# apply_sl_fill(sl_price, direction, slip_std)         -> filled price
# apply_tp_fill(tp_price, direction, slip_std)         -> filled price
#
# SCENARIO PRESETS
# ----------------
# SCENARIO_IDEAL      — spread=0, slippage=0
# SCENARIO_REALISTIC  — normal defaults (1x)
# SCENARIO_WORST_CASE — 2x spread + 2x slippage
# ALL_SCENARIOS       — list of all three, in order
# =============================================================================

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic RNG — single seeded generator for all slippage draws.
# All fill functions below draw from this generator in the order they are
# called, so any run with the same seed AND same call sequence produces
# byte-identical fill prices.  Call reset_rng() before each scenario run
# to get independently reproducible results even when scenarios are chained.
# ---------------------------------------------------------------------------
_RNG: np.random.Generator = np.random.default_rng(42)


def reset_rng(seed: int = 42) -> None:
    """Reset the module-level RNG to a fixed seed.

    Call this once before every top-level backtest run so that each scenario
    (realistic, worst-case, etc.) produces the same results in isolation
    regardless of what was run before it.

    Parameters
    ----------
    seed : int, optional
        Seed value.  Default 42.
    """
    global _RNG
    _RNG = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Per-pair defaults (realistic retail broker costs in price units)
# ---------------------------------------------------------------------------
_SPREAD: dict[str, float] = {
    "EURUSD": 0.00015,
    "GBPUSD": 0.00022,
    "USDJPY": 0.015,
    "AUDUSD": 0.00018,
    "USDCAD": 0.00020,
    "USDCHF": 0.00020,
    "NZDUSD": 0.00022,
}
_DEFAULT_SPREAD = 0.00020

_SLIPPAGE_STD: dict[str, float] = {
    "EURUSD": 0.00008,
    "GBPUSD": 0.00010,
    "USDJPY": 0.008,
    "AUDUSD": 0.00009,
    "USDCAD": 0.00010,
    "USDCHF": 0.00010,
    "NZDUSD": 0.00010,
}
_DEFAULT_SLIPPAGE_STD = 0.00010


def get_default_costs(pair: str) -> tuple[float, float]:
    """Return (spread, slippage_std) defaults for the given currency pair."""
    p = pair.upper()
    return (_SPREAD.get(p, _DEFAULT_SPREAD),
            _SLIPPAGE_STD.get(p, _DEFAULT_SLIPPAGE_STD))


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------
def _abs_slip(slippage_std: float) -> float:
    """Draw a non-negative slippage sample from half-normal distribution.

    Uses the module-level seeded RNG so results are fully deterministic.
    """
    return abs(float(_RNG.normal(0.0, slippage_std))) if slippage_std > 0 else 0.0


# ---------------------------------------------------------------------------
# Fill functions — always return a price slightly WORSE than theoretical
# ---------------------------------------------------------------------------
def apply_entry_fill(price: float, direction: str,
                     spread: float, slippage_std: float) -> float:
    """
    Realistic market-order entry fill.

    Long  (bull): buyer pays ask  → price + spread + |slip|
    Short (bear): seller gets bid → price − spread − |slip|
    """
    s = _abs_slip(slippage_std)
    return (price + spread + s) if direction == "bull" else (price - spread - s)


def apply_exit_fill(price: float, direction: str,
                    spread: float, slippage_std: float) -> float:
    """
    Realistic market-order exit fill.

    Long  exit (sell): receives bid → price − spread − |slip|
    Short exit (buy):  pays ask    → price + spread + |slip|
    """
    s = _abs_slip(slippage_std)
    return (price - spread - s) if direction == "bull" else (price + spread + s)


def apply_sl_fill(sl_price: float, direction: str,
                  slippage_std: float) -> float:
    """
    Realistic stop-loss fill — stop orders gap through the level.

    Long  stop (stop-sell below): sl_price − |slip|
    Short stop (stop-buy above):  sl_price + |slip|
    """
    s = _abs_slip(slippage_std)
    return (sl_price - s) if direction == "bull" else (sl_price + s)


def apply_tp_fill(tp_price: float, direction: str,
                  slippage_std: float) -> float:
    """
    Realistic take-profit fill — limit orders may fill slightly worse.

    Long  TP (limit-sell): tp_price − |slip|
    Short TP (limit-buy):  tp_price + |slip|
    """
    s = _abs_slip(slippage_std)
    return (tp_price - s) if direction == "bull" else (tp_price + s)


# ---------------------------------------------------------------------------
# Scenario presets (Phase 4 stress test)
# ---------------------------------------------------------------------------
SCENARIO_IDEAL: dict = {
    "label":       "ideal",
    "spread_mult": 0.0,
    "slip_mult":   0.0,
    "description": "No spread, no slippage",
}
SCENARIO_REALISTIC: dict = {
    "label":       "realistic",
    "spread_mult": 1.0,
    "slip_mult":   1.0,
    "description": "Normal retail broker costs",
}
SCENARIO_WORST_CASE: dict = {
    "label":       "worst_case",
    "spread_mult": 2.0,
    "slip_mult":   2.0,
    "description": "2x spread + 2x slippage",
}
ALL_SCENARIOS: list[dict] = [SCENARIO_IDEAL, SCENARIO_REALISTIC, SCENARIO_WORST_CASE]
