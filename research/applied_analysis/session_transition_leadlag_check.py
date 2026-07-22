import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.sessions import load_session_returns, load_session_returns_with_first_hour

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT.parent / "data"

PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURGBP", "EURJPY", "EURCHF"]
START = "2011-01-01"
END = "2023-12-31"

TRANSITIONS = [
    ("asian_return", "london_return", 0, "A -> L (same day)"),
    ("asian_return", "ny_return", 0, "A -> U (same day)"),
    ("london_return", "ny_return", 0, "L -> U (same day)"),
    ("london_return", "asian_return", -1, "L -> ndA (next day)"),
    ("ny_return", "asian_return", -1, "U -> ndA (next day)"),
    ("ny_return", "london_return", -1, "U -> ndL (next day)"),
]

print("Session-transition lead-lag check -- full 10-pair universe, descriptive only")
print("Sessions are non-overlapping open-to-open blocks (Asian open -> London open -> New York open -> next Asian open), DST-aware via src/features/sessions.py -- an earlier version defined sessions by each market's own quoted close time and produced a spurious L -> U same-day r=0.60 from shared 1-minute bars during the real London/New York overlap")
print()

pair_data = {}
for pair in PAIRS:
    pair_data[pair] = load_session_returns(pair, START, END, DATA_DIR)

print("Lag-1 autocorrelation of each session return series, averaged across the 10-pair universe")
for col in ["asian_return", "london_return", "ny_return"]:
    autocorrs = [pair_data[pair][col].dropna().autocorr(lag=1) for pair in PAIRS]
    print(f"{col}: mean={np.mean(autocorrs):.4f}  min={np.min(autocorrs):.4f}  max={np.max(autocorrs):.4f}")
print()

for predictor_col, target_col, shift, label in TRANSITIONS:
    pooled_predictor_chunks = []
    pooled_target_chunks = []
    per_pair_results = []
    for pair in PAIRS:
        df = pair_data[pair]
        target = df[target_col].shift(shift) if shift != 0 else df[target_col]
        aligned = pd.concat([df[predictor_col].rename("predictor"), target.rename("target")], axis=1, sort=True).dropna()
        pair_r, _ = pearsonr(aligned["predictor"], aligned["target"])
        per_pair_results.append((pair, pair_r, len(aligned)))
        pooled_predictor_chunks.append(aligned["predictor"].to_numpy())
        pooled_target_chunks.append(aligned["target"].to_numpy())

    pooled_predictor = np.concatenate(pooled_predictor_chunks)
    pooled_target = np.concatenate(pooled_target_chunks)
    pooled_pearson_r, pooled_pearson_p = pearsonr(pooled_predictor, pooled_target)
    pooled_spearman_r, pooled_spearman_p = spearmanr(pooled_predictor, pooled_target)

    print(label)
    print(
        f"Pooled (n={len(pooled_predictor)}): Pearson r={pooled_pearson_r:.4f} p={pooled_pearson_p:.4f}  "
        f"Spearman rho={pooled_spearman_r:.4f} p={pooled_spearman_p:.4f}"
    )
    per_pair_str = "  ".join(f"{pair}={r:.3f}" for pair, r, n in per_pair_results)
    print(f"Per-pair Pearson r: {per_pair_str}")
    same_sign_count = sum(1 for _, r, _ in per_pair_results if np.sign(r) == np.sign(pooled_pearson_r))
    print(f"Pairs agreeing in sign with pooled result: {same_sign_count}/{len(per_pair_results)}")
    print()

print("Cross-pair-average correction: the pooled-concatenation numbers above pseudo-replicate each calendar day 10 times (once per pair), and the pairs are cross-sectionally correlated, so the effective independent sample size is far below the raw pooled n. Averaging session returns across all 10 pairs first gives one observation per trading day instead.")
print()

for predictor_col, target_col, shift, label in TRANSITIONS:
    predictor_avg = pd.concat([pair_data[pair][predictor_col].rename(pair) for pair in PAIRS], axis=1, sort=True).mean(axis=1, skipna=True)
    target_avg = pd.concat([pair_data[pair][target_col].rename(pair) for pair in PAIRS], axis=1, sort=True).mean(axis=1, skipna=True)
    target_avg = target_avg.shift(shift) if shift != 0 else target_avg
    aligned_avg = pd.concat([predictor_avg.rename("predictor"), target_avg.rename("target")], axis=1, sort=True).dropna()
    avg_pearson_r, avg_pearson_p = pearsonr(aligned_avg["predictor"], aligned_avg["target"])
    avg_spearman_r, avg_spearman_p = spearmanr(aligned_avg["predictor"], aligned_avg["target"])
    print(
        f"{label}  cross-pair-average (n={len(aligned_avg)}): Pearson r={avg_pearson_r:.4f} p={avg_pearson_p:.4f}  "
        f"Spearman rho={avg_spearman_r:.4f} p={avg_spearman_p:.4f}"
    )
print()

print("First-hour predictor/target variants: full-session returns bundle in a lot of intra-session noise on both sides of a lead-lag test, so re-run all six transitions using the first 60 minutes of a session in place of the full session, on either or both legs")
print()

first_hour_pair_data = {}
for pair in PAIRS:
    first_hour_pair_data[pair] = load_session_returns_with_first_hour(pair, START, END, DATA_DIR)

FIRST_HOUR_SESSION_TRANSITIONS = [
    ("asian", "london", 0, "A -> L"),
    ("asian", "ny", 0, "A -> U"),
    ("london", "ny", 0, "L -> U"),
    ("london", "asian", -1, "L -> ndA"),
    ("ny", "asian", -1, "U -> ndA"),
    ("ny", "london", -1, "U -> ndL"),
]

FIRST_HOUR_VARIANTS = [
    ("first_hour", "full", "1st-hour predictor -> full predicted"),
    ("first_hour", "first_hour", "1st-hour predictor -> 1st-hour predicted"),
    ("full", "first_hour", "full predictor -> 1st-hour predicted"),
    ("full", "full", "full predictor -> full predicted (baseline)"),
]

first_hour_results = []
for predictor_variant, target_variant, variant_label in FIRST_HOUR_VARIANTS:
    for predictor_session, target_session, shift, transition_label in FIRST_HOUR_SESSION_TRANSITIONS:
        predictor_col = f"{predictor_session}_first_hour_return" if predictor_variant == "first_hour" else f"{predictor_session}_return"
        target_col = f"{target_session}_first_hour_return" if target_variant == "first_hour" else f"{target_session}_return"

        predictor_avg = pd.concat(
            [first_hour_pair_data[pair][predictor_col].rename(pair) for pair in PAIRS], axis=1, sort=True
        ).mean(axis=1, skipna=True)
        target_avg = pd.concat(
            [first_hour_pair_data[pair][target_col].rename(pair) for pair in PAIRS], axis=1, sort=True
        ).mean(axis=1, skipna=True)
        target_avg = target_avg.shift(shift) if shift != 0 else target_avg

        aligned = pd.concat([predictor_avg.rename("predictor"), target_avg.rename("target")], axis=1, sort=True).dropna()
        r, pval = pearsonr(aligned["predictor"], aligned["target"])
        rho, rhop = spearmanr(aligned["predictor"], aligned["target"])
        first_hour_results.append((variant_label, transition_label, r, pval, rho, rhop, len(aligned)))
        print(
            f"{variant_label} / {transition_label}  (n={len(aligned)}): Pearson r={r:.4f} p={pval:.4f}  "
            f"Spearman rho={rho:.4f} p={rhop:.4f}"
        )

print()
print("Specifications with both Pearson and Spearman significant at alpha=0.05 and matching sign:")
survivor_found = False
for variant_label, transition_label, r, pval, rho, rhop, n in first_hour_results:
    if pval < 0.05 and rhop < 0.05 and np.sign(r) == np.sign(rho):
        survivor_found = True
        print(f"  {variant_label} / {transition_label}: Pearson r={r:.4f} p={pval:.4f}  Spearman rho={rho:.4f} p={rhop:.4f}  n={n}")
if not survivor_found:
    print("  none")
