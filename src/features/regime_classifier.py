"""
Composite volatility/trend regime scoring and discrete classification.

The composite score is a construct, not an observable. Any threshold that
turns it into a discrete regime label is a researcher degree of freedom,
and the threshold must be chosen on training data only -- see
`src.signals.regime_refit` for the walk-forward version that respects
that constraint.
"""
import pandas as pd

from src.features.pca import pca

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def compute_composite_regime_score(vol: pd.Series, rate_diff: pd.Series) -> pd.Series:
    """
    2-feature PCA composite regime score: z-scored 78-day realized vol
    combined with the z-scored, publication-lag-adjusted rate differential
    via the 1st principal component, sign-normalized so the vol loading is
    positive.

    Matches `research/strategies/volatility_regime_breakout_mean_revert.md`
    Section 4, item 2. Reuses `src.features.pca.pca` (first-principles
    eigendecomposition, already implemented and tested elsewhere in this
    repo) rather than re-deriving PCA here.

    Note (per the spec's Section 1 hypothesis and Section 4 preamble): this
    must be refit inside each walk-forward window in production -- fitting
    once on the full sample (as this function does when called on a full
    series) is only valid for the Day 43 descriptive threshold-selection
    analysis, not for an actual out-of-sample test.

    Parameters
    ----------
    vol : pd.Series
        Rolling realized volatility (e.g. 78-day std of log returns).
    rate_diff : pd.Series
        Rate differential, already publication-lag-shifted and
        forward-filled to the same daily index as `vol`.

    Returns
    -------
    pd.Series
        Composite z-score, index = intersection of `vol.dropna()` and
        `rate_diff.dropna()` indices (inner join, then z-scored jointly).

    Raises
    ------
    ValueError
        If fewer than 2 overlapping observations remain after dropping NaNs
        (PCA / z-scoring is undefined).
    """
    combined = pd.concat([vol.rename("vol"), rate_diff.rename("rate_diff")], axis=1).dropna()
    if len(combined) < 2:
        raise ValueError(
            f"Need at least 2 overlapping non-NaN (vol, rate_diff) observations, got {len(combined)}."
        )

    z = (combined - combined.mean()) / combined.std()

    components, _explained_variance, _projected = pca(z.to_numpy(), n_components=1)
    pc1 = components[:, 0]
    if pc1[0] < 0:
        pc1 = -pc1

    composite = z.to_numpy() @ pc1
    composite_z = (composite - composite.mean()) / composite.std(ddof=1)

    return pd.Series(composite_z, index=z.index, name="composite_z")


def classify_regime(
    composite_z: pd.Series,
    turbulent_threshold: float = 1.5,
    calm_threshold: float = 1.0,
) -> pd.Series:
    """
    Hard-switch regime classification from the composite z-score.

    Matches Section 4, item 3 (pre-registered Day 43, part of the
    strategy's falsification criteria in Section 1). This is deliberately
    a hard threshold, not hysteresis -- see the Day 46 audit discussion for
    why hysteresis was rejected as an undocumented, post-hoc deviation from
    the pre-registered spec.

    Parameters
    ----------
    composite_z : pd.Series
        Output of `compute_composite_regime_score`.
    turbulent_threshold : float, default 1.5
        `|composite_z| > turbulent_threshold` -> "turbulent".
    calm_threshold : float, default 1.0
        `|composite_z| < calm_threshold` -> "calm". Must be strictly less
        than `turbulent_threshold` or every observation would be classified,
        leaving no deadzone.

    Returns
    -------
    pd.Series
        Object-dtype series of {"turbulent", "calm", "deadzone"}, same index
        as `composite_z`. NaN input rows are labeled "deadzone" conservatively
        (no trade) rather than propagating NaN, since "unknown regime" and
        "confirmed deadzone" should have the same downstream consequence:
        no leg is confidently active.

    Raises
    ------
    ValueError
        If `calm_threshold >= turbulent_threshold`.
    """
    if calm_threshold >= turbulent_threshold:
        raise ValueError(
            f"calm_threshold ({calm_threshold}) must be < turbulent_threshold "
            f"({turbulent_threshold}), else there is no deadzone band."
        )

    abs_z = composite_z.abs()
    regime = pd.Series("deadzone", index=composite_z.index, dtype=object)
    regime[abs_z > turbulent_threshold] = "turbulent"
    regime[abs_z < calm_threshold] = "calm"
    return regime
