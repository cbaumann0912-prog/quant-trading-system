from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.pca import pca

DEFAULT_N_COMPONENTS = 3
SIGN_REFERENCE_PAIR = "USDJPY"


def fit_pc2_loadings(
    train_returns: pd.DataFrame,
    pairs_order: list[str],
    n_components: int = DEFAULT_N_COMPONENTS,
    sign_reference: str = SIGN_REFERENCE_PAIR,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Fit PC2 loadings on training returns, with a fixed sign convention.

    Eigenvectors are determined only up to sign, so loadings estimated on two
    different windows can be identical in structure yet opposite in sign. That
    would invert the signal with no change in the underlying data. Loadings are
    therefore normalised so the ``sign_reference`` entry is positive.

    Parameters
    ----------
    train_returns : pd.DataFrame
        Daily log returns, columns ordered as ``pairs_order``, training period
        only. Passing test-period rows here is a leakage channel that operates
        directly on the quantity under test.
    pairs_order : list[str]
        Column order. Fixes which loading belongs to which pair.
    n_components : int
        Components to retain. PC2 is taken as index 1.
    sign_reference : str
        Pair whose loading is forced positive.

    Returns
    -------
    loadings : np.ndarray
        PC2 loadings, one per pair, in ``pairs_order``.
    loadings_by_pair : dict[str, float]
        Same values keyed by pair.
    train_mean : np.ndarray
        Column means of the training returns, for centring test data.
    """
    components, _explained_variance, _projected = pca(
        train_returns.to_numpy(), n_components=n_components
    )
    loadings = components[:, 1]
    loadings_by_pair = dict(zip(pairs_order, loadings))
    train_mean = train_returns.to_numpy().mean(axis=0)

    if loadings_by_pair[sign_reference] < 0:
        loadings = -loadings
        loadings_by_pair = dict(zip(pairs_order, loadings))

    return loadings, loadings_by_pair, train_mean


def pc2_scores(
    test_returns: pd.DataFrame,
    loadings: np.ndarray,
    train_mean: np.ndarray,
) -> pd.Series:
    """Project test returns onto the PC2 axis, centred on the training mean.

    Centring with the test mean would leak test-period information into the
    out-of-sample scores. Small in magnitude, but it acts directly on the
    quantity being tested, so the training mean is used.
    """
    centered = test_returns.to_numpy() - train_mean
    return pd.Series(centered @ loadings, index=test_returns.index)


def pc2_factor_returns(
    returns: pd.DataFrame,
    loadings_by_pair: dict[str, float],
) -> pd.Series:
    """Factor-mimicking portfolio return for PC2.

    Training loadings applied as weights to contemporaneous pair returns. This
    is a proxy for the return the factor represents, not an investable
    portfolio — no financing, sizing or capacity assumption attaches to it.
    """
    out = None
    for pair, weight in loadings_by_pair.items():
        leg = weight * returns[pair]
        out = leg if out is None else out + leg
    return out


def align_signal_and_forward(
    scores: pd.Series,
    factor_returns: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Pair each score with the next period's factor return.

    The score at ``t`` is matched to the return realised over ``t+1``, so the
    signal is strictly causal with respect to its outcome.
    """
    signal = scores.iloc[:-1]
    forward = factor_returns.iloc[1:]
    forward.index = signal.index

    aligned = pd.concat(
        [signal.rename("signal"), forward.rename("forward_returns")], axis=1
    ).dropna()
    return aligned["signal"], aligned["forward_returns"]
