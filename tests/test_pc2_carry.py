import numpy as np
import pandas as pd

from src.signals.pc2_carry import (
    align_signal_and_forward,
    fit_pc2_loadings,
    pc2_factor_returns,
    pc2_scores,
)

PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]


def _returns(n=400, seed=3):
    rng = np.random.default_rng(seed)
    dollar = rng.normal(0, 0.005, n)
    yen = rng.normal(0, 0.004, n)
    index = pd.date_range("2015-01-01", periods=n, freq="1D")
    return pd.DataFrame({
        "EURUSD": dollar + rng.normal(0, 0.001, n),
        "GBPUSD": dollar + 0.4 * yen + rng.normal(0, 0.001, n),
        "USDJPY": dollar + 1.6 * yen + rng.normal(0, 0.001, n),
    }, index=index)[PAIRS]


def test_sign_convention_forces_reference_pair_positive():
    train = _returns()
    _loadings, by_pair, _mean = fit_pc2_loadings(train, PAIRS)
    assert by_pair["USDJPY"] > 0


def test_sign_convention_holds_when_the_data_is_negated():
    train = _returns()
    _l1, by_pair_a, _m1 = fit_pc2_loadings(train, PAIRS)
    _l2, by_pair_b, _m2 = fit_pc2_loadings(-train, PAIRS)
    assert by_pair_a["USDJPY"] > 0 and by_pair_b["USDJPY"] > 0


def test_loadings_dict_matches_array_order():
    train = _returns()
    loadings, by_pair, _mean = fit_pc2_loadings(train, PAIRS)
    for i, pair in enumerate(PAIRS):
        assert by_pair[pair] == loadings[i]


def test_scores_center_on_train_mean_not_test_mean():
    data = _returns()
    train, test = data.iloc[:300], data.iloc[300:]
    loadings, _by_pair, train_mean = fit_pc2_loadings(train, PAIRS)

    scores = pc2_scores(test, loadings, train_mean)
    expected = (test.to_numpy() - train_mean) @ loadings
    assert np.allclose(scores.to_numpy(), expected)

    leaky = (test.to_numpy() - test.to_numpy().mean(axis=0)) @ loadings
    assert not np.allclose(scores.to_numpy(), leaky)


def test_scores_are_indexed_on_the_test_period():
    data = _returns()
    train, test = data.iloc[:300], data.iloc[300:]
    loadings, _by_pair, train_mean = fit_pc2_loadings(train, PAIRS)
    scores = pc2_scores(test, loadings, train_mean)
    pd.testing.assert_index_equal(scores.index, test.index)


def test_factor_returns_are_the_loading_weighted_sum():
    data = _returns()
    by_pair = {"EURUSD": 0.1, "GBPUSD": 0.5, "USDJPY": 0.9}
    got = pc2_factor_returns(data, by_pair)
    expected = 0.1 * data["EURUSD"] + 0.5 * data["GBPUSD"] + 0.9 * data["USDJPY"]
    assert np.allclose(got.to_numpy(), expected.to_numpy())


def test_alignment_pairs_score_t_with_return_t_plus_one():
    index = pd.date_range("2020-01-01", periods=5, freq="1D")
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)
    factor = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=index)

    signal, forward = align_signal_and_forward(scores, factor)

    assert len(signal) == 4
    assert np.allclose(signal.to_numpy(), [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(forward.to_numpy(), [20.0, 30.0, 40.0, 50.0])


def test_alignment_never_uses_a_contemporaneous_return():
    index = pd.date_range("2020-01-01", periods=6, freq="1D")
    scores = pd.Series(np.arange(6, dtype=float), index=index)
    factor = pd.Series(np.arange(6, dtype=float) * 100, index=index)
    signal, forward = align_signal_and_forward(scores, factor)
    assert not np.allclose(signal.to_numpy() * 100, forward.to_numpy())
