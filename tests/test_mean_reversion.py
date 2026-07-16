import numpy as np
import pandas as pd

from src.signals.mean_reversion import mean_reversion_ladder_signal, price_zscore_signal


def _make_price(values, start="2024-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.DataFrame({"price": values}, index=index)


class TestPriceZscoreSignal:
    def test_warmup_is_nan(self):
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 50))
        data = _make_price(prices)
        lookback = 10
        z = price_zscore_signal(data, lookback)

        assert z.iloc[: lookback - 1].isna().all()
        assert z.iloc[lookback:].notna().any()

    def test_zero_std_masked_not_inf(self):
        data = _make_price([100.0] * 20)
        z = price_zscore_signal(data, lookback=5)

        assert not np.isinf(z).any()
        assert z.iloc[5:].isna().all()

    def test_matches_manual_zscore(self):
        rng = np.random.default_rng(1)
        prices = 100 + np.cumsum(rng.normal(0, 1, 60))
        data = _make_price(prices)
        lookback = 10
        z = price_zscore_signal(data, lookback)
        price = data["price"]
        expected = (price - price.rolling(lookback).mean()) / price.rolling(lookback).std()

        pd.testing.assert_series_equal(z, expected.mask(price.rolling(lookback).std() == 0), check_names=False)


class TestMeanReversionLadderSignal:
    def test_flat_until_entry_threshold_breached(self):
        rng = np.random.default_rng(1)
        base = list(100.0 + rng.normal(0, 0.01, 20))
        prices = base + [50.0] * 5
        data = _make_price(prices)
        exposure = mean_reversion_ladder_signal(data, lookback=10)

        assert exposure.iloc[19] == 0.0
        assert exposure.iloc[20] == 1.0 / 3

    def test_rungs_add_on_deeper_breach_and_cap_at_three(self):
        prices = [100.0] * 30 + [70.0, 40.0, 10.0, 2.0, 1.0, 0.5]
        data = _make_price(prices)

        exposure = mean_reversion_ladder_signal(data, lookback=20)

        assert exposure.iloc[30] == 1 / 3
        assert exposure.iloc[31] == 2 / 3
        assert exposure.iloc[32] == 1.0
        assert exposure.iloc[33] == 1.0

    def test_exit_on_reversion_to_band(self):
        rng = np.random.default_rng(5)
        recovery = list(99.0 + rng.normal(0, 0.05, 10))
        prices = [100.0] * 20 + [70.0] * 5 + recovery
        data = _make_price(prices)
        exposure = mean_reversion_ladder_signal(data, lookback=10)

        assert exposure.iloc[24] != 0.0
        assert exposure.iloc[-1] == 0.0

    def test_exit_on_time_stop(self):
        prices = [100.0] * 20 + [100 - 2 * i for i in range(1, 31)]
        data = _make_price(prices)
        exposure = mean_reversion_ladder_signal(data, lookback=10, time_stop=10)

        assert exposure.iloc[20] != 0.0
        assert exposure.iloc[29] != 0.0
        assert exposure.iloc[30] == 0.0

    def test_nan_while_flat_during_warmup(self):
        data = _make_price([100.0, 101.0, 99.0])
        exposure = mean_reversion_ladder_signal(data, lookback=10)

        assert exposure.isna().all()

    def test_holds_previous_exposure_through_missing_price(self):
        prices = [100.0] * 20 + [60.0, 60.0, np.nan, 60.0, 60.0]
        data = _make_price(prices)
        exposure = mean_reversion_ladder_signal(data, lookback=10)

        assert exposure.iloc[20] == 1 / 3
        assert exposure.iloc[22] == 1 / 3
        assert exposure.iloc[24] == 1 / 3
