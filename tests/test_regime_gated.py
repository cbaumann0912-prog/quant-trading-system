import numpy as np
import pandas as pd

from src.signals.regime_gated import make_regime_gated_signal_fn


def _make_price(values, start="2024-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.DataFrame({"price": values}, index=index)


class TestMakeRegimeGatedSignalFn:
    def test_turbulent_bars_use_momentum(self):
        n = 30
        prices = 100 + np.arange(n)
        data = _make_price(prices)
        regime = pd.Series("turbulent", index=data.index)
        signal_fn = make_regime_gated_signal_fn(regime, reversion_lookback=10)
        signal = signal_fn(data, lookback=10)

        assert (signal.iloc[10:] == 1.0).all()

    def test_calm_bars_build_the_full_ladder(self):
        prices = [100.0] * 30 + [70.0, 40.0, 10.0, 2.0, 1.0, 0.5]
        data = _make_price(prices)
        regime = pd.Series("calm", index=data.index)
        signal_fn = make_regime_gated_signal_fn(regime, reversion_lookback=20)
        signal = signal_fn(data, lookback=10)

        assert signal.iloc[30] == 1 / 3
        assert signal.iloc[31] == 2 / 3
        assert signal.iloc[32] == 1.0

    def test_deadzone_lets_ladder_continue_its_own_lifecycle(self):
        prices = [100.0] * 20 + [100 - 2 * i for i in range(1, 31)]
        data = _make_price(prices)
        regime = pd.Series(
            ["calm"] * 25 + ["deadzone"] * 25, index=data.index
        )
        signal_fn = make_regime_gated_signal_fn(
            regime, reversion_lookback=10, time_stop=10
        )
        signal = signal_fn(data, lookback=10)

        assert signal.iloc[20] != 0.0 
        assert signal.iloc[29] != 0.0
        assert signal.iloc[30] == 0.0

    def test_flip_to_opposite_regime_force_closes_ladder(self):
        prices = [100.0] * 20 + [100 - 2 * i for i in range(1, 31)]
        data = _make_price(prices)
        regime = pd.Series(
            ["calm"] * 25 + ["turbulent"] * 25, index=data.index
        )
        signal_fn = make_regime_gated_signal_fn(
            regime, reversion_lookback=10, time_stop=10
        )
        signal = signal_fn(data, lookback=10)

        assert signal.iloc[24] != 0.0
        assert signal.iloc[25] == -1.0

    def test_leading_nan_before_first_valid_regime(self):
        n = 10
        prices = 100 + np.arange(n)
        data = _make_price(prices)
        regime = pd.Series("deadzone", index=data.index)
        signal_fn = make_regime_gated_signal_fn(regime, reversion_lookback=3)
        signal = signal_fn(data, lookback=3)

        assert signal.isna().all()
