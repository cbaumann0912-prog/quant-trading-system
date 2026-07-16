import numpy as np
import pandas as pd
import pytest

from src.features.regime_classifier import classify_regime, compute_composite_regime_score

def _series(values, start="2024-01-01"):
    index = pd.date_range(start=start, periods=len(values), freq="1D")
    return pd.Series(values, index=index)


class TestComputeCompositeRegimeScore:
    def test_output_is_zscored(self):
        rng = np.random.default_rng(0)
        vol = _series(rng.normal(0.01, 0.002, 200))
        rate_diff = _series(rng.normal(0, 1, 200))
        composite_z = compute_composite_regime_score(vol, rate_diff)

        assert np.isclose(composite_z.mean(), 0.0, atol=1e-8)
        assert np.isclose(composite_z.std(), 1.0, atol=1e-8)

    def test_vol_loading_sign_normalized(self):
        rng = np.random.default_rng(3)
        n = 300
        vol = _series(np.linspace(0.005, 0.05, n) + rng.normal(0, 0.0005, n))
        rate_diff = _series(rng.normal(0, 1, n))
        composite_z = compute_composite_regime_score(vol, rate_diff)
        aligned_vol = vol.reindex(composite_z.index)

        assert composite_z.corr(aligned_vol) > 0

    def test_inner_join_on_overlap(self):
        vol = _series([0.01, 0.02, np.nan, 0.03, 0.015])
        rate_diff = _series([1.0, np.nan, 2.0, 3.0, 1.5])
        composite_z = compute_composite_regime_score(vol, rate_diff)

        assert len(composite_z) == 3 

    def test_too_few_observations_raises(self):
        vol = _series([0.01])
        rate_diff = _series([1.0])

        with pytest.raises(ValueError, match="overlapping"):
            compute_composite_regime_score(vol, rate_diff)


class TestClassifyRegime:
    def test_thresholds_partition_correctly(self):
        composite_z = _series([2.0, -2.0, 0.5, -0.5, 1.2, -1.2])
        regime = classify_regime(composite_z, turbulent_threshold=1.5, calm_threshold=1.0)

        assert list(regime) == [
            "turbulent",
            "turbulent",
            "calm",
            "calm",
            "deadzone",
            "deadzone",
        ]

    def test_nan_input_labeled_deadzone(self):
        composite_z = _series([np.nan, 2.0])
        regime = classify_regime(composite_z)

        assert regime.iloc[0] == "deadzone"

    def test_invalid_thresholds_raise(self):
        composite_z = _series([1.0, 2.0])

        with pytest.raises(ValueError, match="calm_threshold"):
            classify_regime(composite_z, turbulent_threshold=1.0, calm_threshold=1.0)
