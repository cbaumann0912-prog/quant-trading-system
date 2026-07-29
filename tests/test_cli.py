import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from research.run_research import (
    LOCKBOX_START,
    json_safe,
    main,
    spearman_ic,
    summarize,
    truncate_forward_returns,
    window_sharpe,
)

PAIR = "EURUSD"
BASE_ARGS = [
    "--signal", "momentum",
    "--pair", PAIR,
    "--train-start", "2015-01-01",
    "--train-end", "2020-12-31",
    "--windows", "2",
    "--train-years", "3",
    "--test-months", "6",
]


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    """Minute-bar CSV in the loader's expected on-disk format, one pair."""
    index = pd.date_range("2015-01-01", "2021-06-30", freq="4h")
    rng = np.random.default_rng(58)
    price = 1.20 * np.exp(np.cumsum(rng.normal(0.0, 0.002, len(index))))

    frame = pd.DataFrame(
        {
            "Datetime": index.strftime("%Y%m%d %H%M%S"),
            "Open": price,
            "High": price,
            "Low": price,
            "Close": price,
            "Volume": 0,
        }
    )
    frame.to_csv(tmp_path / f"{PAIR}.csv", index=False)
    return tmp_path


def test_cli_runs_without_error(synthetic_data_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        BASE_ARGS
        + ["--data-dir", str(synthetic_data_dir), "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output


def test_output_file_created(synthetic_data_dir, tmp_path):
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        main,
        BASE_ARGS + ["--data-dir", str(synthetic_data_dir), "--output", str(out_dir)],
    )

    assert result.exit_code == 0, result.output

    out_path = out_dir / f"{PAIR}_momentum.json"
    assert out_path.exists()

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pair"] == PAIR
    assert report["signal"] == "momentum"
    assert len(report["window_results"]) == 2
    for key in ("parameters", "sample", "ic_summary", "sharpe_summary", "caveats"):
        assert key in report


def test_lockbox_end_date_is_rejected(synthetic_data_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--signal", "momentum",
            "--pair", PAIR,
            "--train-start", "2015-01-01",
            "--train-end", str(LOCKBOX_START.date()),
            "--data-dir", str(synthetic_data_dir),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "lockbox" in result.output.lower()


def test_unsupported_pair_is_rejected(synthetic_data_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--signal", "momentum",
            "--pair", "XAUUSD",
            "--data-dir", str(synthetic_data_dir),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0


def test_impossible_window_geometry_fails_loudly(synthetic_data_dir, tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--signal", "momentum",
            "--pair", PAIR,
            "--train-start", "2015-01-01",
            "--train-end", "2020-12-31",
            "--windows", "99",
            "--data-dir", str(synthetic_data_dir),
            "--output", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "windows" in result.output.lower()


def test_truncate_forward_returns_masks_the_tail():
    index = pd.date_range("2020-01-01", periods=10, freq="D")
    forward = pd.Series(np.arange(10, dtype=float), index=index)

    truncated = truncate_forward_returns(forward, index, holding_period=3)

    assert truncated.iloc[:7].notna().all()
    assert truncated.iloc[7:].isna().all()


def test_spearman_ic_flags_constant_signal():
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    constant = pd.Series(1.0, index=index)
    forward = pd.Series(np.random.default_rng(0).normal(size=20), index=index)

    ic, status = spearman_ic(constant, forward)

    assert np.isnan(ic)
    assert status == "constant_signal"


def test_spearman_ic_recovers_a_known_monotone_relationship():
    index = pd.date_range("2020-01-01", periods=30, freq="D")
    signal = pd.Series(np.arange(30, dtype=float), index=index)
    forward = pd.Series(np.arange(30, dtype=float) ** 2, index=index)

    ic, status = spearman_ic(signal, forward)

    assert status == "ok"
    assert ic == pytest.approx(1.0)


def test_window_sharpe_uses_lagged_exposure():
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    returns = pd.Series(np.random.default_rng(1).normal(size=50), index=index)
    lookahead_exposure = np.sign(returns)

    sharpe = window_sharpe(lookahead_exposure, returns)

    assert sharpe < 20.0


def test_summarize_ignores_nan():
    result = summarize([1.0, float("nan"), 3.0])

    assert result["n"] == 2
    assert result["mean"] == pytest.approx(2.0)
    assert result["frac_positive"] == pytest.approx(1.0)


def test_json_safe_emits_strict_json():
    payload = {
        "ts": pd.Timestamp("2020-01-01"),
        "nan": float("nan"),
        "inf": float("inf"),
        "np_float": np.float64(1.5),
        "np_int": np.int64(3),
        "nested": [np.bool_(True), float("nan")],
    }

    encoded = json.dumps(json_safe(payload), allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded["ts"] == "2020-01-01T00:00:00"
    assert decoded["nan"] is None
    assert decoded["inf"] is None
    assert decoded["np_float"] == 1.5
    assert decoded["np_int"] == 3
    assert decoded["nested"] == [True, None]
