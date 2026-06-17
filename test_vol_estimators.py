"""
Unit Tests — Historical Volatility & GARCH
============================================
Tests: estimator output shape, positivity, GARCH params, forecast.

Run with: python -m pytest tests/test_vol_estimators.py -v
"""

import pytest
import numpy as np
import pandas as pd
from ml.vol_estimators import (
    hist_vol_close, ewma_vol, parkinson_vol, yang_zhang_vol,
    label_vol_regimes, fit_garch, garch_forecast,
    vol_summary, synthetic_ohlc,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlc():
    return synthetic_ohlc(n=400, seed=42)

@pytest.fixture
def close(ohlc):
    return ohlc["Close"]


# ---------------------------------------------------------------------------
# 1. Close-to-close HV
# ---------------------------------------------------------------------------

class TestHistVol:

    def test_output_is_series(self, close):
        assert isinstance(hist_vol_close(close), pd.Series)

    def test_positive_values(self, close):
        v = hist_vol_close(close).dropna()
        assert (v > 0).all()

    def test_annualised_reasonable(self, close):
        v = hist_vol_close(close).dropna()
        assert v.mean() > 0.05
        assert v.mean() < 1.0

    def test_longer_window_smoother(self, close):
        v21  = hist_vol_close(close, window=21).dropna()
        v63  = hist_vol_close(close, window=63).dropna()
        # Longer window should have smaller std dev
        common = min(len(v21), len(v63))
        assert v63.iloc[-common:].std() <= v21.iloc[-common:].std() + 0.01

    def test_no_annualise(self, close):
        v_ann  = hist_vol_close(close, annualise=True).dropna()
        v_raw  = hist_vol_close(close, annualise=False).dropna()
        ratio = (v_ann / v_raw).dropna()
        assert np.allclose(ratio, np.sqrt(252), atol=0.01)


# ---------------------------------------------------------------------------
# 2. EWMA vol
# ---------------------------------------------------------------------------

class TestEWMAVol:

    def test_output_length(self, close):
        v = ewma_vol(close)
        assert len(v) == len(close) - 1

    def test_positive(self, close):
        assert (ewma_vol(close) > 0).all()

    def test_higher_lambda_smoother(self, close):
        v94 = ewma_vol(close, lam=0.94).dropna()
        v80 = ewma_vol(close, lam=0.80).dropna()
        common = min(len(v94), len(v80))
        assert v94.iloc[-common:].std() <= v80.iloc[-common:].std() + 0.01


# ---------------------------------------------------------------------------
# 3. Parkinson estimator
# ---------------------------------------------------------------------------

class TestParkinsonVol:

    def test_positive(self, ohlc):
        v = parkinson_vol(ohlc["High"], ohlc["Low"]).dropna()
        assert (v > 0).all()

    def test_reasonable_range(self, ohlc):
        v = parkinson_vol(ohlc["High"], ohlc["Low"]).dropna()
        assert v.mean() > 0.01
        assert v.mean() < 1.0

    def test_output_series(self, ohlc):
        assert isinstance(parkinson_vol(ohlc["High"], ohlc["Low"]), pd.Series)


# ---------------------------------------------------------------------------
# 4. Yang-Zhang estimator
# ---------------------------------------------------------------------------

class TestYangZhang:

    def test_positive(self, ohlc):
        v = yang_zhang_vol(ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"]).dropna()
        assert (v > 0).all()

    def test_output_series(self, ohlc):
        v = yang_zhang_vol(ohlc["Open"], ohlc["High"], ohlc["Low"], ohlc["Close"])
        assert isinstance(v, pd.Series)


# ---------------------------------------------------------------------------
# 5. Regime labeling
# ---------------------------------------------------------------------------

class TestVolRegimes:

    def test_only_valid_labels(self, close):
        v = hist_vol_close(close, window=21)
        r = label_vol_regimes(v, window=100)
        valid = {"Low", "Normal", "High", np.nan, None}
        for val in r.dropna().unique():
            assert val in valid

    def test_three_regime_types(self, close):
        v = hist_vol_close(close, window=21)
        r = label_vol_regimes(v, window=100).dropna()
        assert len(r.unique()) <= 3


# ---------------------------------------------------------------------------
# 6. GARCH(1,1)
# ---------------------------------------------------------------------------

class TestGARCH:

    def test_params_present(self, close):
        g = fit_garch(close)
        assert "omega" in g["params"]
        assert "alpha" in g["params"]
        assert "beta"  in g["params"]

    def test_persistence_between_0_and_1(self, close):
        g = fit_garch(close)
        assert 0 <= g["persistence"] <= 1

    def test_cond_vol_positive(self, close):
        g = fit_garch(close)
        assert (g["cond_vol"] > 0).all()

    def test_cond_vol_length(self, close):
        g = fit_garch(close)
        assert len(g["cond_vol"]) == len(close) - 1

    def test_forecast_length(self, close):
        g  = fit_garch(close)
        fc = garch_forecast(g, horizon=10)
        assert len(fc) == 10

    def test_forecast_positive(self, close):
        g  = fit_garch(close)
        fc = garch_forecast(g, horizon=10)
        assert (fc > 0).all()


# ---------------------------------------------------------------------------
# 7. Vol summary
# ---------------------------------------------------------------------------

class TestVolSummary:

    def test_returns_dataframe(self, close):
        v = hist_vol_close(close)
        df = vol_summary({"HV": v})
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, close):
        v  = hist_vol_close(close)
        df = vol_summary({"HV": v})
        for col in ["mean", "median", "std", "min", "max"]:
            assert col in df.columns
