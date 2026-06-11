"""
Unit Tests — Black-Scholes Pricing Model
==========================================
Tests: pricing accuracy, put-call parity, edge cases, vectorised surface.

Run with:  python -m pytest tests/test_bs_model.py -v
"""

import pytest
import numpy as np
from pricing.bs_model import BSMInputs, bs_price, put_call_parity_check, bs_price_vectorised


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_atm_call(**overrides):
    defaults = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="call")
    defaults.update(overrides)
    return BSMInputs(**defaults)


# ---------------------------------------------------------------------------
# 1. Known-value accuracy tests
# ---------------------------------------------------------------------------

class TestKnownValues:
    """Compare against textbook / reference values."""

    def test_atm_call_approx(self):
        """ATM call, 1yr, 20% vol, 5% rate — well-known benchmark ≈ 10.45."""
        inp = make_atm_call()
        result = bs_price(inp)
        assert abs(result.price - 10.45) < 0.10, f"Got {result.price}, expected ~10.45"

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call: price ≈ intrinsic value (S - K * exp(-rT))."""
        inp = make_atm_call(S=200, K=100, T=0.01)
        result = bs_price(inp)
        pv_strike = 100 * np.exp(-0.05 * 0.01)
        assert result.price >= 200 - pv_strike - 0.5

    def test_deep_otm_call_near_zero(self):
        """Deep OTM call with very short expiry should be ~0."""
        inp = make_atm_call(S=50, K=200, T=1/365)
        result = bs_price(inp)
        assert result.price < 0.001

    def test_put_positive_for_otm(self):
        """OTM put (S > K) should still have positive time value."""
        inp = BSMInputs(S=110, K=100, T=0.5, r=0.05, sigma=0.20, option_type="put")
        result = bs_price(inp)
        assert result.price > 0

    def test_itm_put_intrinsic(self):
        """Deep ITM put: price should be close to K - S."""
        inp = BSMInputs(S=50, K=200, T=0.01, r=0.05, sigma=0.20, option_type="put")
        result = bs_price(inp)
        assert result.price >= 150 - 2.0


# ---------------------------------------------------------------------------
# 2. Put-call parity
# ---------------------------------------------------------------------------

class TestPutCallParity:

    def test_atm_parity(self):
        res = put_call_parity_check(100, 100, 1.0, 0.05, 0.20)
        assert res["parity_holds"], f"Parity failed: {res}"

    def test_itm_parity(self):
        res = put_call_parity_check(120, 100, 0.5, 0.03, 0.30)
        assert res["parity_holds"]

    def test_otm_parity(self):
        res = put_call_parity_check(80, 100, 0.25, 0.07, 0.15)
        assert res["parity_holds"]

    def test_parity_difference_tiny(self):
        res = put_call_parity_check(100, 100, 1.0, 0.05, 0.20)
        assert res["difference"] < 1e-4


# ---------------------------------------------------------------------------
# 3. Result properties
# ---------------------------------------------------------------------------

class TestResultProperties:

    def test_call_price_positive(self):
        assert bs_price(make_atm_call()).price > 0

    def test_put_price_positive(self):
        inp = BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert bs_price(inp).price > 0

    def test_time_value_positive(self):
        result = bs_price(make_atm_call())
        assert result.time_value >= 0

    def test_intrinsic_plus_time_equals_price(self):
        result = bs_price(make_atm_call())
        assert abs(result.intrinsic_value + result.time_value - result.price) < 1e-4

    def test_higher_vol_increases_price(self):
        low_vol  = bs_price(make_atm_call(sigma=0.10)).price
        high_vol = bs_price(make_atm_call(sigma=0.40)).price
        assert high_vol > low_vol

    def test_longer_expiry_increases_price(self):
        short = bs_price(make_atm_call(T=0.1)).price
        long  = bs_price(make_atm_call(T=2.0)).price
        assert long > short

    def test_call_increases_with_spot(self):
        p1 = bs_price(make_atm_call(S=90)).price
        p2 = bs_price(make_atm_call(S=110)).price
        assert p2 > p1

    def test_put_decreases_with_spot(self):
        p1 = BSMInputs(S=90, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        p2 = BSMInputs(S=110, K=100, T=1.0, r=0.05, sigma=0.20, option_type="put")
        assert bs_price(p1).price > bs_price(p2).price


# ---------------------------------------------------------------------------
# 4. Validation / edge cases
# ---------------------------------------------------------------------------

class TestValidation:

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError):
            bs_price(BSMInputs(S=-10, K=100, T=1.0, r=0.05, sigma=0.20))

    def test_zero_strike_raises(self):
        with pytest.raises(ValueError):
            bs_price(BSMInputs(S=100, K=0, T=1.0, r=0.05, sigma=0.20))

    def test_zero_time_raises(self):
        with pytest.raises(ValueError):
            bs_price(BSMInputs(S=100, K=100, T=0, r=0.05, sigma=0.20))

    def test_zero_vol_raises(self):
        with pytest.raises(ValueError):
            bs_price(BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.0))

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            bs_price(BSMInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20, option_type="binary"))


# ---------------------------------------------------------------------------
# 5. Vectorised surface
# ---------------------------------------------------------------------------

class TestVectorised:

    def test_output_shape(self):
        strikes  = np.array([90, 100, 110])
        expiries = np.array([0.25, 0.5, 1.0])
        grid = bs_price_vectorised(100, strikes, expiries, 0.05, 0.20)
        assert grid.shape == (3, 3)

    def test_higher_strike_lower_call_price(self):
        strikes  = np.array([90, 100, 110])
        expiries = np.array([1.0])
        grid = bs_price_vectorised(100, strikes, expiries, 0.05, 0.20, "call")
        assert grid[0, 0] > grid[1, 0] > grid[2, 0]

    def test_all_positive(self):
        strikes  = np.linspace(80, 120, 10)
        expiries = np.linspace(0.1, 2.0, 10)
        grid = bs_price_vectorised(100, strikes, expiries, 0.05, 0.20)
        assert (grid > 0).all()
