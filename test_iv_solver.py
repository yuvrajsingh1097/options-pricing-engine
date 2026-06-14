"""
Unit Tests — Implied Volatility Solver
========================================
Tests: round-trip accuracy, method comparison, edge cases, surface recovery.

Run with:  python -m pytest tests/test_iv_solver.py -v
"""

import pytest
import numpy as np
from pricing.bs_model import BSMInputs, bs_price
from pricing.iv_solver import (
    iv_bisection, iv_newton, iv_solver,
    iv_surface, synthetic_market_prices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def market_price(S, K, T, r, sigma, option_type="call"):
    return bs_price(BSMInputs(S, K, T, r, sigma, option_type)).price


# ---------------------------------------------------------------------------
# 1. Round-trip accuracy — recover known vol from B-S price
# ---------------------------------------------------------------------------

class TestRoundTrip:

    @pytest.mark.parametrize("true_vol", [0.10, 0.20, 0.35, 0.50, 0.80])
    def test_hybrid_recovers_vol(self, true_vol):
        mp  = market_price(100, 100, 0.5, 0.05, true_vol)
        res = iv_solver(mp, 100, 100, 0.5, 0.05)
        assert res.converged
        assert abs(res.implied_vol - true_vol) < 1e-4

    @pytest.mark.parametrize("K", [80, 90, 100, 110, 120])
    def test_hybrid_across_strikes(self, K):
        mp  = market_price(100, K, 0.5, 0.05, 0.25)
        res = iv_solver(mp, 100, K, 0.5, 0.05)
        assert res.converged
        assert abs(res.implied_vol - 0.25) < 1e-4

    @pytest.mark.parametrize("T", [0.08, 0.25, 0.5, 1.0, 2.0])
    def test_hybrid_across_expiries(self, T):
        mp  = market_price(100, 100, T, 0.05, 0.20)
        res = iv_solver(mp, 100, 100, T, 0.05)
        assert res.converged
        assert abs(res.implied_vol - 0.20) < 1e-4

    def test_put_round_trip(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.30, "put")
        res = iv_solver(mp, 100, 100, 0.5, 0.05, "put")
        assert res.converged
        assert abs(res.implied_vol - 0.30) < 1e-4


# ---------------------------------------------------------------------------
# 2. Bisection specific
# ---------------------------------------------------------------------------

class TestBisection:

    def test_bisection_converges_atm(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_bisection(mp, 100, 100, 0.5, 0.05)
        assert res.converged
        assert abs(res.implied_vol - 0.20) < 1e-4

    def test_bisection_method_label(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_bisection(mp, 100, 100, 0.5, 0.05)
        assert res.method == "bisection"

    def test_bisection_impossible_price_fails(self):
        res = iv_bisection(9999, 100, 100, 0.5, 0.05)
        assert not res.converged


# ---------------------------------------------------------------------------
# 3. Newton-Raphson specific
# ---------------------------------------------------------------------------

class TestNewton:

    def test_newton_converges_atm(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_newton(mp, 100, 100, 0.5, 0.05)
        assert res.converged
        assert abs(res.implied_vol - 0.20) < 1e-4

    def test_newton_fewer_iterations_than_bisection_atm(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        r_n = iv_newton(mp, 100, 100, 0.5, 0.05)
        r_b = iv_bisection(mp, 100, 100, 0.5, 0.05)
        assert r_n.iterations <= r_b.iterations

    def test_newton_method_label(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_newton(mp, 100, 100, 0.5, 0.05)
        assert res.method == "newton"


# ---------------------------------------------------------------------------
# 4. Hybrid solver
# ---------------------------------------------------------------------------

class TestHybrid:

    def test_hybrid_converges(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_solver(mp, 100, 100, 0.5, 0.05)
        assert res.converged

    def test_hybrid_error_below_tolerance(self):
        mp  = market_price(100, 100, 0.5, 0.05, 0.20)
        res = iv_solver(mp, 100, 100, 0.5, 0.05)
        assert res.error < 1e-5

    def test_hybrid_negative_price_fails(self):
        res = iv_solver(-1.0, 100, 100, 0.5, 0.05)
        assert not res.converged or res.implied_vol is None


# ---------------------------------------------------------------------------
# 5. IV Surface
# ---------------------------------------------------------------------------

class TestIVSurface:

    def setup_method(self):
        self.strikes  = np.array([90.0, 100.0, 110.0])
        self.expiries = np.array([0.25, 0.5,   1.0])
        self.prices   = synthetic_market_prices(
            100, self.strikes, self.expiries, 0.05, 0.20, smile=False
        )
        self.surf = iv_surface(100, self.strikes, self.expiries, self.prices, 0.05)

    def test_output_keys(self):
        assert "iv" in self.surf
        assert "converged" in self.surf
        assert "errors" in self.surf

    def test_shape(self):
        assert self.surf["iv"].shape == (3, 3)

    def test_all_converged_flat_surface(self):
        assert self.surf["converged"].all()

    def test_flat_surface_recovers_vol(self):
        iv = self.surf["iv"]
        assert np.allclose(iv, 0.20, atol=1e-4)

    def test_smile_surface_higher_wings(self):
        prices = synthetic_market_prices(
            100, self.strikes, self.expiries, 0.05, 0.20, smile=True
        )
        surf = iv_surface(100, self.strikes, self.expiries, prices, 0.05)
        iv   = surf["iv"]
        # Wing strikes should have higher IV than ATM
        assert iv[0, 1] > iv[1, 1]   # OTM wing > ATM
        assert iv[2, 1] > iv[1, 1]   # OTM wing > ATM
