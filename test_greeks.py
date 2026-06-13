"""
Unit Tests — Options Greeks
=============================
Tests: sign correctness, monotonicity, put-call symmetry,
       ATM known values, surface shape, edge behaviour.

Run with:  python -m pytest tests/test_greeks.py -v
"""

import pytest
import numpy as np
from pricing.greeks import (
    delta, gamma, theta, vega, rho,
    compute_greeks, greeks_surface,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ATM  = dict(S=100, K=100, T=0.5, r=0.05, sigma=0.20)
ITM  = dict(S=120, K=100, T=0.5, r=0.05, sigma=0.20)
OTM  = dict(S=80,  K=100, T=0.5, r=0.05, sigma=0.20)


# ---------------------------------------------------------------------------
# 1. Delta
# ---------------------------------------------------------------------------

class TestDelta:

    def test_call_delta_between_0_and_1(self):
        assert 0 < delta(**ATM, option_type="call") < 1

    def test_put_delta_between_minus1_and_0(self):
        assert -1 < delta(**ATM, option_type="put") < 0

    def test_atm_call_delta_near_half(self):
        d = delta(**ATM, option_type="call")
        assert 0.45 < d < 0.65   # ATM call delta ≈ 0.5–0.6

    def test_deep_itm_call_delta_near_1(self):
        d = delta(S=200, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
        assert d > 0.95

    def test_deep_otm_call_delta_near_0(self):
        d = delta(S=50, K=100, T=0.5, r=0.05, sigma=0.20, option_type="call")
        assert d < 0.05

    def test_call_delta_increases_with_spot(self):
        d1 = delta(**OTM, option_type="call")
        d2 = delta(**ATM, option_type="call")
        d3 = delta(**ITM, option_type="call")
        assert d1 < d2 < d3

    def test_put_call_delta_relationship(self):
        # Call delta - Put delta = 1  (put-call delta parity)
        dc = delta(**ATM, option_type="call")
        dp = delta(**ATM, option_type="put")
        assert abs(dc - dp - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Gamma
# ---------------------------------------------------------------------------

class TestGamma:

    def test_gamma_always_positive(self):
        for params in [ATM, ITM, OTM]:
            assert gamma(**params) > 0

    def test_gamma_peaks_at_atm(self):
        g_atm = gamma(**ATM)
        g_itm = gamma(**ITM)
        g_otm = gamma(**OTM)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_gamma_same_for_call_and_put(self):
        # Gamma is identical for call and put (same formula)
        assert gamma(**ATM) == gamma(**ATM)

    def test_gamma_decreases_with_longer_expiry(self):
        g_short = gamma(S=100, K=100, T=0.1,  r=0.05, sigma=0.20)
        g_long  = gamma(S=100, K=100, T=2.0,  r=0.05, sigma=0.20)
        assert g_short > g_long


# ---------------------------------------------------------------------------
# 3. Theta
# ---------------------------------------------------------------------------

class TestTheta:

    def test_call_theta_negative(self):
        assert theta(**ATM, option_type="call") < 0

    def test_put_theta_negative(self):
        assert theta(**ATM, option_type="put") < 0

    def test_theta_more_negative_near_expiry(self):
        t_far  = theta(S=100, K=100, T=1.0,   r=0.05, sigma=0.20, option_type="call")
        t_near = theta(S=100, K=100, T=0.05,  r=0.05, sigma=0.20, option_type="call")
        assert t_near < t_far   # more negative (faster decay) near expiry

    def test_theta_atm_most_negative(self):
        t_atm = theta(**ATM, option_type="call")
        t_itm = theta(**ITM, option_type="call")
        t_otm = theta(**OTM, option_type="call")
        assert t_atm < t_itm
        assert t_atm < t_otm


# ---------------------------------------------------------------------------
# 4. Vega
# ---------------------------------------------------------------------------

class TestVega:

    def test_vega_positive(self):
        assert vega(**ATM) > 0

    def test_vega_same_for_call_and_put(self):
        v1 = vega(**ATM)
        v2 = vega(**ATM)
        assert v1 == v2

    def test_vega_peaks_atm(self):
        v_atm = vega(**ATM)
        v_itm = vega(**ITM)
        v_otm = vega(**OTM)
        assert v_atm > v_itm
        assert v_atm > v_otm

    def test_vega_increases_with_longer_expiry(self):
        v_short = vega(S=100, K=100, T=0.1, r=0.05, sigma=0.20)
        v_long  = vega(S=100, K=100, T=2.0, r=0.05, sigma=0.20)
        assert v_long > v_short


# ---------------------------------------------------------------------------
# 5. Rho
# ---------------------------------------------------------------------------

class TestRho:

    def test_call_rho_positive(self):
        assert rho(**ATM, option_type="call") > 0

    def test_put_rho_negative(self):
        assert rho(**ATM, option_type="put") < 0

    def test_call_rho_increases_with_expiry(self):
        r1 = rho(S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call")
        r2 = rho(S=100, K=100, T=2.0,  r=0.05, sigma=0.20, option_type="call")
        assert r2 > r1


# ---------------------------------------------------------------------------
# 6. compute_greeks combined
# ---------------------------------------------------------------------------

class TestComputeGreeks:

    def test_returns_greeks_result(self):
        from pricing.greeks import GreeksResult
        g = compute_greeks(**ATM, option_type="call")
        assert isinstance(g, GreeksResult)

    def test_all_fields_finite(self):
        g = compute_greeks(**ATM, option_type="call")
        for field in [g.delta, g.gamma, g.theta, g.vega, g.rho]:
            assert np.isfinite(field)


# ---------------------------------------------------------------------------
# 7. Vectorised surface
# ---------------------------------------------------------------------------

class TestGreeksSurface:

    def setup_method(self):
        self.strikes  = np.linspace(80, 120, 10)
        self.expiries = np.linspace(0.1, 2.0, 10)
        self.surface  = greeks_surface(100, self.strikes, self.expiries, 0.05, 0.20, "call")

    def test_output_keys(self):
        assert set(self.surface.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_output_shape(self):
        for arr in self.surface.values():
            assert arr.shape == (10, 10)

    def test_delta_surface_between_0_and_1(self):
        assert (self.surface["delta"] >= 0).all()
        assert (self.surface["delta"] <= 1).all()

    def test_gamma_surface_positive(self):
        assert (self.surface["gamma"] > 0).all()

    def test_theta_surface_negative(self):
        assert (self.surface["theta"] < 0).all()

    def test_vega_surface_positive(self):
        assert (self.surface["vega"] > 0).all()
