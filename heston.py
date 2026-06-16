"""
Heston Stochastic Volatility Model
=====================================
The Heston model allows volatility to be stochastic (mean-reverting),
unlike Black-Scholes which assumes constant vol.

Model dynamics:
    dS = μS dt + √v · S dW₁
    dv = κ(θ - v) dt + ξ√v dW₂
    dW₁ dW₂ = ρ dt

Parameters:
    v0    : initial variance  (e.g. 0.04 → 20% initial vol)
    kappa : mean reversion speed
    theta : long-run variance
    xi    : vol of vol
    rho   : correlation between asset and vol Brownian motions

Implements:
    1. Monte Carlo pricing under Heston dynamics (antithetic variates)
    2. Sample path generation for visualisation
    3. Heston vs Black-Scholes price grid comparison
    4. Implied vol smile extraction from Heston prices
"""

import numpy as np
from dataclasses import dataclass
from typing import Literal
from scipy.stats import norm
import warnings


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

@dataclass
class HestonParams:
    v0:    float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi:    float = 0.30
    rho:   float = -0.70

    def validate(self):
        feller = 2 * self.kappa * self.theta > self.xi ** 2
        if not feller:
            warnings.warn(
                f"Feller condition violated: 2*kappa*theta={2*self.kappa*self.theta:.4f} < xi^2={self.xi**2:.4f}."
            )
        if self.v0 <= 0:    raise ValueError("v0 must be > 0")
        if self.kappa <= 0: raise ValueError("kappa must be > 0")
        if self.theta <= 0: raise ValueError("theta must be > 0")
        if self.xi <= 0:    raise ValueError("xi must be > 0")


@dataclass
class HestonResult:
    price:     float
    std_error: float
    ci_lower:  float
    ci_upper:  float
    n_paths:   int
    n_steps:   int


# ---------------------------------------------------------------------------
# Monte Carlo pricer
# ---------------------------------------------------------------------------

def heston_mc(
    S: float,
    K: float,
    T: float,
    r: float,
    params: HestonParams,
    option_type: Literal["call", "put"] = "call",
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int = 42,
) -> HestonResult:
    params.validate()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1 - params.rho ** 2)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    W1 = Z1
    W2 = params.rho * Z1 + rho_bar * Z2

    # Forward paths
    log_S = np.full(n_paths, np.log(S))
    v = np.full(n_paths, params.v0)
    for t in range(n_steps):
        v_pos = np.maximum(v, 0)
        sqrt_v = np.sqrt(v_pos)
        log_S += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W1[:, t]
        v += params.kappa * (params.theta - v_pos) * dt + params.xi * sqrt_v * sqrt_dt * W2[:, t]
    S_T = np.exp(log_S)

    # Antithetic paths
    log_S_a = np.full(n_paths, np.log(S))
    v_a = np.full(n_paths, params.v0)
    for t in range(n_steps):
        v_pos_a = np.maximum(v_a, 0)
        sqrt_va = np.sqrt(v_pos_a)
        log_S_a += (r - 0.5 * v_pos_a) * dt - sqrt_va * sqrt_dt * W1[:, t]
        v_a += params.kappa * (params.theta - v_pos_a) * dt - params.xi * sqrt_va * sqrt_dt * W2[:, t]
    S_T_a = np.exp(log_S_a)

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
        payoff_a = np.maximum(S_T_a - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)
        payoff_a = np.maximum(K - S_T_a, 0)

    combined = (payoff + payoff_a) / 2
    discount = np.exp(-r * T)
    prices = discount * combined

    price   = float(np.mean(prices))
    std_err = float(np.std(prices) / np.sqrt(n_paths))

    return HestonResult(
        price=round(price, 4),
        std_error=round(std_err, 6),
        ci_lower=round(price - 1.96 * std_err, 4),
        ci_upper=round(price + 1.96 * std_err, 4),
        n_paths=n_paths,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# B-S reference
# ---------------------------------------------------------------------------

def _bs_price(S, K, T, r, sigma, option_type="call") -> float:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ---------------------------------------------------------------------------
# Heston vs B-S grid
# ---------------------------------------------------------------------------

def heston_vs_bs_grid(
    S: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    r: float,
    params: HestonParams,
    option_type: str = "call",
    n_paths: int = 20_000,
) -> dict:
    from pricing.iv_solver import iv_solver
    bs_vol = np.sqrt(params.theta)
    n_K, n_T = len(strikes), len(expiries)
    heston_prices = np.zeros((n_K, n_T))
    bs_prices     = np.zeros((n_K, n_T))
    heston_iv     = np.full((n_K, n_T), np.nan)

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            h = heston_mc(S, K, T, r, params, option_type, n_paths=n_paths, n_steps=50, seed=42)
            heston_prices[i, j] = h.price
            bs_prices[i, j]     = _bs_price(S, K, T, r, bs_vol, option_type)
            iv_res = iv_solver(h.price, S, K, T, r, option_type)
            if iv_res.converged:
                heston_iv[i, j] = iv_res.implied_vol

    return {
        "heston_prices": np.round(heston_prices, 4),
        "bs_prices":     np.round(bs_prices, 4),
        "price_diff":    np.round(heston_prices - bs_prices, 4),
        "heston_iv":     np.round(heston_iv, 6),
    }


# ---------------------------------------------------------------------------
# Sample path generator
# ---------------------------------------------------------------------------

def heston_sample_paths(
    S: float,
    T: float,
    r: float,
    params: HestonParams,
    n_paths: int = 10,
    n_steps: int = 252,
    seed: int = 0,
) -> dict:
    params.validate()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    rho_bar = np.sqrt(1 - params.rho ** 2)

    Z1 = rng.standard_normal((n_paths, n_steps))
    Z2 = rng.standard_normal((n_paths, n_steps))
    W1 = Z1
    W2 = params.rho * Z1 + rho_bar * Z2

    S_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S
    v_paths[:, 0] = params.v0

    for t in range(n_steps):
        v_pos = np.maximum(v_paths[:, t], 0)
        sqrt_v = np.sqrt(v_pos)
        S_paths[:, t+1] = S_paths[:, t] * np.exp(
            (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W1[:, t]
        )
        v_paths[:, t+1] = v_paths[:, t] + \
            params.kappa * (params.theta - v_pos) * dt + \
            params.xi * sqrt_v * sqrt_dt * W2[:, t]
        v_paths[:, t+1] = np.maximum(v_paths[:, t+1], 0)

    return {
        "S_paths":   S_paths,
        "v_paths":   v_paths,
        "vol_paths": np.sqrt(np.maximum(v_paths, 0)) * 100,
        "time":      np.linspace(0, T, n_steps + 1),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70)
    result = heston_mc(S=100, K=100, T=0.5, r=0.05, params=params, n_paths=50_000)
    bs_ref = _bs_price(100, 100, 0.5, 0.05, sigma=np.sqrt(params.theta))

    print("=" * 55)
    print("Heston MC Demo  —  ATM Call  S=K=100, T=0.5yr")
    print("=" * 55)
    print(f"  Heston price : {result.price}")
    print(f"  95% CI       : [{result.ci_lower}, {result.ci_upper}]")
    print(f"  Std error    : {result.std_error}")
    print(f"  B-S (flat)   : {bs_ref:.4f}")
    print(f"  Difference   : {result.price - bs_ref:.4f}")
    feller_ok = 2*params.kappa*params.theta > params.xi**2
    print(f"  Feller cond  : {'PASS' if feller_ok else 'FAIL'}")
