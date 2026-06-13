"""
Options Greeks — Black-Scholes Analytical Computation
=======================================================
Computes all five first-order Greeks for European call and put options:
    Delta (Δ)  — sensitivity to spot price
    Gamma (Γ)  — rate of change of delta
    Theta (Θ)  — time decay (per calendar day)
    Vega  (V)  — sensitivity to volatility (per 1% move)
    Rho   (ρ)  — sensitivity to interest rate (per 1% move)

Also provides vectorised surface generation for heatmap plotting.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float   # per calendar day
    vega:  float   # per 1% change in vol
    rho:   float   # per 1% change in rate
    option_type: str


# ---------------------------------------------------------------------------
# Shared d1/d2
# ---------------------------------------------------------------------------

def _d1_d2(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ---------------------------------------------------------------------------
# Individual Greeks
# ---------------------------------------------------------------------------

def delta(S, K, T, r, sigma, option_type="call") -> float:
    """
    Delta — first derivative of option price w.r.t. spot.
    Call delta ∈ (0, 1),  Put delta ∈ (-1, 0).
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1)


def gamma(S, K, T, r, sigma) -> float:
    """
    Gamma — second derivative of price w.r.t. spot (same for call & put).
    Peaks at ATM, decays for ITM/OTM.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def theta(S, K, T, r, sigma, option_type="call") -> float:
    """
    Theta — time decay per calendar day (divide annual theta by 365).
    Almost always negative (options lose value as time passes).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)

    term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))

    if option_type == "call":
        annual = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        annual = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)

    return float(annual / 365)   # per calendar day


def vega(S, K, T, r, sigma) -> float:
    """
    Vega — sensitivity to a 1% change in volatility (same for call & put).
    Divide annual vega by 100 to get per-1%-vol-move value.
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100)


def rho(S, K, T, r, sigma, option_type="call") -> float:
    """
    Rho — sensitivity to a 1% change in risk-free rate.
    Divide by 100 to normalise per 1% rate move.
    """
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)


# ---------------------------------------------------------------------------
# Combined Greeks
# ---------------------------------------------------------------------------

def compute_greeks(S, K, T, r, sigma, option_type="call") -> GreeksResult:
    """Compute all five Greeks and return as a GreeksResult dataclass."""
    return GreeksResult(
        delta=round(delta(S, K, T, r, sigma, option_type), 6),
        gamma=round(gamma(S, K, T, r, sigma), 6),
        theta=round(theta(S, K, T, r, sigma, option_type), 6),
        vega =round(vega(S, K, T, r, sigma), 6),
        rho  =round(rho(S, K, T, r, sigma, option_type), 6),
        option_type=option_type,
    )


# ---------------------------------------------------------------------------
# Vectorised surfaces (strikes × expiries grid)
# ---------------------------------------------------------------------------

def greeks_surface(
    S: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> dict:
    """
    Compute a full Greeks surface over a grid of strikes and expiries.

    Returns a dict of 2-D arrays (shape: len(strikes) × len(expiries)):
        'delta', 'gamma', 'theta', 'vega', 'rho'
    """
    K = strikes[:, None]
    T = expiries[None, :]

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    pdf_d1 = norm.pdf(d1)

    # Delta
    if option_type == "call":
        d = norm.cdf(d1)
    else:
        d = norm.cdf(d1) - 1

    # Gamma (same for call/put)
    g = pdf_d1 / (S * sigma * np.sqrt(T))

    # Theta
    term1 = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        th = (term1 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        th = (term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    # Vega (same for call/put)
    v = S * pdf_d1 * np.sqrt(T) / 100

    # Rho
    if option_type == "call":
        rh = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rh = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": np.round(d,  6),
        "gamma": np.round(g,  6),
        "theta": np.round(th, 6),
        "vega":  np.round(v,  6),
        "rho":   np.round(rh, 6),
    }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    params = dict(S=100, K=100, T=0.5, r=0.05, sigma=0.20)

    print("=" * 50)
    print("Greeks Demo  —  ATM Call  (S=K=100, T=0.5yr)")
    print("=" * 50)
    g = compute_greeks(**params, option_type="call")
    print(f"  Delta : {g.delta:>10.6f}   (₹ per ₹1 move in S)")
    print(f"  Gamma : {g.gamma:>10.6f}   (Δ change per ₹1 move in S)")
    print(f"  Theta : {g.theta:>10.6f}   (₹ per calendar day)")
    print(f"  Vega  : {g.vega:>10.6f}   (₹ per 1% vol move)")
    print(f"  Rho   : {g.rho:>10.6f}   (₹ per 1% rate move)")

    print()
    print("Greeks Demo  —  ATM Put")
    print("=" * 50)
    p = compute_greeks(**params, option_type="put")
    print(f"  Delta : {p.delta:>10.6f}")
    print(f"  Gamma : {p.gamma:>10.6f}  (same as call)")
    print(f"  Theta : {p.theta:>10.6f}")
    print(f"  Vega  : {p.vega:>10.6f}  (same as call)")
    print(f"  Rho   : {p.rho:>10.6f}")
