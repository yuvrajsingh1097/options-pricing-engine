"""
Black-Scholes Options Pricing Model
====================================
Analytical pricing for European call and put options.
Covers: price, intrinsic value, time value, put-call parity check.

Author : yuvrajsingh1097
Project: Options Pricing Engine + LSTM Volatility Surface
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class BSMInputs:
    """All inputs needed to price a European option under Black-Scholes."""
    S: float        # Current stock price
    K: float        # Strike price
    T: float        # Time to expiry in years  (e.g. 30 days → 30/365)
    r: float        # Risk-free rate            (e.g. 0.05 for 5%)
    sigma: float    # Annualised volatility      (e.g. 0.20 for 20%)
    option_type: Literal["call", "put"] = "call"

    def validate(self) -> None:
        if self.S <= 0:
            raise ValueError(f"Stock price must be > 0, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike must be > 0, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be > 0, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be > 0, got {self.sigma}")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'")


@dataclass
class BSMResult:
    """Output of the Black-Scholes pricing engine."""
    price: float
    d1: float
    d2: float
    intrinsic_value: float
    time_value: float
    option_type: str


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float):
    """Compute d1 and d2 terms used in B-S formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_price(inputs: BSMInputs) -> BSMResult:
    """
    Price a European option using the Black-Scholes analytical formula.

    Parameters
    ----------
    inputs : BSMInputs

    Returns
    -------
    BSMResult with price, d1, d2, intrinsic value, and time value.
    """
    inputs.validate()
    S, K, T, r, sigma = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma

    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if inputs.option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        intrinsic = max(S - K, 0.0)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        intrinsic = max(K - S, 0.0)

    time_value = price - intrinsic

    return BSMResult(
        price=round(price, 4),
        d1=round(d1, 6),
        d2=round(d2, 6),
        intrinsic_value=round(intrinsic, 4),
        time_value=round(time_value, 4),
        option_type=inputs.option_type,
    )


# ---------------------------------------------------------------------------
# Put-call parity
# ---------------------------------------------------------------------------

def put_call_parity_check(
    S: float, K: float, T: float, r: float, sigma: float, tol: float = 1e-4
) -> dict:
    """
    Verify put-call parity:  C - P = S - K * exp(-rT)

    Returns a dict with call price, put price, LHS, RHS, and whether parity holds.
    """
    call_inp = BSMInputs(S, K, T, r, sigma, "call")
    put_inp  = BSMInputs(S, K, T, r, sigma, "put")

    call_price = bs_price(call_inp).price
    put_price  = bs_price(put_inp).price

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    holds = abs(lhs - rhs) < tol

    return {
        "call_price": call_price,
        "put_price": put_price,
        "C - P": round(lhs, 6),
        "S - K*exp(-rT)": round(rhs, 6),
        "parity_holds": holds,
        "difference": round(abs(lhs - rhs), 8),
    }


# ---------------------------------------------------------------------------
# Vectorised pricing — useful for surface generation
# ---------------------------------------------------------------------------

def bs_price_vectorised(
    S: float,
    strikes: np.ndarray,
    expiries: np.ndarray,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> np.ndarray:
    """
    Price a grid of options (strikes × expiries) at once.

    Parameters
    ----------
    S        : spot price (scalar)
    strikes  : 1-D array of strike prices
    expiries : 1-D array of expiries in years
    r        : risk-free rate
    sigma    : volatility (scalar — constant surface)
    option_type : 'call' or 'put'

    Returns
    -------
    2-D ndarray shape (len(strikes), len(expiries))
    """
    K = strikes[:, None]   # column vector
    T = expiries[None, :]  # row vector

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        prices = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        prices = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return np.round(prices, 4)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inp = BSMInputs(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type="call")
    result = bs_price(inp)

    print("=" * 50)
    print("Black-Scholes Pricing Demo")
    print("=" * 50)
    print(f"  S={inp.S}, K={inp.K}, T={inp.T:.4f}yr, r={inp.r}, σ={inp.sigma}")
    print(f"  Option type  : {result.option_type.upper()}")
    print(f"  Price        : ₹{result.price}")
    print(f"  Intrinsic    : ₹{result.intrinsic_value}")
    print(f"  Time value   : ₹{result.time_value}")
    print(f"  d1={result.d1}  d2={result.d2}")
    print()

    parity = put_call_parity_check(100, 100, 30/365, 0.05, 0.20)
    print("Put-Call Parity Check:")
    for k, v in parity.items():
        print(f"  {k}: {v}")
