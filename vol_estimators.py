"""
Historical Volatility & GARCH(1,1) Baseline
=============================================
Computes various realised volatility estimators and fits a GARCH(1,1)
model as the baseline forecaster before LSTM is added on Day 6.

Estimators implemented:
    1. Close-to-close historical vol  (standard)
    2. EWMA vol                       (exponentially weighted)
    3. Parkinson estimator            (uses High/Low — more efficient)
    4. Yang-Zhang estimator           (overnight + intraday, most efficient)

Regime labeling:
    Low / Normal / High vol regimes based on rolling percentile thresholds.

GARCH(1,1):
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    Fit using arch library, forecast n-step ahead vol.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Close-to-close historical vol
# ---------------------------------------------------------------------------

def hist_vol_close(
    prices: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """
    Standard close-to-close realised volatility.
    Uses log returns and rolling standard deviation.

    Parameters
    ----------
    prices   : pd.Series of closing prices
    window   : rolling window in trading days (default 21 = 1 month)
    annualise: multiply by sqrt(252) if True

    Returns
    -------
    pd.Series of rolling vol estimates
    """
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window).std()
    if annualise:
        vol = vol * np.sqrt(252)
    return vol.rename(f"hist_vol_{window}d")


# ---------------------------------------------------------------------------
# 2. EWMA volatility (RiskMetrics λ = 0.94)
# ---------------------------------------------------------------------------

def ewma_vol(
    prices: pd.Series,
    lam: float = 0.94,
    annualise: bool = True,
) -> pd.Series:
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics model).
    σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}

    Parameters
    ----------
    prices : pd.Series of closing prices
    lam    : decay factor (0.94 for daily, 0.97 for monthly)
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    n = len(log_ret)
    var = np.zeros(n)
    var[0] = log_ret.iloc[0] ** 2

    for t in range(1, n):
        var[t] = lam * var[t-1] + (1 - lam) * log_ret.iloc[t-1] ** 2

    vol = pd.Series(np.sqrt(var), index=log_ret.index)
    if annualise:
        vol = vol * np.sqrt(252)
    return vol.rename("ewma_vol")


# ---------------------------------------------------------------------------
# 3. Parkinson estimator (High-Low)
# ---------------------------------------------------------------------------

def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """
    Parkinson (1980) estimator using daily High and Low.
    More efficient than close-to-close (uses intraday range).

    σ² = 1/(4·ln2) · E[ln(H/L)²]
    """
    log_hl = np.log(high / low)
    factor = 1.0 / (4.0 * np.log(2))
    var = factor * (log_hl ** 2).rolling(window).mean()
    vol = np.sqrt(var)
    if annualise:
        vol = vol * np.sqrt(252)
    return vol.rename(f"parkinson_vol_{window}d")


# ---------------------------------------------------------------------------
# 4. Yang-Zhang estimator
# ---------------------------------------------------------------------------

def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualise: bool = True,
) -> pd.Series:
    """
    Yang-Zhang (2000) estimator — minimum variance unbiased estimator.
    Accounts for overnight gaps + intraday range.

    σ²_YZ = σ²_overnight + k·σ²_open_close + (1-k)·σ²_RS
    where k = 0.34 / (1.34 + (window+1)/(window-1))
    """
    close_prev = close.shift(1)

    # Overnight return
    log_oc = np.log(open_ / close_prev)
    # Open-to-close return
    log_co = np.log(close / open_)

    # Rogers-Satchell component
    log_hc = np.log(high / close)
    log_ho = np.log(high / open_)
    log_lc = np.log(low  / close)
    log_lo = np.log(low  / open_)
    rs = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))

    var_overnight   = log_oc.rolling(window).var()
    var_open_close  = log_co.rolling(window).var()
    var_yz = var_overnight + k * var_open_close + (1 - k) * rs

    vol = np.sqrt(var_yz.clip(lower=0))
    if annualise:
        vol = vol * np.sqrt(252)
    return vol.rename(f"yang_zhang_vol_{window}d")


# ---------------------------------------------------------------------------
# 5. Vol regime labeling
# ---------------------------------------------------------------------------

def label_vol_regimes(
    vol: pd.Series,
    low_pct: float = 33,
    high_pct: float = 67,
    window: int = 252,
) -> pd.Series:
    """
    Label each day as 'Low', 'Normal', or 'High' volatility regime
    based on rolling percentile thresholds.

    Parameters
    ----------
    vol      : annualised vol series
    low_pct  : percentile below which = 'Low' regime
    high_pct : percentile above which = 'High' regime
    window   : lookback for percentile calculation
    """
    def assign_regime(x):
        if len(x) < 2:
            return np.nan
        v = x.iloc[-1]
        low_thresh  = np.percentile(x.dropna(), low_pct)
        high_thresh = np.percentile(x.dropna(), high_pct)
        if v <= low_thresh:
            return "Low"
        elif v >= high_thresh:
            return "High"
        return "Normal"

    regimes = vol.rolling(window, min_periods=60).apply(
        lambda x: {"Low": 0, "Normal": 1, "High": 2}.get(assign_regime(pd.Series(x)), np.nan),
        raw=False
    )
    label_map = {0: "Low", 1: "Normal", 2: "High"}
    return regimes.map(label_map).rename("vol_regime")


# ---------------------------------------------------------------------------
# 6. GARCH(1,1) baseline
# ---------------------------------------------------------------------------

def fit_garch(
    prices: pd.Series,
    p: int = 1,
    q: int = 1,
) -> dict:
    """
    Fit a GARCH(p,q) model and return fitted params + conditional vol series.

    Requires: arch library  (pip install arch)

    Returns dict with:
        'model'       : fitted arch model result
        'params'      : omega, alpha, beta
        'cond_vol'    : conditional volatility (annualised)
        'persistence' : alpha + beta (< 1 for stationarity)
        'half_life'   : vol shock half-life in days
    """
    try:
        from arch import arch_model
    except ImportError:
        raise ImportError("Install arch: pip install arch --break-system-packages")

    log_ret = np.log(prices / prices.shift(1)).dropna() * 100  # in %

    model = arch_model(log_ret, vol="Garch", p=p, q=q, dist="normal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp="off")

    omega = result.params["omega"]
    alpha = result.params.get("alpha[1]", 0)
    beta  = result.params.get("beta[1]",  0)
    persistence = alpha + beta
    half_life   = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

    cond_vol = result.conditional_volatility / 100 * np.sqrt(252)  # annualised

    return {
        "model":       result,
        "params":      {"omega": omega, "alpha": alpha, "beta": beta},
        "cond_vol":    pd.Series(cond_vol, index=log_ret.index, name="garch_vol"),
        "persistence": round(persistence, 6),
        "half_life":   round(half_life, 2),
        "aic":         round(result.aic, 4),
        "bic":         round(result.bic, 4),
    }


def garch_forecast(
    garch_result: dict,
    horizon: int = 21,
) -> pd.Series:
    """
    Generate n-step ahead GARCH vol forecast.

    Parameters
    ----------
    garch_result : output of fit_garch()
    horizon      : forecast horizon in days

    Returns
    -------
    pd.Series of forecasted annualised vol
    """
    forecast = garch_result["model"].forecast(horizon=horizon, reindex=False)
    var_forecast = forecast.variance.iloc[-1].values  # variance in %²
    vol_forecast = np.sqrt(var_forecast) / 100 * np.sqrt(252)  # annualised

    return pd.Series(
        vol_forecast,
        index=range(1, horizon + 1),
        name="garch_forecast"
    )


# ---------------------------------------------------------------------------
# 7. Summary stats
# ---------------------------------------------------------------------------

def vol_summary(vol_series: dict) -> pd.DataFrame:
    """
    Compute summary statistics for multiple vol estimators.

    Parameters
    ----------
    vol_series : dict of {name: pd.Series}
    """
    rows = []
    for name, s in vol_series.items():
        s_clean = s.dropna()
        rows.append({
            "estimator": name,
            "mean":      round(s_clean.mean() * 100, 2),
            "median":    round(s_clean.median() * 100, 2),
            "std":       round(s_clean.std() * 100, 2),
            "min":       round(s_clean.min() * 100, 2),
            "max":       round(s_clean.max() * 100, 2),
            "n_obs":     len(s_clean),
        })
    return pd.DataFrame(rows).set_index("estimator")


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing without yfinance)
# ---------------------------------------------------------------------------

def synthetic_ohlc(
    n: int = 500,
    S0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLC data with GBM for testing."""
    rng = np.random.default_rng(seed)
    dt  = 1 / 252

    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n)
    close   = S0 * np.exp(np.cumsum(log_ret))
    open_   = np.roll(close, 1); open_[0] = S0

    intraday_range = sigma * np.sqrt(dt) * np.abs(rng.standard_normal(n)) * close
    high = close + intraday_range * 0.6
    low  = close - intraday_range * 0.4

    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close
    }, index=dates)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Historical Volatility & GARCH Demo")
    print("=" * 55)

    df = synthetic_ohlc(n=500)

    hv21  = hist_vol_close(df["Close"], window=21)
    hv63  = hist_vol_close(df["Close"], window=63)
    ewma  = ewma_vol(df["Close"])
    park  = parkinson_vol(df["High"], df["Low"], window=21)
    yz    = yang_zhang_vol(df["Open"], df["High"], df["Low"], df["Close"], window=21)

    print("\nVol Estimator Summary:")
    summary = vol_summary({
        "HV-21d": hv21, "HV-63d": hv63,
        "EWMA":   ewma, "Parkinson": park, "Yang-Zhang": yz
    })
    print(summary.to_string())

    print("\nFitting GARCH(1,1)...")
    garch = fit_garch(df["Close"])
    p = garch["params"]
    print(f"  omega={p['omega']:.6f}  alpha={p['alpha']:.4f}  beta={p['beta']:.4f}")
    print(f"  Persistence : {garch['persistence']:.4f}")
    print(f"  Half-life   : {garch['half_life']:.1f} days")
    print(f"  AIC / BIC   : {garch['aic']} / {garch['bic']}")

    fc = garch_forecast(garch, horizon=10)
    print(f"\nGARCH 10-day forecast (annualised vol %):")
    for d, v in fc.items():
        print(f"  Day {d:2d}: {v*100:.2f}%")
