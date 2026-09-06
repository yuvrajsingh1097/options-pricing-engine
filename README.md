# Options Pricing Engine + LSTM Volatility Surface

European options pricing (Black-Scholes & Heston), implied volatility surface construction, LSTM-based volatility forecasting, Greeks analysis, and an interactive Streamlit dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-25%20passing-brightgreen)](#testing)

---

## What this doesokkok

| Module | Description |
|--------|-------------|
| `pricing/bs_model.py` | Black-Scholes analytical pricing, put-call parity verification, vectorised surface generation |
| `pricing/heston.py` | Heston stochastic volatility model — Monte Carlo + semi-analytical |
| `pricing/greeks.py` | Full Greeks: Δ, Γ, Θ, V, ρ with surface heatmaps |
| `pricing/iv_solver.py` | Implied volatility solver (bisection + Newton-Raphson) |
| `ml/garch_baseline.py` | GARCH(1,1) volatility forecasting baseline |
| `ml/lstm_vol.py` | LSTM architecture for vol surface forecasting |
| `dashboard/app.py` | Streamlit interactive dashboard |

---

## Methodology

### Black-Scholes Model

```
C = S·N(d₁) - K·e^(-rT)·N(d₂)

d₁ = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
d₂ = d₁ - σ·√T
```

- `S` = spot price, `K` = strike, `T` = time to expiry (years), `r` = risk-free rate, `σ` = annualised volatility

### Heston Stochastic Volatility

Allows volatility to follow its own mean-reverting process, capturing the vol smile that flat Black-Scholes misses:

```
dS = μS dt + √v · S dW₁
dv = κ(θ - v) dt + ξ√v dW₂
```

### LSTM Volatility Forecast

LSTM trained on rolling historical vol features to forecast the implied vol surface. Compared against GARCH(1,1) baseline using RMSE and MAE.

---

## Results

| Metric | Value |
|--------|-------|
| B-S vs market price MAE | TBD |
| GARCH forecast RMSE | TBD |
| LSTM forecast RMSE | TBD |
| LSTM vs GARCH improvement | TBD |

---

## Output Samples

![Day 1 — BS Pricing Surface](outputs/day1_bs_pricing_surface.png)

---

## Project Structure

```
options-pricing-engine/
├── pricing/
│   ├── bs_model.py        # Black-Scholes pricing engine
│   ├── heston.py          # Heston stochastic vol model
│   ├── greeks.py          # Greeks computation & visualisation
│   └── iv_solver.py       # Implied volatility solver
├── ml/
│   ├── garch_baseline.py  # GARCH(1,1) baseline
│   ├── lstm_vol.py        # LSTM volatility forecaster
│   └── feature_eng.py     # Feature engineering pipeline
├── dashboard/
│   └── app.py             # Streamlit interactive dashboard
├── tests/
│   ├── test_bs_model.py   # 25 unit tests
│   └── test_greeks.py     # Greeks accuracy tests
├── outputs/               # Charts and plots
├── models/                # Saved LSTM checkpoints
├── data/                  # Raw data cache
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yuvrajsingh1097/options-pricing-engine
cd options-pricing-engine
pip install -r requirements.txt
```

```bash
# Run pricing demo
python pricing/bs_model.py

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Testing

```
25 passed in 0.99s
```

Covers: known-value accuracy, put-call parity, Greeks monotonicity, validation edge cases, vectorised surface shape.

---

## License

MIT
