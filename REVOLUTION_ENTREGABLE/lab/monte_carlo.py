import numpy as np
import pandas as pd

def simulate_price_paths(
    current_price: float,
    historical_returns: pd.Series,
    horizon_days: int = 5,
    n_simulations: int = 10_000
) -> dict:
    """
    Modelo GBM (Geometric Brownian Motion) parametrizado con
    volatilidad y drift reales del ticker.
    """
    historical_returns = historical_returns.dropna()
    
    if len(historical_returns) < 10:
        return {
            "prob_positive": 0.0, "prob_gt_2pct": 0.0, "prob_gt_5pct": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "sigma_anual": 0.0
        }

    mu = historical_returns.mean()
    sigma = historical_returns.std()

    dt = 1
    random_shocks = np.random.normal(
        (mu - 0.5 * sigma**2) * dt,
        sigma * np.sqrt(dt),
        (n_simulations, horizon_days)
    )
    price_paths = current_price * np.exp(np.cumsum(random_shocks, axis=1))
    final_prices = price_paths[:, -1]

    returns = (final_prices - current_price) / current_price

    return {
        "prob_positive":  float((returns > 0).mean()),
        "prob_gt_2pct":   float((returns > 0.02).mean()),
        "prob_gt_5pct":   float((returns > 0.05).mean()),
        "p10": float(np.percentile(returns, 10)),
        "p50": float(np.percentile(returns, 50)),
        "p90": float(np.percentile(returns, 90)),
        "sigma_anual": float(sigma * np.sqrt(252)),
    }
