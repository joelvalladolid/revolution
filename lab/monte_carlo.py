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
            "prob_positive": 0.0, "prob_gt_1pct": 0.0, "prob_gt_2pct": 0.0, "prob_gt_5pct": 0.0,
            "p10": 0.0, "p50": 0.0, "p90": 0.0, "sigma_anual": 0.0
        }

    mu = historical_returns.mean()
    sigma = historical_returns.std()

    # Simulación de caminos usando GBM
    # S_t = S_0 * exp((mu - 0.5 * sigma^2)*t + sigma * W_t)
    drift = mu - 0.5 * sigma**2
    random_shocks = np.random.normal(0, sigma, (horizon_days, n_simulations))
    log_returns = drift + random_shocks
    cumulative_returns = np.sum(log_returns, axis=0)
    final_prices = current_price * np.exp(cumulative_returns)

    returns = (final_prices - current_price) / current_price

    return {
        "prob_positive":  float((returns > 0).mean()),
        "prob_gt_1pct":   float((returns > 0.01).mean()),
        "prob_gt_2pct":   float((returns > 0.02).mean()),
        "prob_gt_5pct":   float((returns > 0.05).mean()),
        "p10": float(np.percentile(returns, 10)),
        "p50": float(np.percentile(returns, 50)),
        "p90": float(np.percentile(returns, 90)),
        "sigma_anual": float(sigma * np.sqrt(252)),
    }
