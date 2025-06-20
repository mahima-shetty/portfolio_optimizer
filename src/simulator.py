
import numpy as np
import pandas as pd

def run_monte_carlo_simulation(prices_df, num_portfolios=10000, risk_free_rate=0.02, seed=42):
    """
    Runs Monte Carlo simulation to generate random portfolios.

    Args:
        prices_df (pd.DataFrame): Adjusted close prices
        num_portfolios (int): Number of random portfolios to simulate
        risk_free_rate (float): Annual risk-free rate for Sharpe ratio
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: Portfolio stats (return, volatility, Sharpe, weights)
        dict: Portfolio with max Sharpe ratio
        dict: Portfolio with min volatility
    """
    np.random.seed(seed)
    
    returns = prices_df.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    tickers = prices_df.columns.tolist()

    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)

        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0

        results.append({
            "Return": port_return,
            "Volatility": port_volatility,
            "SharpeRatio": sharpe,
            **{ticker: w for ticker, w in zip(tickers, weights)}
        })

    results_df = pd.DataFrame(results)

    max_sharpe_port = results_df.loc[results_df["SharpeRatio"].idxmax()].to_dict()
    min_vol_port = results_df.loc[results_df["Volatility"].idxmin()].to_dict()

    return results_df, max_sharpe_port, min_vol_port
