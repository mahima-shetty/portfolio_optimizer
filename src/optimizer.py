
import pandas as pd
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import sample_cov
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def calculate_optimal_portfolio(prices_df, risk_free_rate):
    """
    Calculates optimal portfolio weights using Modern Portfolio Theory.

    Args:
        prices_df (pd.DataFrame): Historical adjusted close prices of assets
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%)

    Returns:
        dict: Optimal weights
        float: Expected annual return
        float: Annual volatility
        float: Sharpe ratio
    """
    # 1. Expected returns and risk
    mu = mean_historical_return(prices_df)
    S = sample_cov(prices_df)

    # 2. Optimization: maximize Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
    cleaned_weights = ef.clean_weights()

    performance = ef.portfolio_performance(verbose=False)
    expected_return, volatility, sharpe_ratio = performance

    return cleaned_weights, expected_return, volatility, sharpe_ratio


def get_discrete_allocation(prices_df, weights, total_portfolio_value):
    """
    Converts fractional weights into actual stock allocations.

    Args:
        prices_df (pd.DataFrame): Historical price data
        weights (dict): Optimal portfolio weights
        total_portfolio_value (float): Total capital to allocate

    Returns:
        dict: Allocation of shares per ticker
        float: Remaining unallocated funds
    """
    latest_prices = get_latest_prices(prices_df)
    da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.lp_portfolio()
    return allocation, leftover
