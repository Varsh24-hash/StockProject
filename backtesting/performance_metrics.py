import numpy as np

# =========================
# 1. Total Return
# =========================
def total_return(portfolio_values):
    """
    Total return of the portfolio.
    """
    return (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]


# =========================
# 2. Sharpe Ratio
# =========================
def sharpe_ratio(portfolio_values):
    """
    Sharpe Ratio - measures risk-adjusted returns.
    """
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)


# =========================
# 3. Max Drawdown
# =========================
def max_drawdown(portfolio_values):
    """
    Max Drawdown - measures the largest peak-to-trough drawdown.
    """
    portfolio_values = np.array(portfolio_values)
    peak = np.argmax(np.maximum.accumulate(portfolio_values))
    trough = np.argmin(portfolio_values[:peak])
    return (portfolio_values[peak] - portfolio_values[trough]) / portfolio_values[peak]


# =========================
# 4. Sortino Ratio
# =========================
def sortino_ratio(portfolio_values):
    """
    Sortino Ratio - similar to Sharpe ratio, but penalizes only downside volatility.
    """
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = np.std(downside_returns)
    return np.mean(daily_returns) / (downside_volatility + 1e-8)


# =========================
# 5. Annualized Return
# =========================
def annualized_return(portfolio_values, time_periods=252):
    """
    Converts total return to annualized return based on the time period.
    """
    total_ret = total_return(portfolio_values)
    return (1 + total_ret) ** (time_periods / len(portfolio_values)) - 1


# =========================
# 6. Performance Summary
# =========================
def performance_summary(portfolio_values):
    """
    Prints out a summary of performance metrics.
    """
    print(f"Total Return: {total_return(portfolio_values):.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio(portfolio_values):.4f}")
    print(f"Max Drawdown: {max_drawdown(portfolio_values):.4f}")
    print(f"Sortino Ratio: {sortino_ratio(portfolio_values):.4f}")
    print(f"Annualized Return: {annualized_return(portfolio_values):.4f}")