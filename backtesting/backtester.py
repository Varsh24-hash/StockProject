import numpy as np
import pandas as pd
import joblib
from ml_price_prediction.predict import predict
from ml_strategy_rl.trading_environment import TradingEnvRL
from ml_strategy_rl.rl_agent import RLAgent
from portfolio.position_sizing import allocate_cash, calculate_shares
from risk_management.risk_controls import apply_stop_loss, apply_position_limit
from risk_management.monte_carlo import monte_carlo_simulation, plot_simulation, compute_var, compute_cvar
import matplotlib.pyplot as plt
from backtesting.performance_metrics import performance_summary



# =========================
# 1. Load Data (Real Data)
# =========================
def load_data(ticker="AAPL"):
    df = pd.read_csv(f"data/processed/{ticker}_features.csv")
    model = joblib.load("models/xgb_model.pkl")
    df_pred = predict(model, df)

    prices = df_pred["close"].values
    predictions = df_pred["probability_up"].values

    return prices, predictions


# =========================
# 2. Backtest RL Strategy
# =========================
def backtest_rl(prices, predictions, episodes=10):
    env = TradingEnvRL(prices, predictions)
    agent = RLAgent(state_size=6, action_size=3)

    # Track portfolio value over time
    portfolio_values = []
    state = env.reset()

    for _ in range(len(prices) - 1):
        action = agent.act(state)  # Get action from RL agent
        next_state, reward, done, _ = env.step(action)

        portfolio_values.append(env.cash + env.shares * prices[env.current_step])

        if done:
            break

        state = next_state

    return np.array(portfolio_values)


# =========================
# 3. Backtest Buy and Hold Strategy
# =========================
def backtest_buy_and_hold(prices, initial_cash=10000):
    shares_to_buy = initial_cash // prices[0]
    portfolio_values = [initial_cash + shares_to_buy * price for price in prices]
    return np.array(portfolio_values)


# =========================
# 4. Performance Metrics
# =========================
def compute_performance_metrics(portfolio_values):
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)

    return total_return, sharpe_ratio


# =========================
# 5. Plot Results
# =========================
def plot_results(rl_portfolio, bh_portfolio):
    plt.figure(figsize=(10, 6))

    plt.plot(rl_portfolio, label="RL Strategy", alpha=0.7)
    plt.plot(bh_portfolio, label="Buy & Hold", alpha=0.7)

    plt.title("Backtest Comparison: RL Strategy vs Buy & Hold")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid()

    plt.show()


# =========================
# 6. Monte Carlo Simulation for Risk Analysis
# =========================
def monte_carlo_analysis(returns, num_simulations=1000, horizon=50):
    simulations = monte_carlo_simulation(returns, num_simulations, horizon)
    plot_simulation(simulations)

    var = compute_var(simulations)
    cvar = compute_cvar(simulations)

    print(f"VaR (95%): {var:.2f}")
    print(f"CVaR (95%): {cvar:.2f}")


# =========================
# 7. Main Backtest Function
# =========================
if __name__ == "__main__":

    # 1. Load data (AAPL example)
    prices, predictions = load_data("AAPL")

    # 2. Backtest RL strategy
    rl_portfolio = backtest_rl(prices, predictions, episodes=10)

    # 3. Backtest Buy & Hold strategy
    bh_portfolio = backtest_buy_and_hold(prices)

    # 4. Compute performance metrics
    rl_return, rl_sharpe = compute_performance_metrics(rl_portfolio)
    bh_return, bh_sharpe = compute_performance_metrics(bh_portfolio)

    # 5. Display results
    print(f"RL Strategy: Total Return = {rl_return:.4f}, Sharpe Ratio = {rl_sharpe:.4f}")
    print(f"Buy & Hold: Total Return = {bh_return:.4f}, Sharpe Ratio = {bh_sharpe:.4f}")

    # 6. Plot results
    plot_results(rl_portfolio, bh_portfolio)

    # 7. Monte Carlo Simulation (Risk Analysis)
    returns = np.diff(rl_portfolio) / rl_portfolio[:-1]
    monte_carlo_analysis(returns)

    # Assuming rl_portfolio is the list of portfolio values over time
    performance_summary(rl_portfolio)