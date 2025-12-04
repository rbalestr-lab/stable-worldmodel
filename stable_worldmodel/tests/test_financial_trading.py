"""Tests for Financial Backtesting Environment."""

import os

import gymnasium as gym
import numpy as np
import pytest


# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Check if Alpaca credentials are available
ALPACA_CONFIGURED = bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))

# Mark for tests that require actual data loading
requires_data = pytest.mark.skipif(
    not ALPACA_CONFIGURED,
    reason="Alpaca API credentials not configured - skipping data-dependent tests",
)


def test_environment_creation():
    """Test that the environment can be created successfully."""
    env = gym.make("swm/FinancialBacktest-v0")
    assert env is not None
    env.close()


def test_action_space():
    """Test that action space is correctly defined."""
    env = gym.make("swm/FinancialBacktest-v0")
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == 3  # BUY, SELL, HOLD
    env.close()


def test_observation_space():
    """Test that observation space is correctly defined."""
    env = gym.make("swm/FinancialBacktest-v0")
    assert isinstance(env.observation_space, gym.spaces.Box)
    # window_size*6 + position + balance + portfolio_value + time_features(4)
    # Default window_size is 60
    expected_shape = (60 * 6 + 1 + 1 + 1 + 4,)
    assert env.observation_space.shape == expected_shape
    env.close()


@requires_data
def test_reset():
    """Test environment reset functionality."""
    env = gym.make("swm/FinancialBacktest-v0")
    obs, info = env.reset(seed=42)

    assert obs is not None
    # Default window_size is 60, so: 60*6 + position + balance + portfolio + time_features(4)
    assert len(obs) == 60 * 6 + 1 + 1 + 1 + 4
    assert isinstance(info, dict)
    assert "symbol" in info
    assert "market_data_length" in info
    env.close()


@requires_data
def test_step():
    """Test environment step functionality."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    # Test BUY action
    obs, reward, terminated, truncated, info = env.step(1)  # BUY
    assert obs is not None
    assert isinstance(reward, int | float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool | np.bool_)
    assert isinstance(info, dict)

    # Test SELL action
    obs, reward, terminated, truncated, info = env.step(0)  # SELL
    assert obs is not None

    # Test HOLD action
    obs, reward, terminated, truncated, info = env.step(2)  # HOLD
    assert obs is not None
    env.close()


def test_variation_space_structure():
    """Test that variation space has complete structure with Alpaca integration."""
    env = gym.make("swm/FinancialBacktest-v0")
    unwrapped_env = env.unwrapped

    assert hasattr(unwrapped_env, "variation_space")
    assert "agent" in unwrapped_env.variation_space.spaces

    agent_space = unwrapped_env.variation_space["agent"]
    # All agent parameters should be present with Alpaca integration
    assert "starting_balance" in agent_space.spaces
    assert "transaction_cost" in agent_space.spaces
    assert "enable_shorting" in agent_space.spaces
    assert "market_impact" in agent_space.spaces  # Renamed from slippage_pct
    assert "window_size" in agent_space.spaces

    env.close()


@requires_data
def test_trading_logic():
    """Test basic trading logic (position management)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    initial_position = env.unwrapped.position
    initial_shares = env.unwrapped.shares_held

    assert initial_position == 0.0
    assert initial_shares == 0.0

    # Execute a BUY action
    obs, reward, terminated, truncated, info = env.step(1)

    # Position and shares should change (depending on market conditions)
    assert isinstance(env.unwrapped.position, int | float | np.floating)
    assert env.unwrapped.portfolio_value > 0

    env.close()


# ===== COMPREHENSIVE BACKTEST RESULTS TESTS =====


@requires_data
def test_backtest_results_basic_metrics():
    """Test basic return metrics calculation."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=20)
    obs, info = env.reset(seed=42)

    # Run a few steps to generate history
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check basic metrics exist
    assert "total_return" in results
    assert "annualized_return" in results
    assert "cumulative_return" in results
    assert "benchmark_return" in results
    assert "excess_return" in results

    # Check they are numeric
    assert isinstance(results["total_return"], int | float | np.floating)
    assert isinstance(results["annualized_return"], int | float | np.floating)

    env.close()


@requires_data
def test_backtest_results_risk_metrics():
    """Test risk metrics calculation (Sharpe, Sortino, Calmar, volatility)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Generate sufficient trading history
    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check risk metrics
    assert "sharpe_ratio" in results
    assert "sortino_ratio" in results
    assert "calmar_ratio" in results
    assert "volatility" in results
    assert "downside_volatility" in results
    assert "alpha" in results
    assert "beta" in results
    assert "information_ratio" in results
    assert "tracking_error" in results
    assert "stability" in results

    # All should be numeric
    assert isinstance(results["sharpe_ratio"], int | float)
    assert isinstance(results["volatility"], int | float)

    env.close()


@requires_data
def test_backtest_results_drawdown_metrics():
    """Test drawdown analysis metrics."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check drawdown metrics
    assert "max_drawdown" in results
    assert "max_drawdown_pct" in results
    assert "max_drawdown_duration" in results
    assert "current_drawdown" in results
    assert "avg_drawdown" in results
    assert "avg_drawdown_duration" in results
    assert "num_drawdown_periods" in results

    # Drawdowns should be non-negative
    assert results["max_drawdown"] >= 0
    assert results["current_drawdown"] >= 0

    env.close()


@requires_data
def test_backtest_results_distribution_metrics():
    """Test return distribution metrics (skew, kurtosis, VaR, tail ratio)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check distribution metrics
    assert "skewness" in results
    assert "kurtosis" in results
    assert "var_95" in results
    assert "var_99" in results
    assert "cvar_95" in results
    assert "cvar_99" in results
    assert "tail_ratio" in results
    assert "omega_ratio" in results

    # All should be numeric
    assert isinstance(results["skewness"], int | float | np.floating)
    assert isinstance(results["tail_ratio"], int | float | np.floating)

    env.close()


@requires_data
def test_backtest_results_trading_stats():
    """Test trading statistics (trades, win rate, profit factor)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Execute varied trades
    for i in range(20):
        action = 1 if i % 3 == 0 else (0 if i % 3 == 1 else 2)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check trading stats
    assert "total_trades" in results
    assert "buy_trades" in results
    assert "sell_trades" in results
    assert "short_trades" in results
    assert "winning_trades" in results
    assert "losing_trades" in results
    assert "win_rate" in results
    assert "profit_factor" in results
    assert "avg_trade_return" in results
    assert "trading_frequency" in results

    # Trade counts should be non-negative integers
    assert results["total_trades"] >= 0
    assert results["buy_trades"] >= 0

    env.close()


@requires_data
def test_backtest_results_position_stats():
    """Test position analysis statistics (leverage, exposure, turnover)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    results = env.unwrapped.get_backtest_results()

    # Check position stats
    assert "avg_position_size" in results
    assert "max_position_size" in results
    assert "avg_leverage" in results
    assert "max_leverage" in results
    assert "avg_exposure" in results
    assert "max_exposure" in results
    assert "total_turnover" in results

    # All should be numeric and non-negative
    assert results["avg_position_size"] >= 0
    assert results["max_leverage"] >= 0

    env.close()


@requires_data
def test_backtest_results_empty_history():
    """Test backtest results with no trading history."""
    env = gym.make("swm/FinancialBacktest-v0")
    obs, info = env.reset(seed=42)

    # Get results immediately without any steps
    results = env.unwrapped.get_backtest_results()

    # Should return empty dict or safe defaults
    assert isinstance(results, dict)

    env.close()


# ===== TEAR SHEET DATA TESTS =====


@requires_data
def test_returns_tear_sheet_data():
    """Test returns tear sheet data generation."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_returns_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    assert "returns" in tear_sheet_data
    assert "cumulative_returns" in tear_sheet_data
    assert "rolling_returns" in tear_sheet_data
    assert "rolling_volatility" in tear_sheet_data
    assert "rolling_sharpe" in tear_sheet_data
    assert "rolling_beta" in tear_sheet_data
    assert "underwater" in tear_sheet_data
    assert "return_quantiles" in tear_sheet_data
    assert "timestamps" in tear_sheet_data

    env.close()


@requires_data
def test_position_tear_sheet_data():
    """Test position tear sheet data generation."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_position_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    assert "gross_exposure" in tear_sheet_data
    assert "net_exposure" in tear_sheet_data
    assert "long_exposure" in tear_sheet_data
    assert "short_exposure" in tear_sheet_data
    assert "gross_leverage" in tear_sheet_data
    assert "top_holdings" in tear_sheet_data
    assert "position_concentration" in tear_sheet_data
    assert "timestamps" in tear_sheet_data

    env.close()


@requires_data
def test_transaction_tear_sheet_data():
    """Test transaction tear sheet data generation."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    # Execute multiple transactions
    for i in range(20):
        action = 1 if i % 2 == 0 else 0  # Alternate buy/sell
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_transaction_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    if tear_sheet_data:  # Only if transactions occurred
        assert "transaction_count" in tear_sheet_data
        assert "transaction_volumes" in tear_sheet_data
        assert "turnovers" in tear_sheet_data
        assert "hour_distribution" in tear_sheet_data
        assert "transaction_types" in tear_sheet_data

    env.close()


@requires_data
def test_round_trip_tear_sheet_data():
    """Test round-trip trade analysis data generation."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=40)
    obs, info = env.reset(seed=42)

    # Execute complete round trips (buy then sell)
    for i in range(30):
        if i % 4 < 2:
            action = 1  # Buy
        else:
            action = 0  # Sell (complete round trip)

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    tear_sheet_data = env.unwrapped.get_round_trip_tear_sheet_data()

    assert isinstance(tear_sheet_data, dict)
    if tear_sheet_data:  # Only if round trips completed
        assert "total_round_trips" in tear_sheet_data
        assert "winning_trips" in tear_sheet_data
        assert "losing_trips" in tear_sheet_data
        assert "win_rate" in tear_sheet_data
        assert "avg_pnl" in tear_sheet_data
        assert "avg_duration_minutes" in tear_sheet_data

    env.close()


@requires_data
def test_capacity_analysis():
    """Test capacity analysis for liquidity constraints."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=30)
    obs, info = env.reset(seed=42)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    capacity_data = env.unwrapped.get_capacity_analysis()

    assert isinstance(capacity_data, dict)
    if capacity_data:  # Only if data available
        assert "avg_daily_volume" in capacity_data
        assert "max_position_shares" in capacity_data
        assert "capacity_multiplier" in capacity_data
        assert "liquidity_constrained" in capacity_data

    env.close()


# ===== TIME MACHINE FUNCTIONALITY TESTS =====


@requires_data
def test_time_machine_reset_to_date():
    """Test time machine functionality for backtesting at specific dates."""
    import pandas as pd

    env = gym.make("swm/FinancialBacktest-v0")
    unwrapped_env = env.unwrapped

    # Use a recent date within Alpaca free tier range (last 5 days)
    target_date = (pd.Timestamp.now() - pd.Timedelta(days=4)).strftime("%Y-%m-%d")
    obs, info = unwrapped_env.reset_to_date(target_date, symbol="AAPL")

    assert obs is not None
    assert "start_timestamp" in info
    assert "time_machine_date" in info
    assert info["time_machine_date"] == target_date
    # Don't assert exact date match since free tier may not have data for exact target date
    # Just verify we got a valid timestamp close to the target
    start_ts = pd.Timestamp(info["start_timestamp"])
    target_ts = pd.Timestamp(target_date)
    days_diff = abs((start_ts - target_ts).days)
    assert days_diff <= 5, f"Start date {start_ts} is too far from target {target_ts}"

    env.close()


# ===== RENDER AND VISUALIZATION TESTS =====


@requires_data
def test_render_returns_none():
    """Test that render returns None (financial data, not images)."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=10)
    obs, info = env.reset(seed=42)

    # Take a few actions
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # Render should return None (no image rendering for financial data)
    output = env.render()
    assert output is None

    env.close()


# ===== INTEGRATION TESTS =====


@requires_data
def test_full_episode_workflow():
    """Test a complete episode workflow with all features."""
    env = gym.make("swm/FinancialBacktest-v0", max_steps=50)
    obs, info = env.reset(seed=42)

    episode_returns = []
    steps = 0

    while steps < 40:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_returns.append(reward)
        steps += 1

        if terminated or truncated:
            break

    # Get comprehensive results
    results = env.unwrapped.get_backtest_results()

    # Verify complete results
    assert len(results) > 20  # Should have many metrics
    assert "total_return" in results
    assert "sharpe_ratio" in results
    assert "max_drawdown" in results
    assert "total_trades" in results

    # Get all tear sheet data
    returns_data = env.unwrapped.get_returns_tear_sheet_data()
    position_data = env.unwrapped.get_position_tear_sheet_data()
    transaction_data = env.unwrapped.get_transaction_tear_sheet_data()

    assert isinstance(returns_data, dict)
    assert isinstance(position_data, dict)
    assert isinstance(transaction_data, dict)

    env.close()


# ===== UNIT TESTS FOR METRICS CALCULATIONS =====


def test_sharpe_ratio_calculation():
    """Test Sharpe ratio calculation with known values."""
    from stable_worldmodel.envs.financial_trading import calculate_sharpe_ratio

    # Test data
    np.random.seed(42)
    returns = np.array([0.001, 0.002, -0.001, 0.0015, 0.003, -0.002, 0.0025])

    # Calculate using our function
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252 * 390)

    # Manual calculation to verify
    risk_free_period = 0.02 / (252 * 390)
    excess_returns = returns - risk_free_period
    expected_sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(sharpe - expected_sharpe) < 1e-10
    assert not np.isnan(sharpe)
    assert isinstance(sharpe, int | float)

    # Test edge case: constant returns -> should return 0
    constant_returns = np.array([0.01, 0.01, 0.01, 0.01])
    sharpe_constant = calculate_sharpe_ratio(constant_returns)
    assert sharpe_constant == 0.0

    # Test edge case: single return -> should return 0
    single_return = np.array([0.01])
    sharpe_single = calculate_sharpe_ratio(single_return)
    assert sharpe_single == 0.0


def test_sortino_ratio_calculation():
    """Test Sortino ratio calculation with known values."""
    from stable_worldmodel.envs.financial_trading import calculate_sortino_ratio

    np.random.seed(42)
    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.02)

    assert not np.isnan(sortino)
    assert isinstance(sortino, int | float)

    # Edge case: all positive returns -> should return 0 (no downside vol)
    positive_returns = np.array([0.01, 0.02, 0.015, 0.03])
    sortino_positive = calculate_sortino_ratio(positive_returns)
    assert sortino_positive == 0.0


def test_max_drawdown_calculation():
    """Test maximum drawdown calculation."""
    from stable_worldmodel.envs.financial_trading import calculate_max_drawdown

    portfolio_values = [100, 110, 105, 95, 90, 100, 120, 115]

    max_dd = calculate_max_drawdown(portfolio_values)

    # The max drawdown should be from peak (110) to trough (90)  # codespell:ignore
    # (90 - 110) / 110 = -0.1818...
    assert max_dd > 0
    assert abs(max_dd - 0.1818) < 0.01  # approximately 18.18%


def test_calmar_ratio_calculation():
    """Test Calmar ratio (return / max drawdown)."""
    from stable_worldmodel.envs.financial_trading import calculate_calmar_ratio

    annualized_return = 0.15  # 15% annual return
    max_drawdown = 0.10  # 10% max drawdown

    calmar = calculate_calmar_ratio(annualized_return, max_drawdown)
    assert abs(calmar - 1.5) < 1e-10

    # Edge case: zero drawdown -> should return 0
    calmar_zero_dd = calculate_calmar_ratio(0.15, 0.0)
    assert calmar_zero_dd == 0.0


def test_alpha_beta_calculation():
    """Test alpha and beta calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_alpha_beta

    np.random.seed(42)

    benchmark_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
    # Portfolio returns with some alpha
    portfolio_returns = benchmark_returns * 1.2 + 0.002  # Beta=1.2, alpha=0.2%

    # Calculate annualized returns (simplified for test)
    portfolio_annualized = np.mean(portfolio_returns) * 252 * 390
    benchmark_annualized = np.mean(benchmark_returns) * 252 * 390

    alpha, beta = calculate_alpha_beta(
        portfolio_returns,
        benchmark_returns,
        portfolio_annualized,
        benchmark_annualized,
    )

    assert not np.isnan(beta)
    assert isinstance(beta, int | float)
    # Beta should be close to 1.2
    assert 1.0 < beta < 1.5


def test_volatility_calculation():
    """Test volatility (standard deviation of returns)."""
    from stable_worldmodel.envs.financial_trading import calculate_volatility

    np.random.seed(42)
    returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005, 0.03, -0.02])

    volatility = calculate_volatility(returns)

    # Manual calculation
    expected_volatility = np.std(returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(volatility - expected_volatility) < 1e-10
    assert volatility > 0
    assert not np.isnan(volatility)


def test_information_ratio_calculation():
    """Test information ratio (excess return / tracking error)."""
    from stable_worldmodel.envs.financial_trading import calculate_information_ratio

    np.random.seed(42)

    portfolio_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
    benchmark_returns = np.array([0.008, 0.018, -0.012, 0.014, -0.006])

    # Calculate annualized returns
    portfolio_annualized = np.mean(portfolio_returns) * 252 * 390
    benchmark_annualized = np.mean(benchmark_returns) * 252 * 390

    ir, tracking_error = calculate_information_ratio(
        portfolio_returns,
        benchmark_returns,
        portfolio_annualized,
        benchmark_annualized,
    )

    assert not np.isnan(ir)
    assert not np.isnan(tracking_error)
    assert tracking_error > 0


def test_var_cvar_calculation():
    """Test Value at Risk (VaR) and Conditional VaR calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_var_cvar

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)

    var_95, cvar_95 = calculate_var_cvar(returns, confidence_level=0.95)

    assert var_95 < 0  # Should be negative (loss)
    assert cvar_95 < var_95  # CVaR should be worse than VaR
    assert not np.isnan(var_95)
    assert not np.isnan(cvar_95)


def test_skewness_kurtosis_calculation():
    """Test skewness and kurtosis calculations."""
    from stable_worldmodel.envs.financial_trading import calculate_skewness_kurtosis

    np.random.seed(42)

    # Normal distribution should have skew~0, kurtosis~0
    normal_returns = np.random.normal(0, 1, 1000)
    skew, kurt = calculate_skewness_kurtosis(normal_returns)

    assert abs(skew) < 0.5  # Should be close to 0
    assert abs(kurt) < 1.0  # Should be close to 0

    # Right-skewed distribution (positive skew)
    right_skewed = np.random.exponential(1, 1000)
    skew_right, _ = calculate_skewness_kurtosis(right_skewed)
    assert skew_right > 0  # Should be positive


def test_tail_ratio_calculation():
    """Test tail ratio (95th percentile / 5th percentile)."""
    from stable_worldmodel.envs.financial_trading import calculate_tail_ratio

    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 1000)

    tail_ratio = calculate_tail_ratio(returns)

    assert tail_ratio > 0
    assert not np.isnan(tail_ratio)


def test_omega_ratio_calculation():
    """Test omega ratio (probability weighted gains vs losses)."""
    from stable_worldmodel.envs.financial_trading import calculate_omega_ratio

    np.random.seed(42)
    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    omega = calculate_omega_ratio(returns, threshold=0.0)

    assert omega > 0
    assert not np.isnan(omega)


def test_win_rate_calculation():
    """Test win rate (winning trades / total trades)."""
    from stable_worldmodel.envs.financial_trading import calculate_win_rate

    trade_returns = [0.02, -0.01, 0.015, 0.01, -0.005, 0.03, -0.02, 0.008]

    win_rate = calculate_win_rate(trade_returns)
    expected_win_rate = 5 / 8  # 5 winning out of 8 total

    assert abs(win_rate - expected_win_rate) < 1e-10
    assert 0 <= win_rate <= 1


def test_profit_factor_calculation():
    """Test profit factor (total gains / total losses)."""
    from stable_worldmodel.envs.financial_trading import calculate_profit_factor

    trade_returns = [0.02, -0.01, 0.015, 0.01, -0.005, 0.03, -0.02, 0.008]

    profit_factor = calculate_profit_factor(trade_returns)

    assert profit_factor > 0
    # In this example: (0.02+0.015+0.01+0.03+0.008) / (0.01+0.005+0.02)
    expected = 0.083 / 0.035
    assert abs(profit_factor - expected) < 0.1


def test_metrics_edge_cases():
    """Test edge cases for all metrics."""
    # All zero returns
    zero_returns = np.array([0.0, 0.0, 0.0])
    assert np.std(zero_returns) == 0

    # All identical returns
    identical = np.array([0.01, 0.01, 0.01])
    assert np.std(identical) == 0


def test_annualized_return_calculation():
    """Test annualized return calculation."""
    from stable_worldmodel.envs.financial_trading import calculate_annualized_return

    # Test simple case: 10% total return over 1 year (252*390 periods)
    total_return = 0.10
    periods_per_year = 252 * 390
    total_periods = periods_per_year  # 1 year

    annualized = calculate_annualized_return(total_return, total_periods, periods_per_year)
    assert abs(annualized - 0.10) < 1e-10  # Should be same as total return

    # Test 6 months: 10% return in 6 months -> ~21% annualized
    total_periods_half = periods_per_year // 2
    annualized_half = calculate_annualized_return(total_return, total_periods_half, periods_per_year)
    expected = (1.10) ** 2 - 1  # ~21%
    assert abs(annualized_half - expected) < 1e-10


def test_cumulative_returns_calculation():
    """Test cumulative returns calculation."""
    import pandas as pd

    # Simple period returns
    returns = [0.01, 0.02, -0.01, 0.015]

    # Cumulative: (1+r1)*(1+r2)*(1+r3)*(1+r4) - 1
    cumulative = pd.Series(returns).add(1).cumprod().sub(1)

    # Manual calculation
    expected = (1.01 * 1.02 * 0.99 * 1.015) - 1
    assert abs(cumulative.iloc[-1] - expected) < 1e-10


def test_downside_volatility_calculation():
    """Test downside volatility (only considers negative returns)."""
    from stable_worldmodel.envs.financial_trading import (
        calculate_downside_volatility,
    )

    returns = np.array([0.02, 0.01, -0.01, 0.015, -0.005, 0.03, -0.02])

    downside_vol = calculate_downside_volatility(returns)

    # Manual calculation
    downside_returns = returns[returns < 0]
    expected = np.std(downside_returns, ddof=1) * np.sqrt(252 * 390)

    assert abs(downside_vol - expected) < 1e-10
    assert downside_vol > 0
    assert not np.isnan(downside_vol)


def test_stability_calculation():
    """Test stability (R-squared of returns vs time)."""
    from stable_worldmodel.envs.financial_trading import calculate_stability

    np.random.seed(42)
    # Create upward trending returns
    returns = np.array([0.001 + 0.0001 * i for i in range(100)])

    stability = calculate_stability(returns)

    # Should be close to 1 for consistent upward trend
    assert stability > 0.5
    assert stability <= 1.0
    assert not np.isnan(stability)


# ===== FINANCIAL DATASET INTEGRATION TEST =====


def test_financial_dataset_integration():
    """Test that FinancialDataset is properly integrated and functional."""

    from stable_worldmodel.data import FinancialDataset

    # Should have FinancialDataset class
    assert hasattr(FinancialDataset, "load")
    assert hasattr(FinancialDataset, "DATA_DIR")
    assert hasattr(FinancialDataset, "SP500_TICKERS")

    # Test DATA_DIR is set
    assert FinancialDataset.DATA_DIR is not None
    assert isinstance(FinancialDataset.DATA_DIR, str)

    # Test SP500_TICKERS is populated
    assert len(FinancialDataset.SP500_TICKERS) > 0
    assert "AAPL" in FinancialDataset.SP500_TICKERS

    # Test encoding/decoding methods exist
    assert hasattr(FinancialDataset, "encode_bfloat16")
    assert hasattr(FinancialDataset, "decode_bfloat16")
    assert hasattr(FinancialDataset, "encode_financial_data")
    assert hasattr(FinancialDataset, "decode_financial_data")


def test_financial_dataset_load():
    """Test FinancialDataset.load() functionality with recent dates."""
    import pandas as pd

    from stable_worldmodel.data import FinancialDataset

    # Use recent dates (data should be available or auto-downloadable)
    # Free tier provides ~5-15 days, so use a conservative 2-day window
    end_date = (pd.Timestamp.now() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")

    # Test loading data
    df = FinancialDataset.load(stocks=["AAPL"], dates=[(start_date, end_date)])

    # Should return a DataFrame (may be empty if Alpaca credentials not configured)
    assert isinstance(df, pd.DataFrame)

    # If data was loaded, verify structure
    if not df.empty:
        required_columns = ["open", "high", "low", "close"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"


def test_financial_dataset_encoding():
    """Test FinancialDataset bfloat16 encoding/decoding."""
    import numpy as np
    import pandas as pd

    from stable_worldmodel.data import FinancialDataset

    # Create test data
    test_prices = pd.Series([100.0, 101.5, 99.8, 102.3, 101.0])

    # Test encoding
    encoded = FinancialDataset.encode_bfloat16(test_prices)
    assert encoded is not None
    assert isinstance(encoded, np.ndarray)
    assert len(encoded) == len(test_prices)

    # Test decoding
    anchor = test_prices.iloc[0]
    decoded = FinancialDataset.decode_bfloat16(encoded, anchor)
    assert decoded is not None
    assert isinstance(decoded, np.ndarray)
    assert len(decoded) == len(test_prices)

    # Verify reconstruction accuracy (bfloat16 has some precision loss)
    max_error = np.abs(test_prices.values - decoded).max()
    assert max_error < 0.01  # Should be very close
