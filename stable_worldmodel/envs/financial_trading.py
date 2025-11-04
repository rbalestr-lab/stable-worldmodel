from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd

import stable_worldmodel as swm


# TODO: Integrate financial data visualization system instead of matplotlib
# Financial environments need specialized charting/metrics displays, not RGB images


DEFAULT_VARIATIONS = (
    "backtest.start_date",
    "backtest.symbol_selection",
    "agent.starting_balance",
)


class FinancialBacktestEnv(gym.Env):
    """
    A comprehensive financial backtesting environment integrated
    for out-of-distribution testing across different market conditions and time periods.

    This environment implements a "time machine" that can go back to any specified date
    and replay historical market data as if it were live. It integrates with the
    high-performance data pipeline for reading minute-level financial data stored
    in compressed parquet format.

    Key Features:
    - Historical backtesting with minute-level precision
    - Support for multiple symbols and market conditions
    - Configurable transaction costs and market impact
    - Out-of-distribution testing through date/symbol/market variations
    - Integration with stable-worldmodel's variation system
    - High-performance data loading from parquet/zarr storage
    - Concurrent data reading for multiple symbols

    TODO: Integrate financial data output format instead of image rendering
    Financial environments should output structured metrics, not visual charts

    Action Space:
    - 0: SELL/SHORT position
    - 1: BUY/LONG position
    - 2: HOLD (maintain current position)

    Observation Space:
    - Historical price features (OHLCV + technical indicators)
    - Position information
    - Portfolio metrics
    - Market metadata
    """

    metadata = {
        # TODO: Integrate financial data output formats instead of render modes
        # Financial environments should output metrics, not images
    }
    reward_range = (-np.inf, np.inf)  # Financial rewards can be unbounded

    def __init__(
        self,
        data_dir: str | Path | None = None,
        window_size: int = 60,  # 1 hour of minute data
        max_steps: int = 1440,  # 1 day of minute data
        symbols: list | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        # TODO: Replace render_mode with financial data output format
        # render_mode: Optional[str] = None,
        enable_shorting: bool = True,
        transaction_cost: float = 0.001,
        market_impact: float = 0.0001,
        seed: int | None = None,
    ):
        super().__init__()

        # Environment configuration
        self.data_dir = Path(data_dir) if data_dir else Path.home() / ".cache" / "financial_data"
        self.window_size = window_size
        self.max_steps = max_steps
        # TODO: Replace render_mode with financial data output configuration
        # self.render_mode = render_mode
        self.enable_shorting = enable_shorting
        self.transaction_cost = transaction_cost
        self.market_impact = market_impact

        # Default symbols and date range
        self.default_symbols = symbols or ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        self.default_start_date = start_date or "2020-01-01"
        self.default_end_date = end_date or "2023-12-31"

        # Action space: 0=SELL/SHORT, 1=BUY/LONG, 2=HOLD
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: [price_features(window_size*6), position(1), balance(1), portfolio_value(1), time_features(4)]
        # Price features: OHLCV + volume_weighted_price per timestep
        obs_size = window_size * 6 + 1 + 1 + 1 + 4  # +4 for time features (hour, day_of_week, month, year_progress)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # Define possible choices for categorical spaces
        self.start_date_choices = [
            "2018-01-01",
            "2019-01-01",
            "2020-01-01",
            "2021-01-01",
            "2022-01-01",
            "2020-03-01",
            "2008-09-01",
            "2000-03-01",  # Include crisis periods
        ]
        self.end_date_choices = [
            "2019-12-31",
            "2020-12-31",
            "2021-12-31",
            "2022-12-31",
            "2023-12-31",
            "2020-06-01",
            "2009-03-01",
            "2002-12-31",
        ]
        self.symbol_selection_choices = [
            "tech_stocks",
            "blue_chip",
            "volatile_stocks",
            "single_stock",
            "sector_rotation",
        ]
        self.market_regime_choices = ["bull", "bear", "sideways", "volatile", "crisis"]
        # TODO: Integrate financial data output format choices instead of chart styles
        # self.output_format_choices = ["json", "dataframe", "structured", "metrics"]

        # Variation space for comprehensive backtesting scenarios
        self.variation_space = swm.spaces.Dict(
            {
                "backtest": swm.spaces.Dict(
                    {
                        "start_date": swm.spaces.Discrete(
                            n=len(self.start_date_choices),
                            init_value=2,  # "2020-01-01"
                        ),
                        "end_date": swm.spaces.Discrete(
                            n=len(self.end_date_choices),
                            init_value=2,  # "2021-12-31"
                        ),
                        "symbol_selection": swm.spaces.Discrete(
                            n=len(self.symbol_selection_choices),
                            init_value=0,  # "tech_stocks"
                        ),
                        "time_acceleration": swm.spaces.Box(
                            low=1.0,
                            high=60.0,  # Up to 1-hour intervals
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "market": swm.spaces.Dict(
                    {
                        "regime": swm.spaces.Discrete(
                            n=len(self.market_regime_choices),
                            init_value=0,  # "bull"
                        ),
                        "liquidity_factor": swm.spaces.Box(
                            low=0.1,
                            high=2.0,
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "slippage_multiplier": swm.spaces.Box(
                            low=0.5,
                            high=3.0,
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "agent": swm.spaces.Dict(
                    {
                        "starting_balance": swm.spaces.Box(
                            low=10000.0,
                            high=1000000.0,
                            init_value=np.array(100000.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "transaction_cost_bps": swm.spaces.Box(
                            low=0.0,
                            high=50.0,  # 0 to 50 basis points
                            init_value=np.array(10.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "max_position_size": swm.spaces.Box(
                            low=0.1,
                            high=1.0,
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "leverage": swm.spaces.Box(
                            low=1.0,
                            high=4.0,
                            init_value=np.array(1.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                # TODO: Integrate financial data display variations instead of visual/color variations
                # Financial environments need structured output format variations:
                # - metrics_format: ["json", "dataframe", "structured"]
                # - report_frequency: ["step", "episode", "batch"]
                # - output_detail: ["basic", "detailed", "comprehensive"]
            },
            sampling_order=[
                "backtest",
                "market",
                "agent",
            ],  # TODO: Add financial data output variations
        )  # Initialize state variables
        self.current_step = 0
        self.current_symbol = "AAPL"
        self.position = 0.0  # Continuous position: negative=short, positive=long, 0=neutral
        self.balance = 100000.0  # Will be set by variation space
        self.portfolio_value = 100000.0
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []

        # Data storage
        self.market_data = None
        self.current_data_index = 0
        self.data_timestamps = None

        # Performance tracking
        self.start_time = None
        self.benchmark_returns = []
        self.portfolio_returns = []

        # TODO: Integrate financial data display/output system
        # Financial environments need metrics dashboards, not matplotlib figures

        # Initialize data reader (mock for now - will integrate with actual data pipeline)
        self._initialize_data_reader()

        # Ensure default variation values are valid
        assert self.variation_space.check(), "Default variation values must be within variation space"

    def _get_start_date_from_variation(self) -> str:
        """Get start date string from variation space index."""
        idx = self.variation_space["backtest"]["start_date"].value
        return self.start_date_choices[idx]

    def _get_end_date_from_variation(self) -> str:
        """Get end date string from variation space index."""
        idx = self.variation_space["backtest"]["end_date"].value
        return self.end_date_choices[idx]

    def _get_symbol_selection_from_variation(self) -> str:
        """Get symbol selection string from variation space index."""
        idx = self.variation_space["backtest"]["symbol_selection"].value
        return self.symbol_selection_choices[idx]

    def _get_market_regime_from_variation(self) -> str:
        """Get market regime string from variation space index."""
        idx = self.variation_space["market"]["regime"].value
        return self.market_regime_choices[idx]

    # TODO: Integrate financial data output format variations instead of chart style
    # def _get_output_format_from_variation(self) -> str:
    #     """Get financial data output format from variation space index."""
    #     idx = self.variation_space["financial_output"]["format"].value
    #     return self.output_format_choices[idx]

    def _initialize_data_reader(self) -> None:
        """TODO: INTEGRATE ALPACA DATA PIPELINE - Initialize real data reading system.

        This should integrate with your Alpaca data pipeline:

        1. Initialize AlpacaHistoricalDataReader with API credentials
        2. Set up data caching directory structure
        3. Load or create metadata index for fast symbol/date queries
        4. Configure parquet/zarr readers for high-performance data loading
        5. Set up concurrent data loading capabilities
        """
        raise NotImplementedError(
            "Real data pipeline initialization required. Please integrate Alpaca API data reader initialization here."
        )

    def _create_sample_metadata(self) -> None:
        """TODO: INTEGRATE ALPACA DATA PIPELINE - Replace with real metadata loading.

        This should integrate with Alpaca API to get real symbol metadata:

        1. Use Alpaca.get_assets() to get available symbols
        2. For each symbol, query available date range from data pipeline
        3. Get data quality metrics from actual stored data
        4. Cache metadata for fast querying

        Expected metadata structure:
        {
            "symbols": {
                "AAPL": {"start_date": "2015-01-01", "end_date": "2024-11-03", "data_quality": 0.99},
                ...
            },
            "date_ranges": {...},
            "data_format": {"frequency": "1min", "columns": [...], "compression": "snappy"}
        }
        """
        raise NotImplementedError(
            "Real metadata loading required. Please integrate Alpaca API metadata querying here."
        )

    def _load_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data for backtesting using optimized data pipeline.

        This method integrates with the high-performance data reading system
        that supports:
        - Fast parquet/zarr reading
        - Concurrent data loading
        - Compression with bf16/float16
        - Metadata queries for symbol availability
        """
        # TODO: INTEGRATE ALPACA DATA PIPELINE HERE
        # This is where the real data pipeline integration should happen:
        #
        # 1. Use AlpacaHistoricalDataReader to load data:
        #    reader = AlpacaHistoricalDataReader(api_key, secret_key)
        #    data = reader.get_bars(symbol, start_date, end_date, timeframe='1Min')
        #
        # 2. Convert to required DataFrame format with columns:
        #    ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        #
        # 3. Apply compression and caching optimizations
        #
        # 4. Return standardized DataFrame with proper indexing

        raise NotImplementedError(
            "Real data pipeline integration required. "
            "Please integrate Alpaca API data loading here. "
            f"Requested: {symbol} from {start_date} to {end_date}"
        )

    # TODO: INTEGRATE ALPACA DATA PIPELINE - All synthetic data generation removed
    # The following methods should be replaced with real data pipeline integration:
    # - Use AlpacaHistoricalDataReader for real market data
    # - Implement proper symbol price/volatility lookup from metadata
    # - Real OHLCV data from Alpaca API instead of synthetic generation

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        # Reset variation space
        self.variation_space.reset()

        # Handle variations
        variations = options.get("variation", DEFAULT_VARIATIONS)
        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variation names to sample")

        self.variation_space.update(variations)
        assert self.variation_space.check(debug=True), "Variation values must be within variation space!"

        # Get backtest configuration from variations
        start_date = self._get_start_date_from_variation()
        end_date = self._get_end_date_from_variation()
        symbol_selection = self._get_symbol_selection_from_variation()

        # Select symbols based on variation
        self.current_symbols = self._select_symbols(symbol_selection)
        self.current_symbol = self.current_symbols[0]  # Start with first symbol

        # Load market data for backtesting
        print(f"Loading historical data for backtesting: {self.current_symbol} from {start_date} to {end_date}")
        self.market_data = self._load_historical_data(self.current_symbol, start_date, end_date)

        if len(self.market_data) < self.window_size:
            raise ValueError(f"Insufficient data: {len(self.market_data)} rows, need at least {self.window_size}")

        # Initialize episode state
        self.current_step = 0
        self.current_data_index = self.window_size  # Start after window
        self.position = 0.0
        self.balance = self.variation_space["agent"]["starting_balance"].value
        self.portfolio_value = self.balance
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []

        # Performance tracking
        self.start_time = self.market_data.index[self.current_data_index]
        self.benchmark_returns = []
        self.portfolio_returns = []

        # Store initial portfolio value for return calculation
        self.initial_portfolio_value = self.portfolio_value

        # Create initial observation
        observation = self._get_observation()
        info = {
            "symbol": self.current_symbol,
            "timestamp": self.start_time,
            "market_data_length": len(self.market_data),
            "backtest_config": {
                "start_date": start_date,
                "end_date": end_date,
                "symbol_selection": symbol_selection,
            },
        }

        return observation, info

    def _select_symbols(self, symbol_selection: str) -> list:
        """Select symbols based on variation configuration."""
        symbol_groups = {
            "tech_stocks": ["AAPL", "GOOGL", "MSFT"],
            "blue_chip": ["AAPL", "MSFT", "SPY"],
            "volatile_stocks": ["TSLA", "GOOGL"],
            "single_stock": ["AAPL"],
            "sector_rotation": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        }
        return symbol_groups.get(symbol_selection, ["AAPL"])

    def step(self, action: int):
        if self.current_data_index >= len(self.market_data) - 1:
            # End of data
            terminated = True
            truncated = False
            reward = 0.0
            return (
                self._get_observation(),
                reward,
                terminated,
                truncated,
                {"reason": "end_of_data"},
            )

        # Get current market data
        current_row = self.market_data.iloc[self.current_data_index]
        current_price = current_row["close"]
        current_timestamp = self.market_data.index[self.current_data_index]

        # Apply time acceleration
        time_accel = self.variation_space["backtest"]["time_acceleration"].value
        self.current_data_index += int(time_accel)

        # Calculate transaction costs and market impact
        transaction_cost_bps = self.variation_space["agent"]["transaction_cost_bps"].value
        slippage_multiplier = self.variation_space["market"]["slippage_multiplier"].value

        # Execute trading action
        action_taken = "hold"

        if action == 1:  # BUY/LONG
            action_taken = self._execute_buy_order(current_price, transaction_cost_bps, slippage_multiplier)

        elif action == 0:  # SELL/SHORT
            action_taken = self._execute_sell_order(current_price, transaction_cost_bps, slippage_multiplier)

        # action == 2 is HOLD (no action needed)

        # Update portfolio value
        previous_portfolio_value = self.portfolio_value
        self.portfolio_value = self.balance + (self.shares_held * current_price)

        # Calculate returns
        portfolio_return = (self.portfolio_value - previous_portfolio_value) / previous_portfolio_value
        benchmark_return = self._calculate_benchmark_return()

        self.portfolio_returns.append(portfolio_return)
        self.benchmark_returns.append(benchmark_return)

        # Calculate reward (risk-adjusted return)
        reward = self._calculate_reward(portfolio_return, benchmark_return)

        # Store trade history
        trade_record = {
            "step": self.current_step,
            "timestamp": current_timestamp,
            "symbol": self.current_symbol,
            "price": current_price,
            "action": action_taken,
            "position": self.position,
            "balance": self.balance,
            "shares_held": self.shares_held,
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "reward": reward,
        }
        self.trade_history.append(trade_record)
        self.price_history.append(
            {
                "timestamp": current_timestamp,
                "price": current_price,
                "volume": current_row["volume"],
            }
        )

        self.current_step += 1

        # Check termination conditions
        max_data_steps = len(self.market_data) - self.window_size - 1
        terminated = self.current_data_index >= max_data_steps or self.current_step >= self.max_steps

        # Check for bankruptcy or margin call
        min_balance = self.variation_space["agent"]["starting_balance"].value * 0.1  # 10% of initial
        truncated = self.portfolio_value < min_balance

        if truncated:
            reward -= 100  # Heavy penalty for bankruptcy

        observation = self._get_observation()
        info = {
            "timestamp": current_timestamp,
            "symbol": self.current_symbol,
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "shares_held": self.shares_held,
            "action_taken": action_taken,
            "current_price": current_price,
            "portfolio_return": portfolio_return,
            "benchmark_return": benchmark_return,
            "total_return": (self.portfolio_value - self.initial_portfolio_value)
            / max(self.initial_portfolio_value, 1e-8),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
        }

        return observation, reward, terminated, truncated, info

    def _execute_buy_order(self, price: float, transaction_cost_bps: float, slippage_multiplier: float) -> str:
        """Execute a buy order with realistic transaction costs."""
        if self.balance <= 0:
            return "hold"  # No money to buy

        # Calculate position size based on available balance and leverage
        max_position_size = self.variation_space["agent"]["max_position_size"].value
        leverage = self.variation_space["agent"]["leverage"].value

        # Calculate shares to buy
        available_capital = self.balance * max_position_size * leverage
        transaction_cost = transaction_cost_bps / 10000.0  # Convert basis points to decimal
        slippage = price * 0.0001 * slippage_multiplier  # Market impact

        effective_price = price + slippage
        shares_to_buy = available_capital / (effective_price * (1 + transaction_cost))

        if shares_to_buy > 0:
            total_cost = shares_to_buy * effective_price * (1 + transaction_cost)

            if total_cost <= self.balance:
                self.shares_held += shares_to_buy
                self.balance -= total_cost
                self.position = min(self.position + shares_to_buy, self.shares_held)  # Update position
                return "buy"

        return "hold"

    def _execute_sell_order(self, price: float, transaction_cost_bps: float, slippage_multiplier: float) -> str:
        """Execute a sell order with realistic transaction costs."""
        if self.shares_held <= 0 and not self.enable_shorting:
            return "hold"  # No shares to sell and shorting disabled

        transaction_cost = transaction_cost_bps / 10000.0
        slippage = price * 0.0001 * slippage_multiplier

        effective_price = price - slippage

        if self.shares_held > 0:
            # Sell existing long position
            revenue = self.shares_held * effective_price * (1 - transaction_cost)
            self.balance += revenue
            self.position -= self.shares_held
            self.shares_held = 0.0
            return "sell"

        elif self.enable_shorting:
            # Enter short position (simplified - real implementation would need margin requirements)
            max_position_size = self.variation_space["agent"]["max_position_size"].value
            leverage = self.variation_space["agent"]["leverage"].value

            short_value = self.balance * max_position_size * leverage
            shares_to_short = short_value / (effective_price * (1 + transaction_cost))

            if shares_to_short > 0:
                # Credit from short sale
                revenue = shares_to_short * effective_price * (1 - transaction_cost)
                self.balance += revenue
                self.position -= shares_to_short  # Negative position for short
                return "short"

        return "hold"

    def _calculate_benchmark_return(self) -> float:
        """Calculate benchmark return (buy-and-hold strategy)."""
        if len(self.price_history) < 2:
            return 0.0

        current_price = self.price_history[-1]["price"]
        previous_price = self.price_history[-2]["price"]

        return (current_price - previous_price) / previous_price

    def _calculate_reward(self, portfolio_return: float, benchmark_return: float) -> float:
        """Calculate reward based on risk-adjusted performance."""
        # Base reward is excess return over benchmark
        excess_return = portfolio_return - benchmark_return

        # Add penalty for high volatility (simplified Sharpe ratio concept)
        if len(self.portfolio_returns) > 10:
            volatility = np.std(self.portfolio_returns[-10:])
            volatility_penalty = volatility * 0.5
        else:
            volatility_penalty = 0.0

        # Scale reward for learning
        reward = (excess_return * 100) - volatility_penalty

        return reward

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio for current performance."""
        if len(self.portfolio_returns) < 2:
            return 0.0

        returns = np.array(self.portfolio_returns)
        excess_returns = returns - 0.02 / (252 * 390)  # Assuming 2% risk-free rate

        if np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 390)  # Annualized

    def _get_observation(self) -> np.ndarray:
        """Create comprehensive observation from current market state."""
        if self.market_data is None or self.current_data_index < self.window_size:
            # Return zero observation if no data available
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get historical price window
        start_idx = max(0, self.current_data_index - self.window_size)
        end_idx = self.current_data_index

        price_window_data = self.market_data.iloc[start_idx:end_idx]

        # Extract OHLCV features
        price_features = []
        for _, row in price_window_data.iterrows():
            # Normalize prices by current close for scale invariance
            current_close = self.market_data.iloc[self.current_data_index]["close"]

            price_features.extend(
                [
                    row["open"] / current_close,
                    row["high"] / current_close,
                    row["low"] / current_close,
                    row["close"] / current_close,
                    np.log(row["volume"] + 1) / 20.0,  # Log-normalized volume
                    (row["high"] + row["low"]) / (2 * current_close),  # Volume-weighted price proxy
                ]
            )

        # Pad if necessary
        expected_price_features = self.window_size * 6
        while len(price_features) < expected_price_features:
            price_features.insert(0, 0.0)

        price_features = price_features[-expected_price_features:]  # Take last window_size elements

        # Portfolio state features
        starting_value = self.variation_space["agent"]["starting_balance"].value
        normalized_position = self.position / 1000.0  # Scale position
        normalized_balance = self.balance / starting_value
        normalized_portfolio = self.portfolio_value / starting_value

        # Time features for intraday patterns
        current_timestamp = self.market_data.index[self.current_data_index]
        hour_of_day = current_timestamp.hour / 24.0
        day_of_week = current_timestamp.weekday() / 7.0
        month_of_year = current_timestamp.month / 12.0
        year_progress = (current_timestamp.dayofyear) / 365.0

        # Combine all features
        observation = np.array(
            price_features
            + [normalized_position]
            + [normalized_balance]
            + [normalized_portfolio]
            + [hour_of_day, day_of_week, month_of_year, year_progress],
            dtype=np.float32,
        )

        # Ensure observation matches expected shape
        if len(observation) != self.observation_space.shape[0]:
            # Pad or truncate to match expected size
            expected_size = self.observation_space.shape[0]
            if len(observation) < expected_size:
                observation = np.pad(observation, (0, expected_size - len(observation)))
            else:
                observation = observation[:expected_size]

        return observation

    def render(self, mode: str | None = None):
        """
        TODO: Integrate financial data output format instead of image rendering.

        Financial environments should output:
        - Portfolio metrics (returns, Sharpe ratio, drawdown)
        - Trade history and performance analytics
        - Market data summaries
        - Risk metrics and position information

        This should integrate with stable-worldmodel's data collection system
        for financial backtesting, not image-based rendering.
        """
        # TODO: Replace with financial data output
        # For now, return None to indicate no image-based rendering
        return None

    def close(self):
        """Clean up environment resources."""
        # TODO: Integrate financial data cleanup instead of matplotlib cleanup
        # Financial environments may need to close data connections,
        # save final metrics, or cleanup temporary files
        pass

    def get_backtest_results(self) -> dict[str, Any]:
        """Get comprehensive backtesting results and performance metrics."""
        if not self.trade_history:
            return {}

        # Basic performance metrics
        total_return = (self.portfolio_value - self.initial_portfolio_value) / max(self.initial_portfolio_value, 1e-8)

        # Calculate benchmark performance (buy and hold)
        initial_price = self.trade_history[0]["price"]
        final_price = self.trade_history[-1]["price"]
        benchmark_return = (final_price - initial_price) / initial_price

        # Risk metrics
        returns = np.array(self.portfolio_returns)
        sharpe_ratio = self._calculate_sharpe_ratio()

        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252 * 390)  # Annualized
            max_drawdown = self._calculate_max_drawdown()
        else:
            volatility = 0.0
            max_drawdown = 0.0

        # Trading statistics
        total_trades = len([t for t in self.trade_history if t["action"] in ["buy", "sell", "short"]])
        buy_trades = len([t for t in self.trade_history if t["action"] == "buy"])
        sell_trades = len([t for t in self.trade_history if t["action"] == "sell"])
        short_trades = len([t for t in self.trade_history if t["action"] == "short"])

        # Time-based metrics
        start_time = self.trade_history[0]["timestamp"]
        end_time = self.trade_history[-1]["timestamp"]
        trading_period = (end_time - start_time).days

        return {
            # Performance
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "benchmark_return": benchmark_return,
            "benchmark_return_pct": benchmark_return * 100,
            "excess_return": total_return - benchmark_return,
            "excess_return_pct": (total_return - benchmark_return) * 100,
            # Risk metrics
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            # Portfolio
            "starting_value": self.initial_portfolio_value,
            "final_value": self.portfolio_value,
            "current_balance": self.balance,
            "current_position": self.position,
            "shares_held": self.shares_held,
            # Trading activity
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "short_trades": short_trades,
            "trading_frequency": total_trades / max(trading_period, 1),
            # Time period
            "start_date": start_time,
            "end_date": end_time,
            "trading_days": trading_period,
            "symbol": self.current_symbol,
            # Market conditions
            "market_regime": self._get_market_regime_from_variation(),
            "transaction_cost_bps": self.variation_space["agent"]["transaction_cost_bps"].value,
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak portfolio value."""
        if not self.trade_history:
            return 0.0

        portfolio_values = [t["portfolio_value"] for t in self.trade_history]
        peak = portfolio_values[0]
        max_drawdown = 0.0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def get_data_pipeline_stats(self) -> dict[str, Any]:
        """TODO: INTEGRATE ALPACA DATA PIPELINE - Get real data loading performance stats.

        This should return actual metrics from the data pipeline:
        - Real compression ratios from parquet/zarr files
        - Actual read speeds and cache performance
        - Memory usage of data readers
        - Number of concurrent readers in use
        """
        raise NotImplementedError(
            "Real data pipeline stats required. Please integrate actual performance metrics from Alpaca data pipeline."
        )

    def reset_to_date(self, target_date: str, symbol: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Time machine functionality: Reset environment to specific date and symbol.

        This is the core backtesting feature that allows going back in time
        to any date and replaying market data chronologically.
        """
        if symbol:
            self.current_symbol = symbol

        # Load historical data for the target date
        extended_end_date = pd.to_datetime(target_date) + timedelta(days=90)  # Load extra data for episode
        self.market_data = self._load_historical_data(
            self.current_symbol, target_date, extended_end_date.strftime("%Y-%m-%d")
        )

        # Reset environment state
        self.current_step = 0
        self.current_data_index = self.window_size
        self.position = 0.0
        self.balance = self.variation_space["agent"]["starting_balance"].value
        self.portfolio_value = self.balance
        self.shares_held = 0.0
        self.trade_history = []
        self.price_history = []
        self.portfolio_returns = []
        self.benchmark_returns = []

        self.start_time = self.market_data.index[self.current_data_index]
        self.initial_portfolio_value = self.portfolio_value

        observation = self._get_observation()
        info = {
            "time_machine_date": target_date,
            "symbol": self.current_symbol,
            "market_data_length": len(self.market_data),
            "start_timestamp": self.start_time,
        }

        return observation, info


# Update the class name in the registration
FinancialTradingEnv = FinancialBacktestEnv  # For backward compatibility
