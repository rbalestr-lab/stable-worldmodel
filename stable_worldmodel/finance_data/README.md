# Stock Market Data Pipeline

Converts stock market data into structured tensors for world models. Data is organized by sector and time, with stocks grouped by sector.

## Quick Start

```bash
python main.py
```

This demonstrates two ways to use the dataset:
1. **Direct loading:** Load entire dataset at once
2. **PyTorch DataLoader:** Iterate through timesteps with automatic caching

## Usage

### Option 1: Direct Loading

Load the entire dataset at once for batch processing:

```python
from process import build_stock_data

sector_config = {
    'Finance': 81,
    'Technology': 64,
    'Health Care': 49,
    'Industrials': 36
}

dataset = build_stock_data(
    start_time="2024-11-17",
    end_time="2025-11-21",
    processing_methods=['return', 'volume', 'trade_count', 'vwap'],
    sector_config=sector_config,
    freq="1min"  # Use pandas frequency strings: '1min', '5min', '1H', '1D', etc.
)

# Output shape: (T, num_sectors, max_stocks_per_sector, channels)
# T = number of timesteps (depends on date range and frequency)
# num_sectors = number of sectors (4 in this example)
# max_stocks_per_sector = max value in sector_config (81 in this example)
# channels = number of processing methods (4 in this example)
```

### Option 2: PyTorch DataLoader (Recommended for World Models)

For sequential processing with automatic caching:

```python
import torch
from torch.utils.data import DataLoader
from main import StockDataset

sector_config = {
    'Finance': 81,
    'Technology': 64,
    'Health Care': 49,
    'Industrials': 36
}

# Create dataset
stock_dataset = StockDataset(
    start_time="2024-11-17",
    end_time="2025-11-21",
    processing_methods=['return', 'volume', 'trade_count', 'vwap'],
    sector_config=sector_config,
    freq="1min"
)

# Create DataLoader
dataloader = DataLoader(
    stock_dataset,
    batch_size=1,      # Process one timestep at a time
    shuffle=False,     # Keep temporal order for world models
    num_workers=0      # Data already cached
)

# Iterate through timesteps
for t_idx, timestep_data in enumerate(dataloader):
    # timestep_data shape: (1, num_sectors, max_stocks_per_sector, channels)
    # Your world model training code here
    pass
```

**Benefits:**
- Fetches each timestep (T) sequentially
- Caches all data on first access for fast subsequent iterations
- Perfect for autoregressive world models
- Memory efficient with `batch_size=1`
- Maintains temporal ordering with `shuffle=False`

## Available Sectors

```
Basic Materials: 76 stocks
Consumer Discretionary: 601 stocks
Consumer Staples: 85 stocks
Energy: 140 stocks
Finance: 782 stocks
Health Care: 500 stocks
Industrials: 407 stocks
Miscellaneous: 31 stocks
Real Estate: 277 stocks
Technology: 463 stocks
Telecommunications: 67 stocks
Utilities: 145 stocks
```

You can specify any combination of these sectors in your `sector_config` dictionary.

## Data Structure

- **Shape:** `(T, num_sectors, max_stocks_per_sector, channels)`
- **Sectors:** Each sector groups related stocks together
- **Variable Sizes:** If sectors have different sizes, smaller sectors are padded with NaN
- **Channels:** Each processing method adds one channel (stackable)
- **Organization:** Data is organized with time as the first dimension, then sectors, then stocks within each sector
- **Caching:** Later timesteps can be cached while fetching new data incrementally

## Processing Methods

### Bar Data Methods

### `return`
Computes log returns:
- Formula: `log((price[t] + ε) / (price[t-1] + ε))`
- Captures relative price changes between consecutive timestamps

### `volume`
Trading volume:
- Raw volume data for each stock
- Indicates market activity and liquidity

### `trade_count`
Number of trades:
- Count of individual trades executed
- Reflects trading frequency and market interest

### `vwap`
Volume Weighted Average Price:
- Average price weighted by volume
- Represents the "true average" trading price

### Level 1 Order Book Methods

### `l1_order`
Raw Level 1 order book data (4 channels):
- `bid_price` - Highest price buyers are willing to pay
- `bid_size` - Volume available at bid price
- `ask_price` - Lowest price sellers are willing to accept
- `ask_size` - Volume available at ask price

### `microprice` (SOTA - Best 1D Compression)
Weighted fair value using order book imbalance:
- Formula: `(ask_price × bid_size + bid_price × ask_size) / (bid_size + ask_size)`
- **Why SOTA:** Uses all 4 L1 variables, captures imbalance, strongest short-term price predictor
- Used by: Jump Trading, Citadel, Optiver, DeepOB, JAX-LOB
- **Best single dimension choice for price prediction**

### `mid_price` (Fair Value Proxy)
Simple average of bid and ask:
- Formula: `(bid_price + ask_price) / 2`
- **Why used:** Very stable, removes spread noise, common baseline in microstructure research
- **Best for stability without directional bias**

### `obi` (Order Book Imbalance - Liquidity Signal)
Normalized buy vs sell pressure:
- Formula: `(bid_size - ask_size) / (bid_size + ask_size)`
- Range: -1 (all sell pressure) to +1 (all buy pressure)
- **Why SOTA:** Directly measures directional pressure, strong micro-alpha signal
- Used in: DeepLOB, DeepOB, ZIB, LOBSTER research
- **Best single scalar for directionality**

### `spread` (Liquidity Cost)
Absolute bid-ask spread:
- Formula: `ask_price - bid_price`
- Measures transaction cost and market liquidity
- Higher spread = lower liquidity
- **Best for liquidity analysis**

### `relative_spread` (Normalized Liquidity)
Spread relative to price level:
- Formula: `(ask_price - bid_price) / mid_price`
- Normalized measure of liquidity cost
- **Why used:** Comparable across different price levels, predicts volatility
- **Best for cross-stock liquidity comparison**

### Normalization

All processing methods apply **z-score normalization** globally across:
- All timestamps (T)
- All sectors/patches
- All stocks

The data is normalized across all timestamps, sectors/patches, stocks per dimension.

**Data Filtering (applies to all methods):**
- Samples `stocks_per_sector * pad_tickers_ratio` tickers per sector initially
- Filters out tickers with leading/trailing zeros across any feature
- Selects first `stocks_per_sector` valid tickers per sector
- Forward fills any remaining missing values across all features


## Ways to Speed Up Data Download

**Batch size:** Controls how many tickers are fetched per API call. Edit in `download.py`:
```python
batch_size = 1000  # Tickers per API call (increase if API allows)
```

**Pad tickers ratio:** Reduces initial ticker selection to minimize data downloads. Edit in `process.py` (`get_sector_tickers` function):
```python
def get_sector_tickers(sector_config, pad_tickers_ratio=1.1):
    # Lower ratio (e.g., 1.05) = fewer tickers downloaded initially
    # Higher ratio (e.g., 1.2) = more buffer for filtering
```
The function initially downloads `stocks_per_sector × pad_tickers_ratio` tickers, then filters to the exact count after validation.

## Data Source

Requires Alpaca Markets API credentials in `.env`:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Data is cached in `dataset.h5` for faster subsequent runs. The caching system supports incremental updates, allowing you to fetch new timesteps while reusing previously downloaded data.
