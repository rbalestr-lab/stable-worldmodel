from .process import build_stock_data
import numpy as np
import random
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader

logger.remove()
logger.add(
    lambda msg: print(msg, end=''),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
    colorize=True
)


class StockDataset(Dataset):
    """
    PyTorch Dataset that loads stock data incrementally by timestep.
    Caches all timesteps after loading for efficient access.
    """
    def __init__(self, start_time, end_time, processing_methods, sector_config, freq):
        self.start_time = start_time
        self.end_time = end_time
        self.processing_methods = processing_methods
        self.sector_config = sector_config
        self.freq = freq
        self.cache = None
        
    def _load_data(self):
        """Load all data and cache it"""
        if self.cache is None:
            logger.info("Loading data into cache...")
            self.cache = build_stock_data(
                start_time=self.start_time,
                end_time=self.end_time,
                processing_methods=self.processing_methods,
                sector_config=self.sector_config,
                freq=self.freq
            )
            logger.success(f"Cached data with shape {self.cache.shape}")
    
    def __len__(self):
        """Return number of timesteps"""
        self._load_data()
        return len(self.cache)
    
    def __getitem__(self, idx):
        """
        Get data for a specific timestep.
        Returns: (num_sectors, max_stocks_per_sector, channels)
        """
        self._load_data()
        return torch.tensor(self.cache[idx], dtype=torch.float32)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logger.info(f"Starting stock dataset generation with seed {SEED}")

# sector_config = {
#     "Basic Materials": 76,
#     "Consumer Discretionary": 601,
#     "Consumer Staples": 85,
#     "Energy": 140,
#     "Finance": 782,
#     "Health Care": 500,
#     "Industrials": 407,
#     "Miscellaneous": 31,
#     "Real Estate": 277,
#     "Technology": 463,
#     "Telecommunications": 67,
#     "Utilities": 145
# }

# sector_config = {
#     'Finance': 81,
#     'Technology': 64,
#     'Health Care': 49,
#     'Industrials': 36
# }

sector_config = {}

# sector_config = {'Finance': 1}


# Option 1: Direct loading (loads all data at once)
# dataset = build_stock_data(
#     start_time="2020-10-24",
#     end_time="2025-10-30",
#     processing_methods=['return', 'volume'], 
#     sector_config=sector_config,
#     freq="1min"
# )
# logger.success(f"Generated {len(dataset)} with shape {dataset.shape}")


# Option 2: PyTorch DataLoader (fetches each T, caches later Ts)
stock_dataset = StockDataset(
    start_time="2020-10-24",
    end_time="2025-10-30",
    processing_methods=['return', 'volume'],
    sector_config=sector_config,
    freq="1min"
)

# Create DataLoader
dataloader = DataLoader(
    stock_dataset,
    batch_size=1,  # Process one timestep at a time
    shuffle=False,  # Keep temporal order
    num_workers=0   # Single process (data is already cached)
)

logger.success(f"Created DataLoader with {len(stock_dataset)} timesteps")

# Example: Iterate through timesteps
for t_idx, timestep_data in enumerate(dataloader):
    # timestep_data shape: (batch_size=1, num_sectors, max_stocks_per_sector, channels)
    logger.info(f"Timestep {t_idx}: {timestep_data.shape}")
    
    # Your world model training code here
    # ...
    
    if t_idx >= 2:  # Just show first few for demo
        logger.info("...")
        break

logger.success(f"Completed processing timesteps")


"""
| Alias        | Meaning             |
| ------------ | ------------------- |
| `B`          | Business day        |
| `C`          | Custom business day |
| `D`          | Calendar day        |
| `W`          | Weekly              |
| `M`          | Month end           |
| `MS`         | Month start         |
| `Q`          | Quarter end         |
| `QS`         | Quarter start       |
| `A` or `Y`   | Year end            |
| `AS` / `YS`  | Year start          |
| `H`          | Hour                |
| `T` or `min` | Minute              |
| `S`          | Second              |
| `L` or `ms`  | Millisecond         |
| `U`          | Microsecond         |
| `N`          | Nanosecond          |
"""

# Save per stock as xarray files