"""Data management and dataset utilities for Stable World Model.

This module provides utilities for managing datasets, models, and world
information. It includes functionality for loading multi-step trajectories,
querying cached data, and retrieving metadata about environments.

The module supports:
    - Multi-step trajectory datasets with frame skipping
    - Dataset and model cache management
    - World environment introspection
    - Gymnasium space metadata extraction
    - Financial data loading with bfloat16 encoding
"""

import os
import re
import shutil
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import gymnasium as gym
import numpy as np
import pandas as pd
import PIL
import stable_pretraining as spt
import torch
from datasets import load_from_disk
from loguru import logger
from rich import print

import stable_worldmodel as swm


try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None
    logger.warning("ml_dtypes not installed - bfloat16 encoding will be disabled")

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed - .env file will not be loaded")

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if api_key and secret_key:
        # Note: StockHistoricalDataClient doesn't have 'paper' parameter
        # Historical data is the same for both paper and live accounts
        _alpaca_client = StockHistoricalDataClient(api_key, secret_key)
        logger.info("Alpaca API client initialized successfully")
    else:
        _alpaca_client = None
        logger.warning("Alpaca API credentials not found - financial data download will be disabled")
except ImportError:
    StockBarsRequest = None
    TimeFrame = None
    _alpaca_client = None
    logger.warning("alpaca-py not installed - financial data download will be disabled")
except Exception as e:
    _alpaca_client = None
    logger.error(f"Failed to initialize Alpaca client: {e}")


class StepsDataset(spt.data.HFDataset):
    """Dataset for loading multi-step trajectory sequences.

    This dataset loads sequences of consecutive steps from episodic data,
    supporting frame skipping and automatic image loading from file paths.
    Inherits from stable_pretraining's HFDataset.

    Attributes:
        data_dir (Path): Directory containing the dataset.
        num_steps (int): Number of steps per sequence.
        frameskip (int): Number of steps to skip between frames.
        episodes (np.ndarray): Array of unique episode indices.
        episode_slices (dict): Mapping from episode index to dataset indices.
        cum_slices (np.ndarray): Cumulative sum of valid samples per episode.
        idx_to_ep (np.ndarray): Mapping from sample index to episode.
        img_cols (set): Set of column names containing image paths.
    """

    def __init__(
        self,
        path,
        *args,
        num_steps=2,
        frameskip=1,
        cache_dir=None,
        **kwargs,
    ):
        """Initialize the StepsDataset.

        Args:
            path (str): Name or path of the dataset within cache directory.
            *args: Additional arguments passed to parent class.
            num_steps (int, optional): Number of consecutive steps per sample. Defaults to 2.
            frameskip (int, optional): Number of steps between sampled frames. Defaults to 1.
            cache_dir (str, optional): Cache directory path. Defaults to None (uses default).
            **kwargs: Additional keyword arguments passed to parent class.

        Raises:
            AssertionError: If required columns are missing from dataset.
            ValueError: If episodes are too short for the requested num_steps and frameskip.
        """
        data_dir = Path(cache_dir or swm.data.get_cache_dir(), path)
        super().__init__(str(data_dir), *args, **kwargs)

        self.data_dir = data_dir
        self.num_steps = num_steps
        self.frameskip = frameskip

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx' column"
        assert "step_idx" in self.dataset.column_names, "Dataset must have 'step_idx' column"
        assert "action" in self.dataset.column_names, "Dataset must have 'action' column"

        self.dataset.set_format("torch")

        # get number of episodes
        ep_indices = self.dataset["episode_idx"][:]
        self.episodes = np.unique(ep_indices)

        # get dataset indices of each episode
        self.episode_slices = {e: self.get_episode_slice(e, ep_indices) for e in self.episodes}

        # start index for each episode
        valid_samples_per_ep = [
            max(0, len(ep_slice) - self.num_steps * self.frameskip + 1) for ep_slice in self.episode_slices.values()
        ]
        self.cum_slices = np.cumsum([0] + valid_samples_per_ep)

        # map from sample to their episode
        self.idx_to_ep = np.searchsorted(self.cum_slices, torch.arange(len(self)), side="right") - 1

        self.img_cols = self.infer_img_path_columns()

    def get_episode_slice(self, episode_idx, episode_indices):
        """Get dataset indices for a specific episode.

        Args:
            episode_idx (int): Episode index to retrieve.
            episode_indices (array-like): Array of episode indices for all steps.

        Returns:
            np.ndarray: Indices of steps belonging to the specified episode.

        Raises:
            ValueError: If episode is too short for num_steps and frameskip.
        """
        indices = np.flatnonzero(episode_indices == episode_idx)
        if len(indices) <= (self.num_steps * self.frameskip):
            raise ValueError(
                f"Episode {episode_idx} is too short ({len(indices)} steps) for {self.num_steps} steps with {self.frameskip} frameskip"
            )
        return indices

    def __len__(self):
        """Get total number of valid step sequences in the dataset.

        Returns:
            int: Number of valid sequences across all episodes.
        """
        return int(self.cum_slices[-1])

    def __getitem__(self, idx):
        """Get a multi-step sequence at the given index.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict: Dictionary containing the sequence data with keys for
                observations, actions, and other dataset fields. Images are
                loaded as PIL Images and actions are reshaped to (num_steps, action_dim).
        """
        ep = self.idx_to_ep[idx]
        episode_indices = self.episode_slices[ep]
        offset = idx - self.cum_slices[ep]
        start = offset
        stop = start + self.num_steps * self.frameskip
        idx_slice = episode_indices[start:stop]
        steps = self.dataset[idx_slice]

        for k, v in steps.items():
            if k == "action":
                continue

            v = v[:: self.frameskip]
            steps[k] = v

            if k in self.img_cols:
                steps[k] = [PIL.Image.open(self.data_dir / img_path) for img_path in v]

        if self.transform:
            steps = self.transform(steps)

        # stack images into a single tensor
        for k in self.img_cols:
            steps[k] = torch.stack(steps[k])

        # reshape action
        steps["action"] = steps["action"].reshape(self.num_steps, -1)

        return steps

    def infer_img_path_columns(self):
        """Infer which dataset columns contain image file paths.

        Checks the first dataset element to identify string columns with
        common image file extensions.

        Returns:
            set: Set of column names containing image file paths.
        """
        IMG_EXTENSIONS = (".jpeg", ".png", ".jpg")

        img_cols = set()
        first_elem = self.dataset[0]
        for col in self.dataset.column_names:
            if isinstance(first_elem[col], str) and first_elem[col].endswith(IMG_EXTENSIONS):
                img_cols.add(col)
        return img_cols


#####################
###     utils     ###
#####################


def is_image(x):
    """Check if input is a valid image array.

    Args:
        x: Input to check.

    Returns:
        bool: True if x is a uint8 numpy array with shape (H, W, C) where
            C is 1 (grayscale), 3 (RGB), or 4 (RGBA).
    """
    return type(x) is np.ndarray and x.ndim == 3 and x.shape[2] in [1, 3, 4] and x.dtype == np.uint8


#####################
###   CLI Info   ####
#####################


class SpaceInfo(TypedDict, total=False):
    """Type specification for Gymnasium space metadata.

    Attributes:
        shape: Dimensions of the space.
        type: Class name of the space (e.g., 'Box', 'Discrete').
        dtype: Data type of the space elements.
        low: Lower bounds for Box spaces.
        high: Upper bounds for Box spaces.
        n: Number of discrete values for Discrete spaces.
    """

    shape: tuple[int, ...]
    type: str
    dtype: str
    low: Any
    high: Any
    n: int


class VariationInfo(TypedDict):
    """Type specification for environment variation metadata.

    Attributes:
        has_variation: Whether the environment supports variations.
        type: Class name of the variation space if it exists.
        names: List of variation parameter names.
    """

    has_variation: bool
    type: str | None
    names: list[str] | None


class WorldInfo(TypedDict):
    """Type specification for world environment information.

    Attributes:
        name: Name/ID of the world environment.
        observation_space: Metadata about the observation space.
        action_space: Metadata about the action space.
        variation: Information about environment variations.
        config: Additional configuration parameters.
    """

    name: str
    observation_space: SpaceInfo
    action_space: SpaceInfo
    variation: VariationInfo
    config: dict[str, Any]


def get_cache_dir() -> Path:
    """Get the cache directory for stable_worldmodel data.

    The cache directory can be customized via the STABLEWM_HOME environment
    variable. If not set, defaults to ~/.stable_worldmodel.

    Returns:
        Path: Path to the cache directory. Directory is created if it doesn't exist.
    """
    cache_dir = os.getenv("STABLEWM_HOME", os.path.expanduser("~/.stable_worldmodel"))
    os.makedirs(cache_dir, exist_ok=True)
    return Path(cache_dir)


def list_datasets():
    """List all cached datasets.

    Returns:
        list[str]: Names of all dataset directories in the cache.
    """
    with os.scandir(get_cache_dir()) as entries:
        return [e.name for e in entries if e.is_dir()]


def list_models():
    """List all cached model checkpoints.

    Searches for files matching the pattern `<name>_weights*.ckpt` or
    `<name>_object.ckpt` and returns unique model names.

    Returns:
        list[str]: Sorted list of model names found in cache.
    """
    pattern = re.compile(r"^(.*?)(?=_(?:weights(?:-[^.]*)?|object)\.ckpt$)", re.IGNORECASE)

    cache_dir = get_cache_dir()
    models = set()

    for fname in os.listdir(cache_dir):
        m = pattern.match(fname)
        if m:
            models.add(m.group(1))

    return sorted(models)


def dataset_info(name):
    """Get metadata about a cached dataset.

    Args:
        name (str): Name of the dataset.

    Returns:
        dict: Dictionary containing dataset metadata including:
            - name: Dataset name
            - num_episodes: Number of unique episodes
            - num_steps: Total number of steps
            - columns: List of column names
            - obs_shape: Shape of observation images
            - action_shape: Shape of action vectors
            - goal_shape: Shape of goal images
            - variation: Dict with variation information

    Raises:
        ValueError: If dataset is not found in cache.
        AssertionError: If required columns are missing.
    """
    # check name exists
    if name not in list_datasets():
        raise ValueError(f"Dataset '{name}' not found. Available: {list_datasets()}")

    dataset = load_from_disk(str(Path(get_cache_dir(), name, "records")))

    dataset.set_format("numpy")

    def assert_msg(col):
        return f"Dataset must have '{col}' column"  # type: ignore

    assert "episode_idx" in dataset.column_names, assert_msg("episode_idx")
    assert "step_idx" in dataset.column_names, assert_msg("step_idx")
    assert "episode_len" in dataset.column_names, assert_msg("episode_len")
    assert "pixels" in dataset.column_names, assert_msg("pixels")
    assert "action" in dataset.column_names, assert_msg("action")
    assert "goal" in dataset.column_names, assert_msg("goal")

    info = {
        "name": name,
        "num_episodes": len(np.unique(dataset["episode_idx"])),
        "num_steps": len(dataset),
        "columns": dataset.column_names,
        "obs_shape": dataset["pixels"][0].shape,
        "action_shape": dataset["action"][0].shape,
        "goal_shape": dataset["goal"][0].shape,
        "variation": {
            "has_variation": any(col.startswith("variation.") for col in dataset.column_names),
            "names": [col.removeprefix("variation.") for col in dataset.column_names if col.startswith("variation.")],
        },
    }

    return info


def list_worlds():
    """List all registered world environments.

    Returns:
        list[str]: Sorted list of world environment IDs.
    """
    return sorted(swm.envs.WORLDS)


def _space_meta(space) -> SpaceInfo | dict[str, SpaceInfo] | list[SpaceInfo]:
    """Extract metadata from a Gymnasium space.

    Recursively processes Dict, Tuple, and Sequence spaces.

    Args:
        space: A Gymnasium space object.

    Returns:
        SpaceInfo | dict | list: Space metadata. Returns a dict for Dict spaces,
            a list for Tuple/Sequence spaces, or a SpaceInfo dict for simple spaces.
    """
    if isinstance(space, gym.spaces.Dict):
        return {k: _space_meta(v) for k, v in space.spaces.items()}

    if isinstance(space, gym.spaces.Sequence) or isinstance(space, gym.spaces.Tuple):
        return [_space_meta(s) for s in space.spaces]

    info: SpaceInfo = {
        "shape": getattr(space, "shape", None),
        "type": type(space).__name__,
    }

    if hasattr(space, "dtype") and getattr(space, "dtype") is not None:
        info["dtype"] = str(space.dtype)
    if hasattr(space, "low"):
        info["low"] = getattr(space, "low", None)
    if hasattr(space, "high"):
        info["high"] = getattr(space, "high", None)
    if hasattr(space, "n"):
        info["n"] = getattr(space, "n")
    return info


@lru_cache(maxsize=128)
def world_info(
    name: str,
    *,
    image_shape: tuple[int, int] = (224, 224),
    render_mode: str = "rgb_array",
) -> WorldInfo:
    """Get metadata about a world environment.

    Creates a temporary world instance to extract observation space, action
    space, and variation information. Results are cached for efficiency.

    Args:
        name: ID of the world environment.
        image_shape: Desired image shape for rendering. Defaults to (224, 224).
        render_mode: Rendering mode for the environment. Defaults to "rgb_array".

    Returns:
        WorldInfo: Dictionary containing world metadata including spaces and variations.

    Raises:
        ValueError: If world name is not registered.
    """
    if name not in swm.envs.WORLDS:
        raise ValueError(f"World '{name}' not found. Available: {', '.join(list_worlds())}")
    world = None

    try:
        world = swm.World(
            name,
            num_envs=1,
            image_shape=image_shape,
            render_mode=render_mode,
            verbose=0,
        )

        obs_space = getattr(world, "single_observation_space", None)
        act_space = getattr(world, "single_action_space", None)
        var_space = getattr(world, "single_variation_space", None)

        variation: VariationInfo = {
            "has_variation": var_space is not None,
            "type": type(var_space).__name__ if var_space is not None else None,
            "names": var_space.names() if hasattr(var_space, "names") else None,
        }

        return {
            "name": name,
            "observation_space": _space_meta(obs_space) if obs_space else {},
            "action_space": _space_meta(act_space) if act_space else {},
            "variation": variation,
        }

    finally:
        if world is not None and hasattr(world, "close"):
            try:
                world.close()
            except Exception:
                pass


def delete_dataset(name):
    """Delete a cached dataset and its associated files.

    Args:
        name (str): Name of the dataset to delete.

    Note:
        Prints success or error messages to console.
    """
    from datasets import logging as ds_logging

    ds_logging.set_verbosity_error()

    try:
        dataset_path = Path(get_cache_dir(), name)

        if not dataset_path.exists():
            raise ValueError(f"Dataset {name} does not exist at {dataset_path}")

        dataset = load_from_disk(str(Path(dataset_path, "records")))

        # remove cache files
        dataset.cleanup_cache_files()

        # delete dataset directory
        shutil.rmtree(dataset_path, ignore_errors=False)

        print(f"🗑️ Dataset {dataset_path} deleted!")

    except Exception as e:
        print(f"[red]Error cleaning up dataset [cyan]{name}[/cyan]: {e}[/red]")


def delete_model(name):
    """Delete cached model checkpoint files.

    Removes all checkpoint files (weights and object files) matching the
    given model name.

    Args:
        name (str): Name of the model to delete.

    Note:
        Prints success or error messages to console for each file deleted.
    """
    pattern = re.compile(rf"^{re.escape(name)}(?:_[^-].*)?\.ckpt$")
    cache_dir = get_cache_dir()

    for fname in os.listdir(cache_dir):
        if pattern.match(fname):
            filepath = os.path.join(cache_dir, fname)
            try:
                os.remove(filepath)
                print(f"🔮 Model {fname} deleted")
            except Exception as e:
                print(f"[red]Error occurred while deleting model [cyan]{name}[/cyan]: {e}[/red]")


# ============================================================================
# Financial Data Pipeline (bfloat16 only)
# ============================================================================


class FinancialDataset:
    """Financial data management with bfloat16 encoding.

    This class encapsulates all financial data operations including:
    - S&P 500 ticker list
    - Data directory configuration
    - bfloat16 encoding/decoding
    - Data loading with automatic download from Alpaca API

    Attributes:
        DATA_DIR: Default data storage directory (~/.stable_worldmodel/data)
        SP500_TICKERS: List of S&P 500 stock symbols
    """

    DATA_DIR = str(Path.home() / ".stable_worldmodel" / "data")

    SP500_TICKERS = [
        "MMM",
        "AOS",
        "ABT",
        "ABBV",
        "ACN",  # codespell:ignore
        "ADBE",
        "AMD",
        "AES",
        "AFL",
        "A",
        "APD",
        "ABNB",
        "AKAM",
        "ALB",
        "ARE",
        "ALGN",
        "ALLE",  # codespell:ignore
        "LNT",
        "ALL",
        "GOOGL",
        "GOOG",
        "MO",
        "AMZN",
        "AMCR",
        "AEE",
        "AEP",
        "AXP",
        "AIG",
        "AMT",
        "AWK",
        "AMP",
        "AME",
        "AMGN",
        "APH",
        "ADI",
        "AON",
        "APA",
        "APO",
        "AAPL",
        "AMAT",
        "APP",
        "APTV",
        "ACGL",
        "ADM",
        "ANET",
        "AJG",
        "AIZ",
        "T",
        "ATO",
        "ADSK",
        "ADP",
        "AZO",
        "AVB",
        "AVY",
        "AXON",
        "BKR",
        "BALL",
        "BAC",
        "BAX",
        "BDX",
    ]

    @staticmethod
    def encode_bfloat16(prices: pd.Series) -> np.ndarray:
        """Encode price series using bfloat16 differential encoding."""
        if ml_dtypes is None:
            raise ImportError("ml_dtypes required for bfloat16 encoding")
        diffs = np.diff(prices.values, prepend=prices.values[0])
        return np.array(diffs, dtype=ml_dtypes.bfloat16).view(np.uint16)

    @staticmethod
    def decode_bfloat16(encoded: np.ndarray, anchor: float) -> np.ndarray:
        """Decode bfloat16 encoded prices."""
        if ml_dtypes is None:
            raise ImportError("ml_dtypes required for bfloat16 decoding")
        bf16 = encoded.astype(np.uint16).view(ml_dtypes.bfloat16)
        return anchor + np.cumsum(bf16.astype("float32"))

    @staticmethod
    def encode_financial_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Encode OHLC columns using bfloat16."""
        metadata = {}
        df_encoded = df.copy()

        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                anchor = float(df[col].iloc[0])
                encoded = FinancialDataset.encode_bfloat16(df[col])
                df_encoded[f"{col}_encoded"] = encoded
                metadata[f"{col}_anchor"] = anchor

        df_encoded = df_encoded.drop(columns=["open", "high", "low", "close"], errors="ignore")
        return df_encoded, metadata

    @staticmethod
    def decode_financial_data(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Decode OHLC columns from bfloat16."""
        df_decoded = df.copy()

        for col in ["open", "high", "low", "close"]:
            enc_col = f"{col}_encoded"
            if enc_col in df.columns:
                anchor = metadata[f"{col}_anchor"]
                df_decoded[col] = FinancialDataset.decode_bfloat16(df[enc_col].values, anchor)
                df_decoded = df_decoded.drop(columns=[enc_col])

        return df_decoded

    @staticmethod
    def load(
        stocks: list[str],
        dates: list[str | tuple[str, str]],
        base_dir: str | None = None,
    ) -> pd.DataFrame:
        """Load financial data for specified stocks and dates.

        Parameters
        ----------
        stocks : list[str]
            List of stock symbols (e.g., ['AAPL', 'TSLA'])
        dates : list[str | tuple[str, str]]
            List of dates or date ranges:
            - Individual dates: ['2024-01-17', '2024-02-20']
            - Date ranges: [('2024-01-01', '2024-01-31')]
        base_dir : str, optional
            Base directory for data storage

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with OHLCV data
        """
        if base_dir is None:
            base_dir = Path(FinancialDataset.DATA_DIR)
        else:
            base_dir = Path(base_dir)

        # Expand date ranges to individual dates
        expanded_dates = []
        for item in dates:
            if isinstance(item, tuple):
                start_date, end_date = item
                current = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                while current <= end:
                    if current.weekday() < 5:  # Weekdays only
                        expanded_dates.append(current.strftime("%Y-%m-%d"))
                    current += timedelta(days=1)
            else:
                expanded_dates.append(item)

        storage_dir = base_dir / "numpyformat_bfloat16"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Check for missing data
        missing_data = []
        available_data = []

        for symbol in stocks:
            for date in expanded_dates:
                file_path = storage_dir / f"{symbol}_{date}.npz"
                if file_path.exists():
                    data = np.load(file_path)
                    metadata = {
                        k: (float(data[k][0]) if k.endswith("_anchor") else data[k])
                        for k in data.keys()
                        if k.endswith("_anchor")
                    }
                    df = pd.DataFrame({"timestamp": data["timestamps"]})
                    for col in ["open", "high", "low", "close"]:
                        if f"{col}_encoded" in data:
                            df[col] = FinancialDataset.decode_bfloat16(
                                data[f"{col}_encoded"], metadata[f"{col}_anchor"]
                            )
                    if "volume" in data:
                        df["volume"] = data["volume"]
                    available_data.append(df)
                else:
                    missing_data.append((symbol, date))

        # Download missing data if needed
        if missing_data and _alpaca_client is not None:
            logger.info(f"Downloading {len(missing_data)} missing data files...")

            for symbol, date in missing_data:
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Minute,
                        start=datetime.fromisoformat(f"{date}T00:00:00"),
                        end=datetime.fromisoformat(f"{date}T23:59:59"),
                    )
                    df_raw = _alpaca_client.get_stock_bars(req).df.reset_index()

                    if df_raw.empty:
                        continue

                    df_raw["symbol"] = symbol
                    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True)
                    df_raw = df_raw[["timestamp", "open", "high", "low", "close", "volume"]]

                    # Encode and save
                    df_encoded, metadata = FinancialDataset.encode_financial_data(df_raw)
                    data_dict = {"timestamps": df_encoded["timestamp"].values}
                    for col in ["open", "high", "low", "close"]:
                        if f"{col}_encoded" in df_encoded.columns:
                            data_dict[f"{col}_encoded"] = df_encoded[f"{col}_encoded"].values
                    if "volume" in df_raw.columns:
                        data_dict["volume"] = df_raw["volume"].values
                    for key, value in metadata.items():
                        data_dict[key] = np.array([value])

                    file_path = storage_dir / f"{symbol}_{date}.npz"
                    np.savez_compressed(file_path, **data_dict)

                    available_data.append(df_raw)

                except Exception as e:
                    logger.error(f"Failed to download {symbol} on {date}: {e}")

        if available_data:
            return pd.concat(available_data, ignore_index=True)
        return pd.DataFrame()
