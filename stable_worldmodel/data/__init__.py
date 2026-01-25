from . import utils
from .dataset import (
    HDF5Dataset,
    ImageDataset,
    VideoDataset,
    MergeDataset,
    ConcatDataset,
)
from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "utils",
    "HDF5Dataset",
    "ImageDataset",
    "VideoDataset",
    "MergeDataset",
    "ConcatDataset",
    "Dataset",
]

@runtime_checkable
class Dataset(Protocol):
    """Protocol defining the interface for episode-based datasets."""

    @property
    def column_names(self) -> list[str]:
        """Return the list of available data column names (e.g. 'observation', 'action')."""
        ...

    def __len__(self) -> int:
        """Return the total number of indexable samples in the dataset."""
        ...

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample as a dict mapping column names to arrays."""
        ...

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray,
    ) -> list[dict]:
        """Load contiguous slices of episodes. Returns one dict per episode segment."""
        ...

    def get_col_data(self, col: str) -> np.ndarray:
        """Return all data for a given column."""
        ...

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return a dict mapping column names to data at the given row index."""
        ...
