import numpy as np
import pytest
import torch
import h5py

from stable_worldmodel.data import Dataset, HDF5Dataset


class ConformingDataset:
    """A minimal dataset that conforms to the Dataset protocol."""

    def __init__(self):
        self._data = {
            "observation": torch.randn(20, 4),
            "action": torch.randn(20, 2),
        }

    @property
    def column_names(self) -> list[str]:
        return list(self._data.keys())

    def __len__(self) -> int:
        return 20

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self._data.items()}

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict[str, torch.Tensor]]:
        chunk = []
        for s, e in zip(start, end):
            chunk.append({k: v[s:e] for k, v in self._data.items()})
        return chunk

    def get_col_data(self, col: str) -> np.ndarray:
        return self._data[col].numpy()

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {k: v[row_idx] for k, v in self._data.items()}


class MissingLen:
    """Dataset missing __len__."""

    @property
    def column_names(self) -> list[str]:
        return []

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {}

    def load_chunk(self, episodes_idx, start, end):
        return []

    def get_col_data(self, col: str) -> np.ndarray:
        return np.array([])

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {}


class MissingGetitem:
    """Dataset missing __getitem__."""

    @property
    def column_names(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return 0

    def load_chunk(self, episodes_idx, start, end):
        return []

    def get_col_data(self, col: str) -> np.ndarray:
        return np.array([])

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {}


class MissingLoadChunk:
    """Dataset missing load_chunk."""

    @property
    def column_names(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {}

    def get_col_data(self, col: str) -> np.ndarray:
        return np.array([])

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {}


class MissingColumnNames:
    """Dataset missing column_names."""

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {}

    def load_chunk(self, episodes_idx, start, end):
        return []

    def get_col_data(self, col: str) -> np.ndarray:
        return np.array([])

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {}


class MissingGetColData:
    """Dataset missing get_col_data."""

    @property
    def column_names(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {}

    def load_chunk(self, episodes_idx, start, end):
        return []

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {}


class MissingGetRowData:
    """Dataset missing get_row_data."""

    @property
    def column_names(self) -> list[str]:
        return []

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {}

    def load_chunk(self, episodes_idx, start, end):
        return []

    def get_col_data(self, col: str) -> np.ndarray:
        return np.array([])


def test_protocol_conforming_dataset():
    """A class implementing all required methods is an instance of Dataset."""
    ds = ConformingDataset()
    assert isinstance(ds, Dataset)


def test_protocol_missing_len():
    """A class missing __len__ does not satisfy the protocol."""
    assert not isinstance(MissingLen(), Dataset)


def test_protocol_missing_getitem():
    """A class missing __getitem__ does not satisfy the protocol."""
    assert not isinstance(MissingGetitem(), Dataset)


def test_protocol_missing_load_chunk():
    """A class missing load_chunk does not satisfy the protocol."""
    assert not isinstance(MissingLoadChunk(), Dataset)


def test_protocol_missing_column_names():
    """A class missing column_names does not satisfy the protocol."""
    assert not isinstance(MissingColumnNames(), Dataset)


def test_protocol_missing_get_col_data():
    """A class missing get_col_data does not satisfy the protocol."""
    assert not isinstance(MissingGetColData(), Dataset)


def test_protocol_missing_get_row_data():
    """A class missing get_row_data does not satisfy the protocol."""
    assert not isinstance(MissingGetRowData(), Dataset)


def test_protocol_len_returns_int():
    """__len__ returns an integer."""
    ds = ConformingDataset()
    assert isinstance(len(ds), int)


def test_protocol_getitem_returns_dict():
    """__getitem__ returns a dict of tensors."""
    ds = ConformingDataset()
    item = ds[0]
    assert isinstance(item, dict)
    for v in item.values():
        assert isinstance(v, torch.Tensor)


def test_protocol_column_names_returns_list():
    """column_names returns a list of strings."""
    ds = ConformingDataset()
    cols = ds.column_names
    assert isinstance(cols, list)
    for c in cols:
        assert isinstance(c, str)


def test_protocol_load_chunk_returns_list():
    """load_chunk returns a list of dicts."""
    ds = ConformingDataset()
    episodes_idx = np.array([0, 0])
    start = np.array([0, 5])
    end = np.array([5, 10])
    chunk = ds.load_chunk(episodes_idx, start, end)
    assert isinstance(chunk, list)
    for item in chunk:
        assert isinstance(item, dict)
        for v in item.values():
            assert isinstance(v, torch.Tensor)


@pytest.fixture
def sample_h5_file(tmp_path):
    """Create a sample HDF5 file for testing."""
    h5_path = tmp_path / "test_dataset.h5"

    ep_lengths = [10, 10]
    ep_offsets = [0, 10]
    total_steps = sum(ep_lengths)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("ep_len", data=np.array(ep_lengths))
        f.create_dataset("ep_offset", data=np.array(ep_offsets))
        f.create_dataset("observation", data=np.random.rand(total_steps, 4).astype(np.float32))
        f.create_dataset("action", data=np.random.rand(total_steps, 2).astype(np.float32))

    return tmp_path, "test_dataset"


def test_hdf5_dataset_conforms_to_protocol(sample_h5_file):
    """HDF5Dataset should satisfy the Dataset protocol."""
    cache_dir, name = sample_h5_file
    dataset = HDF5Dataset(name, cache_dir=str(cache_dir))
    assert isinstance(dataset, Dataset)
