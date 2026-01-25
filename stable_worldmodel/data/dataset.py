import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.utils import get_cache_dir


class HDF5Dataset:
    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.h5_path = Path(cache_dir or get_cache_dir(), f"{name}.h5")
        self.h5_file = None
        self._cache = {}

        self.keys_to_load = keys_to_load
        self.metadata_keys = ["ep_len", "ep_offset"]
        self.keys_to_cache = keys_to_cache or []

        with h5py.File(self.h5_path, "r") as f:
            self.offsets = f["ep_offset"][:]
            self.lengths = f["ep_len"][:]

            if self.keys_to_load is None:
                self.keys_to_load = list(f.keys())

            for key in self.keys_to_cache:
                if key in f:
                    self._cache[key] = f[key][:]
                    logging.info(f"Cached key '{key}' from HDF5 file '{self.h5_path}'")
                else:
                    raise KeyError(f"Key '{key}' not found in HDF5 file '{self.h5_path}'")

        self.transform = transform
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip

        # valid episode indices
        self.clip_indices = []
        for ep_idx, length in enumerate(self.lengths):
            if length >= self.span:
                for start_f in np.linspace(0, length - self.span, dtype=int):
                    self.clip_indices.append((ep_idx, start_f))
        return

    @property
    def column_names(self):
        return [key for key in self.keys_to_load if key not in self.metadata_keys]

    def __len__(self):
        return len(self.clip_indices)

    def _init_h5(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024)
        return

    def load_slice(self, start_idx, end_idx):
        steps = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue

            if col in self._cache:
                data = self._cache[col][start_idx:end_idx]
            else:
                data = self.h5_file[col][start_idx:end_idx]

            # apply frameskip if not action
            if col != "action":
                data = data[:: self.frameskip]

            steps[col] = torch.from_numpy(data)

            # channel first for images
            is_img = len(data.shape) == 4 and data.shape[-1] in [1, 3]
            if is_img:
                steps[col] = steps[col].permute(0, 3, 1, 2)  # TCHW

        if self.transform:
            steps = self.transform(steps)

        return steps

    def __getitem__(self, idx: int):
        ep_idx, local_start = self.clip_indices[idx]
        start = self.offsets[ep_idx] + local_start
        end = start + self.span

        self._init_h5()
        steps = self.load_slice(start, end)

        if "action" in steps:
            act_shape = self.num_steps
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def load_chunk(self, episodes_idx, start, end):
        self._init_h5()
        global_start = self.offsets[episodes_idx] + start
        end = global_start + (end - start)

        chunk = []
        for s_idx, e_idx in zip(global_start, end):
            steps = self.load_slice(s_idx, e_idx)

            # reshape action
            if "action" in steps:
                act_shape = (e_idx - s_idx) // self.frameskip
                steps["action"] = steps["action"].reshape(act_shape, -1)
            chunk.append(steps)

        return chunk

    def get_col_data(self, col: str):
        self._init_h5()
        return self.h5_file[col][:]

    def get_row_data(self, row_idx: int | list[int]):
        self._init_h5()
        sample = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue
            sample[col] = self.h5_file[col][row_idx]

        return sample


class ImageDataset:
    """Dataset that loads data from .npz files and images from folders. """

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
        keys_to_load: list[str] | None = None,
        image_keys: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.data_path = Path(cache_dir or get_cache_dir()) / name
        self._cache = {}

        self.image_keys = image_keys or ["pixels"]
        self.metadata_keys = ["ep_len", "ep_offset"]

        # Load metadata
        self.lengths = np.load(self.data_path / "ep_len.npz")["arr_0"]
        self.offsets = np.load(self.data_path / "ep_offset.npz")["arr_0"]

        # Discover keys if not specified
        if keys_to_load is None:
            keys_to_load = []
            for p in self.data_path.iterdir():
                if p.suffix == ".npz" and p.stem not in self.metadata_keys:
                    keys_to_load.append(p.stem)
                elif p.is_dir():
                    keys_to_load.append(p.name)
        self.keys_to_load = keys_to_load

        # Load non-image data into cache
        for key in self.keys_to_load:
            if key in self.image_keys:
                continue
            npz_path = self.data_path / f"{key}.npz"
            if npz_path.exists():
                self._cache[key] = np.load(npz_path)["arr_0"]
                logging.info(f"Cached key '{key}' from '{npz_path}'")

        self.transform = transform
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip

        # Build valid clip indices
        self.clip_indices = []
        for ep_idx, length in enumerate(self.lengths):
            if length >= self.span:
                for start_f in np.linspace(0, length - self.span, dtype=int):
                    self.clip_indices.append((ep_idx, start_f))

    @property
    def column_names(self):
        return [key for key in self.keys_to_load if key not in self.metadata_keys]

    def __len__(self):
        return len(self.clip_indices)

    def _load_image(self, ep_idx: int, step_idx: int, key: str) -> np.ndarray:
        """Load a single image from disk."""
        img_path = self.data_path / key / f"ep_{ep_idx}_step_{step_idx}.jpeg"
        img = Image.open(img_path)
        return np.array(img)

    def load_slice(self, ep_idx: int, local_start: int, local_end: int):
        """Load a slice of data from an episode."""
        steps = {}
        global_start = self.offsets[ep_idx] + local_start
        global_end = self.offsets[ep_idx] + local_end

        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue

            if col in self.image_keys:
                # Load images from folder
                images = []
                for step in range(local_start, local_end, self.frameskip if col != "action" else 1):
                    img = self._load_image(ep_idx, step, col)
                    images.append(img)
                data = np.stack(images)
            else:
                # Load from cache
                data = self._cache[col][global_start:global_end]
                if col != "action":
                    data = data[:: self.frameskip]

            steps[col] = torch.from_numpy(data)

            # Channel first for images (THWC -> TCHW)
            is_img = len(data.shape) == 4 and data.shape[-1] in [1, 3]
            if is_img:
                steps[col] = steps[col].permute(0, 3, 1, 2)

        if self.transform:
            steps = self.transform(steps)

        return steps

    def __getitem__(self, idx: int):
        ep_idx, local_start = self.clip_indices[idx]
        local_end = local_start + self.span

        steps = self.load_slice(ep_idx, local_start, local_end)

        if "action" in steps:
            act_shape = self.num_steps
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray):
        """Load contiguous slices of episodes."""
        chunk = []
        for ep_idx, s, e in zip(episodes_idx, start, end):
            steps = self.load_slice(ep_idx, s, e)
            if "action" in steps:
                act_shape = (e - s) // self.frameskip
                steps["action"] = steps["action"].reshape(act_shape, -1)
            chunk.append(steps)
        return chunk

    def get_col_data(self, col: str) -> np.ndarray:
        """Return all data for a given column."""
        if col in self._cache:
            return self._cache[col]
        raise KeyError(f"Column '{col}' not found in cache. Image columns not supported.")

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return data at the given row index."""
        sample = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue
            if col in self._cache:
                sample[col] = self._cache[col][row_idx]
        return sample


class VideoDataset:
    """Dataset that loads data from .npz files and video frames from folders."""

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
        keys_to_load: list[str] | None = None,
        video_keys: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.data_path = Path(cache_dir or get_cache_dir()) / name
        self._cache = {}

        self.video_keys = video_keys or ["video"]
        self.metadata_keys = ["ep_len", "ep_offset"]

        # Load metadata
        self.lengths = np.load(self.data_path / "ep_len.npz")["arr_0"]
        self.offsets = np.load(self.data_path / "ep_offset.npz")["arr_0"]

        # Discover keys if not specified
        if keys_to_load is None:
            keys_to_load = []
            for p in self.data_path.iterdir():
                if p.suffix == ".npz" and p.stem not in self.metadata_keys:
                    keys_to_load.append(p.stem)
                elif p.is_dir():
                    keys_to_load.append(p.name)
        self.keys_to_load = keys_to_load

        # Load non-video data into cache
        for key in self.keys_to_load:
            if key in self.video_keys:
                continue
            npz_path = self.data_path / f"{key}.npz"
            if npz_path.exists():
                self._cache[key] = np.load(npz_path)["arr_0"]
                logging.info(f"Cached key '{key}' from '{npz_path}'")

        self.transform = transform
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip

        # Build valid clip indices
        self.clip_indices = []
        for ep_idx, length in enumerate(self.lengths):
            if length >= self.span:
                for start_f in np.linspace(0, length - self.span, dtype=int):
                    self.clip_indices.append((ep_idx, start_f))

    @property
    def column_names(self):
        return [key for key in self.keys_to_load if key not in self.metadata_keys]

    def __len__(self):
        return len(self.clip_indices)

    def _load_frame(self, ep_idx: int, step_idx: int, key: str) -> np.ndarray:
        """Load a single video frame from disk."""
        frame_path = self.data_path / key / f"ep_{ep_idx}_step_{step_idx}.jpeg"
        img = Image.open(frame_path)
        return np.array(img)

    def load_slice(self, ep_idx: int, local_start: int, local_end: int):
        """Load a slice of data from an episode."""
        steps = {}
        global_start = self.offsets[ep_idx] + local_start
        global_end = self.offsets[ep_idx] + local_end

        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue

            if col in self.video_keys:
                # Load video frames from folder
                frames = []
                for step in range(local_start, local_end, self.frameskip if col != "action" else 1):
                    frame = self._load_frame(ep_idx, step, col)
                    frames.append(frame)
                data = np.stack(frames)
            else:
                # Load from cache
                data = self._cache[col][global_start:global_end]
                if col != "action":
                    data = data[:: self.frameskip]

            steps[col] = torch.from_numpy(data)

            # Channel first for video frames (THWC -> TCHW)
            is_video = len(data.shape) == 4 and data.shape[-1] in [1, 3]
            if is_video:
                steps[col] = steps[col].permute(0, 3, 1, 2)

        if self.transform:
            steps = self.transform(steps)

        return steps

    def __getitem__(self, idx: int):
        ep_idx, local_start = self.clip_indices[idx]
        local_end = local_start + self.span

        steps = self.load_slice(ep_idx, local_start, local_end)

        if "action" in steps:
            act_shape = self.num_steps
            steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray):
        """Load contiguous slices of episodes."""
        chunk = []
        for ep_idx, s, e in zip(episodes_idx, start, end):
            steps = self.load_slice(ep_idx, s, e)
            if "action" in steps:
                act_shape = (e - s) // self.frameskip
                steps["action"] = steps["action"].reshape(act_shape, -1)
            chunk.append(steps)
        return chunk

    def get_col_data(self, col: str) -> np.ndarray:
        """Return all data for a given column."""
        if col in self._cache:
            return self._cache[col]
        raise KeyError(f"Column '{col}' not found in cache. Video columns not supported.")

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return data at the given row index."""
        sample = {}
        for col in self.keys_to_load:
            if col in self.metadata_keys:
                continue
            if col in self._cache:
                sample[col] = self._cache[col][row_idx]
        return sample



class MergeDataset:
    """Dataset that merges data from multiple datasets efficiently.

    Designed for cases where you have separate datasets for different modalities
    (e.g., images, audio) that share common keys (e.g., action, proprioception).

    Args:
        datasets: List of datasets to merge.
        keys_from_dataset: Optional list of lists specifying which keys to load
            from each dataset. If None, loads all keys (with later datasets
            overwriting earlier ones for shared keys).
        shared_keys_source: Index of the dataset to use for shared keys when
            keys_from_dataset is not specified. Defaults to 0 (first dataset).

    Example:
        # Dataset A has: pixels, action, observation
        # Dataset B has: audio, action, observation
        merged = MergeDataset(
            datasets=[ds_a, ds_b],
            keys_from_dataset=[
                ["pixels", "action", "observation"],  # from ds_a
                ["audio"],  # from ds_b (skip redundant action/observation)
            ]
        )
    """

    def __init__(
        self,
        datasets: list,
        keys_from_dataset: list[list[str]] | None = None,
    ):
        if not datasets:
            raise ValueError("MergeDataset requires at least one dataset")

        self.datasets = datasets
        self._length = len(datasets[0])

        # Verify all datasets have the same length
        for i, ds in enumerate(datasets[1:], 1):
            if len(ds) != self._length:
                raise ValueError(
                    f"All datasets must have the same length. "
                    f"Dataset 0 has length {self._length}, dataset {i} has length {len(ds)}"
                )

        # Determine which keys to load from each dataset
        if keys_from_dataset is not None:
            if len(keys_from_dataset) != len(datasets):
                raise ValueError(
                    f"keys_from_dataset must have same length as datasets. "
                    f"Got {len(keys_from_dataset)} vs {len(datasets)}"
                )
            self.keys_from_dataset = keys_from_dataset
        else:
            # Default: load all keys, later datasets overwrite earlier
            # But track which keys come from which dataset to avoid redundancy
            seen_keys = set()
            self.keys_from_dataset = []
            for ds in datasets:
                ds_keys = []
                for col in ds.column_names:
                    if col not in seen_keys:
                        ds_keys.append(col)
                        seen_keys.add(col)
                self.keys_from_dataset.append(ds_keys)

        # Build column_names list (preserving order)
        self._column_names = []
        for keys in self.keys_from_dataset:
            for k in keys:
                if k not in self._column_names:
                    self._column_names.append(k)

    @property
    def column_names(self) -> list[str]:
        """Return list of all column names that will be loaded."""
        return self._column_names

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        """Return merged dict, loading only specified keys from each dataset."""
        merged = {}
        for ds, keys in zip(self.datasets, self.keys_from_dataset):
            if not keys:
                continue
            item = ds[idx]
            for k in keys:
                if k in item:
                    merged[k] = item[k]
        return merged

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        """Load and merge chunks, loading only specified keys from each dataset."""
        # Collect chunks from each dataset
        all_chunks = []
        for ds, keys in zip(self.datasets, self.keys_from_dataset):
            if not keys:
                all_chunks.append([{} for _ in range(len(episodes_idx))])
                continue
            chunks = ds.load_chunk(episodes_idx, start, end)
            # Filter to only requested keys
            filtered = [{k: c[k] for k in keys if k in c} for c in chunks]
            all_chunks.append(filtered)

        # Merge chunks
        merged_chunks = []
        for items in zip(*all_chunks):
            merged = {}
            for item in items:
                merged.update(item)
            merged_chunks.append(merged)
        return merged_chunks

    def get_col_data(self, col: str) -> np.ndarray:
        """Return column data from the dataset assigned to provide it."""
        for ds, keys in zip(self.datasets, self.keys_from_dataset):
            if col in keys:
                return ds.get_col_data(col)
        raise KeyError(f"Column '{col}' not assigned to any dataset")

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return merged row data, loading only specified keys from each dataset."""
        merged = {}
        for ds, keys in zip(self.datasets, self.keys_from_dataset):
            if not keys:
                continue
            data = ds.get_row_data(row_idx)
            for k in keys:
                if k in data:
                    merged[k] = data[k]
        return merged


class ConcatDataset:
    """Dataset that concatenates multiple datasets.

    Length is the sum of all individual dataset lengths.
    """

    def __init__(self, datasets: list):
        if not datasets:
            raise ValueError("ConcatDataset requires at least one dataset")

        self.datasets = datasets
        self._lengths = [len(ds) for ds in datasets]
        self._cumulative_lengths = np.cumsum([0] + self._lengths)
        self._total_length = sum(self._lengths)

    @property
    def column_names(self) -> list[str]:
        """Return union of all column names from all datasets."""
        cols = []
        for ds in self.datasets:
            for col in ds.column_names:
                if col not in cols:
                    cols.append(col)
        return cols

    def __len__(self) -> int:
        return self._total_length

    def _get_dataset_and_idx(self, idx: int) -> tuple[int, int]:
        """Map global index to (dataset_index, local_index)."""
        if idx < 0:
            idx = self._total_length + idx
        if idx < 0 or idx >= self._total_length:
            raise IndexError(f"Index {idx} out of range for ConcatDataset of length {self._total_length}")

        # Binary search for the dataset
        ds_idx = np.searchsorted(self._cumulative_lengths[1:], idx, side="right")
        local_idx = idx - self._cumulative_lengths[ds_idx]
        return ds_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        """Return item from the appropriate dataset."""
        ds_idx, local_idx = self._get_dataset_and_idx(idx)
        return self.datasets[ds_idx][local_idx]

    def load_chunk(self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray) -> list[dict]:
        # Note: load_chunk assumes episode indices are within a single dataset.
        return self.datasets[0].load_chunk(episodes_idx, start, end)

    def get_col_data(self, col: str) -> np.ndarray:
        """Return concatenated column data from all datasets."""
        col_data = []
        for ds in self.datasets:
            if col in ds.column_names:
                col_data.append(ds.get_col_data(col))
        if not col_data:
            raise KeyError(f"Column '{col}' not found in any dataset")
        return np.concatenate(col_data, axis=0)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        """Return row data from the appropriate dataset."""
        if isinstance(row_idx, int):
            ds_idx, local_idx = self._get_dataset_and_idx(row_idx)
            return self.datasets[ds_idx].get_row_data(local_idx)
        else:
            # For list of indices, map each to appropriate dataset
            results = {}
            for idx in row_idx:
                ds_idx, local_idx = self._get_dataset_and_idx(idx)
                data = self.datasets[ds_idx].get_row_data(local_idx)
                for k, v in data.items():
                    if k not in results:
                        results[k] = []
                    results[k].append(v)
            # Stack the results
            return {k: np.stack(v) for k, v in results.items()}

