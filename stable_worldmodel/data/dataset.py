import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from PIL import Image

from stable_worldmodel.data.utils import get_cache_dir


class Dataset:
    """Base class for episode-based datasets."""

    def __init__(
        self,
        lengths: np.ndarray,
        offsets: np.ndarray,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
    ):
        self.lengths = lengths
        self.offsets = offsets
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.span = num_steps * frameskip
        self.transform = transform
        self.clip_indices = [
            (ep, start)
            for ep, length in enumerate(lengths)
            if length >= self.span
            for start in np.linspace(0, length - self.span, dtype=int)
        ]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]
        steps = self._load_slice(ep_idx, start, start + self.span)
        if 'action' in steps:
            steps['action'] = steps['action'].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for ep, s, e in zip(episodes_idx, start, end):
            steps = self._load_slice(ep, s, e)
            if 'action' in steps:
                steps['action'] = steps['action'].reshape(
                    (e - s) // self.frameskip, -1
                )
            chunk.append(steps)
        return chunk

    def get_col_data(self, col: str) -> np.ndarray:
        raise NotImplementedError

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        raise NotImplementedError


class HDF5Dataset(Dataset):
    """Dataset loading from HDF5 file."""

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
        self.h5_path = Path(cache_dir or get_cache_dir(), f'{name}.h5')
        self.h5_file = None
        self._cache = {}

        with h5py.File(self.h5_path, 'r') as f:
            lengths, offsets = f['ep_len'][:], f['ep_offset'][:]
            self._keys = keys_to_load or [
                k for k in f.keys() if k not in ('ep_len', 'ep_offset')
            ]
            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]
                logging.info(f"Cached '{key}' from '{self.h5_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(
                self.h5_path, 'r', swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            src = self._cache if col in self._cache else self.h5_file
            data = src[col][g_start:g_end]
            if col != 'action':
                data = data[:: self.frameskip]
            steps[col] = torch.from_numpy(data)
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        self._open()
        return self.h5_file[col][:]

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        return {col: self.h5_file[col][row_idx] for col in self._keys}


class FolderDataset(Dataset):
    """Dataset loading from folder with images/videos."""

    def __init__(
        self,
        name: str,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable | None = None,
        keys_to_load: list[str] | None = None,
        folder_keys: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self.path = Path(cache_dir or get_cache_dir()) / name
        self.folder_keys = folder_keys or []
        self._cache = {}

        lengths = np.load(self.path / 'ep_len.npz')['arr_0']
        offsets = np.load(self.path / 'ep_offset.npz')['arr_0']

        if keys_to_load is None:
            keys_to_load = [
                p.stem if p.suffix == '.npz' else p.name
                for p in self.path.iterdir()
                if p.stem not in ('ep_len', 'ep_offset')
            ]
        self._keys = keys_to_load

        for key in self._keys:
            if key not in self.folder_keys:
                npz = self.path / f'{key}.npz'
                if npz.exists():
                    self._cache[key] = np.load(npz)['arr_0']
                    logging.info(f"Cached '{key}' from '{npz}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _load_file(self, ep_idx: int, step: int, key: str) -> np.ndarray:
        return np.array(
            Image.open(self.path / key / f'ep_{ep_idx}_step_{step}.jpeg')
        )

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            if col in self.folder_keys:
                data = np.stack(
                    [
                        self._load_file(ep_idx, s, col)
                        for s in range(start, end, self.frameskip)
                    ]
                )
            else:
                data = self._cache[col][g_start:g_end]
                if col != 'action':
                    data = data[:: self.frameskip]
            steps[col] = torch.from_numpy(data)
            if data.ndim == 4 and data.shape[-1] in (1, 3):
                steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col not in self._cache:
            raise KeyError(f"'{col}' not in cache")
        return self._cache[col]

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        return {
            c: self._cache[c][row_idx] for c in self._keys if c in self._cache
        }


class ImageDataset(FolderDataset):
    """Convenience alias for FolderDataset with image defaults."""

    def __init__(self, name: str, image_keys: list[str] | None = None, **kw):
        super().__init__(name, folder_keys=image_keys or ['pixels'], **kw)


class VideoDataset(FolderDataset):
    """Dataset loading video frames from MP4 files."""

    def __init__(self, name: str, video_keys: list[str] | None = None, **kw):
        try:
            import decord

            decord.bridge.set_bridge('torch')
            self._decord = decord
        except ImportError:
            raise ImportError('VideoDataset requires decord')
        super().__init__(name, folder_keys=video_keys or ['video'], **kw)

    @lru_cache(maxsize=8)
    def _reader(self, ep_idx: int, key: str):
        return self._decord.VideoReader(
            str(self.path / key / f'ep_{ep_idx}.mp4'), num_threads=1
        )

    def _load_file(self, ep_idx: int, step: int, key: str) -> np.ndarray:
        return self._reader(ep_idx, key)[step].numpy()

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        g_start, g_end = (
            self.offsets[ep_idx] + start,
            self.offsets[ep_idx] + end,
        )
        steps = {}
        for col in self._keys:
            if col in self.folder_keys:
                frames = self._reader(ep_idx, col).get_batch(
                    list(range(start, end, self.frameskip))
                )
                steps[col] = frames.permute(0, 3, 1, 2)
            else:
                data = self._cache[col][g_start:g_end]
                if col != 'action':
                    data = data[:: self.frameskip]
                steps[col] = torch.from_numpy(data)
        return self.transform(steps) if self.transform else steps


class MergeDataset:
    """Merges multiple datasets of same length."""

    def __init__(
        self, datasets: list, keys_from_dataset: list[list[str]] | None = None
    ):
        if not datasets:
            raise ValueError('Need at least one dataset')
        self.datasets = datasets
        self._len = len(datasets[0])

        if keys_from_dataset:
            self.keys_map = keys_from_dataset
        else:
            # Auto-deduplicate: each dataset provides keys not seen in previous datasets
            seen = set()
            self.keys_map = []
            for ds in datasets:
                keys = [c for c in ds.column_names if c not in seen]
                seen.update(keys)
                self.keys_map.append(keys)

    @property
    def column_names(self) -> list[str]:
        cols = []
        for keys in self.keys_map:
            cols.extend(keys)
        return cols

    @property
    def lengths(self) -> np.ndarray:
        """Episode lengths from first dataset (all merged datasets share same structure)."""
        return self.datasets[0].lengths

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        out = {}
        for ds, keys in zip(self.datasets, self.keys_map):
            item = ds[idx]
            for k in keys:
                if k in item:
                    out[k] = item[k]
        return out

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        all_chunks = [
            ds.load_chunk(episodes_idx, start, end) for ds in self.datasets
        ]

        merged = []
        for items in zip(*all_chunks):
            combined = {}
            for item in items:
                combined.update(item)
            merged.append(combined)
        return merged

    def get_col_data(self, col: str) -> np.ndarray:
        for ds, keys in zip(self.datasets, self.keys_map):
            if col in keys:
                return ds.get_col_data(col)
        raise KeyError(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        out = {}
        for ds, keys in zip(self.datasets, self.keys_map):
            data = ds.get_row_data(row_idx)
            for k in keys:
                if k in data:
                    out[k] = data[k]
        return out


class ConcatDataset:
    """Concatenates multiple datasets."""

    def __init__(self, datasets: list):
        if not datasets:
            raise ValueError('Need at least one dataset')
        self.datasets = datasets

        # Cumulative lengths for index mapping: [0, len(ds0), len(ds0)+len(ds1), ...]
        lengths = [len(ds) for ds in datasets]
        self._cum = np.cumsum([0] + lengths)

        # Cumulative episode counts for load_chunk mapping
        ep_counts = [len(ds.lengths) for ds in datasets]
        self._ep_cum = np.cumsum([0] + ep_counts)

    @property
    def column_names(self) -> list[str]:
        seen = set()
        cols = []
        for ds in self.datasets:
            for c in ds.column_names:
                if c not in seen:
                    seen.add(c)
                    cols.append(c)
        return cols

    def __len__(self) -> int:
        return self._cum[-1]

    def _loc(self, idx: int) -> tuple[int, int]:
        """Map global index to (dataset_index, local_index)."""
        if idx < 0:
            idx += len(self)
        ds_idx = np.searchsorted(self._cum[1:], idx, side='right')
        local_idx = idx - self._cum[ds_idx]
        return ds_idx, local_idx

    def __getitem__(self, idx: int) -> dict:
        ds_idx, local_idx = self._loc(idx)
        return self.datasets[ds_idx][local_idx]

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        episodes_idx = np.asarray(episodes_idx)
        start = np.asarray(start)
        end = np.asarray(end)

        # Map global episode indices to dataset indices
        ds_indices = np.searchsorted(self._ep_cum[1:], episodes_idx, side='right')
        local_eps = episodes_idx - self._ep_cum[ds_indices]

        # Group by dataset and collect results
        results: list[dict | None] = [None] * len(episodes_idx)
        for ds_idx in range(len(self.datasets)):
            mask = ds_indices == ds_idx
            if not np.any(mask):
                continue

            chunks = self.datasets[ds_idx].load_chunk(
                local_eps[mask], start[mask], end[mask]
            )

            # Place results back in original order
            for i, chunk in zip(np.where(mask)[0], chunks):
                results[i] = chunk

        return results  # type: ignore[return-value]

    def get_col_data(self, col: str) -> np.ndarray:
        data = []
        for ds in self.datasets:
            if col in ds.column_names:
                data.append(ds.get_col_data(col))
        if not data:
            raise KeyError(col)
        return np.concatenate(data)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        if isinstance(row_idx, int):
            ds_idx, local_idx = self._loc(row_idx)
            return self.datasets[ds_idx].get_row_data(local_idx)

        # Multiple indices: collect and stack results
        results = {}
        for idx in row_idx:
            ds_idx, local_idx = self._loc(idx)
            row = self.datasets[ds_idx].get_row_data(local_idx)
            for k, v in row.items():
                if k not in results:
                    results[k] = []
                results[k].append(v)

        return {k: np.stack(v) for k, v in results.items()}
