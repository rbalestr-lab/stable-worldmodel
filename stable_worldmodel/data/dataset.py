import logging
from collections.abc import Callable
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

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

    def get_chunk_data(self, episodes_idx, start, end):
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
