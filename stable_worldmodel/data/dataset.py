from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torchcodec.decoders import VideoDecoder
from torchvision.io import decode_image

from stable_worldmodel.data.utils import get_cache_dir


class Dataset:
    def __init__(self, name, frameskip=1, num_steps=1, decode_columns=None, cache_dir=None):
        self.data_dir = Path(cache_dir or get_cache_dir(), name)
        self.dataset = load_from_disk(self.data_dir)
        self.frameskip = frameskip
        self.num_steps = num_steps
        self.dataset.set_format("torch")
        self.complete_traj = num_steps < 0

        self.transform = None  # TODO

        if type(decode_columns) is str:
            decode_columns = [decode_columns]
        self.decode_columns = decode_columns or []

        assert "episode_idx" in self.dataset.column_names, "Dataset must have 'episode_idx' column"
        assert "step_idx" in self.dataset.column_names, "Dataset must have 'step_idx' column"
        assert "action" in self.dataset.column_names, "Dataset must have 'action' column"

        episode_col = self.dataset["episode_idx"][:]

        self.episodes = np.unique(episode_col)
        self.episode_indices = {ep: np.flatnonzero(episode_col == ep) for ep in self.episodes}

        self.clip_len = max(frameskip * num_steps, 1) if not self.complete_traj else 0

        if any(len(self.episode_indices[ep]) < self.clip_len for ep in self.episodes):
            raise ValueError(f"All episodes must have at least {self.clip_len} steps")

        episode_max_end = [max(0, len(ep) - self.clip_len + 1) for ep in self.episode_indices.values()]
        self.episode_starts = np.cumsum([0] + episode_max_end)
        self.idx_to_episode = np.searchsorted(self.episode_starts, np.arange(len(self)), side="right") - 1

        return

    def __len__(self):
        return int(self.episode_starts[-1]) if not self.complete_traj else len(self.episodes)

    def decode(self, col_data, indices):
        raise NotImplementedError("Dataset.decode must be implemented in subclass")

    def load_chunk(self, episode, start, end):
        if type(episode) is int:
            episode = [episode]

        if type(start) is int:
            start = [start] * len(episode)

        if type(end) is int:
            end = [end] * len(episode)

        if not (self.frameskip == 1 and self.num_steps == 1):
            raise NotImplementedError("Dataset.load_chunk need only be have frameskip=1 and num_steps=1")

        chunks = []

        for ep, s, en in zip(episode, start, end):
            episode_indices = self.episode_indices[ep]

            if ep > len(self.episodes) or ep < 0:
                raise ValueError(f"Episode {ep} index out of range [0, {len(self.episodes)})")

            if en > len(episode_indices) or s < 0 or en <= s:
                raise ValueError(f"Invalid start/end indices for episode {ep}: [{s}, {en})")

            episode_mask = self.dataset["episode_idx"][:] == ep
            steps_mask = self.dataset["step_idx"][:]
            steps_mask = (steps_mask >= s) & (steps_mask < en)

            indices = np.flatnonzero(episode_mask & steps_mask)
            steps = self.dataset[indices]

            for col, data in steps.items():
                if col == "action":
                    continue

                data = data[:: self.frameskip]
                steps[col] = data

                if col in self.decode_columns:
                    steps[col] = self.decode(steps[col], start=s, end=en)

            if self.transform:
                steps = self.transform(steps)

            # stack frames
            for col in self.decode_columns:
                if col not in steps:
                    continue
                steps[col] = torch.stack(steps[col])

            # reshape action
            act_shape = en - s
            steps["action"] = steps["action"].reshape(act_shape, -1)

            chunks.append(steps)

        return chunks


class FrameDataset(Dataset):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.decode_columns = self.decode_columns or self.determine_img_columns(self.dataset[0])

    def decode(self, col_data, start=0, end=-1):
        return [decode_image(self.data_dir / img_path) for img_path in col_data]

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_indices = self.episode_indices[episode]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps[col])

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(steps[col])

        # reshape action
        act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode])
        steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def determine_img_columns(self, sample):
        # TODO: support other image formats
        img_columns = {k for k in sample.keys() if isinstance(sample[k], str) and k.endswith(".jpeg")}
        return img_columns


class VideoDataset(Dataset):
    def __init__(self, name, *args, device="cpu", **kwargs):
        super().__init__(name, *args, **kwargs)
        self.device = device
        self.decode_columns = self.decode_columns or self.determine_video_columns(self.dataset[0])

    def decode(self, col_data, start=0, end=-1):
        video = VideoDecoder(self.data_dir / col_data[0], device=self.device)
        return list(video[start : end : self.frameskip])

    def __getitem__(self, index):
        episode = self.idx_to_episode[index]
        episode_indices = self.episode_indices[episode]
        offset = index - self.episode_starts[episode]

        # determine clip bounds
        start = offset if not self.complete_traj else 0
        stop = start + self.clip_len if not self.complete_traj else len(self.episode_indices[episode])
        step_slice = episode_indices[start:stop]
        steps = self.dataset[step_slice]

        for col, data in steps.items():
            if col == "action":
                continue

            data = data[:: self.frameskip]
            steps[col] = data

            if col in self.decode_columns:
                steps[col] = self.decode(steps[col], start=start, end=stop)

        if self.transform:
            steps = self.transform(steps)

        # stack frames
        for col in self.decode_columns:
            if col not in steps:
                continue
            steps[col] = torch.stack(steps[col])

        # reshape action
        act_shape = self.num_steps if not self.complete_traj else len(self.episode_indices[episode])
        steps["action"] = steps["action"].reshape(act_shape, -1)

        return steps

    def determine_video_columns(self, sample):
        # TODO: support other video formats
        video_columns = {k for k in sample.keys() if isinstance(sample[k], str) and k.endswith(".mp4")}
        return video_columns


# TODO check framedataset and videodataset give the same results
