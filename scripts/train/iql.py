import dataclasses
import datetime
import os
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from time import time
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import ogbench
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import OmegaConf
from PIL import Image, ImageEnhance

# requires torch==2.6.0 and tensordict-nightly==2025.1.1 and torchrl-nightly==2025.1.1
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from termcolor import colored
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from tqdm import trange


os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["TORCH_LOGS"] = "+recompiles"

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


# ============================================================================
# Utils
# ============================================================================
def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def cfg_to_dataclass(cfg, frozen=False):
    """
    Converts an OmegaConf config to a dataclass object.
    This prevents graph breaks when used with torch.compile.
    """
    cfg_dict = OmegaConf.to_container(cfg)
    fields = []
    for key, value in cfg_dict.items():
        fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
    dataclass_name = "Config"
    dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)

    def get(self, val, default=None):
        return getattr(self, val, default)

    dataclass.get = get
    return dataclass()


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
    """
    Parses a Hydra config. Mostly for convenience.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v is None:
                v = True
        except Exception:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except Exception:
            pass

    # Seed
    if cfg.seed == "random":
        cfg.seed = random.randint(0, 50_000)

    # Environment
    # assert cfg.task in ['cube-v0', 'scene-v0'], f'Currentlly only OGBench tasks are supported.'
    # cfg.env_kwargs = cfg.env_kwargs["cube"] if "cube" in cfg.task else \
    # 				 cfg.env_kwargs["scene"] if "scene" in cfg.task else \
    # 				 {}

    assert cfg.obs in ["state", "rgb"]
    assert not cfg.multiview, "Currently multiview is not supported for IQL training."

    # Agent
    agent_cfg = {"iql": cfg.iql}.get(cfg.agent, {})
    cfg.alpha = agent_cfg.get("alpha")
    cfg.policy_type = agent_cfg.get("policy_type")
    cfg.pi_noise = agent_cfg.get("pi_noise")

    # Paths
    cfg.work_dir = Path(hydra.utils.get_original_cwd())
    # cfg.data_dir = Path(hydra.utils.get_original_cwd()) / f"{cfg.data_dir}"
    cfg.log_dir = Path(hydra.utils.get_original_cwd()) / "output" / "train" / f"{cfg.agent}_{cfg.task}_{str(cfg.seed)}"
    cfg.checkpoint = Path(hydra.utils.get_original_cwd()) / f"{cfg.checkpoint}"
    cfg.eval_log_dir = (
        Path(hydra.utils.get_original_cwd()) / "output" / "eval" / f"{cfg.agent}_{cfg.task}_{str(cfg.seed)}"
    )
    cfg.eval_checkpoint = Path(hydra.utils.get_original_cwd()) / f"{cfg.eval_checkpoint}"

    # Convenience
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.exp_name = f"{cfg.agent}_{cfg.task}"

    return cfg_to_dataclass(cfg)


# ============================================================================
# Logger
# ============================================================================
CONSOLE_FORMAT = [
    ("iteration", "I", "int"),
    ("episode", "E", "int"),
    ("step", "I", "int"),
    ("episode_reward", "R", "float"),
    ("overall_success", "S", "float"),
    ("overall_success_frac", "SF", "float"),
    ("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
    "pretrain": "yellow",
    "train": "blue",
    "eval": "green",
}


def make_dir(dir_path):
    """Create directory if it does not already exist."""
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def print_run(cfg):
    """
    Pretty-printing of current run information.
    Logger calls this method at initialization.
    """
    prefix, color, attrs = "  ", "green", ["bold"]

    def _limstr(s, maxlen=36):
        return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

    def _pprint(k, v):
        print(prefix + colored(f"{k.capitalize() + ':':<15}", color, attrs=attrs), _limstr(v))

    kvs = [
        ("task", cfg.task_title),
        ("steps", f"{int(cfg.steps):,}"),
        ("observation", cfg.obs),
        ("actions", cfg.action_dim),
        ("agent", cfg.agent),
        ("experiment", cfg.exp_name),
    ]
    w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
    div = "-" * w
    print(div)
    for k, v in kvs:
        _pprint(k, v)
    print(div)


def cfg_to_group(cfg, return_list=False):
    """
    Return a wandb-safe group name for logging.
    Optionally returns group name as list.
    """
    lst = [cfg.task, cfg.agent, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
    return lst if return_list else "-".join(lst)


def reshape_video(v, n_cols=5):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def frames_list_to_array(frames_list=None, n_cols=5):
    """

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
            frames_list: List of videos. Each video should be a numpy array of shape (t, h, w, c).
            n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(frames) for frames in frames_list])
    for i, frames in enumerate(frames_list):
        assert frames.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = frames[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(frames), axis=0)
        frames_list[i] = np.concatenate([frames, pad], axis=0)

        # Add borders.
        frames_list[i] = np.pad(frames_list[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode="constant", constant_values=0)
    frames_array = np.array(frames_list)  # (n, t, h, w, c)

    frames_array = reshape_video(frames_array, n_cols)  # (t, c, nr * h, nc * w)

    return frames_array


class Logger:
    """Primary logging object. Logs either locally or using wandb."""

    def __init__(self, cfg):
        self._log_dir = make_dir(cfg.log_dir)
        self._model_dir = make_dir(self._log_dir / "models")
        self._save_agent = cfg.save_agent
        self._save_video = cfg.save_video
        self._group = cfg_to_group(cfg)
        self._seed = cfg.seed
        print_run(cfg)
        self.project = cfg.get("wandb_project", "none")
        self.entity = cfg.get("wandb_entity", "none")
        if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
            print(colored("Wandb disabled.", "blue", attrs=["bold"]))
            cfg.save_agent = False
            cfg.save_video = False
            self._wandb = None
            self._video = None
            return
        os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
        import wandb

        wandb.init(
            project=self.project,
            entity=self.entity,
            name=f"{cfg.exp_name}{cfg.wandb_name_suffix}",
            group=self._group,
            tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
            dir=self._log_dir,
            config=dataclasses.asdict(cfg),
        )
        print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
        self._wandb = wandb

    def save_agent(self, agent, buffer, identifier="latest"):
        if self._save_agent and agent:
            fp = self._model_dir / f"{str(identifier)}.pt"
            agent.save(fp, buffer.obs_mean, buffer.obs_std)
            # if self._wandb:
            # 	artifact = self._wandb.Artifact(
            # 		self._group + '-' + str(self._seed) + '-' + str(identifier),
            # 		type='model',
            # 	)
            # 	artifact.add_file(fp)
            # 	self._wandb.log_artifact(artifact)

    def finish(self, agent, buffer):
        try:
            self.save_agent(agent, buffer)
        except Exception as e:
            print(colored(f"Failed to save model: {e}", "red"))
        if self._wandb:
            self._wandb.finish()

    def _format(self, key, value, ty):
        if ty == "int":
            return f"{colored(key + ':', 'blue')} {int(value):,}"
        elif ty == "float":
            return f"{colored(key + ':', 'blue')} {value:.01f}"
        elif ty == "time":
            value = str(datetime.timedelta(seconds=int(value)))
            return f"{colored(key + ':', 'blue')} {value}"
        else:
            raise f"invalid log format type: {ty}"

    def _print(self, d, category):
        category = colored(category, CAT_TO_COLOR[category])
        pieces = [f" {category:<14}"]
        for k, disp_k, ty in CONSOLE_FORMAT:
            if k in d:
                pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
        print("   ".join(pieces))

    def _log_video(self, frames_list, step):
        frames_array = frames_list_to_array(frames_list, n_cols=len(frames_list))
        video = self._wandb.Video(frames_array, fps=15, format="mp4")
        self._wandb.log({"eval_video": video}, step=step)

    def log(self, d, category="train", frames_list=None):
        assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
        if self._wandb:
            step = d["step"]
            _d = {}
            for k, v in d.items():
                _d[category + "/" + k] = v
            self._wandb.log(_d, step)

            if category == "eval" and self._save_video:
                self._log_video(frames_list, step)

        self._print(d, category)


# ============================================================================
# Env Wrappers
# ============================================================================
class TorchObsWrapper(gym.Wrapper):
    """Preprocess environment observations to be compatible with agent input."""

    def __init__(self, env, cfg):
        super().__init__(env)

        self.cfg = cfg
        self.obs_mode = cfg.obs

        obs, info = self.env.reset()

        obs = self.preprocess_obs(self._obs_to_tensor(obs))
        obs = obs.numpy()

        if self.obs_mode == "rgb":
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs.shape, dtype=obs.dtype)
        else:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype)

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32))

    def _try_f32_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            if x.dtype == torch.float64:
                x = x.float()
        return x

    def _obs_to_tensor(self, obs):
        if isinstance(obs, dict):
            for k in obs.keys():
                obs[k] = self._try_f32_tensor(obs[k])
        else:
            obs = self._try_f32_tensor(obs)
        return obs

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)

        # preprocess goal
        goal = info.get("goal")
        goal = self.preprocess_obs(self._obs_to_tensor(goal))
        info["goal"] = goal

        # preprocess observation
        observation = self.preprocess_obs(self._obs_to_tensor(observation))

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action.numpy())

        # preprocess observation
        observation = self.preprocess_obs(self._obs_to_tensor(observation))

        return observation, torch.tensor(reward, dtype=torch.float32), terminated, truncated, info

    def preprocess_obs(self, obs, batch=False):
        if not batch:
            obs = obs.unsqueeze(0)

        if self.obs_mode == "state":
            pass

        elif self.obs_mode == "rgb":
            obs = obs.permute(0, 3, 1, 2)

        else:
            raise NotImplementedError

        if not batch:
            obs = obs.squeeze(0)

        return obs


# ============================================================================
# Dataset and Buffer
# ============================================================================
def load_dataset_to_buffer(cfg, np_dataset):
    """Load dataset to buffer for offline training."""

    # Get dataset attributes
    capacity = np_dataset["terminals"].size
    cfg.data_episode_length = int(np.nonzero(np_dataset["terminals"])[0][0]) + 1

    # Convert dataset to torch tensordict
    obs = torch.from_numpy(np_dataset["observations"])
    if cfg.obs == "rgb":
        obs = obs.permute(0, 3, 1, 2)

    dataset = TensorDict(
        {
            "obs": obs,
            "action": torch.from_numpy(np_dataset["actions"]),
        },
        batch_size=capacity,
    )

    # Create buffer for sampling
    buffer = Buffer(cfg, capacity)
    buffer.load(dataset)

    expected_episodes = capacity // (cfg.data_episode_length + 1)
    if buffer.num_eps != expected_episodes:
        print(
            f"WARNING: buffer has {buffer.num_eps} episodes, expected {expected_episodes} episodes for {cfg.task} task."
        )

    return buffer


class RMSNormalizer:
    """
    Calculates the running mean and std (RMS) of a data stream for normalization purposes.
    """

    def __init__(self, shape, epsilon=1e-6, disable=False):
        self.mean = torch.zeros(shape, dtype=torch.float32)
        self.var = torch.ones(shape, dtype=torch.float32)
        self.count = epsilon
        self.epsilon = epsilon if not disable else 0
        self.disable = disable

    def normalize(self, obs):
        if self.disable:
            return obs
        else:
            return torch.clip(
                (obs - self.mean.to(obs.device)) / torch.sqrt(self.var.to(obs.device) + self.epsilon), -5, 5
            )

    def unnormalize(self, obs):
        if self.disable:
            return obs
        else:
            return (obs * torch.sqrt(self.var.to(obs.device) + self.epsilon)) + self.mean.to(obs.device)

    def update(self, obs):
        if self.disable:
            return
        else:
            obs = obs.view(-1, obs.shape[-1])
            batch_mean = torch.mean(obs, dim=0)
            batch_var = torch.var(obs, dim=0)
            batch_count = obs.shape[0]
            self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def set_stats(self, obs):
        if self.disable:
            return
        else:
            obs = obs.view(-1, obs.shape[-1])
            self.mean = torch.mean(obs, dim=0)
            self.var = torch.var(obs, dim=0)
            self.count = obs.shape[0]


class Buffer:
    """
    Replay buffer for RL training, based on torchrl.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg, capacity):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._capacity = capacity
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key="episode",
            truncated_key=None,
            strict_length=True,
            cache_values=True,  # NOTE: should be True in offline training
            compile=cfg.compile,
        )
        self.normalizer = RMSNormalizer(
            self.cfg.obs_shape[-1], disable=(self.cfg.obs == "rgb")
        )  # initialize RMS normalizer

        self._batch_size = cfg.batch_size * cfg.data_episode_length
        self._num_eps = 0
        if self.cfg.geom_sample_value or self.cfg.geom_sample_policy:
            self._geometric_dist = torch.distributions.Geometric(
                torch.tensor([1 - self.cfg.gamma], device=self._device)
            )
        self.horizon = 1  # NOTE: can change if want to sample slices longer than 1 transition

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    @property
    def obs_mean(self):
        return self.normalizer.mean

    @property
    def obs_std(self):
        return torch.sqrt(self.normalizer.var)

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, td):
        """Initialize the replay buffer. Use the first tensordict to estimate storage requirements."""
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in td.values()
            ]
        ) / len(td)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes / 1e9:.2f} GB")
        # Heuristic: decide whether to use CUDA or CPU memory
        storage_device = "cuda:0" if 2.0 * total_bytes < mem_free else "cpu"  # 2.5 is the original value
        print(f"Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(LazyTensorStorage(self._capacity, device=self._storage_device))

    def load(self, td):
        """
        Load a batch of episodes into the buffer. This is useful for loading data from disk,
        and is more efficient than adding episodes one by one.
        """
        num_new_eps = len(td) // (self.cfg.data_episode_length + 1)
        td["episode"] = torch.arange(self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64).repeat_interleave(
            self.cfg.data_episode_length + 1
        )
        if self._num_eps == 0:
            self._buffer = self._init(td[0].unsqueeze(0))
            self.normalizer.set_stats(td["obs"])  # set RMS normalizer stats once from offline data
        self._buffer.extend(td)
        self._num_eps += num_new_eps
        return self._num_eps

    def add(self, td):
        """Add an episode to the buffer."""
        td["episode"] = torch.full_like(td["reward"], self._num_eps, dtype=torch.int64)
        if self._num_eps == 0:
            self._buffer = self._init(td)
        self._buffer.extend(td)
        self.normalizer.update(td["obs"])  # update RMS normalizer stats
        self._num_eps += 1
        return self._num_eps

    def sample(self):
        # sample full episodes
        episodes = self._buffer.sample().view(-1, self.cfg.data_episode_length)
        # sample state indices
        idxs = torch.randint(
            self.cfg.data_episode_length - self.horizon, size=(self.cfg.batch_size,), device=self._storage_device
        )
        # batch_idxs = torch.arange(self.cfg.batch_size, device=self._device)
        batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device).unsqueeze(
            1
        )  # shape: (batch_size, 1)
        slice_idxs = idxs.unsqueeze(1) + torch.arange(self.horizon, device=self._storage_device).unsqueeze(0)
        # sample policy goal indices
        policy_goal_batch_idxs, policy_goal_idxs = self._sample_goal(
            idxs + self.horizon - 1,
            self.cfg.p_curgoal_policy,
            self.cfg.p_trajgoal_policy,
            self.cfg.p_randgoal_policy,
            self.cfg.geom_sample_policy,
        )
        # sample value goal indices
        value_goal_batch_idxs, value_goal_idxs = self._sample_goal(
            idxs + self.horizon - 1,
            self.cfg.p_curgoal_value,
            self.cfg.p_trajgoal_value,
            self.cfg.p_randgoal_value,
            self.cfg.geom_sample_value,
        )
        # retrieve data
        obs = episodes.get("obs")[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
        action = episodes.get("action")[batch_idxs, slice_idxs].contiguous().to(self._device, non_blocking=True)
        next_obs = episodes.get("obs")[batch_idxs, slice_idxs + 1].contiguous().to(self._device, non_blocking=True)
        value_goal = (
            episodes.get("obs")[value_goal_batch_idxs, value_goal_idxs]
            .unsqueeze(1)
            .repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape))
            .contiguous()
            .to(self._device, non_blocking=True)
        )
        policy_goal = (
            episodes.get("obs")[policy_goal_batch_idxs, policy_goal_idxs]
            .unsqueeze(1)
            .repeat(1, self.horizon, *[1] * len(self.cfg.obs_shape))
            .contiguous()
            .to(self._device, non_blocking=True)
        )
        reward, done = self._compute_reward(
            episodes, idxs, batch_idxs, slice_idxs, value_goal_batch_idxs, value_goal_idxs
        )
        # save in tensordict
        td = TensorDict(
            obs=self.normalizer.normalize(obs),
            action=action,
            next_obs=self.normalizer.normalize(next_obs),
            reward=reward,
            done=done,
            value_goal=self.normalizer.normalize(value_goal),
            policy_goal=self.normalizer.normalize(policy_goal),
            batch_size=(obs.shape[0], obs.shape[1]),
            device=self._device,
        )
        return td.squeeze(
            1
        )  # NOTE: this is to remove the extra 'horizon' dimension when it is not used (currently not used)

    def _sample_goal(self, idxs, p_curgoal, p_trajgoal, p_randgoal, geom_sample):
        # random goals
        randgoal_idxs = torch.randint(
            self.cfg.data_episode_length - 1, size=(self.cfg.batch_size,), device=self._storage_device
        )
        randgoal_batch_idxs = torch.randint(
            self.cfg.batch_size, size=(self.cfg.batch_size,), device=self._storage_device
        )

        # goals from the same episode
        if geom_sample:
            offsets = self._geometric_dist.sample((self.cfg.batch_size,)).to(self._storage_device, non_blocking=True)
            trajgoal_idxs = torch.minimum(
                idxs + offsets.squeeze(-1), torch.full_like(idxs, self.cfg.data_episode_length - 1)
            ).to(torch.int)
        else:
            distances = torch.rand(self.cfg.batch_size, device=self._storage_device)  # in [0, 1)
            trajgoal_idxs = torch.round(
                (idxs + 1) * distances + (self.cfg.data_episode_length - 1) * (1 - distances)
            ).to(torch.int)

        # goals at the next state
        curgoal_idxs = idxs + 1

        batch_idxs = torch.arange(self.cfg.batch_size, device=self._storage_device)
        goal_batch_idxs = torch.where(
            batch_idxs >= self.cfg.batch_size * (1 - p_randgoal), randgoal_batch_idxs, batch_idxs
        )

        goal_idxs = torch.where(batch_idxs < self.cfg.batch_size * p_curgoal, curgoal_idxs, trajgoal_idxs)
        goal_idxs = torch.where(batch_idxs >= self.cfg.batch_size * (1 - p_randgoal), randgoal_idxs, goal_idxs)

        return goal_batch_idxs, goal_idxs

    def _compute_reward(self, episodes, idxs, batch_idxs, slice_idxs, goal_batch_idxs, goal_idxs):
        # self-supervised reward
        episode_id = (
            episodes.get("episode")[batch_idxs.squeeze(1), idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
        )
        goal_episode_id = (
            episodes.get("episode")[goal_batch_idxs, idxs].unsqueeze(1).repeat(1, self.horizon).contiguous()
        )
        goal_idxs = goal_idxs.unsqueeze(1).repeat(1, self.horizon).contiguous()
        reward = (
            torch.where((episode_id == goal_episode_id) & (slice_idxs + 1 == goal_idxs), 0.0, -1.0)
            .unsqueeze(-1)
            .contiguous()
            .to(self._device, non_blocking=True)
        )
        done = torch.where(reward == 0, 1, 0).unsqueeze(-1).contiguous().to(self._device, non_blocking=True)

        done = done if self.cfg.done_signal else torch.zeros_like(reward)

        return reward, done


# ============================================================================
# Actor-Critic Networks
# ============================================================================
class ResnetStack(nn.Module):
    def __init__(self, c_in, c, device=None):
        super().__init__()

        self.conv_in = nn.Conv2d(c_in, c, kernel_size=3, stride=1, padding=1, device=device)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = self.conv_in(x)
        x_in = self.maxpool(x)
        # residual block
        x = self.conv1(F.relu(x_in))
        x = self.conv2(F.relu(x)) + x_in
        return x


class ImpalaEncoder(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()

        channels, out_dim = cfg.arch_mlp["conv_channels"], cfg.arch_mlp["h_dim"]
        c = [6, *channels]  # 6 for 3 input channels of state and goal images

        self.resnet = nn.Sequential(*[ResnetStack(c[i], c[i + 1], device) for i in range(len(c) - 1)])

        # conv_out_flat_dim = 2048  # TODO: replace hard-coded value
        with torch.no_grad():
            dummy_input = torch.randn(1, 6, 64, 64, device=device)  # adjust based on expected input size
            conv_out_flat_dim = self.resnet(dummy_input).view(-1).shape[0]

        self.linear = nn.Linear(conv_out_flat_dim, out_dim, device=device)

    def forward(self, x):
        x = F.relu(self.resnet(x))
        x = F.gelu(self.linear(x.view(*x.shape[:-3], -1)))
        return x


class Actor(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp["h_dim"]

        if cfg.obs == "rgb":
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        # self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        self.fc_mu = nn.Linear(h_dim, a_dim, device=device)

    def forward(self, x, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        # x = F.gelu(self.fc4(x))
        x = self.fc_mu(x).tanh()
        return x


class QNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, a_dim, h_dim = cfg.obs_shape[-1], cfg.action_dim, cfg.arch_mlp["h_dim"]

        if cfg.obs == "rgb":
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim + a_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim + a_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        # self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        # self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, a, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = torch.cat([x, a], dim=-1)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        # x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x


class VNetwork(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        obs_dim, h_dim = cfg.obs_shape[-1], cfg.arch_mlp["h_dim"]

        if cfg.obs == "rgb":
            self.gc_encoder = ImpalaEncoder(cfg, device=device)
            in_dim = h_dim
        else:
            self.gc_encoder = None
            in_dim = 2 * obs_dim

        self.fc1 = nn.Linear(in_dim, h_dim, device=device)
        self.ln1 = nn.LayerNorm(h_dim, device=device)
        self.fc2 = nn.Linear(h_dim, h_dim, device=device)
        self.ln2 = nn.LayerNorm(h_dim, device=device)
        self.fc3 = nn.Linear(h_dim, h_dim, device=device)
        self.ln3 = nn.LayerNorm(h_dim, device=device)
        # self.fc4 = nn.Linear(h_dim, h_dim, device=device)
        # self.ln4 = nn.LayerNorm(h_dim, device=device)
        self.fc_out = nn.Linear(h_dim, 1, device=device)

    def forward(self, x, g):
        x = torch.cat([x, g], dim=1)
        if self.gc_encoder is not None:
            x = x.float() / 255.0
            x = self.gc_encoder(x)
        x = self.ln1(F.gelu(self.fc1(x)))
        x = self.ln2(F.gelu(self.fc2(x)))
        x = self.ln3(F.gelu(self.fc3(x)))
        # x = self.ln4(F.gelu(self.fc4(x)))
        x = self.fc_out(x)
        return x


# ============================================================================
# IQL Agent
# ============================================================================
class IQL(nn.Module):
    def __init__(self, cfg, Actor, QNetwork, VNetwork):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda:0")

        # hyperparameters
        self.tau = self.cfg.iql["tau"]
        self.expectile = self.cfg.iql["expectile"]
        self.alpha = self.cfg.iql["alpha"]
        self.grad_clip_norm = self.cfg.grad_clip_norm
        self.policy_type = self.cfg.iql["policy_type"]
        self.pi_noise = self.cfg.iql["pi_noise"]
        self.pi_noise_clip = self.cfg.iql["pi_noise_clip"]

        # actor
        self.pi = Actor(self.cfg, self.device)

        self.pi_detach = Actor(self.cfg, self.device)
        from_module(self.pi).data.to_module(self.pi_detach)  # Copy params to pi_detach without grad

        # value
        self.vf = VNetwork(self.cfg, self.device)

        self.vf_detach = VNetwork(self.cfg, self.device)
        from_module(self.vf).data.to_module(self.vf_detach)  # Copy params to vf_detach without grad

        # q-value
        qf1 = QNetwork(self.cfg, self.device)
        qf2 = QNetwork(self.cfg, self.device)

        self.qnet_params = from_modules(qf1, qf2, as_module=True)
        self.qnet_target_params = self.qnet_params.data.clone()
        # discard params of net
        self.qnet = QNetwork(self.cfg, device="meta")
        self.qnet_params.to_module(self.qnet)

        # optimizers
        self.actor_optimizer = optim.Adam(
            list(self.pi.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        self.v_optimizer = optim.Adam(
            list(self.vf.parameters()), lr=cfg.lr, capturable=cfg.cudagraphs and not cfg.compile
        )

        self.q_optimizer = optim.Adam(
            self.qnet_params.values(include_nested=True, leaves_only=True),
            lr=self.cfg.lr,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
        )

        # torch.compile
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self._update_q = torch.compile(self._update_q, mode=mode)
            self._update_v = torch.compile(self._update_v, mode=mode)
            self._update_pi = torch.compile(self._update_pi, mode=mode)
            self._act = torch.compile(self._act, mode=mode)

        if self.cfg.cudagraphs:
            self._update_q = CudaGraphModule(self._update_q, in_keys=[], out_keys=[], warmup=5)
            self._update_v = CudaGraphModule(self._update_v, in_keys=[], out_keys=[], warmup=5)
            self._update_pi = CudaGraphModule(self._update_pi, in_keys=[], out_keys=[], warmup=5)
            self._act = CudaGraphModule(self._act, in_keys=[], out_keys=[], warmup=5)

    @torch.no_grad()
    def act(self, td):
        td_out = self._act(td.to(self.device, non_blocking=True))
        action = td_out["action"]
        return action, {}

    @torch.no_grad()
    def _act(self, td):
        obs = td["obs"]
        goal = td["goal"]
        action = self.pi_detach(obs, goal)
        if self.policy_type == "noisy":
            noise = torch.randn_like(action)
            clipped_noise = noise.mul(self.pi_noise).clamp(
                -self.pi_noise_clip, self.pi_noise_clip
            )  # NOTE: action scale is assumed to be 1
            action = (action + clipped_noise).clamp(self.cfg.action_low, self.cfg.action_high)
        elif self.policy_type == "deterministic":
            pass
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")
        return TensorDict(action=action.detach())

    def update(self, batch, step):
        # update value
        info = self._update_v(batch)

        # update q-value
        info.update(self._update_q(batch))

        # update actor
        info.update(self._update_pi(batch))

        # update the target networks
        # lerp is defined as x' = x + w (y-x), which is equivalent to x' = (1-w) x + w y
        self.qnet_target_params.lerp_(self.qnet_params.data, self.tau)

        return info.detach().mean()

    def _update_q(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        done = batch["done"]
        goal = batch["value_goal"]

        vf_next_target = self.vf_detach(next_obs, goal)
        next_q_value = (
            reward.flatten() + (1 - done.flatten()) * self.cfg.gamma * vf_next_target.flatten()
        )  # whether a "done" signal is received depends on config
        next_q_value = torch.clamp(
            next_q_value, min=-1 / (1 - self.cfg.gamma), max=0
        )  # NOTE: clamp to possible range based on the reward, assumes reward in [-1, 0]

        qf_loss = torch.vmap(self._batched_qf, (0, None, None, None, None))(
            self.qnet_params, obs, action, goal, next_q_value
        )
        qf_loss = qf_loss.sum(0)

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.qnet_params.values(include_nested=True, leaves_only=True), self.grad_clip_norm
        )
        self.q_optimizer.step()

        return TensorDict(qf_loss=qf_loss.detach(), qf_gradnorm=grad_norm)

    def _batched_qf(self, params, obs, action, goal, next_q_value=None):
        with params.to_module(self.qnet):
            vals = self.qnet(obs, action, goal)
            if next_q_value is not None:
                loss_val = F.mse_loss(vals.view(-1), next_q_value)
                return loss_val
            return vals

    def _update_v(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["value_goal"]

        qf_target = torch.vmap(self._batched_qf, (0, None, None, None))(self.qnet_target_params, obs, action, goal)
        min_qf_target = qf_target.min(0).values

        value = self.vf(obs, goal)

        value_loss = torch.mean(
            torch.abs(self.expectile - ((min_qf_target - value) < 0).float()) * (min_qf_target - value) ** 2
        )  # expectile loss

        self.v_optimizer.zero_grad()
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.vf.parameters(), self.grad_clip_norm)
        self.v_optimizer.step()

        return TensorDict(vf_loss=value_loss.detach(), vf_gradnorm=grad_norm)

    def _update_pi(self, batch):
        obs = batch["obs"]
        action = batch["action"]
        goal = batch["policy_goal"]

        # NOTE: different from OGBench in that uses qf1 and not min(qf1, qf2) for policy training
        self.actor_optimizer.zero_grad()
        with self.qnet_params.data[0].to_module(self.qnet):
            pi_out = self.pi(obs, goal)
            q_out = self.qnet(obs, pi_out, goal)

            q_loss = -q_out.mean() / (q_out.abs().mean().detach() + 1e-6)
            bc_loss = F.mse_loss(pi_out, action)

            pi_loss = q_loss + self.alpha * bc_loss

        pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.pi.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        return TensorDict(
            pi_loss=pi_loss.detach(),
            pi_q_loss=q_loss.detach(),
            pi_bc_loss=bc_loss.detach(),
            pi_gradnorm=grad_norm,
            q_scale=q_out.abs().mean().detach(),
        )

    def save(self, fp, obs_mean=0, obs_std=1):
        """
        Save state dict of the agent to filepath.

        Args:
                fp (str): Filepath to save state dict to.
        """
        torch.save(
            {
                "pi": self.pi.state_dict(),
                "vf": self.vf.state_dict(),
                "qfs": self.qnet_params.state_dict(),
                "obs_mean": obs_mean,
                "obs_std": obs_std,
            },
            fp,
        )

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent for inference.

        Args:
                fp (str or dict): Filepath or state dict to load.
        """
        state_dict = torch.load(fp)
        self.pi.load_state_dict(state_dict["pi"])
        self.vf.load_state_dict(state_dict["vf"])
        self.qnet_params.load_state_dict(state_dict["qfs"])
        self.qnet_target_params = self.qnet_params.data.clone()
        self.qnet_params.to_module(self.qnet)
        self.obs_mean = state_dict["obs_mean"]
        self.obs_std = state_dict["obs_std"]

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Trainer
# ============================================================================
class Trainer:
    """Trainer class for offline agents"""

    def __init__(self, cfg, env, buffer, agent, logger):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.logger = logger
        self.buffer = buffer

        print(f"Learnable parameters: {self.agent.total_params:,}")
        self._start_time = time()

    def _time_metrics(self, step):
        """Return a dictionary of currentt time metrics."""
        elapsed_time = time() - self._start_time
        return {"step": step, "elapsed_time": elapsed_time, "steps_per_second": step / elapsed_time}

    def _eval(self):
        """OGBench evaluation protocol"""
        metrics = {}
        overall_metrics = defaultdict(list)
        frames_list = []

        task_infos = (
            self.env.unwrapped.task_infos if hasattr(self.env.unwrapped, "task_infos") else self.env.task_infos
        )

        for task_id in [1, 2, 3, 4, 5]:
            cur_metrics = defaultdict(list)
            cur_frames = []

            for i in trange(self.cfg.num_eval_episodes + self.cfg.num_video_episodes):
                should_render = i >= self.cfg.num_eval_episodes

                obs, goal_info = self.env.reset(options={"task_id": task_id, "render_goal": should_render})

                goal = goal_info.get("goal")
                goal = self.buffer.normalizer.normalize(goal.unsqueeze(0))

                done = False
                step = 0
                frames = []
                while not done:
                    torch.compiler.cudagraph_mark_step_begin()
                    obs = self.buffer.normalizer.normalize(obs.unsqueeze(0))
                    action, act_info = self.agent.act(TensorDict(obs=obs, goal=goal, step=step))
                    action = action.squeeze(0).cpu()

                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    step += 1

                    if should_render and (step % self.cfg.video_frame_skip == 0 or done):
                        frames.append(self.env.render().copy())

                    obs = next_obs

                if i < self.cfg.num_eval_episodes:
                    add_to(cur_metrics, flatten(info))
                else:
                    cur_frames.append(np.array(frames))

            frames_list.extend(cur_frames)

            for k, v in cur_metrics.items():
                cur_metrics[k] = np.mean(v)

            task_name = task_infos[task_id - 1]["task_name"]
            metric_names = ["success"]
            metrics.update({f"{task_name}_{k}": v for k, v in cur_metrics.items() if k in metric_names})
            for k, v in cur_metrics.items():
                if k in metric_names:
                    overall_metrics[k].append(v)

        for k, v in overall_metrics.items():
            metrics[f"overall_{k}"] = np.mean(v)

        return metrics, frames_list  # NOTE: videos should be None if we don't want to save them

    def train(self):
        print(f"Training agent for {self.cfg.steps} iterations...")
        best_success_rate = 0
        for i in range(self.cfg.steps):
            # sample batched data
            batch = self.buffer.sample()

            # update agent
            train_metrics = self.agent.update(batch, i)

            # evaluate agent periodically
            if i % self.cfg.eval_freq == 0 or i % 10_000 == 0:
                train_metrics.update(self._time_metrics(i))
                self.logger.log(train_metrics, "train")
                if i % self.cfg.eval_freq == 0:
                    eval_metrics, eval_videos = self._eval()
                    eval_metrics.update(self._time_metrics(i))
                    self.logger.log(eval_metrics, "eval", eval_videos)
                    if i > 0:
                        self.logger.save_agent(self.agent, self.buffer, identifier="latest")
                        if eval_metrics["overall_success"] > best_success_rate:
                            self.logger.save_agent(self.agent, self.buffer, identifier="best")
                            best_success_rate = eval_metrics["overall_success"]

        self.logger.finish(self.agent, self.buffer)


# ============================================================================
# Main Entry Point
# ============================================================================
@hydra.main(config_name="config_iql", config_path=".")
def train(cfg: dict):
    """
    Script for training an offline RL agent.

    Most relevant args:
            `task`: task name (see OGBench documentation)
            `agent`: type of agent to train
            `steps`: number of training steps (default: 10M)
            `seed`: random seed (if 'random' will choose randomly from [0, 50_000], default: 'random')

    See config.yaml for a full list of args.
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    # NOTE: currently only supports OGBench environments
    env, np_dataset, _ = ogbench.make_env_and_datasets(cfg.task, dataset_dir=cfg.dataset_dir, compact_dataset=True)

    cfg.env_episode_length = env._max_episode_steps

    env = TorchObsWrapper(env, cfg)

    try:  # Dict
        cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
    except Exception:  # Box
        cfg.obs_shape = env.observation_space.shape

    cfg.action_dim = env.action_space.shape[0]
    cfg.action_high = env.action_space.high[0].item()
    cfg.action_low = env.action_space.low[0].item()

    buffer = load_dataset_to_buffer(cfg, np_dataset)

    trainer = Trainer(
        cfg=cfg,
        env=env,
        buffer=buffer,
        agent=IQL(cfg, Actor, QNetwork, VNetwork),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
