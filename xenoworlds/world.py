import gymnasium as gym
import numpy as np
from loguru import logger as logging

from torchvision import transforms


class World:
    def __init__(
        self,
        env_name,
        num_envs,
        wrappers: list,
        goal_wrappers: list = None,
        seed: int = 2349867,
        max_episode_steps: int = 100,
        sample_goal_every_k_steps: int = -1,
    ):
        self.envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=wrappers,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )

        self.goal_envs = gym.make_vec(
            env_name,
            num_envs=num_envs,
            vectorization_mode="sync",
            wrappers=goal_wrappers if goal_wrappers else wrappers,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )

        logging.info("WORLD INITIALIZED")
        logging.info(f"ACTION SPACE: {self.envs.action_space}")
        logging.info(f"OBSERVATION SPACE: {self.envs.observation_space}")
        self.num_envs = num_envs
        self.seed = seed
        self.goal_seed = seed + 2344

        # note if sample_goal_every_k_steps is set to -1, will sample goal once per episode
        # TODO implement sample_goal_every_k_steps

    @property
    def observation_space(self):
        return self.envs.observation_space

    @property
    def action_space(self):
        return self.envs.action_space

    @property
    def single_action_space(self):
        return self.envs.single_action_space

    @property
    def single_observation_space(self):
        return self.envs.single_observation_space

    def close(self, **kwargs):
        return self.envs.close(**kwargs)

    # TEMOPORARY, need to delete!!!
    def denormalize(self, x):
        # x is (B,C,H,W) in [-1,1]
        return (x * 0.5) + 0.5

    def __iter__(self):
        self.terminations = np.array([False] * self.num_envs)
        self.truncations = np.array([False] * self.num_envs)
        self.rewards = None
        logging.info(f"Resetting the ({self.num_envs}) world(s)!")
        self.states, finfos = self.envs.reset(seed=self.seed)
        self.goal_states, _ = self.goal_envs.reset(seed=self.goal_seed)

        # save goal pixels as png image
        to_pil = transforms.ToPILImage()  # expects CxHxW (uint8) or HxWxC (uint8)
        goal_pixels = self.goal_states["pixels"]  # (B, C, H, W), dtype=torch.uint8
        goal_pixels = goal_pixels.transpose(0, 2, 3, 1)
        goal_pixels = (self.denormalize(goal_pixels) * 255.0).astype(np.uint8)

        for i in range(goal_pixels.shape[0]):
            img = to_pil(goal_pixels[i])  # one image from batch
            img.save(f"./videos/goal_{i}.png")

        return self

    def __next__(self):
        if not all(self.terminations) and not all(self.truncations):
            return (
                self.states,
                self.goal_states,
                self.rewards,
            )
        else:
            raise StopIteration

    def step(self, actions):
        (
            self.states,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        ) = self.envs.step(actions)
