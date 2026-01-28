"""Random action sampling solver for planning problems."""

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class RandomSolver:
    """Random action sampling solver for model-based planning."""

    def __init__(self) -> None:
        """Initialize an unconfigured RandomSolver."""
        self._configured = False
        self._action_space: gym.Space | None = None
        self._n_envs: int | None = None
        self._action_dim: int | None = None
        self._config: Any = None

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment and planning specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def solve(
        self, info_dict: dict[str, Any], init_action: torch.Tensor | None = None
    ) -> dict[str, Any]:
        """Generate random action sequences for the planning horizon."""
        outputs = {}
        actions = init_action

        # -- no actions provided, sample
        if actions is None:
            actions = torch.zeros((self.n_envs, 0, self.action_dim))

        # fill remaining actions with random sample
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            total_sequence = remaining * self._config.action_block
            action_sequence = np.stack([self._action_space.sample() for _ in range(total_sequence)], axis=1)

            new_action = torch.from_numpy(action_sequence).view(self.n_envs, remaining, self.action_dim)
            actions = torch.cat([actions, new_action], dim=1)

        outputs["actions"] = actions
        return outputs
