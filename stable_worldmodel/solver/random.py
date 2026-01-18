"""Random action sampling solver for planning problems."""

import numpy as np
import torch


class RandomSolver:
    """Random action sampling solver for model-based planning."""

    def __init__(self):
        """Initialize an unconfigured RandomSolver."""
        self._configured = False
        self._action_space = None
        self._n_envs = None
        self._action_dim = None
        self._config = None

    def configure(self, *, action_space, n_envs: int, config) -> None:
        """Configure the solver with environment and planning specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

    @property
    def n_envs(self) -> int:
        """Number of parallel environments the solver plans for."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Total action dimensionality including action blocking."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in steps."""
        return self._config.horizon

    def __call__(self, *args, **kwargs) -> dict:
        """Make the solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def solve(self, info_dict, init_action=None) -> dict:
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
