from typing import Protocol, runtime_checkable

import gymnasium as gym
import torch


class Costable(Protocol):
    """Protocol for world model cost functions."""

    def criterion(self, info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Compute the cost criterion for action candidates."""

    def get_cost(info_dict: dict, action_candidates: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Compute cost for given action candidates based on info dictionary."""
        ...


@runtime_checkable
class Solver(Protocol):
    """Protocol for model-based planning solvers."""

    def configure(self, *, action_space: gym.Space, n_envs: int, config) -> None:
        """Configure the solver with environment and planning specifications."""
        ...

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        ...

    @property
    def n_envs(self) -> int:
        """Number of parallel environments being planned for."""
        ...

    @property
    def horizon(self) -> int:
        """Planning horizon length in timesteps."""
        ...

    def solve(self, info_dict, init_action=None) -> dict:
        """Solve the planning optimization problem to find optimal actions."""
        ...
