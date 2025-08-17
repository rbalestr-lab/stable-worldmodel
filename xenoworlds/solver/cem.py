import torch
import nevergrad as ng
import numpy as np
from .base import BaseSolver


# TODO implement dino-wm cem solver
class CEMSolver(BaseSolver):
    """Cross Entropy Method Solver"""

    def __init__(self, world_model, n_steps: int, action_space, horizon: int):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.action_space = action_space
        self.horizon = horizon


class CEMNevergrad(BaseSolver):
    def __init__(
        self,
        world_model: torch.nn.Module,
        n_steps: int,
        action_space,
        planning_horizon: int,
    ):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.planning_horizon = planning_horizon
        init = torch.from_numpy(
            np.stack([action_space.sample() for _ in range(planning_horizon)], 0)
        )
        self.register_parameter("init", torch.nn.Parameter(init))

    def solve(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        """Solve the planning optimization problem using CEM."""
        # Define the action space
        with torch.no_grad():
            init = torch.from_numpy(
                np.stack(
                    [action_space.sample() for _ in range(self.planning_horizon)], 0
                )
            )
            self.init.copy_(init)
        # Initialize the optimizer
        optimizer = ng.optimizers.CMA(
            parametrization=ng.p.Array(
                shape=self.init.shape,
                lower=np.stack(
                    [action_space.low for _ in range(self.planning_horizon)], 0
                ),
                upper=np.stack(
                    [action_space.high for _ in range(self.planning_horizon)], 0
                ),
            ),
            budget=self.n_steps,
        )
        # Run the optimization
        for _ in range(self.n_steps):
            candidate = optimizer.ask()
            actions = torch.from_numpy(candidate.value.astype(np.float32))
            rewards = self.evaluate_action_sequence(
                states, actions, goals
            )  # todo how does it works? visual l2 distance with goals is enough? what about other metrics e.g SNR?
            # Negate rewards to minimize
            optimizer.tell(candidate, [-r for r in rewards])
        # Get the best action sequence
        best_action_sequence = optimizer.provide_recommendation().value
        return torch.from_numpy(best_action_sequence.astype(np.float32))

    def evaluate_action_sequence(self, states, actions, goals):
        with torch.inference_mode():
            preds = self.world_model(states, actions.unbind(0))
            rewards = (preds - goals).square().mean(1)
            return rewards
