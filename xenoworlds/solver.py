"""Planning optimization solvers implementation."""

import torch
import nevergrad as ng
import numpy as np

from loguru import logger as logging
from torch.nn import functional as F

from einops import rearrange

## -- Solver (For Planning)
### GradientSolver
### CEMSolver (navergrad, evotorch, mppi) MPC


class BaseSolver(torch.nn.Module):
    """Base class for planning solvers"""

    # the idea for solver is to implement different methods for solving planning optimization problems
    def __init__(self, world_model):
        super().__init__()

        # disable gradients for the world model
        self.world_model = world_model
        self.world_model.requires_grad_(False)

    def __call__(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        return self.solve(states, action_space, goals)

    def solve(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        """Solve the planning optimization problem given states, action space, and goals."""
        raise NotImplementedError("Solver must implement the solve method.")


class GDSolver(BaseSolver):
    """Gradient Descent Solver"""

    def __init__(
        self,
        world_model,
        horizon: int,
        n_steps: int,
        action_space,
        init_action=None,
        criterion=F.mse_loss,
        action_noise=0.003,
        mpc_n_actions=1,
    ):
        super().__init__(world_model)
        self.horizon = horizon
        self.n_steps = n_steps
        self.criterion = criterion
        self.device = world_model.device
        self.action_noise = action_noise
        self.mpc_n_actions = mpc_n_actions

        # starting point for the optimization
        self.init_action(action_space, init_action)

    def init_action(self, action_space, initial_action=None):
        """Initialize the action tensor for the solver.
        set sel.init - initial action sequences (n_envs, horizon, action_dim)
        """
        action_dim = self.world_model.action_dim
        actions = initial_action
        # -- no actions provided, sample
        if actions is None:
            # (n_envs, 1, action_dim)
            n_envs = action_space.shape[0]
            actions = torch.zeros((n_envs, 0, action_dim)).float()

        # fill remaining action with random sample!
        remaining = self.horizon - actions.shape[1]

        if remaining > 0:
            new_actions = torch.stack(
                [torch.from_numpy(action_space.sample()) for _ in range(remaining)],
                dim=1,
            ).float()

            new_actions = new_actions.unsqueeze(1)  # temporal dim

            # TODO double check, the wm should have all the stats from its pretraining
            new_actions = self.world_model.normalize_actions(new_actions)
            new_actions = rearrange(new_actions, "... f d -> ... (f d)")  # frameskip
            actions = torch.cat([actions, new_actions], dim=1)

        actions = actions.to(self.device)

        # -- reset the initial action
        if hasattr(self, "init"):
            self.init.copy_(actions)
        else:
            self.register_parameter(
                "init",
                torch.nn.Parameter(actions),
            )

    def cost_function(self, z_pred, z_target, proprio_scale=1.0):
        """Compute the cost function for the optimization.
        Args:
            z_pred: Predicted latent observations dict .
            z_target: Target latent observations dict (goals).
        Returns:
            torch.Tensor: Computed cost. (B,)
        """

        z_pixel = z_pred["pixels"]
        z_proprio = z_pred["proprio"]

        z_goal_pixel = z_target["pixels"]
        z_goal_proprio = z_target["proprio"]

        # cost for the last prediction
        loss_pixel = self.criterion(
            z_pixel[:, -1:], z_goal_pixel, reduction="none"
        ).mean(dim=tuple(range(1, z_pixel.ndim)))

        # cost for the last proprioceptive observation
        loss_proprio = self.criterion(
            z_proprio[:, -1:], z_goal_proprio, reduction="none"
        ).mean(dim=tuple(range(1, z_proprio.ndim)))

        # Combine the losses
        loss = loss_pixel + proprio_scale * loss_proprio
        return loss

    def solve(
        self, obs_0: dict, action_space, goals: dict, init_action=None
    ) -> torch.Tensor:
        """Solve the planning optimization problem using gradient descent."""

        # -- reinitialize the initial action and optim
        with torch.no_grad():
            self.init_action(action_space, init_action)
        optim = torch.optim.SGD([self.init], lr=1.0)

        # -- encode goal states
        z_goals = self.world_model.encode_obs(goals)
        z_goals = {k: v.detach() for k, v in z_goals.items()}

        # -- solve optimization
        for step_i in range(self.n_steps):
            z_obs_i, z = self.world_model.rollout(obs_0, self.init)
            loss = self.cost_function(z_obs_i, z_goals).sum()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)

            self.init.data += torch.randn_like(self.init) * self.action_noise

        # TODO add logger here
        # TODO break solving if finished
        logging.info(f"Final gradient solver loss: {loss.item()}")

        # -- return the actions
        mpc_actions = self.init.detach().cpu()

        # squeeze time dimension if needed
        if mpc_actions.ndim == 3:
            mpc_actions = mpc_actions.squeeze(1)

        mpc_actions = rearrange(
            mpc_actions,
            "... (f d) -> ... f d",
            f=self.horizon,
            d=self.world_model.original_action_dim,
        )

        # keep first action
        mpc_actions = mpc_actions[:, 0]

        return mpc_actions.numpy()


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
