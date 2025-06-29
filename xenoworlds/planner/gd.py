import torch
from .base import BasePlanner
from loguru import logger as logging


class GD(BasePlanner):
    def __init__(self, world_model: torch.nn.Module, n_steps: int, action_space):
        super().__init__(world_model)
        self.n_steps = n_steps
        self.register_parameter(
            "init",
            torch.nn.Parameter(torch.from_numpy(action_space.sample()).float()),
        )

    def plan(self, states, action_space, goals):
        with torch.no_grad():
            self.init.copy_(torch.from_numpy(action_space.sample()).float())
        optim = torch.optim.SGD([self.init], lr=1)
        for step in range(self.n_steps):
            preds = self.world_model(states, self.init)
            loss = (preds - goals).square().mean(1).sum()
            loss.backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
        logging.info(f"Final gradient planner loss: {loss.item()}")
        return self.init.detach()
