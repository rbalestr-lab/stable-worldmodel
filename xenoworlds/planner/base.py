import torch
import gymnasium as gym


class BasePlanner(torch.nn.Module):
    def __init__(self, world_model):
        super().__init__()
        self.world_model = world_model
        self.world_model.requires_grad_(False)

    def plan(
        self, states: torch.Tensor, action_space, goals: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError
