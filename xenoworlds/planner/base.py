import torch
import gymnasium as gym


class BasePlanner(torch.nn.Module):
    def __init__(self, world_model, action_space):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.world_model = world_model.to(self.device)
        self.world_model.requires_grad_(False)

        self.action_space = action_space
        self.action_dim = self.action_space.shape[0]

    def plan(self, obs: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
