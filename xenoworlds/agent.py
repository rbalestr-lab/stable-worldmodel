import gymnasium as gym
import numpy as np
import torch
from .planner import BasePlanner

import gymnasium as gym
import torch
import torchvision.transforms.v2 as transforms
from .world import World

## -- Policy
### BasePolicy(env, planner)
### RandomPolicy
### OptimalPolicy (Expert)
### PlanningPolicy (wm, solver)


## -- Solver (For Planning)
### GradientSolver
### CEMSolver (navergrad, evotorch, mppi) MPC

## -- Evaluator / Collector
### Evaluator(env, policy)

### DataSetUpload (download using stable_ssl)


class Agent:  ### Evaluator(env, policy)
    def __init__(self, planner: BasePlanner, world: World):
        self.world = world
        self.planner = planner
        self.goals = self.planner.world_model.encode(
            torch.from_numpy(self.world.envs.reset()[0]["pixels"])
        )

    def run(self, episodes=1, max_steps=100):
        for episode in range(episodes):
            for states, rewards in self.world:
                pixels = torch.from_numpy(states["pixels"])
                actions = self.planner.plan(pixels, self.world.action_space, self.goals)
                if actions.ndim == 2:
                    actions.unsqueeze_(0)
                for a in actions.unbind(0):
                    self.world.step(a.numpy())
            print(f"Episode {episode + 1} finished ")
        self.world.close()
