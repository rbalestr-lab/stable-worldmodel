from .solver import *
from .world import World

### DataSetUpload (download using stable_ssl)

# class Agent:
#     def __init__(self, planner: BasePlanner, world: World):
#         self.world = world
#         self.planner = planner
#         self.goals = self.planner.world_model.encode(
#             torch.from_numpy(self.world.envs.reset()[0]["pixels"])
#         )

#     def run(self, episodes=1, max_steps=100):
#         for episode in range(episodes):
#             for states, rewards in self.world:
#                 pixels = torch.from_numpy(states["pixels"])
#                 actions = self.planner.plan(pixels, self.world.action_space, self.goals)
#                 if actions.ndim == 2:
#                     actions.unsqueeze_(0)
#                 for a in actions.unbind(0):
#                     self.world.step(a.numpy())
#             print(f"Episode {episode + 1} finished ")
#         self.world.close()


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(self, world: World, solver: BaseSolver, goal_dist=None):
        self.world = world
        self.solver = solver

        # sample goal
        self.goal_dist = goal_dist

    def run(self, episodes=1, max_steps=100):
        pass
