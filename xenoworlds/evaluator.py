import torch
from .policy import BasePolicy
from .world import World


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(self, world: World, policy: BasePolicy, goal_dist=None):
        self.world = world
        self.policy = policy
        # sample goal
        self.goal_dist = goal_dist

    def run(self, episodes=1):
        # todo return interested logging data
        data = {}

        for episode in range(episodes):
            # sample goals and get their representations
            goals = None  # self.goal_dist.sample()
            # goals = self.solver.world_model.encode(goals).unsqueeze(0)

            for states, rewards in self.world:
                # get action from the policy wih optional goal specification
                pixels = torch.from_numpy(states["pixels"])  # get the state
                actions = self.policy.get_action(pixels, goals=goals)

                # actions = actions.squeeze(0) if actions.ndim == 2 else actions
                # apply actions in the env
                # for a in actions.unbind(0):
                #     self.world.step(a.numpy())
                self.world.step(actions)

            print(f"Episode {episode + 1} finished ")

        self.world.close()
        return data


### DataSetUpload (download using stable_ssl)
