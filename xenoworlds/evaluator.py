import torch
from .policy import BasePolicy
from .world import World


## -- Evaluator / Collector
### Evaluator(env, policy)
class Evaluator:
    # the role of evaluator is to determine perf of the policy in the env
    def __init__(self, world: World, policy: BasePolicy):
        self.world = world
        self.policy = policy

    def run(self, episodes=1):
        # todo return interested logging data
        data = {}

        for episode in range(episodes):
            for obs, rewards in self.world:
                # get action from the policy wih optional goal specification
                pixels = torch.from_numpy(obs["pixels"])  # get the state
                goal = torch.from_numpy(obs["goal"]) if "goal" in obs else None

                actions = self.policy.get_action(pixels, goals=goal)

                # actions = actions.squeeze(0) if actions.ndim == 2 else actions
                # apply actions in the env
                # for a in actions.unbind(0):
                #     self.world.step(a.numpy())

                # make actions double precision (np array)
                actions = (
                    actions.double().numpy()
                    if isinstance(actions, torch.Tensor)
                    else actions
                )

                self.world.step(actions)

            print(f"Episode {episode + 1} finished ")
            self.world.close()

        return data


### DataSetUpload (download using stable_ssl)
