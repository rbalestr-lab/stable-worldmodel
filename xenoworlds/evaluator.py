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
            for states, rewards in self.world:
                # -- get observations and goal images
                obs = torch.from_numpy(states["pixels"])
                goal_obs = torch.from_numpy(states["goal_pixels"])

                # for k, v in states.items():
                #     print(f"State {k}: {v.shape}")

                # -- get actions from the policy
                actions = self.policy.get_action(obs, goals=goal_obs)

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

                # TODO SHOULD GET SOME DATA FROM THE ENV TO KNOW HOW GOOD
                self.world.step(actions)

            print(f"Episode {episode + 1} finished ")
            self.world.close()

        return data

    def sample_goal(self):
        pass


### DataSetUpload (download using stable_ssl)
