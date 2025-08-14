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
            for states, goal_states, rewards in self.world:
                # -- get observations and goal images
                # goal_obs = torch.from_numpy(goal_states["pixels"])

                # for k, v in states.items():
                #     print(f"State {k}: {v.shape}")

                # -- get actions from the policy
                actions = self.policy.get_action(states, goal_states=goal_states)

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
    # def eval_state(self, goal_state, cur_state):
    #     """
    #     Return True if the goal is reached
    #     [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]
    #     from: https://github.com/gaoyuezhou/dino_wm/blob/main/env/pusht/pusht_wrapper.py
    #     """
    #     # if position difference is < 20, and angle difference < np.pi/9, then success
    #     pos_diff = np.linalg.norm(goal_state[:4] - cur_state[:4])
    #     angle_diff = np.abs(goal_state[4] - cur_state[4])
    #     angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    #     success = pos_diff < 20 and angle_diff < np.pi / 9
    #     state_dist = np.linalg.norm(goal_state - cur_state)
    #     return success, state_dist
