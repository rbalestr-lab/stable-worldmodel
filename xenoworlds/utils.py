from .world import World


def set_state(world: World, state):
    for i in range(world.num_envs):
        if hasattr(world.envs.envs[i].unwrapped, "state"):
            world.envs.envs[i].unwrapped.state = state[i]
        elif hasattr(world.envs.envs[i].unwrapped, "s"):
            world.envs.envs[i].unwrapped.s = state[i]
        elif hasattr(world.envs.envs[i].unwrapped.sim, "set_state_from_flattened"):
            world.envs.envs[i].unwrapped.sim.set_state_from_flattened(state[i].view(-1))


import gymnasium as gym
import gymnasium_robotics
import numpy as np
import xenoworlds
import matplotlib.pyplot as plt

wrappers = [
    # lambda x: RecordVideo(x, video_folder="./videos"),
    lambda x: xenoworlds.wrappers.AddRenderObservation(x, render_only=False),
    lambda x: xenoworlds.wrappers.TransformObservation(x),
]
# print(ogbench.make_env_and_datasets(env, env_only=True))
world = xenoworlds.World(
    "FrozenLake-v1", num_envs=2, wrappers=wrappers, max_episode_steps=2
)

world.envs.reset()
img = world.envs.envs[0].render()
plt.subplot(2, 4, 1)
plt.imshow(img)
plt.subplot(2, 4, 5)
img = world.envs.envs[1].render()
plt.imshow(img)

xenoworlds.set_state(world, np.arange(2))
img = world.envs.envs[0].render()
plt.subplot(2, 4, 2)
plt.imshow(img)
plt.subplot(2, 4, 6)
img = world.envs.envs[1].render()
plt.imshow(img)


world.envs.reset()
img = world.envs.envs[0].render()
plt.subplot(2, 4, 3)
plt.imshow(img)
plt.subplot(2, 4, 7)
img = world.envs.envs[1].render()
plt.imshow(img)

xenoworlds.set_state(world, np.asarray([0, 3]))
img = world.envs.envs[0].render()
plt.subplot(2, 4, 4)
plt.imshow(img)
plt.subplot(2, 4, 8)
img = world.envs.envs[1].render()
plt.imshow(img)
plt.savefig("test_delete.png")
world.close()
