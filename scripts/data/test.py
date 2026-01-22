import os


os.environ["MUJOCO_GL"] = "egl"

import stable_worldmodel as swm


world = swm.World(env_name="swm/FingerDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
print("finger")
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )

world = swm.World(env_name="swm/ManipulatorDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
print("manipulator")
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )

world = swm.World(env_name="swm/BallInCupDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
print("ball_in_cup")
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )
