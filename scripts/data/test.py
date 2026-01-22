import os


os.environ["MUJOCO_GL"] = "egl"

import stable_worldmodel as swm


world = swm.World(env_name="swm/ReacherDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
world.record_video(
    "./",
    seed=42,
    options={"variation": ("all",)},
)
