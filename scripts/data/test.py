import os


os.environ["MUJOCO_GL"] = "egl"

import stable_worldmodel as swm


world = swm.World(env_name="swm/CheetahDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("agent.color",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("agent.arm_density",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("agent.finger_density",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("target.color",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("target.shape",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("floor.color",)},
#     )
# for i in range(5):
#     world.record_video(
#         "./",
#         seed=i,
#         options={"variation": ("light.intensity",)},
#     )
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )

world = swm.World(env_name="swm/HopperDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )

world = swm.World(env_name="swm/HumanoidDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )

world = swm.World(env_name="swm/WalkerDMControl-v0", image_shape=(224, 224), num_envs=1, max_episode_steps=1000)
world.set_policy(swm.policy.RandomPolicy())
for i in range(10):
    world.record_video(
        "./",
        seed=i,
        options={"variation": ("all",)},
    )
