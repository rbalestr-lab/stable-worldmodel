import os


os.environ['MUJOCO_GL'] = 'egl'

import stable_worldmodel as swm


world = swm.World(
    env_name='swm/QuadrupedDMControl-v0',
    image_shape=(224, 224),
    num_envs=1,
    max_episode_steps=1000,
)
world.set_policy(swm.policy.RandomPolicy())
print('quadruped')
for i in range(10):
    world.record_video(
        './',
        seed=i,
        options={'variation': ('all',)},
    )


world = swm.World(
    env_name='swm/WalkerDMControl-v0',
    image_shape=(224, 224),
    num_envs=1,
    max_episode_steps=1000,
)
world.set_policy(swm.policy.RandomPolicy())
print('walker')
for i in range(10):
    world.record_video(
        './',
        seed=i,
        options={'variation': ('all',)},
    )
