import os
import shutil

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
import pytest

import stable_worldmodel as swm


os.environ["MUJOCO_GL"] = "egl"
swm_env = [env_id for env_id in gym.envs.registry.keys() if env_id.startswith("swm/")]


@pytest.fixture(scope="session")
def temp_path(tmp_path_factory, request):
    tmp_dir = tmp_path_factory.mktemp("data")

    def cleanup():
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    request.addfinalizer(cleanup)
    return tmp_dir


@pytest.mark.parametrize("env", swm_env)
def test_each_env(env, temp_path):
    EPISODE_LENGTH = 10

    world = swm.World(
        env,
        num_envs=1,
        image_shape=(224, 224),
        max_episode_steps=EPISODE_LENGTH,
        render_mode="rgb_array",
        verbose=0,
    )

    print(f"Testing env {env} with temp path {temp_path}")

    ds_name = f"tmp-{env.replace('swm/', '').lower()}"

    world.set_policy(swm.policy.RandomPolicy())

    world.record_dataset(ds_name, episodes=1, seed=2347, cache_dir=temp_path)
    world.record_video(f"{temp_path}/{ds_name}", seed=2347)
    world.record_video_from_dataset(f"{temp_path}/{ds_name}", ds_name, episode_idx=0, cache_dir=temp_path)

    # assert all files are created
    assert os.path.exists(f"{temp_path}/{ds_name}/dataset_info.json")
    assert os.path.exists(f"{temp_path}/{ds_name}/env_0.mp4")
    assert os.path.exists(f"{temp_path}/{ds_name}/episode_0.mp4")

    # load both video
    recorded_video = iio.imread(f"{temp_path}/{ds_name}/env_0.mp4", index=None)
    dataset_video = iio.imread(f"{temp_path}/{ds_name}/episode_0.mp4", index=None)

    assert isinstance(recorded_video, np.ndarray)
    assert isinstance(dataset_video, np.ndarray)
    assert recorded_video.dtype == np.uint8
    assert dataset_video.dtype == np.uint8
    assert recorded_video.shape[0] == EPISODE_LENGTH
    assert recorded_video.shape[3] == 3  # RGB channels
    assert recorded_video.shape == dataset_video.shape  # both videos should have the same shape
    # assert np.array_equal(recorded_video, dataset_video)

    return
