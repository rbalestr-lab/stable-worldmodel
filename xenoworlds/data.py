import minari
from .utils import set_state
from tqdm.rich import tqdm


def cache_minari_with_pixels(name, env):
    dataset = minari.load_dataset(name)
    N = len(dataset)
    for i, episode_data in enumerate(dataset.iterate_episodes()):
        env.reset()
        observations = episode_data.observations
        actions = episode_data.actions
        rewards = episode_data.rewards
        terminations = episode_data.terminations
        truncations = episode_data.truncations
        infos = episode_data.infos
        pixels = []
        assert "observation" in observations
        for action, observation in zip(actions, observations["observation"]):
            set_state(env, observation)
            env.step(action)
            pixels.append(env.render())


if __name__ == "__main__":
    import gymnasium as gym
    import gymnasium_robotics
    import xenoworlds
    from gymnasium.wrappers import RecordVideo

    env = gym.make("AntMaze_Large-v1", render_mode="rgb_array")
    env = RecordVideo(
        env, video_folder="test_videos_data_caching", episode_trigger=lambda x: True
    )
    data = xenoworlds.data.cache_minari_with_pixels("D4RL/antmaze/large-play-v1", env)
