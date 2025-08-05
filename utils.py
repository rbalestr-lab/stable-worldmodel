from .world import World

import requests
from PIL import Image
from io import BytesIO


def create_pil_image_from_url(image_url):
    """
    Creates a PIL Image object from a given image URL.

    Args:
        image_url (str): The URL of the image.

    Returns:
        PIL.Image.Image: The PIL Image object, or None if an error occurs.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        image_data = BytesIO(response.content)
        img = Image.open(image_data)
        return img
    except requests.exceptions.RequestException as e:
        print(image_url)
        print(f"Error downloading image from URL: {e}")
        return None
    except IOError as e:
        print(image_url)
        print(f"Error opening image data with PIL: {e}")
        return None


def set_state(world: World, state):
    if isinstance(world, World):
        envs = world.envs.envs
    else:
        envs = [world.env]
        state = [state]
    for i, env in enumerate(envs):
        if hasattr(env.unwrapped, "state"):
            env.unwrapped.state = state[i]
        elif hasattr(env.unwrapped, "s"):
            env.unwrapped.s = state[i]
        elif hasattr(env.unwrapped, "sim") and hasattr(
            env.unwrapped.sim, "set_state_from_flattened"
        ):
            env.unwrapped.sim.set_state_from_flattened(state[i].view(-1))


if __name__ == "__main__":
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
