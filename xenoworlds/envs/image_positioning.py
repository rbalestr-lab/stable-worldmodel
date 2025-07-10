from typing import Optional
import numpy as np
import gymnasium as gym
import pygame
from PIL import Image, ImageOps

COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


class ImagePositinoing(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        resolution: int,
        images: list[Image],
        render_mode: str = None,
    ):
        self.resolution = resolution

        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "current_locations": gym.spaces.Box(
                    0, 1, shape=(len(images), 2), dtype=float
                ),
                "current_rotations": gym.spaces.Box(
                    0, 2 * np.pi, shape=(len(images), 1), dtype=float
                ),
                "target_locations": gym.spaces.Box(
                    0, 1, shape=(len(images), 2), dtype=float
                ),
                "target_rotations": gym.spaces.Box(
                    0, 2 * np.pi, shape=(len(images), 1), dtype=float
                ),
            }
        )

        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._current_locations = np.empty(
            self.observation_space["current_locations"].shape, dtype=np.float
        )
        self._target_locations = np.array(
            self.observation_space["target_locations"].shape, dtype=np.float
        )
        self._current_rotations = np.empty(
            self.observation_space["current_rotations"].shape, dtype=np.float
        )
        self._target_rotations = np.array(
            self.observation_space["target_rotations"].shape, dtype=np.float
        )

        # Define what actions are available (4 directions)
        self.action_space = gym.spaces.Dict(
            {
                "delta_locations": gym.spaces.Box(
                    low=-0.1, high=0.1, shape=(len(images), 2)
                ),
                "delta_rotations": gym.spaces.Box(
                    low=-0.1, high=0.1, shape=(len(images), 1)
                ),
            }
        )
        self.images = [
            ImageOps.expand(img, border=10, fill=c).convert("RGBA")
            for img, c in zip(images, COLORS)
        ]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {
            "current_locations": self._current_locations,
            "current_rotations": self._current_rotations,
            "target_locations": self._target_locations,
            "target_rotations": self._target_rotations,
        }

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "location_distance": np.linalg.norm(
                self._current_locations - self._target_locations, ord=1
            ),
            "rotation_distance": np.linalg.norm(
                self._current_rotations - self._target_rotations, ord=1
            ),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-3) to a movement direction

        # Update agent position, ensuring it stays within grid bounds
        # np.clip prevents the agent from walking off the edge
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # get the image
        new_background = Image.new(
            "RGBA", (self.resolution, self.resolution), (255, 255, 255)
        )
        for i, img in enumerate(self.images):
            new_background.paste(
                img.rotate(self._current_rotations[i, 0]), self._current_locations[i]
            )

        # get the surface
        # Get image data, size, and mode from PIL Image
        image_bytes = new_background.tobytes()
        image_size = new_background.size
        image_mode = new_background.mode

        # Create a Pygame Surface from the PIL image data
        pygame_surface = pygame.image.frombytes(image_bytes, image_size, image_mode)
        self.window.blit(pygame_surface, (0, 0))  # Blit at position (0,0)

        # Update the display
        pygame.display.flip()
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    