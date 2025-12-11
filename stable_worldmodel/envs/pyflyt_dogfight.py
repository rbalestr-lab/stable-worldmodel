"""PyFlyt Dogfight Environment Wrapper for stable-worldmodel.

This wrapper integrates the PyFlyt MA Fixedwing Dogfight environment (multi-agent aerial combat)
with the stable-worldmodel framework, providing domain randomization capabilities.

The environment simulates aerial combat between fixed-wing aircraft with the following features:
- Multi-agent: 2 teams of aircraft (default 2 vs 2)
- Physics-based flight dynamics using PyBullet
- Cannon-only combat (no missiles)
- Automatic weapon firing when conditions are met
- Health-based and collision-based termination

Note: This wrapper requires PyFlyt and pybullet to be installed:
    pip install PyFlyt pybullet
"""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

import stable_worldmodel as swm

try:
    from PyFlyt.pz_envs import MAFixedwingDogfightEnvV2
except ImportError as e:
    raise ImportError(
        "PyFlyt is required for this environment. Install it with: pip install PyFlyt pybullet"
    ) from e


DEFAULT_VARIATIONS = ("team1.color", "team2.color")


class PyFlytDogfightEnv(gym.Env):
    """PyFlyt MA Fixedwing Dogfight Environment with domain randomization.

    This environment wraps the PyFlyt multi-agent dogfight simulation and adds
    stable-worldmodel's variation space for domain randomization.

    Args:
        team_size: Number of aircraft per team (default: 2)
        render_mode: Rendering mode ("rgb_array" or None)
        render_resolution: Tuple of (width, height) for rendering (default: (640, 480))
        max_duration_seconds: Maximum episode duration in seconds (default: 60.0)
        damage_per_hit: Damage dealt per hit per physics step (default: 0.003)
        lethal_distance: Maximum range for weapon effectiveness (default: 20.0)
        lethal_angle_radians: Cone of fire width in radians (default: 0.07)
        assisted_flight: Use RPYT commands vs raw actuator commands (default: True)
        aggressiveness: Reward shaping for aggressive behavior (0-1, default: 0.5)
        cooperativeness: Reward shaping for team cooperation (0-1, default: 0.5)
        sparse_reward: Use sparse rewards instead of dense (default: False)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        team_size: int = 2,
        render_mode: str = "rgb_array",
        render_resolution: tuple[int, int] = (640, 480),
        max_duration_seconds: float = 60.0,
        damage_per_hit: float = 0.003,
        lethal_distance: float = 20.0,
        lethal_angle_radians: float = 0.07,
        assisted_flight: bool = True,
        aggressiveness: float = 0.5,
        cooperativeness: float = 0.5,
        sparse_reward: bool = False,
    ):
        super().__init__()
        self.team_size = team_size
        self.render_mode = render_mode
        self.render_resolution = render_resolution
        self.max_duration_seconds = max_duration_seconds
        self.damage_per_hit = damage_per_hit
        self.lethal_distance = lethal_distance
        self.lethal_angle_radians = lethal_angle_radians
        self.assisted_flight = assisted_flight
        self.aggressiveness = aggressiveness
        self.cooperativeness = cooperativeness
        self.sparse_reward = sparse_reward

        # The PyFlyt environment will be created in _setup()
        self.pz_env = None
        self.agents = []
        self.current_agent_idx = 0

        # Define observation and action spaces based on PyFlyt's specs
        # Note: These are approximate and may need adjustment based on actual PyFlyt specs
        # The observation includes: position (3), rotation (4 quat), velocity (3), angular_velocity (3),
        # and relative observations of other agents
        obs_dim = 13 + (team_size * 2 - 1) * 7  # Base obs + relative obs per other agent
        self.observation_space = gym.spaces.Dict({
            "proprio": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
        })

        # Action space: [roll, pitch, yaw, throttle] for RPYT mode
        # or raw actuator commands (elevator, aileron, rudder, throttle)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(4,),
            dtype=np.float32,
        )

        # Define variation space for domain randomization
        self.variation_space = swm.spaces.Dict(
            {
                "team1": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([255, 0, 0], dtype=np.uint8)  # Red team
                        ),
                        "spawn_height_range": swm.spaces.Box(
                            low=50.0,
                            high=150.0,
                            init_value=np.array([50.0, 100.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                        "spawn_radius": swm.spaces.Box(
                            low=10.0,
                            high=50.0,
                            init_value=np.array(30.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "team2": swm.spaces.Dict(
                    {
                        "color": swm.spaces.RGBBox(
                            init_value=np.array([0, 0, 255], dtype=np.uint8)  # Blue team
                        ),
                        "spawn_height_range": swm.spaces.Box(
                            low=50.0,
                            high=150.0,
                            init_value=np.array([50.0, 100.0], dtype=np.float32),
                            shape=(2,),
                            dtype=np.float32,
                        ),
                        "spawn_radius": swm.spaces.Box(
                            low=10.0,
                            high=50.0,
                            init_value=np.array(30.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "environment": swm.spaces.Dict(
                    {
                        "spawn_separation": swm.spaces.Box(
                            low=50.0,
                            high=200.0,
                            init_value=np.array(100.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                        "bounds_size": swm.spaces.Box(
                            low=200.0,
                            high=500.0,
                            init_value=np.array(300.0, dtype=np.float32),
                            shape=(),
                            dtype=np.float32,
                        ),
                    }
                ),
                "lighting": swm.spaces.Dict(
                    {
                        "ambient_color": swm.spaces.RGBBox(
                            init_value=np.array([200, 200, 200], dtype=np.uint8)
                        ),
                    }
                ),
            },
            sampling_order=["environment", "team1", "team2", "lighting"],
        )

        self._goal = None

        # Validate default variation values
        assert self.variation_space.check(), "Default variation values must be valid"

    def _setup(self):
        """Initialize the PyFlyt environment with current variation values."""
        # Close existing environment if any
        if self.pz_env is not None:
            try:
                self.pz_env.close()
            except Exception:
                pass

        # Create new PyFlyt environment
        # Note: PyFlyt Dogfight only supports 'human' render mode which requires a display.
        # In headless environments (Docker), we use None to avoid errors, though this means
        # no visual rendering. The environment still works for state-based observations.
        self.pz_env = MAFixedwingDogfightEnvV2(
            render_mode=None,  # PyFlyt doesn't support headless rgb_array rendering
            team_size=self.team_size,
            max_duration_seconds=self.max_duration_seconds,
            damage_per_hit=self.damage_per_hit,
            lethal_distance=self.lethal_distance,
            lethal_angle_radians=self.lethal_angle_radians,
            assisted_flight=self.assisted_flight,
            aggressiveness=self.aggressiveness,
            cooperativeness=self.cooperativeness,
            sparse_reward=self.sparse_reward,
        )

        # Reset the PettingZoo environment
        self.pz_env.reset()
        self.agents = self.pz_env.agents.copy()
        self.current_agent_idx = 0

    def reset(self, seed=None, options=None):
        """Reset the environment with variation sampling.

        Args:
            seed: Random seed for reproducibility
            options: Dictionary of options, may include "variation" key
                    with list of variation parameters to sample

        Returns:
            observation: Dict with "proprio" and "state" keys
            info: Dictionary with metadata including "goal" image
        """
        super().reset(seed=seed, options=options)

        if hasattr(self, "variation_space"):
            self.variation_space.seed(seed)

        options = options or {}

        # Reset variation space to initial values
        self.variation_space.reset()

        # Get variations to sample
        variations = options.get("variation", DEFAULT_VARIATIONS)

        if not isinstance(variations, Sequence):
            raise ValueError("variation option must be a Sequence containing variation names to sample")

        # Sample the specified variations
        self.variation_space.update(variations)

        # Validate all variations
        assert self.variation_space.check(debug=True), "Variation values must be valid!"

        # Setup/reset the PyFlyt environment
        self._setup()

        # Generate goal image (initial state)
        self._goal = self.render()

        # Get initial observation from the first agent
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute one step in the environment.

        Args:
            action: Action for the current agent (shape: (4,) for RPYT/actuator commands)

        Returns:
            observation: Dict with "proprio" and "state" keys
            reward: Float reward value
            terminated: Boolean indicating if episode ended
            truncated: Boolean indicating if episode was truncated
            info: Dictionary with metadata
        """
        if self.pz_env is None or not self.agents:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Get current agent
        current_agent = self.agents[self.current_agent_idx]

        # Step the current agent in PettingZoo environment
        self.pz_env.step({current_agent: action})

        # After stepping, get the rewards, terminations, truncations from the PettingZoo API
        rewards_dict = getattr(self.pz_env, 'rewards', {})
        terminations_dict = getattr(self.pz_env, 'terminations', {})
        truncations_dict = getattr(self.pz_env, 'truncations', {})

        # Move to next agent (round-robin)
        self.current_agent_idx = (self.current_agent_idx + 1) % len(self.agents)

        # Check if episode is done (all agents terminated or no agents left)
        terminated = all(terminations_dict.values()) if terminations_dict else not bool(self.pz_env.agents)
        truncated = all(truncations_dict.values()) if truncations_dict else False

        # Get observation and reward
        obs = self._get_obs()
        # Sum rewards across all agents for single-agent interface
        reward = sum(rewards_dict.values()) if rewards_dict else 0.0

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get observation from the current or last agent.

        Returns:
            Dict with "proprio" and "state" keys containing observation arrays
        """
        if self.pz_env is None:
            # Return zero observation if environment not initialized
            zero_obs = np.zeros(self.observation_space["proprio"].shape, dtype=np.float32)
            return {"proprio": zero_obs, "state": zero_obs.copy()}

        if not self.agents:
            # Return zero observation if no agents left
            zero_obs = np.zeros(self.observation_space["proprio"].shape, dtype=np.float32)
            return {"proprio": zero_obs, "state": zero_obs.copy()}

        # Get observation from current agent
        current_agent = self.agents[self.current_agent_idx]
        if current_agent in self.pz_env.observations:
            raw_obs = self.pz_env.observations[current_agent]
            # Ensure observation matches our space
            if isinstance(raw_obs, dict):
                # If PyFlyt returns dict, extract relevant parts
                obs_array = np.concatenate([v.flatten() for v in raw_obs.values()], dtype=np.float32)
            else:
                obs_array = np.array(raw_obs, dtype=np.float32).flatten()

            # Pad or truncate to match observation space
            expected_size = self.observation_space["proprio"].shape[0]
            if len(obs_array) < expected_size:
                obs_array = np.pad(obs_array, (0, expected_size - len(obs_array)))
            elif len(obs_array) > expected_size:
                obs_array = obs_array[:expected_size]
        else:
            obs_array = np.zeros(self.observation_space["proprio"].shape, dtype=np.float32)

        return {"proprio": obs_array, "state": obs_array.copy()}

    def _get_info(self):
        """Get info dictionary with metadata.

        Returns:
            Dict with "goal" key containing the goal/initial state image
        """
        info = {"goal": self._goal if self._goal is not None else np.zeros((480, 640, 3), dtype=np.uint8)}

        # Add additional info from PyFlyt if available
        if self.pz_env is not None and hasattr(self.pz_env, "infos"):
            for agent_id, agent_info in self.pz_env.infos.items():
                info[f"agent_{agent_id}"] = agent_info

        return info

    def render(self):
        """Render the current state of the environment.

        Returns:
            RGB image array of shape (height, width, 3) with dtype uint8
        """
        if self.pz_env is None:
            # Return blank image if not initialized
            return np.zeros((*self.render_resolution[::-1], 3), dtype=np.uint8)

        if self.render_mode == "rgb_array":
            # PyFlyt should handle rendering
            try:
                img = self.pz_env.render()
                if img is not None:
                    # Ensure correct format
                    if not isinstance(img, np.ndarray):
                        img = np.array(img)
                    if img.dtype != np.uint8:
                        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                    return img
            except Exception as e:
                print(f"Warning: Rendering failed: {e}")

        # Return blank image as fallback
        return np.zeros((*self.render_resolution[::-1], 3), dtype=np.uint8)

    def close(self):
        """Clean up environment resources."""
        if self.pz_env is not None:
            try:
                self.pz_env.close()
            except Exception:
                pass
            self.pz_env = None
        self.agents = []
