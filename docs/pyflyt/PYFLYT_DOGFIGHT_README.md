# PyFlyt Dogfight Environment Integration

This document describes the integration of the PyFlyt MA Fixedwing Dogfight environment into stable-worldmodel.

## Overview

The PyFlyt Dogfight environment is a multi-agent aerial combat simulator featuring fixed-wing aircraft. This integration provides a wrapper that adds stable-worldmodel's domain randomization capabilities to the PyFlyt environment.

## Files Added

### 1. [stable_worldmodel/envs/pyflyt_dogfight.py](stable_worldmodel/envs/pyflyt_dogfight.py)

Main environment wrapper that implements:
- **Observation Space**: Dict with "proprio" and "state" keys containing flight dynamics
- **Action Space**: Box(4,) for [roll, pitch, yaw, throttle] commands
- **Variation Space**: Domain randomization for:
  - Team colors (team1, team2)
  - Spawn positions (height range, radius, separation)
  - Environment bounds
  - Lighting/ambient colors

### 2. [stable_worldmodel/envs/__init__.py](stable_worldmodel/envs/__init__.py)

Updated to register the new environment:
```python
register(
    id="swm/PyFlytDogfight-v0",
    entry_point="stable_worldmodel.envs.pyflyt_dogfight:PyFlytDogfightEnv",
)
```

## Environment Features

### Multi-Agent Combat
- Default: 2 vs 2 teams (configurable via `team_size` parameter)
- Physics-based flight dynamics using PyBullet
- Cannon-only combat (no missiles)
- Automatic weapon firing when conditions are met

### Termination Conditions
Agents are eliminated when they:
1. Hit anything (collision)
2. Fly out of bounds
3. Lose all health

Episodes also terminate after `max_duration_seconds` (default: 60s).

### Configuration Parameters

```python
env = gym.make("swm/PyFlytDogfight-v0",
    team_size=2,                      # Aircraft per team
    render_mode="rgb_array",          # Rendering mode
    render_resolution=(640, 480),     # Render size
    max_duration_seconds=60.0,        # Max episode duration
    damage_per_hit=0.003,             # Damage per hit
    lethal_distance=20.0,             # Weapon range
    lethal_angle_radians=0.07,        # Cone of fire
    assisted_flight=True,             # RPYT vs actuator commands
    aggressiveness=0.5,               # Reward shaping (0-1)
    cooperativeness=0.5,              # Team cooperation reward (0-1)
    sparse_reward=False,              # Dense vs sparse rewards
)
```

### Variation Space

The environment supports domain randomization through the `variation_space`:

```python
variation_space = swm.spaces.Dict({
    "team1": {
        "color": RGBBox (default: red [255, 0, 0])
        "spawn_height_range": Box([50.0, 100.0])
        "spawn_radius": Box(30.0)
    },
    "team2": {
        "color": RGBBox (default: blue [0, 0, 255])
        "spawn_height_range": Box([50.0, 100.0])
        "spawn_radius": Box(30.0)
    },
    "environment": {
        "spawn_separation": Box(100.0)  # Distance between teams
        "bounds_size": Box(300.0)        # Arena size
    },
    "lighting": {
        "ambient_color": RGBBox ([200, 200, 200])
    },
})
```

## Installation Requirements

### Dependencies

1. **PyFlyt**: Multi-agent flight simulator
   ```bash
   pip install PyFlyt
   ```

2. **pybullet**: Physics engine (required by PyFlyt)
   ```bash
   # On macOS with ARM64, pybullet may need to be compiled from source
   # Install cmake first:
   brew install cmake

   # Then install pybullet:
   pip install pybullet
   ```

   **Note**: pybullet compilation on macOS ARM64 can be challenging. If you encounter build errors, you may need to:
   - Ensure Xcode Command Line Tools are installed: `xcode-select --install`
   - Try conda: `conda install -c conda-forge pybullet`
   - Or use a pre-compiled wheel if available

3. **FFmpeg**: Required for video encoding (torchcodec dependency)
   ```bash
   brew install ffmpeg
   ```

4. **NumPy compatibility**: PyFlyt requires numpy<2.0.0
   ```bash
   pip install "numpy<2.0.0"
   ```

### Full Installation

```bash
# Install system dependencies
brew install cmake ffmpeg

# Install Python packages
pip install PyFlyt pybullet
pip install "numpy<2.0.0"  # Ensure numpy compatibility

# Install stable-worldmodel in editable mode
pip install -e .
```

## Usage Example

```python
import gymnasium as gym
import stable_worldmodel as swm

# Create environment
env = gym.make("swm/PyFlytDogfight-v0", render_mode="rgb_array")

# Create world wrapper
world = swm.World(
    "swm/PyFlytDogfight-v0",
    num_envs=1,
    image_shape=(224, 224),
    max_episode_steps=100,
    render_mode="rgb_array",
)

# Set random policy
policy = swm.policy.RandomPolicy(seed=42)
world.set_policy(policy)

# Record dataset
world.record_dataset("dogfight_data", episodes=10, seed=42)

# Record video
world.record_video("dogfight_video.mp4", seed=42)
```

### With Variation Sampling

```python
# Reset with specific variations
obs, info = env.reset(
    seed=42,
    options={
        "variation": ("team1.color", "team2.color", "environment.spawn_separation")
    }
)

# Step through environment
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

## Testing

### Run Validation Tests

```bash
# Run check_world.py to validate the environment
pytest stable_worldmodel/tests/check_world.py::test_each_env[swm/PyFlytDogfight-v0] -v
```

This will test:
- Environment creation
- Episode rollout
- Dataset recording
- Video generation
- Observation/action space compliance
- Deterministic behavior with seeds

## Implementation Notes

### Multi-Agent to Single-Agent Wrapper

The PyFlyt environment is a PettingZoo multi-agent environment. This wrapper adapts it to Gymnasium's single-agent interface by:
1. Stepping through agents in round-robin fashion
2. Summing rewards across all agents
3. Terminating when all agents are done

### Observation Space

The observation includes:
- Base dynamics: position (3), rotation quaternion (4), velocity (3), angular velocity (3)
- Relative observations of other agents (position, velocity, etc.)

Total dimension: `13 + (team_size * 2 - 1) * 7`

### Action Space

For assisted flight mode (default):
- `action[0]`: Roll command [-1, 1]
- `action[1]`: Pitch command [-1, 1]
- `action[2]`: Yaw command [-1, 1]
- `action[3]`: Throttle [0, 1]

## Known Issues & Limitations

1. **pybullet Installation**: Compilation on macOS ARM64 can fail. This is a known issue with the pybullet package.

2. **NumPy Version Conflicts**: PyFlyt requires numpy<2.0, which conflicts with some other packages (jax, opencv-python). You may see dependency warnings.

3. **Multi-Agent Adaptation**: The current wrapper treats the multi-agent environment as single-agent by cycling through agents. For true multi-agent training, consider using the PyFlyt environment directly with PettingZoo-compatible algorithms.

4. **Rendering**: PyFlyt's rendering may not work in headless environments. Ensure you have a display or use virtual framebuffer (xvfb) on Linux.

## References

- **PyFlyt Documentation**: https://taijunjet.com/PyFlyt/
- **Dogfight Environment**: https://taijunjet.com/PyFlyt/documentation/pz_envs/ma_fixedwing_dogfight_env.html
- **PettingZoo**: https://pettingzoo.farama.org/
- **stable-worldmodel**: Current repository

## Future Improvements

1. **True Multi-Agent Support**: Implement proper multi-agent interface compatible with PettingZoo
2. **Enhanced Variation Space**: Add more domain randomization options (weather, wind, terrain)
3. **Curriculum Learning**: Implement difficulty progression (team size, bounds, etc.)
4. **Expert Demonstrations**: Add scripted policies for behavior cloning
5. **Metrics & Analysis**: Add combat-specific metrics (kill/death ratio, accuracy, etc.)

## Contributing

When modifying this environment, please ensure:
1. Tests pass: `pytest stable_worldmodel/tests/check_world.py`
2. Variation space remains valid
3. Documentation is updated
4. Follows stable-worldmodel patterns (see [CONTRIBUTING.md](CONTRIBUTING.md))
