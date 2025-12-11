# PyFlyt MA Fixedwing Dogfight Environment

This directory contains documentation for the PyFlyt MA Fixedwing Dogfight environment integration with stable-worldmodel.

## Overview

The PyFlyt Dogfight environment provides a multi-agent aerial combat simulation using PyBullet physics. The integration adapts PyFlyt's PettingZoo multi-agent API to Gymnasium's single-agent interface with full stable-worldmodel support.

## Implementation

**Location**: `stable_worldmodel/envs/pyflyt_dogfight.py`

**Registration**: `swm/PyFlytDogfight-v0`

**Key Features**:
- Multi-agent to single-agent adapter (round-robin stepping)
- State-based observations (34-dimensional proprio/state vectors)
- RPYT action space (roll, pitch, yaw, throttle)
- Domain randomization (team colors, spawn positions, lighting)
- Reward aggregation across agents

## Usage

```python
import stable_worldmodel as swm

world = swm.World(
    'swm/PyFlytDogfight-v0',
    num_envs=4,
    image_shape=(224, 224),
    max_episode_steps=200,
)

policy = swm.policy.RandomPolicy(42)
world.set_policy(policy)

# Reset with domain randomization
world.reset(seed=42, options={'variation': ['all']})

# Record dataset
world.record_dataset('dogfight_data', episodes=100, seed=42)
```

## Configurable Parameters

- `team_size`: Aircraft per team (default: 2)
- `damage_per_hit`: Weapon damage (default: 0.003)
- `lethal_distance`: Weapon range (default: 20.0)
- `lethal_angle_radians`: Weapon cone (default: 0.07)
- `assisted_flight`: RPYT vs raw controls (default: True)
- `aggressiveness`: Reward shaping (default: 0.5)
- `cooperativeness`: Team cooperation (default: 0.5)
- `sparse_reward`: Dense vs sparse rewards (default: False)
- `max_duration_seconds`: Episode time limit (default: 60.0)

## Variation Space

Domain randomization parameters:
- Team 1/2 colors, spawn heights, spawn radii
- Environment bounds size, spawn separation
- Lighting ambient color

## Docker Setup

Required for ARM64 macOS due to pybullet compilation issues:

```bash
docker-compose build
docker-compose up -d
docker-compose exec pyflyt pytest stable_worldmodel/tests/test_dogfight_validation.py -v
```

## Testing

**Validation Tests**: `stable_worldmodel/tests/test_dogfight_validation.py`

Run tests:
```bash
pytest stable_worldmodel/tests/test_dogfight_validation.py -v
```

All tests pass successfully.

## Important Limitation

**Visual Rendering**: PyFlyt Dogfight only supports 'human' render mode (GUI window). Headless pixel rendering is not available. The environment works perfectly for state-based observations but cannot provide pixel observations in headless environments like Docker.

For pixel-based world model training, use other stable-worldmodel environments (PushT, TwoRoom, etc.) that support 'rgb_array' render mode.

## Documentation Files

- `PYFLYT_ENVIRONMENT_STATUS.md`: Complete implementation status and features
- `VISUAL_SETUP.md`: Explanation of visual rendering limitations
- `CONDA_SETUP_GUIDE.md`: Local conda environment setup

## Dependencies

- PyFlyt
- pybullet >= 3.2.7
- numpy < 1.25
- scipy < 1.13

All dependencies are installed automatically via Docker.
